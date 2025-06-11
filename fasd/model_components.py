import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .mlp import MLP


class FASD:

    def __init__(
        self,
        task: str,
        target_dim: int,
        input_dims: list[int],
        predictor_config: dict,
        vae_config: dict,
        decoder_config: dict,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor_config = predictor_config
        self.vae_config = vae_config
        self.decoder_config = decoder_config

        self.task = task

        self.fasd_predictor = Predictor(
            task=task,
            input_dim=sum(input_dims),
            output_dim=target_dim,
            predictor_config=predictor_config,
            device=self.device,
            random_state=random_state,
        )
        self.fasd_decoder = Decoder(
            input_dim=self.predictor_config["representation_dim"],
            output_dims=input_dims,
            decoder_config=decoder_config,
            device=self.device,
            random_state=random_state,
        )
        self.fasd_generator = VAE(
            input_dims=[1]
            * self.predictor_config[
                "representation_dim"
            ],  # continuous representations as input to the VAE
            vae_config=vae_config,
            device=self.device,
            random_state=random_state,
        )

    def _train(self, X: pd.DataFrame, y: pd.Series):
        self.X_cols = X.columns
        self.y_col = y.name
        # turn input into torch tensors
        Xt = torch.from_numpy(X.to_numpy()).float()
        yt = torch.from_numpy(y.to_numpy()).float()

        # train the predictor
        self.fasd_predictor._train(Xt, yt)

        # extract representations from the predictor by passing X through encoder portion
        Xt_rep = self.fasd_predictor.encoder(Xt)
        if self.predictor_config["representations_nonlin"].lower() == "none":
            Xt_rep = torch.clamp(Xt_rep, -1, 1)

        # train the generator (self-supervised on representations)
        self.fasd_generator._train(Xt_rep)

        # train the decoder (predict original data from representations)
        self.fasd_decoder._train(Xt_rep, Xt)

        return self

    def _generate(self, n: int):

        # set to inference mode
        with torch.no_grad():
            # generate representations from generator
            Xt_syn_rep = self.fasd_generator._generate(n)

            # decode representations to original data space using decoder
            Xt_syn = self.fasd_decoder(Xt_syn_rep)

            # use predictor to predict y
            yt_syn = self.fasd_predictor.predictor(Xt_syn_rep)

        # reshape categorical predictions to labels
        if self.task == "classification":
            yt_syn = torch.argmax(yt_syn, dim=1)

        # turn X and y into dataframes (note that before this all data was still on-device)
        X = pd.DataFrame(Xt_syn.detach().cpu().numpy(), columns=self.X_cols)
        y = pd.Series(yt_syn.detach().cpu().numpy(), name=self.y_col)

        return X, y


class Predictor(nn.Module):

    def __init__(
        self,
        task: str,  # "classification" or "regression"
        input_dim: int,
        output_dim: int,
        predictor_config: dict,
        device,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.task = task
        self.input_dim = input_dim
        self.random_state = random_state
        self.predictor_config = predictor_config

        encoder_output_config = [
            (1, predictor_config["representations_nonlin"])
            for x in list(range(predictor_config["hidden_layers"][-1]))
        ]
        self.encoder = MLP(
            input_dim=self.input_dim,
            output_config=encoder_output_config,
            hidden_dims=predictor_config["hidden_layers"][
                :-1
            ],  # last one is output layer
            nonlinearity=predictor_config["nonlin"],
            dropout=predictor_config["dropout"],
            residual=False,
            batch_norm=False,
        ).to(self.device)

        if self.task == "classification":
            output_nonlin = "softmax"
            self.criterion = nn.CrossEntropyLoss()
        else:
            output_nonlin = "tanh"
            self.criterion = nn.MSELoss()

        self.predictor = MLP(
            input_dim=predictor_config["hidden_layers"][-1],
            output_config=[(output_dim, output_nonlin)],
            hidden_dims=[],  # only output layer
            nonlinearity="none",
            dropout=0,
            residual=False,
            batch_norm=False,
        ).to(self.device)

    def forward(self, X: torch.Tensor):
        X = self.encoder(X)
        # ensure reasonable range of representations if no hidden activation
        if self.predictor_config["representations_nonlin"].lower() == "none":
            X = torch.clamp(X, -1, 1)
        return self.predictor(X)

    def _train(self, X: torch.Tensor, y: torch.Tensor):
        # create (stratified) validation split
        X_pd = pd.DataFrame(X.detach().cpu().numpy(), columns=list(range(X.shape[-1])))
        y_pd = pd.Series(y.detach().cpu().numpy().squeeze(), name="target")
        stratify = None
        if self.task == "classification":
            stratify = y_pd
        X, X_val, y, y_val = train_test_split(
            X_pd,
            y_pd,
            stratify=stratify,
            train_size=0.8,
            random_state=self.random_state,
        )

        # create tensors and send to device
        Xt = self._check_tensor(X)
        Xt_val = self._check_tensor(X_val)
        yt = self._check_tensor(y)
        yt_val = self._check_tensor(y_val)

        # create dataloaders
        loader = DataLoader(
            dataset=TensorDataset(Xt, yt),
            batch_size=self.predictor_config["batch_size"],
            pin_memory=False,
        )
        val_loader = DataLoader(
            dataset=TensorDataset(Xt_val, yt_val),
            batch_size=self.predictor_config["batch_size"],
            pin_memory=False,
        )

        # --- perform the training loop
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.predictor_config["lr"],
            weight_decay=self.predictor_config["weight_decay"],
            betas=self.predictor_config["opt_betas"],
        )

        best_state_dict = None
        best_loss = float("inf")
        patience = 0
        for epoch in tqdm(range(self.predictor_config["epochs"])):
            self.train()
            train_loss = 0
            for inputs, targets in loader:
                outputs = self.forward(inputs)
                if self.task == "classification":
                    targets = targets.long()
                loss = self.criterion(outputs, targets)
                optimizer.zero_grad()
                if self.predictor_config["clipping_value"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.predictor_config["clipping_value"]
                    )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = self.validate(val_loader)

            if val_loss >= best_loss:
                patience += 1
            else:
                best_loss = val_loss
                best_state_dict = self.state_dict()
                patience = 0

            if (
                patience >= self.predictor_config["patience"]
                and epoch >= self.predictor_config["n_iter_min"]
            ):
                break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return self

    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                if self.task == "classification":
                    targets = targets.long()
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).float().to(self.device)


class Decoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dims: list[int],  # list of dimensions of each original feature
        decoder_config: dict,
        device,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.device = device
        self.random_state = random_state
        self.decoder_config = decoder_config

        decoder_output_nonlin = [
            "gumbel_softmax" if x > 1 else "tanh" for x in self.output_dims
        ]
        self.decoder_output_config = [
            (d, n) for d, n in zip(self.output_dims, decoder_output_nonlin)
        ]

        self.decoder = MLP(
            input_dim=self.input_dim,
            output_config=self.decoder_output_config,
            hidden_dims=self.decoder_config["hidden_layers"],
            nonlinearity=self.decoder_config["nonlin"],
            dropout=self.decoder_config["dropout"],
            residual=False,
            batch_norm=False,
        ).to(self.device)

    def forward(self, X: torch.Tensor):
        return self.decoder(X)

    def _train(self, X: torch.Tensor, y: torch.Tensor):
        # X contains representations, y contains original data

        # create validation split
        X_pd = pd.DataFrame(X.detach().cpu().numpy(), columns=list(range(X.shape[-1])))
        y_pd = pd.DataFrame(y.detach().cpu().numpy(), columns=list(range(y.shape[-1])))

        X, X_val, y, y_val = train_test_split(
            X_pd,
            y_pd,
            train_size=0.8,
            random_state=self.random_state,
        )

        # create tensors and send to device
        Xt = self._check_tensor(X)
        Xt_val = self._check_tensor(X_val)
        yt = self._check_tensor(y)
        yt_val = self._check_tensor(y_val)

        # create dataloaders
        loader = DataLoader(
            dataset=TensorDataset(Xt, yt),
            batch_size=self.decoder_config["batch_size"],
            pin_memory=False,
        )
        val_loader = DataLoader(
            dataset=TensorDataset(Xt_val, yt_val),
            batch_size=self.decoder_config["batch_size"],
            pin_memory=False,
        )

        # --- perform the training loop
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.decoder_config["lr"],
            weight_decay=self.decoder_config["weight_decay"],
            betas=self.decoder_config["opt_betas"],
        )

        best_state_dict = None
        best_loss = float("inf")
        patience = 0
        for epoch in tqdm(range(self.decoder_config["epochs"])):
            self.train()
            train_loss = 0
            for inputs, targets in loader:
                outputs = self.forward(inputs)
                loss = self._loss_function(outputs, targets)
                optimizer.zero_grad()
                if self.decoder_config["clipping_value"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.decoder_config["clipping_value"]
                    )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = self.validate(val_loader)

            if val_loss >= best_loss:
                patience += 1
            else:
                best_loss = val_loss
                best_state_dict = self.state_dict()
                patience = 0

            if (
                patience >= self.decoder_config["patience"]
                and epoch >= self.decoder_config["n_iter_min"]
            ):
                break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return self

    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                loss = self._loss_function(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).float().to(self.device)

    def _loss_function(
        self,
        reconstructed: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:

        # code adapted from Synthcity library

        step = 0
        loss = []
        for length, activation in self.decoder_output_config:
            step_end = step + length
            # reconstructed is after the activation
            if "softmax" in activation:
                discr_loss = nn.NLLLoss(reduction="sum")(
                    torch.log(reconstructed[:, step:step_end] + 1e-8),
                    torch.argmax(real[:, step:step_end], dim=-1),
                )
                loss.append(discr_loss)
            else:
                diff = reconstructed[:, step:step_end] - real[:, step:step_end]
                cont_loss = (
                    50 * diff**2
                ).sum()  # inspired from Synthcity's loss function

                loss.append(cont_loss)
            step = step_end

        reconstruction_loss = torch.sum(torch.stack(loss)) / real.shape[0]

        if torch.isnan(reconstruction_loss):
            raise RuntimeError("NaNs detected in the reconstruction_loss")

        return reconstruction_loss


class VAE(nn.Module):

    def __init__(
        self,
        input_dims: list[int],  # list of dimensions of each original feature
        vae_config: dict,
        device,
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dims = input_dims
        self.device = device
        self.vae_config = vae_config
        self.random_state = random_state

        encoder_output_config = [
            (1, self.vae_config["encoder_nonlin"])
            for x in list(range(self.vae_config["encoder_hidden_layers"][-1]))
        ]
        self.encoder = MLP(
            input_dim=len(input_dims),
            output_config=encoder_output_config,
            hidden_dims=self.vae_config["encoder_hidden_layers"],
            nonlinearity=self.vae_config["encoder_nonlin"],
            dropout=self.vae_config["encoder_dropout"],
            residual=False,
            batch_norm=False,
        ).to(self.device)
        self.mu_fc = nn.Linear(
            self.vae_config["encoder_hidden_layers"][-1],
            self.vae_config["embedding_size"],
        ).to(self.device)
        self.logvar_fc = nn.Linear(
            self.vae_config["encoder_hidden_layers"][-1],
            self.vae_config["embedding_size"],
        ).to(self.device)

        decoder_output_nonlin = [
            "gumbel_softmax" if x > 1 else "tanh" for x in self.input_dims
        ]
        self.decoder_output_config = [
            (d, n) for d, n in zip(self.input_dims, decoder_output_nonlin)
        ]
        self.decoder = MLP(
            input_dim=self.vae_config["embedding_size"],
            output_config=self.decoder_output_config,
            hidden_dims=self.vae_config["decoder_hidden_layers"],
            nonlinearity=self.vae_config["decoder_nonlin"],
            dropout=self.vae_config["decoder_dropout"],
            residual=False,
            batch_norm=False,
        ).to(self.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X: torch.Tensor):
        # input to representation spaceW
        X = self.encoder(X)
        mu = self.mu_fc(X)
        logvar = self.logvar_fc(X)
        logvar = torch.clamp(logvar, -30, 20)

        # reparametrization trick
        z = self.reparameterize(mu, logvar)

        # decode to data space
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def _train(self, X: torch.Tensor):
        # minmax scale VAE input ->  relevant when representation_nonlin is not "tanh" or "none" (in the latter case it's clamped to [-1,1])
        self.min_x, self.max_x = (
            X.min(dim=0, keepdim=True).values,
            X.max(dim=0, keepdim=True).values,
        )
        denom = self.max_x - self.min_x
        denom[denom == 0] = 1
        X = 2 * (X - self.min_x) / denom - 1
        # create validation split
        X_pd = pd.DataFrame(X.detach().cpu().numpy(), columns=list(range(X.shape[-1])))
        X, X_val = train_test_split(
            X_pd,
            train_size=0.8,
            random_state=self.random_state,
        )

        # create tensors and send to device
        Xt = self._check_tensor(X)
        Xt_val = self._check_tensor(X_val)

        # create dataloaders
        loader = DataLoader(
            dataset=TensorDataset(Xt),
            batch_size=self.vae_config["batch_size"],
            pin_memory=False,
        )
        val_loader = DataLoader(
            dataset=TensorDataset(Xt_val),
            batch_size=self.vae_config["batch_size"],
            pin_memory=False,
        )

        # --- perform the training loop
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.vae_config["lr"],
            weight_decay=self.vae_config["weight_decay"],
            betas=self.vae_config["opt_betas"],
        )

        best_state_dict = None
        best_loss = float("inf")
        patience = 0
        for epoch in tqdm(range(self.vae_config["epochs"])):
            self.train()
            train_loss = 0
            for inputs in loader:
                inputs = inputs[0]
                outputs, mu, logvar = self.forward(inputs)
                loss = self._loss_function(outputs, inputs, mu, logvar)
                optimizer.zero_grad()
                if self.vae_config["clipping_value"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.vae_config["clipping_value"]
                    )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = self.validate(val_loader)

            if val_loss >= best_loss:
                patience += 1
            else:
                best_loss = val_loss
                best_state_dict = self.state_dict()
                patience = 0

            if (
                patience >= self.vae_config["patience"]
                and epoch >= self.vae_config["n_iter_min"]
            ):
                break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

    def _generate(self, n: int):
        batches = n // self.vae_config["batch_size"] + 1
        data = []

        for idx in range(batches):
            mean = torch.zeros(
                self.vae_config["batch_size"], self.vae_config["embedding_size"]
            )
            std = torch.ones(
                self.vae_config["batch_size"], self.vae_config["embedding_size"]
            )
            noise = torch.normal(mean=mean, std=std).to(self.device)
            fake = self.decoder(noise)
            data.append(fake)

        data = torch.cat(data, dim=0)
        data = data[:n]
        # reverse the [-1,1] scaling
        denom = self.max_x - self.min_x
        denom[denom == 0] = 1
        data = 2 * (data - self.min_x) / denom - 1
        data = ((data + 1) * denom / 2) + self.min_x
        return data

    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs[0]
                outputs, mu, logvar = self.forward(inputs)
                loss = self._loss_function(outputs, inputs, mu, logvar)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def _loss_function(
        self,
        reconstructed: torch.Tensor,
        real: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        # code adapted from Synthcity library

        step = 0

        loss = []
        for length, activation in self.decoder_output_config:
            step_end = step + length
            # reconstructed is after the activation
            if "softmax" in activation:
                discr_loss = nn.NLLLoss(reduction="sum")(
                    torch.log(reconstructed[:, step:step_end] + 1e-8),
                    torch.argmax(real[:, step:step_end], dim=-1),
                )
                loss.append(discr_loss)
            else:
                diff = reconstructed[:, step:step_end] - real[:, step:step_end]
                cont_loss = (50 * diff**2).sum()

                loss.append(cont_loss)
            step = step_end

        if step != reconstructed.size()[1]:
            raise RuntimeError(
                f"Invalid reconstructed features. Expected {step}, got {reconstructed.shape}"
            )

        reconstruction_loss = torch.sum(torch.stack(loss)) / real.shape[0]
        KLD_loss = (-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())) / real.shape[0]

        if torch.isnan(reconstruction_loss):
            raise RuntimeError("NaNs detected in the reconstruction_loss")
        if torch.isnan(KLD_loss):
            raise RuntimeError("NaNs detected in the KLD_loss")

        return reconstruction_loss * self.vae_config["loss_factor"] + KLD_loss

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).float().to(self.device)
