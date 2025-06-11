import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


from .base import BaseGenerator
from .model_components import FASD


class TabularFASD(BaseGenerator):
    """
    Generator object to train a Fidelity Agnostic Synthetic Data algorithm and synthesize new data.
    """

    def __init__(
        self,
        target_column: str,
        representation_dim: int = 100,
        predictor_hidden_layers: list = [],
        predictor_nonlin: str = "relu",
        representations_nonlin: str = "tanh",
        predictor_dropout: float = 0,
        predictor_epochs: int = 100,
        predictor_batch_size: int = 512,
        predictor_lr: float = 1e-3,
        predictor_opt_betas: tuple = (0.9, 0.999),
        predictor_weight_decay: float = 1e-3,
        predictor_early_stopping: bool = True,
        predictor_n_iter_min: int = 10,
        predictor_patience: int = 50,
        predictor_clipping_value: float = 0,
        decoder_hidden_layers: list = [100],
        decoder_nonlin: str = "relu",
        decoder_dropout: float = 0,
        decoder_epochs: int = 100,
        decoder_batch_size: int = 512,
        decoder_lr: float = 1e-3,
        decoder_opt_betas: tuple = (0.9, 0.999),
        decoder_weight_decay: float = 1e-3,
        decoder_early_stopping: bool = True,
        decoder_n_iter_min: int = 10,
        decoder_patience: int = 50,
        decoder_clipping_value: float = 0,
        vae_encoder_hidden_layers: list = [128, 128],
        vae_encoder_nonlin: str = "relu",
        vae_encoder_dropout: float = 0,
        vae_decoder_hidden_layers: list = [128, 128],
        vae_decoder_nonlin: str = "relu",
        vae_decoder_dropout: float = 0,
        vae_embedding_size: int = 128,
        vae_loss_factor: int = 1,
        vae_epochs: int = 100,
        vae_batch_size: int = 512,
        vae_lr: float = 1e-3,
        vae_opt_betas: tuple = (0.9, 0.999),
        vae_weight_decay: float = 1e-3,
        vae_early_stopping: bool = True,
        vae_n_iter_min: int = 10,
        vae_patience: int = 50,
        vae_clipping_value: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_column = target_column
        predictor_hidden_layers.append(representation_dim)

        self.predictor_config = dict(
            representation_dim=representation_dim,
            hidden_layers=predictor_hidden_layers,
            epochs=predictor_epochs,
            clipping_value=predictor_clipping_value,
            patience=predictor_patience,
            n_iter_min=predictor_n_iter_min,
            early_stopping=predictor_early_stopping,
            weight_decay=predictor_weight_decay,
            opt_betas=predictor_opt_betas,
            batch_size=predictor_batch_size,
            nonlin=predictor_nonlin,
            representations_nonlin=representations_nonlin,
            lr=predictor_lr,
            dropout=predictor_dropout,
        )

        self.vae_config = dict(
            epochs=vae_epochs,
            clipping_value=vae_clipping_value,
            patience=vae_patience,
            n_iter_min=vae_n_iter_min,
            early_stopping=vae_early_stopping,
            weight_decay=vae_weight_decay,
            opt_betas=vae_opt_betas,
            lr=vae_lr,
            batch_size=vae_batch_size,
            encoder_hidden_layers=vae_encoder_hidden_layers,
            encoder_nonlin=vae_encoder_nonlin,
            encoder_dropout=vae_encoder_dropout,
            decoder_hidden_layers=vae_decoder_hidden_layers,
            decoder_nonlin=vae_decoder_nonlin,
            decoder_dropout=vae_decoder_dropout,
            embedding_size=vae_embedding_size,
            loss_factor=vae_loss_factor,
        )

        self.decoder_config = dict(
            epochs=decoder_epochs,
            clipping_value=decoder_clipping_value,
            patience=decoder_patience,
            n_iter_min=decoder_n_iter_min,
            early_stopping=decoder_early_stopping,
            weight_decay=decoder_weight_decay,
            opt_betas=decoder_opt_betas,
            lr=decoder_lr,
            batch_size=decoder_batch_size,
            hidden_layers=decoder_hidden_layers,
            nonlin=decoder_nonlin,
            dropout=decoder_dropout,
        )

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        """
        Fit the FASD algorithm on the provided tabular data.
        """
        if self.target_column not in X.columns.tolist():
            raise Exception("Provided target column name not found in dataset.")
        self.ori_cols = X.columns.tolist()

        # split into X and y
        y = X[self.target_column].copy()
        X = X.drop(self.target_column, axis=1)

        # do FASD specific preprocessing (one-hot encoding, minmax-scaling)
        self.encoders_x = []
        data = []
        input_dims = []
        for col in X.columns:
            if col in discrete_features:
                encoder = OneHotEncoder(sparse_output=False)
                input_dims.append(X[col].nunique())
            else:
                encoder = MinMaxScaler(feature_range=(-1, 1))
                input_dims.append(1)
            d = encoder.fit_transform(X[[col]])

            d = pd.DataFrame(d, columns=encoder.get_feature_names_out())
            data.append(d)
            self.encoders_x.append(encoder)

        x = pd.concat(data, axis=1, ignore_index=True)  # preprocessed input data

        # preprocess target feature
        if self.target_column in discrete_features:
            encoder = LabelEncoder()  # classifiers expect integer labels
            self.task = "classification"
            target_dim = y.nunique()
            y = encoder.fit_transform(y)
        else:
            if y.nunique() < 15:
                raise Warning(
                    "Found less than 15 unique values in the target column. "
                    "Are you sure the target should not be handled as discrete? "
                    "If so, please add it to the list of discrete features..."
                )
            self.task = "regression"
            encoder = MinMaxScaler(feature_range=(-1, 1))
            target_dim = 1
            y = encoder.fit_transform(y.to_frame())
        y = pd.Series(y.squeeze(), name=self.target_column)
        self.encoder_y = encoder

        self.fasd = FASD(
            task=self.task,
            target_dim=target_dim,
            input_dims=input_dims,
            predictor_config=self.predictor_config,
            vae_config=self.vae_config,
            decoder_config=self.decoder_config,
            random_state=self.random_state,
        )

        self.fasd._train(x, y)
        return self

    def _generate_data(self, n: int):
        """
        Generate `n` synthetic samples using the trained FASD model.
        """

        # generate data
        syn_X, syn_y = self.fasd._generate(n)

        # reverse FASD-specific preprocessing (inverse transform onehot and minmax scaling)
        syn_x = []
        i = 0
        for encoder, col in zip(self.encoders_x, self.ori_cols):
            if isinstance(encoder, OneHotEncoder):
                n_cols = len(encoder.get_feature_names_out([col]))
                data = encoder.inverse_transform(syn_X.iloc[:, i : i + n_cols])
                syn_x.append(pd.Series(data.flatten(), name=col))
                i += n_cols
            elif isinstance(encoder, MinMaxScaler):
                data = encoder.inverse_transform(syn_X.iloc[:, i].values.reshape(-1, 1))
                syn_x.append(pd.Series(data.flatten(), name=col))
                i += 1
            else:
                raise Exception("Unknown encoder encountered during posprocessing")

        syn_X = pd.concat(syn_x, axis=1)

        # reverse y preprocessing
        if self.task == "regression":
            syn_y = syn_y.to_frame()
        syn_y = self.encoder_y.inverse_transform(syn_y)
        syn_y = pd.Series(syn_y.squeeze(), name=self.target_column)

        syn = pd.concat((syn_X, syn_y), axis=1)
        syn = syn[self.ori_cols]

        return syn
