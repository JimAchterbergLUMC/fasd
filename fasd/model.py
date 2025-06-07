import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


from .base import BaseGenerator
from .model_components import FASD_Generator


class FASD(BaseGenerator):

    def __init__(
        self,
        target_column: str,
        representation_dim=100,
        predictor_hidden_layers=[],
        predictor_nonlin="relu",
        representations_nonlin="tanh",
        predictor_dropout=0,
        decoder_hidden_layers=[100],
        decoder_nonlin="relu",
        decoder_dropout=0,
        vae_encoder_hidden_layers=[128, 128],
        vae_encoder_nonlin="relu",
        vae_encoder_dropout=0,
        vae_decoder_hidden_layers=[128, 128],
        vae_decoder_nonlin="relu",
        vae_decoder_dropout=0,
        vae_embedding_size=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_column = target_column
        self.representation_dim = representation_dim

        self.predictor_hidden_layers = predictor_hidden_layers
        self.predictor_hidden_layers.append(self.representation_dim)
        self.predictor_nonlin = predictor_nonlin
        self.predictor_dropout = predictor_dropout

        self.representations_nonlin = representations_nonlin

        self.decoder_hidden_layers = decoder_hidden_layers
        self.decoder_nonlin = decoder_nonlin
        self.decoder_dropout = decoder_dropout

        self.vae_encoder_hidden_layers = vae_encoder_hidden_layers
        self.vae_encoder_nonlin = vae_encoder_nonlin
        self.vae_encoder_dropout = vae_encoder_dropout

        self.vae_decoder_hidden_layers = vae_decoder_hidden_layers
        self.vae_decoder_nonlin = vae_decoder_nonlin
        self.vae_decoder_dropout = vae_decoder_dropout

        self.vae_embedding_size = vae_embedding_size

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):
        if self.target_column not in X.columns.tolist():
            raise Exception("Provided target column name not found in dataset.")
        self.ori_cols = X.columns.tolist()

        # split into X and y
        y = X[[self.target_column]].copy()
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
            task = "classification"
            target_dim = y.squeeze().nunique()
        else:
            if y.squeeze().nunique() < 15:
                raise Warning(
                    "Found less than 15 unique values in the target column. "
                    "Are you sure the target should not be handled as discrete? "
                    "If so, please add it to the list of discrete features..."
                )
            task = "regression"
            encoder = MinMaxScaler(feature_range=(-1, 1))
            target_dim = 1
        y = encoder.fit_transform(y)
        y = pd.DataFrame(y, columns=[self.target_column])
        self.encoder_y = encoder

        self.fasd = FASD_Generator(
            task=task,
            target_dim=target_dim,
            input_dims=input_dims,
            representation_dim=self.representation_dim,
            predictor_hidden_layers=self.predictor_hidden_layers,
            predictor_nonlin=self.predictor_nonlin,
            predictor_batch_norm=False,
            predictor_residual=False,
            representations_nonlin=self.representations_nonlin,
            predictor_dropout=self.predictor_dropout,
            decoder_hidden_layers=self.decoder_hidden_layers,
            decoder_batch_norm=False,
            decoder_residual=False,
            decoder_nonlin=self.decoder_nonlin,
            decoder_dropout=self.decoder_dropout,
            vae_encoder_hidden_layers=self.vae_encoder_hidden_layers,
            vae_batch_norm=False,
            vae_residual=False,
            vae_encoder_nonlin=self.vae_encoder_nonlin,
            vae_encoder_dropout=self.vae_encoder_dropout,
            vae_decoder_hidden_layers=self.vae_decoder_hidden_layers,
            vae_decoder_nonlin=self.vae_decoder_nonlin,
            vae_decoder_dropout=self.vae_decoder_dropout,
            vae_embedding_size=self.vae_embedding_size,
            random_state=self.random_state,
        )

        self.fasd._train(x, y)
        return self

    def _generate_data(self, n: int):

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
        syn_y = self.encoder_y.inverse_transform(syn_y)
        syn_y = pd.Series(syn_y.flatten(), name=self.target_column)

        syn = pd.concat((syn_X, syn_y), axis=1)
        syn = syn[self.ori_cols]

        return syn
