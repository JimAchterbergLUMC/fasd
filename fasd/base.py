import pandas as pd

from .utils import preprocess, postprocess, get_discretes


class BaseGenerator:

    def __init__(self, impute_nan: bool = True, random_state: int = 0):
        self.impute_nan = impute_nan
        self.random_state = random_state
        self.missing_suffix = "_missing"
        self.missing_indicators = None

    def fit(self, X: pd.DataFrame, discrete_features: list = None):
        self.X_ori = X.copy()
        self.discrete_features = discrete_features
        if self.discrete_features is None:
            self.discrete_features = get_discretes(df=X, discrete_threshold=5)

        X[self.discrete_features] = X[self.discrete_features].astype(str)

        self.numerical_features = [
            col for col in X.columns if col not in self.discrete_features
        ]

        try:
            X[self.numerical_features].astype(float)
        except:
            raise Exception(
                "Some features are discrete but were not passed as discrete in the discrete_features list."
            )

        # if desired, impute NaNs randomly and create an additional indicator column to later reinstate NaNs
        if self.impute_nan:
            nan_cols = [col for col in self.numerical_features if X[col].isnull().any()]
            self.missing_indicators = X[nan_cols].isnull().astype(int)
            self.missing_indicators.columns = [
                col + self.missing_suffix for col in nan_cols
            ]
            self.discrete_features.extend(
                self.missing_indicators.columns
            )  # generators regard indicator columns as discrete
        else:
            # else we simply drop NaNs for numerical columns (discrete NaNs get one-hot encoded)
            X = X.dropna(subset=self.numerical_features)

        self.X = preprocess(
            df=X.copy(),
            missing_indicators=self.missing_indicators,
        )

        self._fit_model(X=self.X, discrete_features=self.discrete_features)

        return self

    def generate(self, n: int):
        syn_X = self._generate_data(n)

        syn_X = postprocess(
            df_syn=syn_X,
            df_ori=self.X_ori,
            discrete_features=[
                x for x in self.discrete_features if x in self.X_ori.columns
            ],  # exclude indicator cols
            missing_indicators=self.missing_indicators,
            missing_suffix=self.missing_suffix,
        )

        return syn_X

    def _fit_model(self, X: pd.DataFrame, discrete_features: list = None):
        raise NotImplementedError("Subclasses must implement _fit_model")

    def _generate_data(self, n: int) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _generate_data")
