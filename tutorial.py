from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from fasd.utils import set_seed

from fasd import FASD


seed = 0
set_seed(0)  # sets random, numpy, and torch seeds

# load a dataset
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
target_col = y.name
X = pd.concat((X, y), axis=1)

# generate synthetic data
generator = FASD(target_column=target_col, impute_nan=True, random_state=seed)
generator.fit(X)
syn = generator.generate(len(X))

exit()

# evaluate ML efficacy
y = X[target_col].copy()
X = X.drop(target_col, axis=1)
y_syn = syn[target_col].copy()
X_syn = syn.drop(target_col, axis=1)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, stratify=y, train_size=0.7, random_state=seed
)
X_syn_tr, X_syn_te, y_syn_tr, y_syn_te = train_test_split(
    X_syn, y_syn, stratify=y_syn, train_size=0.7, random_state=seed
)

model = HistGradientBoostingClassifier(max_depth=3)
model.fit(X_tr, y_tr)
preds = model.predict(X_te)
score = roc_auc_score(y_te, preds)
print(f"Train Real Test Real: {score}")

model = HistGradientBoostingClassifier(max_depth=3)
model.fit(X_syn_tr, y_syn_tr)
preds = model.predict(X_te)
score = roc_auc_score(y_te, preds)
print(f"Train Synthetic Test Real: {score}")
