"""The fraud detection workflow from Anomaly_Detection_Evaluation.ipynb and
Turning_Scores_into_Decisions.ipynb, written as a Flyte 2 pipeline.

Each function decorated with @env.task is one step of the pipeline. Flyte runs
the steps locally with flyte.init() (no Docker needed) or on a cluster with
flyte.init_from_config(), without changing this file.
"""
import asyncio
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import flyte
from flyte.io import HashFunction

FEATURES = [f'V{i}' for i in range(1, 29)] + ['log_amount', 'hour']

# Flyte caches a task's output and reuses it when the task is called again with the same inputs.
# Hashing DataFrames by their contents lets the cache recognize the same data across runs.
HashedDataFrame = Annotated[pd.DataFrame, HashFunction.from_fn(lambda df: str(pd.util.hash_pandas_object(df).sum()))]

# The hyperparameters tried for each model family. Every entry becomes one parallel task.
EXPERIMENTS = {
    'Logistic Regression': [{'C': C} for C in [0.1, 1.0, 10.0]],
    'Random Forest': [{'max_depth': d, 'min_samples_leaf': 5} for d in [8, 16]],
    'Gradient-Boosted Trees': [{'learning_rate': lr} for lr in [0.05, 0.1]],
}

env = flyte.TaskEnvironment(
    name='fraud-pipeline',
    image=flyte.Image.from_debian_base().with_pip_packages('scikit-learn', 'pandas', 'pyarrow'),
    resources=flyte.Resources(cpu=2, memory='4Gi'),
    cache='auto',
)


def make_model(model_name: str, params: dict):
    models = {
        'Logistic Regression': lambda: LogisticRegression(max_iter=1000, **params),
        'Random Forest': lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, **params),
        'Gradient-Boosted Trees': lambda: HistGradientBoostingClassifier(class_weight='balanced', random_state=0, **params),
    }
    return make_pipeline(StandardScaler(), models[model_name]())


@env.task
def engineer_features(df: HashedDataFrame) -> HashedDataFrame:
    """Add the features from Anomaly_Detection_Evaluation.ipynb: hour of the day and log amount."""
    df['hour'] = (df['Time'] // 3600) % 24
    df['log_amount'] = np.log1p(df['Amount'])
    return df[FEATURES + ['Amount', 'Class']]


@env.task
def split_data(df: HashedDataFrame) -> tuple[HashedDataFrame, HashedDataFrame, HashedDataFrame]:
    """60% train, 20% validation (choose the threshold), 20% test (final results)."""
    train, temp = train_test_split(df, test_size=0.4, stratify=df['Class'], random_state=0)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['Class'], random_state=0)
    return train, val, test


@env.task
def evaluate_model(train: HashedDataFrame, model_name: str, params: dict) -> dict:
    """Cross-validate one model and one set of hyperparameters on the training set."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = cross_validate(make_model(model_name, params), train[FEATURES], train['Class'],
                            cv=cv, scoring=['roc_auc', 'average_precision'])
    return {'model': model_name,
            'params': params,
            'ROC-AUC': float(scores['test_roc_auc'].mean()),
            'PR-AUC': float(scores['test_average_precision'].mean())}


@env.task
def score_best_model(train: HashedDataFrame, val: HashedDataFrame, test: HashedDataFrame,
                     model_name: str, params: dict) -> dict:
    """Refit the best model on the training set and score the validation and test sets."""
    model = make_model(model_name, params).fit(train[FEATURES], train['Class'])
    return {'val_labels': val['Class'].tolist(),
            'val_amounts': val['Amount'].tolist(),
            'val_scores': model.predict_proba(val[FEATURES])[:, 1].tolist(),
            'test_labels': test['Class'].tolist(),
            'test_amounts': test['Amount'].tolist(),
            'test_scores': model.predict_proba(test[FEATURES])[:, 1].tolist()}


# The parent task is not cached, so every run shows which child tasks were reused.
@env.task(cache='disable')
async def fraud_pipeline(raw: HashedDataFrame, model_names: list[str]) -> dict:
    """The whole workflow. The model comparison runs as parallel tasks."""
    df = await engineer_features.aio(raw)
    train, val, test = await split_data.aio(df)

    experiments = [(name, params) for name in model_names for params in EXPERIMENTS[name]]
    results = await asyncio.gather(*[evaluate_model.aio(train, name, params)
                                     for name, params in experiments])

    best = max(results, key=lambda r: r['PR-AUC'])
    scores = await score_best_model.aio(train, val, test, best['model'], best['params'])
    return {'results': results, 'best': best, **scores}
