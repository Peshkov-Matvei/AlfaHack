import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier


def train_ensemble(X_train, y_train):
    lgb_model = lgb.LGBMClassifier()
    xgb_model = xgb.XGBClassifier(eval_metric="logloss")
    cat_model = CatBoostClassifier(verbose=0)

    ensemble = VotingClassifier(
        estimators=[('lgb', lgb_model), ('xgb', xgb_model), ('cat', cat_model)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    return ensemble
