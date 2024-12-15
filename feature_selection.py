from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def select_important_features(X, y, top_n=30):
    model = RandomForestClassifier()
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importances.nlargest(top_n).index
