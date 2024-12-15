from data_processing import load_data, preprocess_data
from feature_engineering import create_time_features, add_custom_features
from feature_selection import select_important_features
from model_ensemble import train_ensemble
from evaluation import evaluate_model
from submission import create_submission_file
from sklearn.model_selection import train_test_split

df = load_data("data.csv")
df = preprocess_data(df)
df = create_time_features(df, "last_transaction_date")
df = add_custom_features(df)

target = "churn"
X = df.drop(columns=[target])
y = df[target]

important_features = select_important_features(X, y)
X = X[important_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_ensemble(X_train, y_train)

evaluate_model(model, X_test, y_test)

test_df = load_data("test_data.csv")
test_df = preprocess_data(test_df)
test_df = test_df[important_features]
test_predictions = model.predict_proba(test_df)[:, 1]

create_submission_file(test_df, test_predictions, "submission.csv")
