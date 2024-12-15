import pandas as pd


def create_submission_file(test_df, predictions, output_file):
    submission = pd.DataFrame({
        "client_id": test_df["client_id"],
        "churn_probability": predictions
    })
    submission.to_csv(output_file, index=False)
