import seaborn as sns
import matplotlib.pyplot as plt


def visualize_data(df):
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()


def visualize_time_trends(df, time_column, target_column):
    df[time_column] = pd.to_datetime(df[time_column])
    trend = df.groupby(df[time_column].dt.to_period("M"))[target_column].mean()
    trend.plot(kind='line', title='Monthly Trend of Target')
    plt.show()
