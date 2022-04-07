from plotly.subplots import make_subplots

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> tuple:
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pd.read_csv(filename, index_col=0)

    # remove empty house
    df = df[df['date'] != "0.0"]
    df = df.dropna()

    price_vector = df.loc[:, ["price"]]
    calc_house_age(df)

    df = df.loc[:,
         ["bedrooms", "bathrooms", "sqft_living", "sqft_lot",
          "floors",
          "waterfront", "view", "lat", "condition", "grade",
          "long",
          "sqft_living15", "sqft_lot15", "sqft_basement",
          "sqft_above", "age", "renovated", "zipcode"]]

    df = pd.get_dummies(df, columns=["zipcode"])

    return df, price_vector.squeeze()


def calc_house_age(df):
    df["renovated"] = np.where(df["yr_renovated"], 1, 0)

    # todo need to avoid int(str(obg))
    df["age"] = np.where(df["renovated"], df["yr_renovated"] - df["yr_built"],
                         df["date"].str[:4].astype(float) - df["yr_built"])


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    y_axis = []
    x_axis = X.columns

    # fixme not good
    y_np = y.to_numpy()
    y_np = y_np.ravel()

    for i, feature in enumerate(X.columns):
        y_axis.append(
            np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y)))

    NUM_TO_BREAK = 15
    for i, feature in enumerate(X):

        if i == NUM_TO_BREAK:
            break
        create_scatter_for_feature(X[feature], y_np, round(y_axis[i], 3),
                                   feature,
                                   output_path)


def create_scatter_for_feature(X_feature: pd.DataFrame, prices, title,
                               feature_name: str, output_path: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_feature, y=prices, mode="markers"))
    fig.update_layout(title="Pirson: " + str(title),
                      xaxis_title=feature_name,
                      yaxis_title="price")

    fig.show()
    # fig.write_image(output_path + feature_name+".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, prices = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, prices, "")

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, prices)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    liner_reg = LinearRegression()

    avg_loss = np.ndarray(91, )
    std = np.ndarray(91, )

    for percent in range(10, 101):
        tmp_loss = np.ndarray(10, )
        for i in range(10):
            liner_reg.fit(X_train.sample(random_state=i,
                                         frac=(percent / 100)).to_numpy(),
                          y_train.sample(random_state=i,
                                         frac=(percent / 100)).to_numpy())

            tmp_loss[i] = liner_reg.loss(X_test.to_numpy(), y_test.to_numpy())

        avg_loss[percent - 10] = np.mean(tmp_loss)
        std[percent - 10] = np.std(tmp_loss)

    fig = go.Figure([
        go.Scatter(
            name='Mean value',
            x=np.arange(10, 101),
            y=avg_loss,
            mode='markers+lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=np.arange(10, 101),
            y=avg_loss + 2 * std,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=np.arange(10, 101),
            y=avg_loss - 2 * std,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])

    fig.update_layout(
        yaxis_title='mean values',
        title='Mean loss vs sample percent',
        hovermode="x"
    )
    fig.show()
