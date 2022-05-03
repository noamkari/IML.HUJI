from datetime import datetime, date

import plotly.graph_objects as go
import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])

    df = df[df['Temp'] > -72]
    df = df.dropna()

    df["day_of_year"] = df['Date'].dt.dayofyear

    return df, df.loc[:, ["Temp"]]


def color_helper(year):
    return


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_data = X[X["Country"] == "Israel"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=israel_data["day_of_year"], y=israel_data["Temp"],
                   mode="markers", marker=dict(
                color=israel_data["Year"])))

    fig.update_layout(title="temp in israel by day of year",
                      xaxis_title="day_of_year",
                      yaxis_title="Temp")
    fig.show()

    px.bar(israel_data.groupby("Month").agg(np.std), y="Temp").show()

    # Question 3 - Exploring differences between countries
    tmp_X = X.groupby(["Country", "Month"])["Temp"].agg(
        [np.mean, np.std]).reset_index()

    px.line(tmp_X,
            x="Month",
            y="mean",
            color="Country",
            error_y="std").show()

    # Question 4 - Fitting model for different values of `k`

    train_x, train_y, test_x, test_y = split_train_test(
        israel_data.loc[:, ["day_of_year"]],
        israel_data["Temp"])

    loss_arr = {"Degree": [], "MSE": []}
    for k in range(1, 11):
        poly_r = PolynomialFitting(k)
        poly_r.fit(train_x.to_numpy(), train_y.to_numpy())

        tmp_loss = poly_r.loss(test_x.to_numpy(), test_y.to_numpy())
        loss_arr["Degree"] += [k]
        loss_arr["MSE"] += [tmp_loss]
        print(tmp_loss)

    px.bar(loss_arr, x="Degree", y="MSE", title="MSE VS Degree").show()

    # Question 5 - Evaluating fitted model on different countries

    CHOOSE_K = 3
    poly_r = PolynomialFitting(CHOOSE_K)
    poly_r.fit(train_x.to_numpy(), train_y.to_numpy())

    dct_error = {"Country": [], "Error": []}
    for country in set(X["Country"]):
        data = X[X["Country"] == country]
        dct_error["Country"].append(country)
        dct_error["Error"].append(poly_r.loss(
            (data.loc[:, ["day_of_year"]]).to_numpy(),
            data["Temp"].to_numpy()))

    px.bar(dct_error, x="Country", y="Error",
           title="error sum of model train on Israel").show()
