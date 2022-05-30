from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    X = np.linspace(-1.2, 2, n_samples)

    noise_vec = [np.random.normal(0, noise) for _ in range(n_samples)]
    y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X),
                                                        pd.Series(
                                                            y + noise_vec),
                                                        2 / 3)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    fig = go.Figure(data=[go.Scatter(x=X, y=y, mode="markers", name="true"),
                          go.Scatter(x=X_train.flatten(), y=y_train,
                                     mode="markers",
                                     name="train"),
                          go.Scatter(x=X_test.flatten(), y=y_test,
                                     mode="markers",
                                     name="test")])
    fig.update_layout(title=f"value of {n_samples} samples with {noise} noise")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    MAX_DEGREE = 10

    train_lost = []
    test_lost = []

    for degree in range(MAX_DEGREE + 1):
        cur_lost, cur_valid = cross_validate(PolynomialFitting(degree),
                                             X_train, y_train,
                                             mean_square_error)
        train_lost.append(cur_lost)
        test_lost.append(cur_valid)

    fig = go.Figure(data=[
        go.Bar(x=np.arange(MAX_DEGREE + 1), y=train_lost, name="train lost"),
        go.Bar(x=np.arange(MAX_DEGREE + 1), y=test_lost, name="test lost")])
    fig.update_layout(
        title=f"loss as func of degree on {n_samples} samples with {noise} noise")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    chosen_degree = 4

    pol_fitting = PolynomialFitting(chosen_degree)
    pol_fitting.fit(X_train, y_train)
    pol_fitting.loss(X_test, y_test)


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, train_size=n_samples)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    model_dct = {"RidgeRegression": RidgeRegression,
                 "Lasso": Lasso}
    ranges = [(0, 3), (0, 100), (0, 0.25)]

    for cur_range in ranges:

        fig = go.Figure()
        for model_name, model in model_dct.items():

            train_lost = []
            test_lost = []

            for lam in np.linspace(cur_range[0], cur_range[1],
                                   num=n_evaluations):
                cur_lost, cur_valid = cross_validate(
                    model(lam),
                    X_train, y_train,
                    mean_square_error)

                train_lost.append(cur_lost)
                test_lost.append(cur_valid)

            fig.add_trace(go.Scatter(
                x=np.linspace(cur_range[0], cur_range[1], num=n_evaluations),
                y=train_lost,
                name=f"train lost {model_name}"))
            fig.add_trace(
                go.Scatter(x=np.linspace(cur_range[0], cur_range[1],
                                         num=n_evaluations),
                           y=test_lost, name=f"test lost {model_name}"))

            fig.update_layout(title=f"{cur_range}")
        fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_for_lasso = 0.3006
    best_lam_for_ridge = 0.024

    model_dct["Least Squares Regression"] = LinearRegression()
    model_dct["RidgeRegression"] = RidgeRegression(best_lam_for_ridge)
    model_dct["Lasso"] = Lasso(best_lam_for_lasso)

    X = X.to_numpy()
    y = y.to_numpy()
    lost = {}

    for model_name, model in model_dct.items():
        model.fit(X, y)
        lost[model_name] = mean_square_error(y, model.predict(X))

    print(lost)


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(1500, 10)
    select_regularization_parameter()
