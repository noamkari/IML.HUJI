from __future__ import annotations
import numpy as np
import pandas as pd
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

    noise = [np.random.normal(0, noise) for _ in range(n_samples)]
    y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y + noise),
                                                        2 / 3)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers", name="true"))
    fig.add_trace(
        go.Scatter(x=X_train.flatten(), y=y_train, mode="markers",
                   name="train"))
    fig.add_trace(
        go.Scatter(x=X_test.flatten(), y=y_test, mode="markers",
                   name="test"))
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

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=np.arange(MAX_DEGREE + 1), y=train_lost, name="train lost"))
    fig.add_trace(
        go.Bar(x=np.arange(MAX_DEGREE + 1), y=test_lost, name="test lost"))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    chosen_degree = 5

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
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
