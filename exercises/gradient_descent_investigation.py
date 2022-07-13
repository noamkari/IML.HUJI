import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection.cross_validate import *

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange]
         over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's
     value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding
         the objective's value and parameters at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weights_lst = []
    values_lst = []

    def helper(**kwargs):
        weights_lst.append(kwargs["weights"])
        values_lst.append(kwargs["val"])

    return helper, values_lst, weights_lst


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    model_dct = {L1: "L1", L2: "L2"}
    loss_graph = {}

    for eta in etas:
        for model in model_dct.keys():
            callback = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(eta),
                                 callback=callback[0])
            gd.fit(model(init.copy()), None, None)
            plot_descent_path(model, np.array(callback[2]),
                              f"{model_dct[model]} model with eta={eta}").show()

            loss_graph[f"{model_dct[model]} with eta={eta}"] = go.Scatter(
                x=np.arange(gd.max_iter_),
                y=callback[1],
                mode="markers",
                name="")

            print(f"model={model_dct[model]} eta={eta}"
                  f" loss={callback[1][np.argmin(callback[1])]}")

    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=list(loss_graph.keys()))
    for i, graph in enumerate(loss_graph.values()):
        fig.add_trace(graph, row=1 if i < 4 else 2, col=(i % 4) + 1)
    fig.update_layout(
        title_text="loss as func of iteration in different model and eta")
    fig.show()


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate

    scatters = []

    for gama in gammas:
        loss = []
        gd = GradientDescent(ExponentialLR(eta, gama),
                             callback=lambda **kwargs: loss.append(
                                 kwargs["val"]))
        gd.fit(L1(init.copy()), None, None)

        scatters.append(go.Scatter(x=np.arange(gd.max_iter_),
                                   y=loss,
                                   name=gama))
        print(
            f"model=L1 ete={eta} gama={gama} loss={loss[np.argmin(loss)]}")

    # Plot algorithm's convergence for the different values of gamma
    fig = go.Figure(data=scatters)
    fig.update_layout(title="loss as func of gama in L1 model")
    fig.show()

    # Plot descent path for gamma=0.95
    callback = get_gd_state_recorder_callback()
    gd = GradientDescent(ExponentialLR(eta, 0.95),
                         callback=callback[0])
    gd.fit(L1(init.copy()), None, None)
    plot_descent_path(L1, np.array(callback[2]),
                      f"L1 model with eta={eta} and gama=0.95").show()


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset

    X_train, y_train, X_test, y_test = list(map(lambda x: x.to_numpy(),
                                                list(load_data())))

    # Plotting convergence rate of logistic regression over SA heart disease data


    # amit
    gd = GradientDescent(FixedLR(1e-4), max_iter=20000)
    model = LogisticRegression(solver=gd)
    model.fit(np.array(X_train), np.array(y_train))
    y_prob = model.predict_proba(np.array(X_train))
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()
    # end amit

    # q9 find best a
    print(thresholds[np.argmax(tpr - fpr)])

    l = []
    for lam in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:

        for p in ["l1", "l2"]:
            model = LogisticRegression(
                solver=GradientDescent(FixedLR(1e-4), max_iter=20000),
                penalty=p, alpha=0.5, lam=lam)

        l.append(cross_validate(model, X_train, y_train,
                                scoring=misclassification_error))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
