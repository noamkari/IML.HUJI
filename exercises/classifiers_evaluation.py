from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from IMLearn.metrics import accuracy

pio.templates.default = "simple_white"


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    full_data = np.load(filename)
    return np.split(full_data, [-1], axis=1)

    # raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
     linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss
    values (y-axis) as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration

        losses = []
        perceptron = Perceptron(
            callback=lambda p, X, y: losses.append(p._loss(X, y)))
        perceptron.fit(X, y)

        # Plot figure

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(perceptron.max_iter_), y=losses,
                                 mode="lines"))
        fig.update_layout(title=n,
                          xaxis_title="Sample value",
                          yaxis_title="loss percent")

        fig.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    model_dct = {"lda": LDA(), "gaussian_naive": GaussianNaiveBayes()}

    for f in ["gaussian1.npy", "gaussian2.npy"]:

        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set

        pred_dct = {}
        symbol_dct = {label: color for
                      color, label in enumerate(np.unique(y))}

        for name, model in model_dct.items():
            model.fit(X, y)
            pred_dct[name] = (model.predict(X))

        y = y.reshape(y.shape[0])
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"{name} accuracy is {accuracy(y, pred)}"
            for name, pred in pred_dct.items()])

        for i, model in enumerate(model_dct.keys()):
            fig.add_trace(go.Scatter(x=X.T[0], y=X.T[1],
                                     mode="markers",
                                     marker=dict(
                                         symbol=[symbol_dct[l] for l in y],
                                         color=pred_dct[model])),
                          row=1, col=i + 1)

            fig.add_trace(go.Scatter(x=model_dct[model].mu_[:, 0],
                                     y=model_dct[model].mu_[:, 1],
                                     mode='markers',
                                     marker=dict(color="black", symbol="x")),
                          row=1, col=i + 1)

        for i in range(model_dct["lda"].classes_.size):
            fig.add_trace(get_ellipse(
                model_dct["lda"].mu_[i],
                model_dct["lda"].cov_), row=1, col=1)

        for i in range(model_dct["gaussian_naive"].classes_.size):
            fig.add_trace(get_ellipse(
                model_dct["gaussian_naive"].mu_[i],
                np.diag(model_dct["gaussian_naive"].vars_[i])), row=1, col=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
