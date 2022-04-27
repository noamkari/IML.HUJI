from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


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

        # TODO remove
        fig.write_image(f"losses in {n}.png")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)

        gaussian_naive = GaussianNaiveBayes()
        gaussian_naive.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot
        # titles should specify algorithm and accuracy

        from IMLearn.metrics import accuracy

        exm = np.array([[1, 2, 3, 4],
                        [0, 0, 5, 6]])

        true_exm = [1, -1, -1, -1]
        pred_exm = [1, -1, 1, -1]

        np_exm = np.array([1, -1, 1, 1])

        # [(x_1,y_1), (x_2,y_2), (x_3,y_3)]
        # [color_1, color_2, color_3]

        symbols = {color: label for color, label in enumerate(set(np_exm))}

        fig = go.Figure([go.Scatter(x=exm[0], y=exm[1],
                                    mode="markers",
                                    marker=dict(
                                        symbol=[symbols[i] for i in true_exm],
                                        color=pred_exm))])

        fig.update_layout(title="some example",
                          xaxis_title="Sample value",
                          yaxis_title="loss percent"
                          )

        fig.show()


if __name__ == '__main__':
    # np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()

    price = 0
    for i in range(1, 667):
        if i < 10:
            price += 1
        elif i < 100:
            price += 2
        elif i < 1000:
            price += 3

    print(price)
