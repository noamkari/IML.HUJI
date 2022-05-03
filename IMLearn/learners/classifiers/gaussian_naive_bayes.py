from typing import NoReturn

from .. import MultivariateGaussian
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        y = y.reshape(y.shape[0])
        self.pi_ = np.zeros(self.classes_.shape[0])

        for c in self.classes_:
            a_sum = 0
            for l in y:
                if l == c:
                    a_sum += 1
            np.put(self.pi_, c, float(a_sum) / y.size)

        self.mu_ = np.array(
            [np.mean(X[y == i], axis=0) for i in self.classes_])
        self.vars_ = np.zeros(shape=[self.classes_.size, X.shape[1]])

        for c in range(self.classes_.size):
            curr = X[y == self.classes_[c]]
            for l in range(curr.shape[1]):
                self.vars_[c][l] = np.var(curr.T[l])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        maximum = np.argmax(self.likelihood(X) * self.pi_, axis=1)
        arg_max = np.ndarray(X.shape[0])

        for i in range(arg_max.size):
            arg_max[i] = self.classes_[maximum[i]]

        return arg_max

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        like = None
        for a_class in range(self.classes_.size):
            gaus = MultivariateGaussian()
            gaus.fitted_ = True
            gaus.mu_ = self.mu_[a_class]
            gaus.cov_ = np.diag(self.vars_[a_class])
            pdf = gaus.pdf(X)

            if like is None:
                like = pdf
            else:
                like = np.c_[like, pdf]

        return like

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
