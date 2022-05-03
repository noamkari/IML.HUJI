from typing import NoReturn

from .. import MultivariateGaussian
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error

from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.fitted_ = True
        y = y.reshape(y.shape[0])
        self.classes_ = np.unique(y)
        self.pi_ = np.zeros(self.classes_.shape[0])

        for c in self.classes_:
            a_sum = 0
            for l in y:
                if l == c:
                    a_sum += 1
            np.put(self.pi_, c, float(a_sum) / y.size)

        self.mu_ = np.array(
            [np.mean(X[y == i], axis=0) for i in self.classes_])
        v = np.zeros(shape=[X.shape[1], X.shape[1]])

        for i in range(self.classes_.size):
            for x in X[y == self.classes_[i]]:
                curr = x - self.mu_[i]
                curr = np.asmatrix(curr)
                v = v + curr.T @ curr

        self.cov_ = v * (1 / (X.shape[0] - self.classes_.size))
        self._cov_inv = inv(self.cov_)

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

        max_ = np.ndarray.argmax(self.likelihood(X) * self.pi_, axis=1)
        arg_max = np.ndarray(X.shape[0])

        for arg_i in range(arg_max.size):
            arg_max[arg_i] = self.classes_[max_[arg_i]]

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
        for i in range(self.classes_.size):
            m_gaussian = MultivariateGaussian()
            m_gaussian.fitted_ = True
            m_gaussian.mu_, m_gaussian.cov_ = self.mu_[i], self.cov_
            temp_ptf = m_gaussian.pdf(X)

            if like is None:
                like = temp_ptf
            else:
                like = np.c_[like, temp_ptf]

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
