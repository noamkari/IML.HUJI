from plotly.subplots import make_subplots

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    samples = np.random.normal(10, 1, 1000)

    ug = UnivariateGaussian()
    ug.fit(samples)
    print(ug.mu_, ug.var_)

    # Question 2 - Empirically showing sample mean is consistent

    y_axis = []
    x_axis = []

    for i in range(0, samples.size, 10):
        ug.fit(samples[:i + 10])
        y_axis.append(abs(10 - ug.mu_))
        x_axis.append(i + 10)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=y_axis))
    fig.update_layout(title="Exception difference depends on sample size",
                      xaxis_title="Sample size",
                      yaxis_title="Exception")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model

    x_axis = samples
    ug.fit(x_axis)
    y_axis = ug.pdf(x_axis)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="markers"))
    fig.update_layout(title="Empirical PDF of fitted model",
                      xaxis_title="Sample value",
                      yaxis_title="PDF")
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    mu = [0, 0, 4, 0]

    samples = np.random.multivariate_normal(mu, cov, size=1000)
    mug = MultivariateGaussian()
    mug.fit(samples)
    print(mug.mu_, "\n", mug.cov_)

    # Question 5 - Likelihood evaluation

    mat = np.linspace(-10, 10, 200)

    likelihood_array = np.empty(shape=(mat.size, mat.size))

    max_index = [mat[0], mat[0]]
    max_arg = mug.log_likelihood(np.array([mat[0], 0, mat[0], 0]), cov,
                                 samples)

    for i in range(mat.size):
        for j in range(mat.size):
            tmp_arg = mug.log_likelihood(np.array([mat[i], 0, mat[j], 0]),
                                         cov,
                                         samples)
            if tmp_arg > max_arg:
                max_arg = tmp_arg
                max_index = [mat[i], mat[j]]
            likelihood_array[i][j] = tmp_arg

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=mat, y=mat, z=likelihood_array))
    fig.update_layout(title="logliklihood of [f1, 0, f3, 0]",
                      xaxis_title="f1 val",
                      yaxis_title="f3 val")
    fig.show()

    # Question 6 - Maximum likelihood
    print(max_arg)
    print(max_index)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
