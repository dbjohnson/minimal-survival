import numpy as np
import pandas as pd
from scipy.linalg import pinv


def fit_coefs(
    df,
    timeline_column,
    event_column,
    covariates,
    maxiter=1000,
    tol=1e-9
):
    """
    h = h0(t) * exp(b1 * x1 + b2 * x2 + ... + bk * xk)
    h0(t) is the baseline hazard function
    https://en.wikipedia.org/wiki/Proportional_hazards_model

    use newton's method - we'll need the negative log likelihood function
    for a given set of weights, and the first and second derivatives
    (gradient and hessian) of the negative log likelihood function
    We use the negative log likelihood because the log likelihood
    is by definition negative (likelihood is in [0, 1], and our optimizer
    is a minimizer
    # NOTE: we are optimizing for clarity of interpretation, not speed!
    """
    w = np.zeros(len(covariates))
    w_prev = w
    loss = float('inf')
    i = 0
    while True and i < maxiter:
        i += 1
        loss_new, gradient, hessian = _log_loss_and_derivates(
            w,
            df,
            timeline_column,
            event_column,
            covariates
        )

        # solve for Ax = b
        # where:
        #  A = hessian,
        #  b = gradient
        #  x = delta, the step to take
        delta = gradient.dot(pinv(hessian))

        if loss_new > loss:
            # perform step-halving if negative log-likelihood does not decrease
            w = (w_prev + w) / 2
            continue

        w, w_prev = w - delta, w

        if np.abs(1 - (loss_new / loss)) < tol:
            # converged
            break

        loss = loss_new

    return pd.Series(w, index=covariates)


def _log_loss_and_derivates(
    weights,
    df,
    timeline_column,
    event_column,
    covariates
):
    """
    Calculate negative log-likehood and its first and second derivatives
    for a given set of weights.

    Cribbed from:
    https://github.com/sebp/scikit-survival/blob/d5fcbc9e06f4824f987b5cf550ce0bf82665f28e/sksurv/linear_model/coxph.py#L131
    """
    # sort descending
    idx = np.argsort(-df[timeline_column])
    time = df[timeline_column].values[idx]
    X = df[covariates].values[idx]
    event = df[event_column].values[idx]

    n_samples = X.shape[0]
    xw = np.dot(X, weights)
    exp_xw = np.exp(xw)
    n_samples, n_features = X.shape

    gradient = np.zeros((1, n_features))
    hessian = np.zeros((n_features, n_features))

    inv_n_samples = 1. / n_samples
    loss = 0
    risk_set = 0
    risk_set_x = np.zeros((1, n_features))
    risk_set_xx = np.zeros((n_features, n_features))
    k = 0
    # iterate time in descending order
    while k < n_samples:
        ti = time[k]
        n_events = 0
        numerator = 0
        numerator_log_loss = 0
        while k < n_samples and ti == time[k]:
            # preserve 2D shape of row vector
            xk = X[k:k + 1]

            # outer product
            xx = np.dot(xk.T, xk)

            risk_set += exp_xw[k]
            risk_set_x += exp_xw[k] * xk
            risk_set_xx += exp_xw[k] * xx
            if event[k]:
                numerator_log_loss += xw[k]
                numerator += xk
                n_events += 1
            k += 1

        if n_events > 0:
            loss -= (numerator_log_loss - n_events * np.log(risk_set)) / n_samples
            z = risk_set_x / risk_set
            gradient -= (numerator - n_events * z) * inv_n_samples

            a = risk_set_xx / risk_set
            # outer product
            b = np.dot(z.T, z)

            hessian += n_events * (a - b) * inv_n_samples

    return (
        loss,
        gradient.ravel(),
        hessian
    )
