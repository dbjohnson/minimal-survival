import numpy as np
import pandas as pd
from scipy.linalg import pinv
from scipy.optimize import dual_annealing

from src import survival


class Cox:
    def __init__(self, df, timeline_column, event_column, covariates):
        self.df_train = df
        self.timeline_column = timeline_column
        self.event_column = event_column
        self.covariates = covariates
        self.coefs = []
        self.cumulative_baseline_hazard = survival.cumulative_hazard(
            df, timeline_column, event_column
        )

    def fit_coefs(
        self,
        maxiter=1000,
        tol=1e-9
    ):
        """
        h = h0(t) * exp(b1 * x1 + b2 * x2 + ... + bk * xk)
        h0(t) is the baseline hazard function
        https://en.wikipedia.org/wiki/Proportional_hazards_model

        use Newton's method to optimize the coefficients by maximizing
        the likelihood function (or minimizing the negative log-likelihood)
        which is the product of the individual event probabilities given
        individual risk scores and the total remaining risk at each time point.

        See https://en.wikipedia.org/wiki/Proportional_hazards_model#Likelihood_for_unique_times
        for more details on the likelihood function.

        We'll need the negative log likelihood function for a given set of coefs,
        and the first and second derivatives (gradient and Hessian) of the
        negative log likelihood function.
        We use the negative log likelihood because the log likelihood
        is by definition negative (likelihood is in [0, 1], and our optimizer
        is a minimizer
        """
        w = np.zeros(len(self.covariates))
        w_prev = w
        loss = float('inf')
        i = 0
        while True and i < maxiter:
            i += 1
            loss_new, gradient, hessian = self._log_loss_and_derivates(w)

            # https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization#Higher_dimensions
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

        self.coefs = w
        return pd.Series(w, index=self.covariates)

    def _log_loss_and_derivates(
        self,
        coefs,
    ):
        """
        Calculate negative log-likehood and its first and second derivatives
        for a given set of coefs.

        Cribbed from:
        https://github.com/sebp/scikit-survival/blob/d5fcbc9e06f4824f987b5cf550ce0bf82665f28e/sksurv/linear_model/coxph.py#L131
        """
        # sort descending
        idx = np.argsort(-self.df_train[self.timeline_column])
        time = self.df_train[self.timeline_column].values[idx]
        X = self.df_train[self.covariates].values[idx]
        event = self.df_train[self.event_column].values[idx]

        n_samples = X.shape[0]
        xw = np.dot(X, coefs)
        exp_xw = np.exp(xw)
        n_samples, n_features = X.shape

        gradient = np.zeros((1, n_features))
        hessian = np.zeros((n_features, n_features))

        loss = 0
        risk_set = 0
        risk_set_x = np.zeros((1, n_features))
        risk_set_xx = np.zeros((n_features, n_features))
        k = 0
        # iterate time in descending order
        while k < n_samples:
            ti = time[k]
            n_events = 0
            covariate_sum = 0
            numerator = 0
            while k < n_samples and time[k] == ti:
                # partial hazard for sample k
                risk_set += exp_xw[k]

                # covariates for sample k
                xk = X[k].reshape(1, -1)
                risk_set_x += exp_xw[k] * xk
                risk_set_xx += exp_xw[k] * np.dot(xk.T, xk)
                if event[k]:
                    numerator += xw[k]  # log partial hazard for sample k
                    covariate_sum += xk  # covariates for sample k
                    n_events += 1
                k += 1

            if n_events > 0:
                loss += (
                    numerator - n_events * np.log(risk_set)
                ) / n_samples
                z = risk_set_x / risk_set
                gradient -= (covariate_sum - n_events * z) / n_samples

                a = risk_set_xx / risk_set
                b = np.dot(z.T, z)

                hessian += n_events * (a - b) / n_samples

        return (
            -loss,
            gradient[0],  # flatten 2d array
            hessian
        )

    def fit_naive(self):
        """
        The _log_loss_and_derivates method used to fit the coefficients via
        Newton's method can be a bit difficult to follow.

        Let's implement a fitter using naive parameter search instead.  It'll
        be much slower - and not exact - but hopefully will be easier to
        understand.
        """
        def loss_function(coefs):
            """
            https://en.wikipedia.org/wiki/Proportional_hazards_model#Likelihood_for_unique_times
            """            
            log_risk_scores = np.dot(
                self.df_train[self.covariates],
                coefs
            )
            risk_scores = np.exp(log_risk_scores)

            def log_partial_likelihood_at_time(t):

                event_idx = (
                    (self.df_train[self.timeline_column] == t) &
                    self.df_train[self.event_column]
                )
                # denominator / risk set is the sum of risk scores
                # for all samples who remain at risk at time t
                risk_set = risk_scores[
                    self.df_train[self.timeline_column] >= t
                ].sum()
                return (
                    log_risk_scores - np.log(risk_set)
                )[event_idx].sum()

            log_likelihood = sum([
                log_partial_likelihood_at_time(t)
                for t in self.df_train[self.timeline_column].unique()
            ])
            return -log_likelihood  # minimize

        self.coefs = dual_annealing(
            loss_function,
            bounds=[[-2, 2]] * len(self.covariates),
            maxiter=100,
        ).x
        return pd.Series(self.coefs, index=self.covariates)

    def _risk_scores(self, df):
        """
        exp(coefs * (covariates - covariates_mean))
        """
        assert len(self.coefs), 'Model has not been fit'
        return np.exp(
            (
                df[self.covariates] - self.df_train[self.covariates].mean()
            ).values.dot(self.coefs)
        )

    def predict_cumulative_hazard(self, df):
        """
        Nelson-Aalen baseline cumulative hazard function modulated
        by risk scores for each data point
        """
        return pd.DataFrame(
            np.outer(
                self.cumulative_baseline_hazard,
                self._risk_scores(df)
            ),
            index=self.cumulative_baseline_hazard.index,
        )

    def predict_survival_function(self, df):
        """
        Breslow's survival function estimator modulated by risk scores for each
        data point
        """
        return np.exp(-self.predict_cumulative_hazard(df))
