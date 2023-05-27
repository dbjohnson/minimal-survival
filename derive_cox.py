import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter
from lifelines import datasets
from sksurv.linear_model.coxph import CoxPHOptimizer
from matplotlib import pyplot as plt
from src import survival, cox


# Explore using the lifelines Larynx example dataset
df = datasets.load_larynx().sort_values(by='time')
covariates = [c for c in df.columns if c not in ['time', 'death']]


# km plotting function we'll use a lot
def plot(series, label):
    plt.step(
        series.index,
        series.values,
        'ro',
        where='post',
        label=label
    )
    plt.legend()
    plt.show()


# 1) estimate baseline survival
# let lifelines do it first...
km = KaplanMeierFitter()
km.fit(df['time'], event_observed=df['death'])
km.plot(label='Lifelines KaplanMeierFitter', ci_show=False)

# now let's do it ourselves
plot(
    survival.baseline_survival(df, 'time', 'death'),
    'Our baseline survival estimate'
)

# 2) estimate cumulative hazard
# let lifelines do it first...
na = NelsonAalenFitter()
na.fit(df['time'], event_observed=df['death'])
na.plot(label='Lifelines NelsonAalenFitter', ci_show=False)

# now let's do it ourselves
plot(
    survival.cumulative_hazard(df, 'time', 'death'),
    label='Our cumulative hazard estimate',
)

# 3) optimize cox model coefficients
# let lifelines do it first...
model = CoxPHFitter()
model.fit(df, 'time', event_col='death')
print('Cox model coefficients:')
print('theirs:')
print(model.summary['coef'])

# now let's do it ourselves
print('\nours:')
print(cox.fit_coefs(
    df,
    'time',
    'death',
    covariates
))

# 4) look at the details - estimate negative log likelihood, gradient, and Hessian
# use sksurv as the reference (these are exposed better than in lifelines)
model = CoxPHOptimizer(
    df[covariates].values,
    df['death'].values,
    df['time'].values,
    alpha=np.float64(0),
    ties='breslow',
)
dir(model)

weights = np.ones(len(covariates))
nlog_theirs = model.nlog_likelihood(weights)
# compute gradient and hessian
model.update(weights)

# we'll calculate these all at once...
nlog_ours, gradient_ours, hessian_ours = cox._log_loss_and_derivates(
    weights,
    df,
    'time',
    'death',
    covariates
)

print('\n\nnegative log-likelihood:')
print(f'theirs: {nlog_theirs}')
print(f'ours: {nlog_ours}')

print('\ngradient:')
print(f'theirs: {model.gradient}')
print(f'ours: {gradient_ours}')

print('\nhessian:')
print(f'theirs: {model.hessian}')
print(f'ours: {hessian_ours}')
