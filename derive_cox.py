from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines import datasets
from matplotlib import pyplot as plt
from src import survival


# Explore using the lifelines Larynx example dataset
df = datasets.load_larynx()


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
    'Out baseline survival estimate'
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
