import pandas as pd


def baseline_survival(*args, **kwargs):
    """
    Kaplan-Meier estimator
    """
    return kaplan_meier(*args, **kwargs)


def kaplan_meier(df, timeline_column, event_column):
    """
    https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator
    """
    return _event_table(df, timeline_column, event_column).apply(
        lambda x: (x['at_risk'] - x['events']) / x['at_risk'],
        axis=1
    ).cumprod()


def cumulative_hazard(df, timeline_column, event_column):
    """
    Nelson-Allen estimator
    https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html#estimating-hazard-rates-using-nelson-aalen
    """
    dfe = _event_table(df, timeline_column, event_column)
    return (dfe['events'] / dfe['at_risk']).cumsum()


def _event_table(df, timeline_column, event_column):
    return pd.DataFrame([{
        'time': t,
        'events': df[df[timeline_column] == t][event_column].sum(),
        'at_risk': len(df) - df[df[timeline_column] < t][event_column].count()
    }
        for t in df[timeline_column].unique()
    ] + [{
        'time': 0,
        'events': 0,
        'at_risk': len(df)
    }]).set_index('time').sort_index()
