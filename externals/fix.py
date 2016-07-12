"""Azure scikit-learn is old, so we need to patch it."""
from collections import defaultdict
from sklearn.externals import six
from sklearn.pipeline import Pipeline as OrigPipeline


class Pipeline(OrigPipeline):
    @property
    def classes_(self):
        return self.steps[-1][-1].classes_


def make_pipeline(*steps):
    """Construct a Pipeline from the given estimators.
    This is a shorthand for the Pipeline constructor; it does not require, and
    does not permit, naming the estimators. Instead, they will be given names
    automatically based on their types.
    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> make_pipeline(StandardScaler(), GaussianNB())
    Pipeline(steps=[('standardscaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('gaussiannb', GaussianNB())])
    Returns
    -------
    p : Pipeline
    """
    return Pipeline(_name_estimators(steps))


def _name_estimators(estimators):
    """Generate names for estimators."""

    names = [type(estimator).__name__.lower() for estimator in estimators]
    namecount = defaultdict(int)
    for est, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(six.iteritems(namecount)):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))
