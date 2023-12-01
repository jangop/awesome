import sklearn.utils.estimator_checks

from awesome import AwesomeClassifier


def test_classifier_almost_manually():
    X = [[1, 2], [3, 4]]
    y = [0, 1]
    clf = AwesomeClassifier()
    clf.fit(X, y)
    assert all(clf.predict(X) == [0, 0])


@sklearn.utils.estimator_checks.parametrize_with_checks([AwesomeClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
