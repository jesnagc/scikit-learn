import pytest
from unittest import mock
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from collections import defaultdict

@pytest.fixture
def mock_data():
    with mock.patch('sklearn.datasets.make_classification') as mock_make_classification:
        mock_make_classification.return_value = (np.random.rand(100_000, 20), np.random.randint(0, 2, 100_000))
        yield mock_make_classification

@pytest.fixture
def mock_train_test_split():
    with mock.patch('sklearn.model_selection.train_test_split') as mock_split:
        mock_split.return_value = (np.random.rand(1000, 20), np.random.rand(99000, 20), np.random.randint(0, 2, 1000), np.random.randint(0, 2, 99000))
        yield mock_split

@pytest.fixture
def mock_logistic_regression():
    with mock.patch('sklearn.linear_model.LogisticRegression') as mock_lr:
        instance = mock_lr.return_value
        instance.fit.return_value = None
        instance.predict_proba.return_value = np.random.rand(99000, 2)
        instance.predict.return_value = np.random.randint(0, 2, 99000)
        yield instance

@pytest.fixture
def mock_gaussian_nb():
    with mock.patch('sklearn.naive_bayes.GaussianNB') as mock_gnb:
        instance = mock_gnb.return_value
        instance.fit.return_value = None
        instance.predict_proba.return_value = np.random.rand(99000, 2)
        instance.predict.return_value = np.random.randint(0, 2, 99000)
        yield instance

@pytest.fixture
def mock_linear_svc():
    with mock.patch('sklearn.svm.LinearSVC') as mock_svc:
        instance = mock_svc.return_value
        instance.fit.return_value = None
        instance.decision_function.return_value = np.random.rand(1000)
        instance.predict_proba.return_value = np.random.rand(99000, 2)
        instance.predict.return_value = np.random.randint(0, 2, 99000)
        yield instance

@pytest.fixture
def mock_calibrated_classifier_cv():
    with mock.patch('sklearn.calibration.CalibratedClassifierCV') as mock_calibrated:
        instance = mock_calibrated.return_value
        instance.fit.return_value = None
        instance.predict_proba.return_value = np.random.rand(99000, 2)
        yield instance

@pytest.fixture
def mock_calibration_display():
    with mock.patch('sklearn.calibration.CalibrationDisplay') as mock_display:
        mock_display.from_estimator.return_value = mock.Mock()
        yield mock_display

@pytest.fixture
def mock_defaultdict():
    with mock.patch('collections.defaultdict') as mock_dd:
        yield mock_dd

# happy_path - test_train_test_split - Test that train_test_split splits the dataset into train and test sets with specified test size.
def test_train_test_split(mock_train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)
    assert X_train.shape == (1000, 20)
    assert X_test.shape == (99000, 20)
    assert len(y_train) == 1000
    assert len(y_test) == 99000

# happy_path - test_fit - Test that fit method trains the classifier on the training data.
def test_fit(mock_logistic_regression):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    assert clf.fit.called

# happy_path - test_predict_proba - Test that predict_proba returns probability estimates for test data.
def test_predict_proba(mock_logistic_regression):
    clf = LogisticRegression()
    probabilities = clf.predict_proba(X_test)
    assert probabilities.shape == (99000, 2)

# happy_path - test_from_estimator - Test that from_estimator plots calibration curve for the classifier.
def test_from_estimator(mock_calibration_display):
    clf = LogisticRegression()
    display = CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=10, name="Logistic")
    assert mock_calibration_display.from_estimator.called

# edge_case - test_make_classification_zero_samples - Test that make_classification handles zero samples gracefully.
def test_make_classification_zero_samples():
    X, y = make_classification(n_samples=0, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    assert X.shape == (0, 20)
    assert len(y) == 0

# edge_case - test_train_test_split_invalid_test_size - Test that train_test_split raises an error with invalid test size.
def test_train_test_split_invalid_test_size():
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5, random_state=42)

# edge_case - test_fit_empty_data - Test that fit method handles empty training data without crashing.
def test_fit_empty_data(mock_logistic_regression):
    clf = LogisticRegression()
    with pytest.raises(ValueError):
        clf.fit([], [])

# edge_case - test_predict_proba_empty_data - Test that predict_proba returns empty array for empty test data.
def test_predict_proba_empty_data(mock_logistic_regression):
    clf = LogisticRegression()
    probabilities = clf.predict_proba([])
    assert probabilities.shape == (0, 2)

# edge_case - test_from_estimator_empty_data - Test that from_estimator handles empty data without plotting.
def test_from_estimator_empty_data(mock_calibration_display):
    clf = LogisticRegression()
    display = CalibrationDisplay.from_estimator(clf, [], [], n_bins=10, name="Logistic")
    assert not mock_calibration_display.from_estimator.called

