import pytest
from unittest import mock
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

@pytest.fixture
def mock_make_blobs():
    with mock.patch('sklearn.datasets.make_blobs') as mock_blobs:
        mock_blobs.return_value = (np.random.rand(2000, 2), np.random.randint(0, 3, 2000))
        yield mock_blobs

@pytest.fixture
def mock_random_forest_classifier():
    with mock.patch('sklearn.ensemble.RandomForestClassifier') as mock_rf:
        mock_rf_instance = mock_rf.return_value
        mock_rf_instance.fit.return_value = None
        mock_rf_instance.predict_proba.return_value = np.random.rand(1000, 3)
        yield mock_rf

@pytest.fixture
def mock_calibrated_classifier_cv():
    with mock.patch('sklearn.calibration.CalibratedClassifierCV') as mock_calibrated:
        mock_calibrated_instance = mock_calibrated.return_value
        mock_calibrated_instance.fit.return_value = None
        mock_calibrated_instance.predict_proba.return_value = np.random.rand(1000, 3)
        yield mock_calibrated

@pytest.fixture
def mock_log_loss():
    with mock.patch('sklearn.metrics.log_loss') as mock_log:
        mock_log.return_value = 0.5
        yield mock_log

@pytest.fixture
def mock_matplotlib():
    with mock.patch('matplotlib.pyplot.figure') as mock_fig, \
         mock.patch('matplotlib.pyplot.arrow') as mock_arrow, \
         mock.patch('matplotlib.pyplot.plot') as mock_plot, \
         mock.patch('matplotlib.pyplot.annotate') as mock_annotate, \
         mock.patch('matplotlib.pyplot.grid') as mock_grid, \
         mock.patch('matplotlib.pyplot.show') as mock_show:
        yield {
            'figure': mock_fig,
            'arrow': mock_arrow,
            'plot': mock_plot,
            'annotate': mock_annotate,
            'grid': mock_grid,
            'show': mock_show
        }

# happy_path - test_make_blobs_shape - Test that make_blobs generates a dataset with correct number of samples and features.
def test_make_blobs_shape(mock_make_blobs):
    X, y = make_blobs(n_samples=2000, n_features=2, centers=3, random_state=42, cluster_std=5.0)
    assert X.shape == (2000, 2)
    assert y.shape == (2000,)

# happy_path - test_random_forest_fit - Test that fit method trains RandomForestClassifier on train_valid dataset.
def test_random_forest_fit(mock_random_forest_classifier):
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(X[:1000], y[:1000])
    mock_random_forest_classifier.fit.assert_called_once_with(X[:1000], y[:1000])

# happy_path - test_predict_proba_output_shape - Test that predict_proba returns probabilities for each class.
def test_predict_proba_output_shape(mock_random_forest_classifier):
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(X[:1000], y[:1000])
    probs = clf.predict_proba(X[1000:])
    assert probs.shape == (1000, 3)

# happy_path - test_calibrated_classifier_cv - Test that CalibratedClassifierCV calibrates the classifier using sigmoid method.
def test_calibrated_classifier_cv(mock_calibrated_classifier_cv):
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(X[:600], y[:600])
    cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
    cal_clf.fit(X[600:1000], y[600:1000])
    mock_calibrated_classifier_cv.fit.assert_called_once_with(X[600:1000], y[600:1000])

# happy_path - test_log_loss_uncalibrated - Test that log_loss computes log loss correctly for uncalibrated classifier.
def test_log_loss_uncalibrated(mock_log_loss):
    clf_probs = np.random.rand(1000, 3)
    log_loss_value = log_loss(y[1000:], clf_probs)
    assert isinstance(log_loss_value, float)

# edge_case - test_make_blobs_negative_samples - Test that make_blobs raises error with negative number of samples.
def test_make_blobs_negative_samples():
    with pytest.raises(ValueError):
        make_blobs(n_samples=-10, n_features=2, centers=3)

# edge_case - test_fit_empty_data - Test that fit method raises error when input data is empty.
def test_fit_empty_data(mock_random_forest_classifier):
    clf = RandomForestClassifier(n_estimators=25)
    with pytest.raises(ValueError):
        clf.fit(np.array([]), np.array([]))

# edge_case - test_predict_proba_unfitted_model - Test that predict_proba raises error when model is not fitted.
def test_predict_proba_unfitted_model(mock_random_forest_classifier):
    clf = RandomForestClassifier(n_estimators=25)
    with pytest.raises(NotFittedError):
        clf.predict_proba(X[1000:])

# edge_case - test_calibrated_classifier_cv_invalid_method - Test that CalibratedClassifierCV raises error with invalid method.
def test_calibrated_classifier_cv_invalid_method(mock_calibrated_classifier_cv):
    clf = RandomForestClassifier(n_estimators=25)
    clf.fit(X[:600], y[:600])
    with pytest.raises(ValueError):
        CalibratedClassifierCV(clf, method='invalid_method', cv='prefit').fit(X[600:1000], y[600:1000])

# edge_case - test_log_loss_mismatched_lengths - Test that log_loss raises error with mismatched true and predicted lengths.
def test_log_loss_mismatched_lengths(mock_log_loss):
    clf_probs = np.random.rand(1000, 3)
    with pytest.raises(ValueError):
        log_loss(y[1000:1500], clf_probs)

