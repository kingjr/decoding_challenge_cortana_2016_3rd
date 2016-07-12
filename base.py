# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from pandas import DataFrame, concat

from externals.fix import make_pipeline  # else bug in old sklearn version
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

from externals.mne.decoding import PSDEstimator, GeneralizationLight
from externals.mne import set_log_level
set_log_level(False)

from externals.pyriemann.estimation import ERPCovariances, Xdawn
from externals.pyriemann.tangentspace import TangentSpace
from externals.pyriemann.channelselection import FlatChannelRemover

from externals.transformers import (Baseliner, TimeFreqDecomposer, TimePadder,
                                    TimeSelector, Reshaper, CustomEnsemble,
                                    Filterer, force_predict)

n_jobs = 1

transform_tf = make_pipeline(
    FlatChannelRemover(),
    Baseliner(),
    Xdawn(10, estimator='oas'),
    TimePadder(1500),
    TimeFreqDecomposer(sfreq=1000, frequencies=np.logspace(.5, 2.5, 20),
                       decim=slice(1500, -1500, 2), n_jobs=n_jobs),
    Reshaper(), LogisticRegression()
)


transform_gat = make_pipeline(
    FlatChannelRemover(),
    Baseliner(),
    TimePadder(400),
    Filterer(sfreq=1000, l_freq=None, h_freq=20, n_jobs=n_jobs),
    TimeSelector(slice(800, -400, 20)),
    Xdawn(15, estimator='oas'),
    GeneralizationLight(force_predict(SVC(probability=True)), n_jobs=-1),
    Reshaper(), LogisticRegression()
)


transform_xdawn = make_pipeline(
    FlatChannelRemover(),
    Baseliner(),
    TimePadder(400),
    Filterer(sfreq=1000, l_freq=None, h_freq=20, n_jobs=n_jobs),
    TimeSelector(slice(800, -400, 10)),
    Xdawn(10, estimator='oas'),
    Reshaper(), LogisticRegression()
    )


transform_cov = make_pipeline(
    FlatChannelRemover(),  # check if present??
    ERPCovariances(estimator='oas'),
    TangentSpace(),
    Reshaper(), LogisticRegression()
)


transform_psd = make_pipeline(
    FlatChannelRemover(),
    Baseliner(),
    TimePadder(400),
    Xdawn(20, estimator='oas'),
    PSDEstimator(sfreq=1000, fmax=200, n_jobs=n_jobs),
    Reshaper(), LogisticRegression()
)

pipe = make_pipeline(
    Reshaper(),
    BaggingClassifier(make_pipeline(
        Reshaper([64, 800]),
        CustomEnsemble([
            make_pipeline(transform_xdawn),
            make_pipeline(transform_tf),
            make_pipeline(transform_gat),
            make_pipeline(transform_psd),
            make_pipeline(transform_cov),
            ]),
        LogisticRegression()
    )))


def get_data(csv, patient=0):
    """Read and Epoch data."""
    from externals.mne.io import RawArray
    from externals.mne import Epochs, create_info, find_events

    # prepare structure
    patients = np.unique(csv['PatientID'])
    ch_eeg = ['Electrode_%i' % ch for ch in range(1, 65)]
    ch_stim = ['Stimulus_Type', 'Stimulus_ID']
    ch_names = ch_eeg + ch_stim
    ch_types = np.r_[np.tile(['ecog'], len(ch_eeg)),
                     np.tile(['stim'], len(ch_stim))]
    sel = np.where(csv['PatientID'] == patients[patient])
    data = csv[ch_names].iloc[sel].as_matrix().T

    when = data[-2, :]
    when[when == 101] = 0
    when = when[1:] - when[:-1]
    when[when < 0] = 0
    when = np.where(when)[0] + 1

    what_type = data[-2, when]
    what_id = data[-1, when]
    data[-2:, :] *= 0
    for tim, this_id, this_type in zip(when, what_id, what_type):
        data[-2, tim:(tim + 20)] = (this_type > 50) + 1
        data[-1, tim:(tim + 20)] = this_id

    info = create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
    raw = RawArray(data, info=info)

    # epochs
    events = find_events(raw, stim_channel='Stimulus_Type')
    epochs = Epochs(raw=raw, events=events, baseline=None,
                    tmin=-.400, tmax=.399, preload=True)
    y = find_events(raw, stim_channel='Stimulus_Type')[:, 2] == 2
    stim_id = find_events(raw, stim_channel='Stimulus_ID')[:, 2]

    # output as X and y
    X = epochs._data[:, :64, :]
    return X, y, stim_id, what_type - 1


def azureml_main(csv_train=None, csv_test=None):
    """Fit and predict each patient separately.

    Parameters
    ----------
    csv_train : csv
        The public training dataset
    csv_test : csv
        Either the public training dataset (to check that there is no error),
        or the private test dataset.

    Returns
    -------
    all_preds : DataFrame
        The output DataFrame containing 'PatientID', 'Stimulus_ID' and
        'Scored Labels' for each trial.
    """
    all_preds = list()
    for patient in range(4):
        # fit on labelled data
        X_train, y_train, stim_id_train, _ = get_data(csv_train, patient)
        pipe.fit(X_train, y=y_train)

        # Predict test data
        X_test, _, stim_id_test, _ = get_data(csv_test, patient)
        y_pred_proba = pipe.predict_proba(X_test)[:, 1]

        # Store predictions in required format
        y_pred = np.array(y_pred_proba > np.median(y_pred_proba), int) + 1
        y_pred = DataFrame({'PatientID': ['p%i' % (patient + 1)] * len(X_test),
                            'Stimulus_ID': stim_id_test,
                            'Scored Labels': y_pred})
        all_preds.append(y_pred)
    all_preds = concat(all_preds)

    return all_preds
