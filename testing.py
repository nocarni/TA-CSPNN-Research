import logging
import os.path
import time
from collections import OrderedDict
import sys

import numpy as np


from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
)
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne


data_folder = './bci_2a'
subject_id = 1
low_cut_hz = 4

ival = [500, 2500]
max_epochs = 500
max_increase_epochs = 160
batch_size = 60
high_cut_hz = 40
factor_new = 1e-3
init_block_size = 1000
valid_set_fraction = 0.1

train_filename = "A{:02d}T.gdf".format(subject_id)
test_filename = "A{:02d}E.gdf".format(subject_id)
train_filepath = os.path.join(data_folder, train_filename)
test_filepath = os.path.join(data_folder, test_filename)
train_label_filepath = train_filepath.replace(".gdf", ".mat")
test_label_filepath = test_filepath.replace(".gdf", ".mat")

train_loader = BCICompetition4Set2A(
    train_filepath, labels_filename=train_label_filepath
)
test_loader = BCICompetition4Set2A(
    test_filepath, labels_filename=test_label_filepath
)
train_cnt = train_loader.load()
test_cnt = test_loader.load()

train_cnt = train_cnt.drop_channels(
        ["EOG-left", "EOG-central", "EOG-right"]
    )
assert len(train_cnt.ch_names) == 22
# lets convert to millvolt for numerical stability of next operations
train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
train_cnt = mne_apply(
    lambda a: bandpass_cnt(
        a,
        low_cut_hz,
        high_cut_hz,
        train_cnt.info["sfreq"],
        filt_order=3,
        axis=1,
    ),
    train_cnt,
)

train_cnt = mne_apply(
    lambda a: exponential_running_standardize(
        a.T,
        factor_new=factor_new,
        init_block_size=init_block_size,
        eps=1e-4,
    ).T,
    train_cnt,
)
test_cnt = test_cnt.drop_channels(["EOG-left", "EOG-central", "EOG-right"])
assert len(test_cnt.ch_names) == 22
test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
test_cnt = mne_apply(
    lambda a: bandpass_cnt(
        a,
        low_cut_hz,
        high_cut_hz,
        test_cnt.info["sfreq"],
        filt_order=3,
        axis=1,
    ),
    test_cnt,
)
test_cnt = mne_apply(
    lambda a: exponential_running_standardize(
        a.T,
        factor_new=factor_new,
        init_block_size=init_block_size,
        eps=1e-4,
    ).T,
    test_cnt,
)

marker_def = OrderedDict([
        ("Left Hand", [1]),
        ("Right Hand", [2]),
        ("Foot", [3]),
        ("Tongue", [4]),
    ])

train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

train_set, valid_set = split_into_two_sets(
    train_set, first_set_fraction=1 - valid_set_fraction
)

print(type(train_set.X))
print(type(train_set.y))
print(train_set.X.shape)