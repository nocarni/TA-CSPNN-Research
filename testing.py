import logging
import os.path
import time
from collections import OrderedDict
import sys

import numpy as np

from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A



data_folder = './bci_2a'
subject_id = 2
low_cut_hz = 4

ival = [-500, 4000]
max_epochs = 1600
max_increase_epochs = 160
batch_size = 60
high_cut_hz = 38
factor_new = 1e-3
init_block_size = 1000
valid_set_fraction = 0.2

train_filename = "A{:02d}T.gdf".format(subject_id)
test_filename = "A{:02d}E.gdf".format(subject_id)
train_filepath = os.path.join(data_folder, train_filename)
test_filepath = os.path.join(data_folder, test_filename)
train_label_filepath = train_filepath.replace(".gdf", ".mat")
test_label_filepath = test_filepath.replace(".gdf", ".mat")


print(train_filepath)
print(train_label_filepath)
train_loader = BCICompetition4Set2A(
    train_filepath, labels_filename=train_label_filepath
)
train_cnt = train_loader.load()
