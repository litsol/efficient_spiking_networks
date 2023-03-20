#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0
# flake8: noqa
# pylint: skip-file
# type: ignore
# REUSE-IgnoreStart

import inspect
import os
import pprint
import sys
from pathlib import Path

import IPython.display as ipd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from utilities.genfind import gen_find

pp = pprint.PrettyPrinter(indent=4, compact=True, width=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATAROOT = Path("DATA")

noise_files = [*gen_find("*.wav", DATAROOT / "train" / "_background_noise_")]
print(noise_files)

silence_folder = Path(DATAROOT / "train" / "_silence_").mkdir(
    parents=True, exist_ok=True
)

breakpoint()


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(
            "/export/scratch2/guravage/google/",
            download=True,
            url="speech_commands_v0.01",
        )

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in fileobj
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list(
                "testing_list.txt"
            )
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")
validation_set = SubsetSC("validation")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.plot(waveform.t().numpy())

waveform_first, *_ = train_set[0]
ipd.Audio(waveform_first.numpy(), rate=sample_rate)

waveform_second, *_ = train_set[1]
ipd.Audio(waveform_second.numpy(), rate=sample_rate)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.0
    )
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False
training_labels = labels = sorted(
    list(set(datapoint[2] for datapoint in train_set))
)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

testing_labels = labels = sorted(
    list(set(datapoint[2] for datapoint in train_set))
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

validation_labels = labels = sorted(
    list(set(datapoint[2] for datapoint in train_set))
)
validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)

print(word_start, "-->", index, "-->", word_recovered)


train_features, train_labels = next(iter(train_loader))
print(f"Train Feature batch shape: {train_features.size()}")
print(f"Train Labels batch shape: {train_labels.size()}")
print(f"Train labels, i.e. indices:\n{pp.pformat(train_labels)}")
print(
    f"Training labels[{len(training_labels)}]:\n{pp.pformat(training_labels)}"
)

test_features, test_labels = next(iter(test_loader))
print(f"Test Feature batch shape: {test_features.size()}")
print(f"Test Labels batch shape: {test_labels.size()}")
print(f"Test labels, i.e. indices:\n{pp.pformat(test_labels)}")
print(f"Test labels[{len(testing_labels)}]:\n{pp.pformat(testing_labels)}")

val_features, val_labels = next(iter(validation_loader))
print(f"Validation Feature batch shape: {val_features.size()}")
print(f"Validation Labels batch shape: {val_labels.size()}")
print(f"Validation labels, i.e. indices:\n{pp.pformat(val_labels)}]")
print(
    f"Validation labels[{len(validation_labels)}]:\n{pp.pformat(validation_labels)}"
)

breakpoint()
# REUSE-IgnoreEnd
# finis

# Local Variables:
# compile-command: "pyflakes pta.py; pylint-3 -d E0401 -f parseable pta.py" # NOQA, pylint: disable=C0301
# End:
