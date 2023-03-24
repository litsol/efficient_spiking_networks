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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchvision
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.utils import _load_waveform

from GSC.data import Pad  # pylint: disable=C0301
from GSC.data import MelSpectrogram, Normalize, Rescale, SpeechCommandsDataset

# from GSC.utils import generate_random_silence_files
from GSC.utils import generate_noise_files
from utilities.genfind import gen_find

pp = pprint.PrettyPrinter(indent=4, compact=True, width=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Here's where we'll find our data
GSC_URL = "speech_commands_v0.02"
DATAROOT = Path("google")
GSC = DATAROOT / "SpeechCommands" / GSC_URL

BATCH_SIZE = 32
SIZE = 16000
SR = 16000  # Sampling Rate 16Hz ?

DELTA_ORDER = 2
FMAX = 4000
FMIN = 20
HOP_LENGTH = int(10e-3 * SR)
N_FFT = int(30e-3 * SR)
N_MELS = 40
STACK = True

# Turn wav files into Melspectrograms
melspec = MelSpectrogram(
    SR, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX, DELTA_ORDER, stack=STACK
)

pad = Pad(SIZE)
rescale = Rescale()
normalize = Normalize()
transforms = torchvision.transforms.Compose([pad, melspec, rescale])


# Retrieve the Google Speech Commands Dataset
gsc_dataset = torchaudio.datasets.SPEECHCOMMANDS(
    DATAROOT, url=GSC_URL, folder_in_archive="SpeechCommands", download=True
)

# Compose a list of the GSC background noise files
background_noise_files = [*gen_find("*.wav", GSC / "_background_noise_")]

# This is the wav file we'll use to generate all our other white noise files.
background_noise_file = GSC / "_background_noise_" / "white_noise.wav"

# Create the folder where we'll write our white noise files
silence_folder = GSC / "_silence_"
silence_folder.mkdir(parents=True, exist_ok=True)

# generate_random_silence_files(nb_files=2560,
#                               noise_files=background_noise_files,
#                               size=16000,
#                               output_folder=silence_folder,
#                               file_prefix="rd_silence_",
#                               sr=16000)

generate_noise_files(
    nb_files=2560,
    noise_file=background_noise_file,
    output_folder=silence_folder,
    file_prefix="rd_silence_",
    sr=16000,
)

silence_files = [*gen_find("*.wav", silence_folder)]
with open(GSC / "silence_validation_list.txt", "w") as f:
    for filename in silence_files[:260]:
        f.write(f"{filename}\n")


class GSC_SSubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, transform=None):
        super().__init__(
            DATAROOT,
            url=GSC_URL,
            folder_in_archive="SpeechCommands",
            download=True,
        )
        self.transform = transform

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in fileobj
                ]

        if subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "validation":
            self._walker = load_list("validation_list.txt") + load_list(
                "silence_validation_list.txt"
            )
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list(
                "testing_list.txt"
            )
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

    def __getitem__(self, n):
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        m = torch.max(torch.abs(waveform))

        if m > 0:
            waveform /= m

        if self.transform is not None:
            waveform = self.transform(waveform.squeeze())
            # waveform = torch.from_numpy(waveform)
        return (waveform,) + metadata[1:]
        # return item, label


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(
            DATAROOT,
            url=GSC_URL,
            folder_in_archive="SpeechCommands",
            download=True,
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
gsc_test_set = GSC_SSubsetSC("testing", transform=transforms)

waveform, sample_rate, label, speaker_id, utterance_number = gsc_test_set[0]
print(f"Shape of gsc_test_set waveform: {waveform.shape}")
print(f"Sample rate of gsc_test_set  waveform: {sample_rate}")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
print(f"Shape of train_set waveform: {waveform.size()}")
print(f"Sample rate of train_set waveform: {sample_rate}")


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


def gsc_collate_fn(data):
    """
    Collate function docscting
    """
    x_batch = np.array([d[0] for d in data])  # pylint: disable=C0103
    std = x_batch.std(axis=(0, 2), keepdims=True)
    x_batch = torch.tensor(x_batch / std)  # pylint: disable=E1101
    y_batch = torch.tensor([d[1] for d in data])  # pylint: disable=C0103,E1101

    return x_batch, y_batch


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
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
train_features, train_labels = next(iter(train_loader))
print(f"Train Feature batch shape: {train_features.size()}")
print(f"Train Labels batch shape: {train_labels.size()}")
print(f"Train labels, i.e. indices:\n{pp.pformat(train_labels)}")
print(
    f"Training labels[{len(training_labels)}]:\n{pp.pformat(training_labels)}"
)


testing_labels = labels = sorted(
    list(set(datapoint[2] for datapoint in train_set))
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_features, test_labels = next(iter(test_loader))
print(f"Test Feature batch shape: {test_features.size()}")
print(f"Test Labels batch shape: {test_labels.size()}")
print(f"Test labels, i.e. indices:\n{pp.pformat(test_labels)}")
print(f"Test labels[{len(testing_labels)}]:\n{pp.pformat(testing_labels)}")


validation_labels = labels = sorted(
    list(set(datapoint[2] for datapoint in train_set))
)
validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
val_features, val_labels = next(iter(validation_loader))
print(f"Validation Feature batch shape: {val_features.size()}")
print(f"Validation Labels batch shape: {val_labels.size()}")
print(f"Validation labels, i.e. indices:\n{pp.pformat(val_labels)}]")
print(
    f"Validation labels[{len(validation_labels)}]:\n{pp.pformat(validation_labels)}"
)


gsc_test_loader = torch.utils.data.DataLoader(
    gsc_test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=gsc_collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
gsc_features, gsc_labels = next(iter(gsc_test_loader))
print(f"GSC Feature batch shape: {gsc_features.size()}")
print(f"GSC Labels batch shape: {gsc_labels.size()}")
print(f"GSC labels, i.e. indices:\n{pp.pformat(gsc_labels)}]")
print(f"GSC labels[{len(validation_labels)}]:\n{pp.pformat(gsc_labels)}")


breakpoint()
# REUSE-IgnoreEnd
# finis

# Local Variables:
# compile-command: "pyflakes pta.py; pylint-3 -d E0401 -f parseable pta.py" # NOQA, pylint: disable=C0301
# End:
