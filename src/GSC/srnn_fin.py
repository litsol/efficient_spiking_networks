#! /usr/bin/env python

# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

"""
This is a functional recurrent spiking neural network

"""

import os
import pprint
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from loguru import logger
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import efficient_spiking_networks.srnn_layers.spike_dense as sd
import efficient_spiking_networks.srnn_layers.spike_neuron as sn
import efficient_spiking_networks.srnn_layers.spike_rnn as sr
from GSC.data import Pad  # pylint: disable=C0301
from GSC.data import MelSpectrogram, Normalize, Rescale, SpeechCommandsDataset
from GSC.utils import generate_random_silence_files

# import snoop
# import deeplake
# from tqdm import tqdm_notebo

# Setup pretty printing
pp = pprint.PrettyPrinter(indent=4, width=41, compact=True)

# Setup logger level
logger.remove()
logger.add(sys.stderr, level="INFO")

sys.path.append("..")

# device = torch.device("cpu")
device = torch.device(  # pylint: disable=E1101
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
logger.info(f"{device=}")


# Setup number of workers dependent upon where the code is run
NUMBER_OF_WORKERS = 4 if device.type == "cpu" else 8
logger.info(f"The Dataloader will spawn {NUMBER_OF_WORKERS} worker processes.")

# Data Directories
TRAIN_DATA_ROOT = "./DATA/train"
TEST_DATA_ROOT = "./DATA/test"

# Specify the learning rate
LEARNING_RATE = 3e-3  # 1.2e-2

EPOCHS = 1

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
transform = torchvision.transforms.Compose([pad, melspec, rescale])


# Define the overall RNN network
class RecurrentSpikingNetwork(nn.Module):  # pylint: disable=R0903

    """
    Class docstring
    """

    def __init__(
        self,
    ):
        """
        Constructor docstring
        """
        super().__init__()
        N = 256  # pylint: disable=C0103
        # IS_BIAS=False

        # Here is what the network looks like
        self.dense_1 = sd.SpikeDENSE(
            40 * 3,
            N,
            tau_adp_inital_std=50,
            tau_adp_inital=200,
            tau_m=20,
            tau_m_inital_std=5,
            device=device,
            bias=IS_BIAS,
        )
        self.rnn_1 = sr.SpikeRNN(
            N,
            N,
            tau_adp_inital_std=50,
            tau_adp_inital=200,
            tau_m=20,
            tau_m_inital_std=5,
            device=device,
            bias=IS_BIAS,
        )
        self.dense_2 = sd.ReadoutIntegrator(
            N, 12, tau_m=10, tau_m_inital_std=1, device=device, bias=IS_BIAS
        )

        # self.dense_2 = sr.spike_rnn(
        #     N,
        #     12,
        #     tauM=10,
        #     tauM_inital_std=1,
        #     device=device,
        #     bias=IS_BIAS, #10
        # )

        # Please comment this code
        self.thr = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.thr, 5e-2)

        # Initialize the network layers
        torch.nn.init.kaiming_normal_(self.rnn_1.recurrent.weight)

        torch.nn.init.xavier_normal_(self.dense_1.dense.weight)
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)

        if IS_BIAS:
            torch.nn.init.constant_(self.rnn_1.recurrent.bias, 0)
            torch.nn.init.constant_(self.dense_1.dense.bias, 0)
            torch.nn.init.constant_(self.dense_2.dense.bias, 0)

    def forward(self, inputs):  # pylint: disable=R0914
        """
        Forward member function docstring
        """
        # What is this that returns 4 values?
        # What is b?
        # Stereo channels?
        (
            b,  # pylint: disable=C0103
            channel,
            seq_length,
            inputs_dim,
        ) = inputs.shape
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.rnn_1.set_neuron_state(b)

        fr_1 = []
        fr_2 = []
        # fr_3 = []
        output = 0

        # inputs_s = inputs
        # Why multiply by 1?
        inputs_s = (
            thr_func(inputs - self.thr) * 1.0
            - thr_func(-self.thr - inputs) * 1.0
        )

        # For every timestep update the membrane potential
        for i in range(seq_length):
            inputs_x = inputs_s[:, :, i, :].reshape(b, channel * inputs_dim)
            (
                mem_layer1,  # mem_layer1 unused! pylint: disable=W0612,C0301
                spike_layer1,
            ) = self.dense_1.forward(inputs_x)
            (
                mem_layer2,  # mem_layer2 unused! pylint: disable=W0612,C0301
                spike_layer2,
            ) = self.rnn_1.forward(spike_layer1)
            # mem_layer3,spike_layer3 = self.dense_2.forward(spike_layer2)
            mem_layer3 = self.dense_2.forward(spike_layer2)

            # #tracking #spikes (firing rate)
            output += mem_layer3
            fr_1.append(spike_layer1.detach().cpu().numpy().mean())
            fr_2.append(spike_layer2.detach().cpu().numpy().mean())
            # fr_3.append(spike_layer3.detach().cpu().numpy().mean())

        output = F.log_softmax(output / seq_length, dim=1)
        return output, [
            np.mean(np.abs(inputs_s.detach().cpu().numpy())),
            np.mean(fr_1),
            np.mean(fr_2),
        ]


# Please comment this code
def collate_fn(data):
    """
    Collate function docscting
    """

    x_batch = np.array([d[0] for d in data])  # pylint: disable=C0103
    std = x_batch.std(axis=(0, 2), keepdims=True)
    x_batch = torch.tensor(x_batch / std)  # pylint: disable=E1101
    y_batch = torch.tensor([d[1] for d in data])  # pylint: disable=C0103,E1101

    return x_batch, y_batch


def test(data_loader, is_show=0):
    """
    test function docstring
    """

    test_acc = 0.0
    sum_sample = 0.0
    fr_ = []
    for _, (images, labels) in enumerate(data_loader):
        images = images.view(-1, 3, 101, 40).to(device)

        labels = labels.view((-1)).long().to(device)
        predictions, fr = model(images)  # pylint: disable=C0103
        fr_.append(fr)
        values, predicted = torch.max(  # pylint: disable=W0612,E1101
            predictions.data, 1
        )
        labels = labels.cpu()
        predicted = predicted.cpu().t()

        test_acc += (predicted == labels).sum()
        sum_sample += predicted.numel()
    mean_fr = np.mean(fr_, axis=0)
    if is_show:
        logger.info(f"Mean FR: {mean_fr}")

    return test_acc.data.cpu().numpy() / sum_sample, mean_fr


def train(
    epochs, criterion, optimizer, scheduler=None
):  # pylint: disable=R0914
    """
    train function docstring
    """
    acc_list = []
    best_acc = 0

    path = "../model/"  # .pth'
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0
        for _, (images, labels) in enumerate(train_dataloader):
            # if i ==0:
            images = images.view(-1, 3, 101, 40).to(device)

            labels = labels.view((-1)).long().to(device)
            optimizer.zero_grad()

            predictions, _ = model(images)
            values, predicted = torch.max(  # pylint: disable=W0612,E1101
                predictions.data, 1
            )

            logger.debug(f"predictions:\n{pp.pformat(predictions)}]")
            logger.debug(f"labels:\n{pp.pformat(labels)}]")
            train_loss = criterion(predictions, labels)

            logger.debug(f"{predictions=}\n{predicted=}")

            train_loss.backward()
            train_loss_sum += train_loss.item()
            optimizer.step()

            labels = labels.cpu()
            predicted = predicted.cpu().t()

            train_acc += (predicted == labels).sum()
            sum_sample += predicted.numel()

        if scheduler:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy() / sum_sample
        valid_acc, _ = test(test_dataloader, 1)
        train_loss_sum += train_loss

        acc_list.append(train_acc)
        logger.info(f"{optimizer.param_groups[0]['lr']=}")

        if valid_acc > best_acc and train_acc > 0.890:
            best_acc = valid_acc
            torch.save(model, path + str(best_acc)[:7] + "-srnn-v3.pth")
        logger.info(f"{model.thr=}")

        training_loss = train_loss_sum / len(train_dataloader)
        logger.info(
            f"{epoch=:}, {training_loss=}, {train_acc=:.4f}, {valid_acc=:.4f}"
        )

    return acc_list


# Definitions complete - let's get going!

# list the directories and folders in TRAIN_DATA_ROOT folder
training_words = os.listdir(TRAIN_DATA_ROOT)

# Isolate the directories in the train_date_root
training_words = [
    x
    for x in training_words  # pylint: disable=C0103
    if os.path.isdir(os.path.join(TRAIN_DATA_ROOT, x))
]

# Ignore those that begin with an underscore
training_words = [
    x
    for x in training_words  # pylint: disable=C0103
    if os.path.isdir(os.path.join(TRAIN_DATA_ROOT, x))
    if x[0] != "_"
]
logger.info(
    f"traiing words[{len(training_words)}]:\n{pp.pformat(training_words)}]"
)

# list the directories and folders in TEST_DATA_ROOT folder
testing_words = os.listdir(TEST_DATA_ROOT)

# Look for testing_word directories in TRAIN_DATA_ROOT so that we only
# select test data for selected training classes.
testing_words = [
    x
    for x in testing_words  # pylint: disable=C0103
    if os.path.isdir(os.path.join(TRAIN_DATA_ROOT, x))
]

# Ignore those that begin with an underscore
testing_words = [
    x
    for x in testing_words  # pylint: disable=C0103
    if os.path.isdir(os.path.join(TRAIN_DATA_ROOT, x))
    if x[0] != "_"
]
logger.info(
    f"testing words[{len(testing_words)}]:\n{pp.pformat(testing_words)}]"
)

# Create a dictionary whose keys are
# testing_words(in the TRAIN_DATA_ROOT)
# and whose values are the words' ordianal
# position in the original list.

label_dct = {
    k: i for i, k in enumerate(testing_words + ["_silence_", "_unknown_"])
}

# Look for training directories in testing directories.
for w in training_words:
    label = label_dct.get(w)
    if label is None:
        label_dct[w] = label_dct["_unknown_"]

# Dictionary of testing words plus training words not in testing words.
logger.info(pp.pformat(f"{len(label_dct)=}, {label_dct=}"))

noise_path = os.path.join(TRAIN_DATA_ROOT, "_background_noise_")
noise_files = []
for f in os.listdir(noise_path):
    if f.endswith(".wav"):
        full_name = os.path.join(noise_path, f)
        noise_files.append(full_name)

logger.info(f"noise_files[{len(noise_files)}]:\n{pp.pformat(noise_files)}]")


# generate silence training and validation data

silence_folder = os.path.join(TRAIN_DATA_ROOT, "_silence_")
if not os.path.exists(silence_folder):
    os.makedirs(silence_folder)
    # 260 validation / 2300 training
    generate_random_silence_files(
        2560, noise_files, SIZE, os.path.join(silence_folder, "rd_silence")
    )

    # save 260 files for validation
    silence_files = list(os.listdir(silence_folder))
    silence_lines = [
        "_silence_/" + fname + "\n" for fname in silence_files[:260]
    ]
    silence_filename = os.path.join(
        TRAIN_DATA_ROOT, "silence_validation_list.txt"
    )
    with open(silence_filename, "a", encoding="utf-8") as fp:
        fp.writelines(silence_lines)


# Collect the training, testing and validation data

train_dataset = SpeechCommandsDataset(
    TRAIN_DATA_ROOT,
    label_dct,
    transform=transform,
    mode="train",
    max_nb_per_class=None,
)

item, label = train_dataset[0]
logger.info(f"Shape of train item: {item.shape}")
logger.info(f"Label of train item: {label}")

train_sampler = torch.utils.data.WeightedRandomSampler(
    train_dataset.weights, len(train_dataset.weights)
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUMBER_OF_WORKERS,
    sampler=train_sampler,
    collate_fn=collate_fn,
)

train_features, train_labels = next(iter(train_dataloader))
logger.info(f"Train Feature batch shape: {train_features.size()}")
logger.info(f"Train Labels batch shape: {train_labels.size()}")
logger.info(f"Train labels:\n{pp.pformat(train_labels)}]")

valid_dataset = SpeechCommandsDataset(
    TRAIN_DATA_ROOT,
    label_dct,
    transform=transform,
    mode="valid",
    max_nb_per_class=None,
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUMBER_OF_WORKERS,
    collate_fn=collate_fn,
)

test_dataset = SpeechCommandsDataset(
    TEST_DATA_ROOT, label_dct, transform=transform, mode="test"
)

item, label = test_dataset[0]
logger.info(f"Shape of test item: {item.shape}")
logger.info(f"Label of test item: {label}")

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUMBER_OF_WORKERS,
    collate_fn=collate_fn,
)

test_features, test_labels = next(iter(test_dataloader))
logger.info(f"Test Feature batch shape: {test_features.size()}")
logger.info(f"Test Labels batch shape: {test_labels.size()}")
logger.info(f"Test labels:\n{pp.pformat(test_labels)}]")

# Specify the function that will apply the forward and backward passes
thr_func = sn.ActFunADP.apply
IS_BIAS = True

# Instantiate the model
model = RecurrentSpikingNetwork()
criterion_f = nn.CrossEntropyLoss()  # nn.NLLLoss()

model.to(device)


test_acc_before_training = test(test_dataloader)
logger.info(f"{test_acc_before_training=}")

if IS_BIAS:
    base_params = [
        model.dense_1.dense.weight,
        model.dense_1.dense.bias,
        model.rnn_1.dense.weight,
        model.rnn_1.dense.bias,
        model.rnn_1.recurrent.weight,
        model.rnn_1.recurrent.bias,
        # model.dense_2.recurrent.weight,
        # model.dense_2.recurrent.bias,
        model.dense_2.dense.weight,
        model.dense_2.dense.bias,
    ]
else:
    base_params = [
        model.dense_1.dense.weight,
        model.rnn_1.dense.weight,
        model.rnn_1.recurrent.weight,
        model.dense_2.dense.weight,
    ]

optimizer_f = torch.optim.Adam(
    [
        {"params": base_params, "lr": LEARNING_RATE},
        {"params": model.thr, "lr": LEARNING_RATE * 0.01},
        {"params": model.dense_1.tau_m, "lr": LEARNING_RATE * 2},
        {"params": model.dense_2.tau_m, "lr": LEARNING_RATE * 2},
        {"params": model.rnn_1.tau_m, "lr": LEARNING_RATE * 2},
        {"params": model.dense_1.tau_adp, "lr": LEARNING_RATE * 2.0},
        #   {'}params': model.dense_2.tau_adp, 'lr': LEARNING_RATE * 10},
        {"params": model.rnn_1.tau_adp, "lr": LEARNING_RATE * 2.0},
    ],
    lr=LEARNING_RATE,
)

# scheduler_f = StepLR(optimizer_f, step_size=20, gamma=.5) # 20
scheduler_f = StepLR(optimizer_f, step_size=10, gamma=0.1)  # 20
# scheduler_f = LambdaLR(optimizer_f,lr_lambda=lambda epoch: 1-epoch/70)
# scheduler_f = ExponentialLR(optimizer_f, gamma=0.85)

train_acc_training_complete = train(
    EPOCHS, criterion_f, optimizer_f, scheduler_f
)
logger.info(f"{train_acc_training_complete=}")

logger.info("TRAINING COMPLETE")

test_acc_after_training = test(test_dataloader)
logger.info(f"{test_acc_after_training}")

logger.info("TESTING COMPLETE")

# finis

# Local Variables:
# compile-command: "pyflakes srnn_fin.py; pylint-3 -d E0401 -f parseable srnn_fin.py" # NOQA, pylint: disable=C0301
# End:
