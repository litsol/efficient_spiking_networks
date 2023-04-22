#! /usr/bin/env python
# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
# SPDX-License-Identifier: MPL-2.0

"""
This is the BPTT spiking recurrent neural network (srnn)
example using the Google Speech Commands dataset.
"""

import pprint
import random
import sys
from pathlib import Path

import numpy as np
import tomli

# import snoop  # for debugging when something goes wrong.
import torch
import torch.nn.functional as F
import torchaudio
import torchvision
import typer
from loguru import logger
from torch import nn
from torch.optim.lr_scheduler import StepLR

# srnn specific modules.
import efficient_spiking_networks.srnn_layers.spike_dense as sd
import efficient_spiking_networks.srnn_layers.spike_neuron as sn
import efficient_spiking_networks.srnn_layers.spike_rnn as sr
from GSC.data import (  # noqa: E501 pylint: disable=C0301
    GSCSSubsetSC,
    MelSpectrogram,
    Pad,
    Rescale,
)

# modules that are associated with this example.
from GSC.utils import generate_noise_files

# Two generator functions; part of our utilities suite.
from utilities.gendfind import gen_dfind  # pylint: disable=C0411
from utilities.genfind import gen_find  # pylint: disable=C0411

# Setup pretty printing.
pp = pprint.PrettyPrinter(indent=4, compact=True, width=42)


# Two class_dict helper functions.
def label_to_index(class_dict: dict, word: str) -> int:
    """
    Return the position of the word in labels.
    """
    return class_dict[word]


def index_to_label(class_dict: dict, index: int) -> str:
    """
    Return the word corresponding to the index in labels.
    This is the inverse of label_to_index.
    """

    return list(class_dict.keys())[list(class_dict.values()).index(index)]


def read_configuration(filename: Path) -> dict:
    """
    Several experiment parameters are defined in a TOML
    configuration file; whose filename is an argument to the the
    simulation code.

    This function reads the TOML configuration file, validate its's
    contents against a schema, and return a configuration dictionary.
    """
    with open(filename, mode="rb") as fp:  # pylint: disable=C0103
        config = tomli.load(fp)

    match config:
        case {
            "data": {"dataroot": str(), "gsc_url": str()},
            "srnn": {
                "network_size": int(),
                "learning_rate": float(),
                "epochs": int(),
                "batch_size": int(),
                "size": int(),
                "sample_rate": int(),
                "bias": bool(),
            },
            "mel": {
                "delta_order": int(),
                "fmax": int(),
                "fmin": int(),
                "n_mels": int(),
                "stack": bool(),
            },
            "logger": {"level": str()},
            "cuda": {"cuda": bool()},
        }:
            pass
        case _:
            raise ValueError(f"invalid configuration: {config}")
    return config


# Here begins the Spiking Recurrent Neural Network specific code.


def collate_fn(data):
    """
    This custom collate function is called with a list of data samples.
    It collates the input samples into a batch for yielding from the
    data loader iterator.

    Dividing the batch by its standard deviation yields a distribution with
    standard deviation equal to 1 (and a variance equal to 1^2=1).
    """
    x_batch = np.array([d[0] for d in data])  # pylint: disable=C0103
    std = x_batch.std(axis=(0, 2), keepdims=True)
    x_batch = torch.tensor(x_batch / std)  # pylint: disable=E1101
    y_batch = torch.tensor([d[1] for d in data])  # pylint: disable=C0103,E1101
    # y_batch = [d[1] for d in data]  # pylint: disable=C0103,E1101

    return x_batch, y_batch


# Definition of the overall RNN network.
class RecurrentSpikingNetwork(nn.Module):  # pylint: disable=R0903

    """
    This class defines an SRNN.
    """

    def __init__(self, device, bias: bool, thr_func, network_size: int = 256):
        """
        Constructor docstring
        """
        super().__init__()
        self.bias = bias
        self.thr_func = thr_func

        # Here is what the network looks like
        self.dense_1 = sd.SpikeDENSE(
            40 * 3,
            network_size,
            tau_adp_inital_std=50,
            tau_adp_inital=200,
            tau_m=20,
            tau_m_inital_std=5,
            device=device,
            bias=self.bias,
        )

        self.dense_2 = sd.ReadoutIntegrator(
            network_size,
            12,
            tau_m=10,
            tau_m_inital_std=1,
            device=device,
            bias=self.bias,
        )

        self.rnn_1 = sr.SpikeRNN(
            network_size,
            network_size,
            tau_adp_inital_std=50,
            tau_adp_inital=200,
            tau_m=20,
            tau_m_inital_std=5,
            device=device,
            bias=self.bias,
        )

        # Please comment this code.
        self.thr = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.thr, 5e-2)

        # Initialize the network layers.
        torch.nn.init.kaiming_normal_(self.rnn_1.recurrent.weight)

        torch.nn.init.xavier_normal_(self.dense_1.dense.weight)
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)

        if bias:
            torch.nn.init.constant_(self.rnn_1.recurrent.bias, 0)
            torch.nn.init.constant_(self.dense_1.dense.bias, 0)
            torch.nn.init.constant_(self.dense_2.dense.bias, 0)

    def forward(self, inputs):  # pylint: disable=R0914
        """
        The forward pass.
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
        inputs_s = self.thr_func(inputs - self.thr) - self.thr_func(
            -self.thr - inputs
        )

        # For every timestep update the membrane potential
        for i in range(seq_length):
            inputs_x = inputs_s[:, :, i, :].reshape(b, channel * inputs_dim)
            _, spike_layer1 = self.dense_1.forward(inputs_x)
            (
                _,
                spike_layer2,
            ) = self.rnn_1.forward(spike_layer1)
            # mem_layer3,spike_layer3 = self.dense_2.forward(spike_layer2)
            mem_layer3 = self.dense_2.forward(spike_layer2)

            # tracking number of spikes (firing rate).
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


def test(data_loader, device, model, is_show=0):
    """
    Test the network against the testing data.
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
    train_data_loader,
    test_data_loader,
    device,
    model,
    epochs,
    criterion,
    optimizer,
    scheduler=None,
):  # pylint: disable=R0913,R0914
    """
    Train the network with by the standard forward pass - loss
    calculation - backward propogation cycle.
    """
    acc_list = []
    best_acc = 0

    path = "../model/"  # .pth'
    for epoch in range(epochs):
        logger.info(f"{epoch=}")
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0
        for _, (images, labels) in enumerate(train_data_loader):
            # if i == 0:
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
        valid_acc, _ = test(test_data_loader, device, model, 1)  # what?!
        train_loss_sum += train_loss

        acc_list.append(train_acc)
        logger.info(f"{optimizer.param_groups[0]['lr']=}")

        if valid_acc > best_acc and train_acc > 0.890:
            best_acc = valid_acc
            torch.save(model, path + str(best_acc)[:7] + "-srnn-v3.pth")
        logger.info(f"{model.thr=}")

        training_loss = train_loss_sum / len(train_data_loader)
        logger.info(
            f"{epoch=:}, {training_loss=}, {train_acc=:.4f}, {valid_acc=:.4f}"
        )

    return acc_list


app = typer.Typer()


@app.command()
def main(config_file: Path) -> None:  # pylint: disable=R0914,R0915
    """
    Spiking Recurrent Neural Networks
    """
    # Read the configuration file.
    config = read_configuration(config_file)

    # Setup logger level.
    logger.remove()
    logger.add(sys.stderr, level=config["logger"]["level"])

    # Use cuda if it's available.
    device = torch.device(  # pylint: disable=E1101
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if config["cuda"]["cuda"] is False:
        device = torch.device("cpu")  # pylint: disable=E1101
    logger.info(f"{device=}")

    # Setup number of workers dependent upon where the code is run.
    number_of_workers = 4 if device.type == "cpu" else 8
    pin_memory = device.type == "cuda"

    logger.info(
        f"The Dataloader will spawn {number_of_workers} worker processes."
    )
    logger.info(f"{pin_memory=}")

    # Specify the several paramaters that we'll use throughout this example.
    # Paths to the data.
    dataroot = Path(config["data"]["dataroot"])
    gsc_url = config["data"]["gsc_url"]
    gsc = dataroot / "SpeechCommands" / gsc_url
    logger.info("\n".join([f"{dataroot=}", f"{gsc_url=}", f"{gsc=}"]))

    # Specify the learning rate, etc.
    network_size = config["srnn"]["network_size"]
    learning_rate = config["srnn"]["learning_rate"]
    epochs = config["srnn"]["epochs"]
    batch_size = config["srnn"]["batch_size"]
    size = config["srnn"]["size"]
    sample_rate = config["srnn"]["sample_rate"]
    bias = config["srnn"]["bias"]
    logger.info(
        "\n".join(
            [
                f"\n{learning_rate=}",
                f"{epochs=}",
                f"{batch_size=}",
                f"{size=}",
                f"{sample_rate=}",
                f"{bias=}",
            ]
        )
    )

    # Parameters for converting a wav into a mel-scaled spectrogram.
    # This is one of the transformations applied to each dataset.
    delta_order = config["mel"]["delta_order"]
    fmax = config["mel"]["fmax"]
    fmin = config["mel"]["fmin"]
    hop_length = int(10e-3 * sample_rate)
    n_fft = int(30e-3 * sample_rate)
    n_mels = config["mel"]["n_mels"]
    stack = config["mel"]["stack"]
    melspec = MelSpectrogram(
        sample_rate,
        n_fft,
        hop_length,
        n_mels,
        fmin,
        fmax,
        delta_order,
        stack=stack,
    )
    logger.info(
        "\n".join(
            [
                f"\n{delta_order=}",
                f"{fmax=}",
                f"{fmin=}",
                f"{hop_length=}",
                f"{n_fft=}",
                f"{n_mels=}",
                f"{stack=}",
            ]
        )
    )

    # Compose transformations applied to each dataset.
    pad = Pad(size)
    rescale = Rescale()
    transforms = torchvision.transforms.Compose([pad, melspec, rescale])

    # Specify our custom autograd function that defines how forward and
    # backward passes are performed.
    thr_func = sn.ActFunADP.apply
    logger.info(f"{thr_func=}")

    # Specify our loss function.
    criterion_f = nn.CrossEntropyLoss()  # nn.NLLLoss()
    logger.info(f"{criterion_f=}")

    # Retrieve the Google Speech Commands Dataset.
    torchaudio.datasets.SPEECHCOMMANDS(
        dataroot,
        url=gsc_url,
        folder_in_archive="SpeechCommands",
        download=True,
    )

    # Create random noise files for training and validation.
    silence_folder = gsc / "_silence_"
    if not silence_folder.exists():
        # Create the folder where we will write white noise files.
        silence_folder.mkdir(parents=True, exist_ok=True)

        # Compose a list of the GSC background noise files.

        # Four of the six files envoked a warning when read.
        # This is why we'"ll not choose from among these six
        # but use one file generate our noise files.
        # background_noise_files = [*gen_find("*.wav", gsc / "_background_noise_")]  # noqa: E501 pylint: disable=C0301

        # Instead of choosing among many, this is the one wav file
        # we will use to generate our white noise files.
        background_noise_file = gsc / "_background_noise_" / "white_noise.wav"

        # 260 validation / 2300 training.
        generate_noise_files(
            nb_files=2560,
            noise_file=background_noise_file,
            output_folder=silence_folder,
            file_prefix="rd_silence_",
            sr=sample_rate,
        )

        # Compose a list of the new noise files.  Write the first 260
        # names to silence_validation_list.txt, and the remaining 2300
        # to silence_training_list.txt.

        silence_files = [*gen_find("*.wav", silence_folder)]

        # Write the first 260 filenames to silence_validation_list.txt.
        with open(
            gsc / "silence_validation_list.txt", mode="w", encoding="utf-8"
        ) as fp:  # pylint: disable=C0103
            for filename in silence_files[:260]:
                filename = Path(*Path(filename).parts[-2:])  # Relative path
                fp.write(f"{filename}\n")

        # Write the remaining 2300 filenames to silence_training_list.txt.
        with open(
            gsc / "silence_training_list.txt", mode="w", encoding="utf-8"
        ) as fp:  # pylint: disable=C0103
            for filename in silence_files[260:]:
                filename = Path(*Path(filename).parts[-2:])  # Relative path
                fp.write(f"{filename}\n")

        logger.info("{Successfully created silence random noise files.")

    # Create Class Label Dictionary.

    # The dictionary's keys:value pairs are category names gleaned from
    # the GSC directory structure and integers, i.e. [0-9, 10, 11].  The
    # first ten keys or categories, whether chozen ordinally or drawn
    # randomly, recieve as values the first ten integers. The next
    # two key:value pairs are {'_silencee_':10, 'unknown':11}. The
    # remaining key or categories all recieve the value 11.
    # The values [0-10] represent testing categories.

    # Beginning at GSC find directories without a leading underscore.
    class_labels = list(
        {Path(dir).parts[-1] for dir in gen_dfind(r"^(?!_).*", gsc)}
    )
    logger.info(
        f"Class Labels[{len(class_labels)}]:\n{pp.pformat(class_labels)}"
    )

    # Compose the class dictionary by choosing
    # the first twn categories sequentially.
    # class_dict = dict(
    #     {j: i for i, j in enumerate(class_labels[:10])},
    #     **{"_silence_": 10},
    #     **{"_unknown_": 11},
    #     **{j: 11 for _, j in enumerate(class_labels[11:])},
    # )

    # Compose the class dictionary by choosing
    # the first ten categories randomly.
    # fmt: off
    class_dict = dict(
        {j: i for i, j in enumerate([class_labels.pop(random.randrange(len(class_labels))) for _ in range(10)])},  # noqa: E501 pylint: disable=C0301
        **{"_silence_": 10},
        **{"_unknown_": 11},
        **{i: 11 for i in class_labels})
    # fmt: on
    logger.info(f"class dict[{len(class_dict)}]:\n{pp.pformat(class_dict)}")

    # Reading and preprocessing the data.

    # The training dataset.
    # Note that the transformations specified here are applied in
    # the __getitem__ dunder method of the custom the GSCSSubsetSC class.

    gsc_training_dataset = GSCSSubsetSC(
        root=dataroot,
        url=gsc_url,
        download=True,
        subset="training",
        transform=transforms,
        class_dict=class_dict,
    )
    logger.info(
        f"The training data consists of {len(gsc_training_dataset)} samples."
    )

    waveform, idx = gsc_training_dataset[0]
    logger.info(f"Shape of gsc_training_set waveform: {waveform.shape}")
    logger.info(f"Waveform label: {index_to_label(class_dict, idx)}")
    # labels = sorted(list(set(index_to_label(class_dict, datapoint[1]) for datapoint in gsc_training_dataset)))  # noqa: E501 pylint: disable=C0301
    # logger.info(f"training labels:\n{pp.pformat(labels)}]")

    # The training dataloader.
    gsc_training_dataloader = torch.utils.data.DataLoader(
        gsc_training_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
    )
    gsc_features, gsc_labels = next(iter(gsc_training_dataloader))
    logger.info(f"Training Feature batch shape: {gsc_features.size()}")
    logger.info(f"Training Labels batch shape: {gsc_labels.size()}")
    logger.info(f"Training labels, i.e. indices:\n{pp.pformat(gsc_labels)}]")
    # logger.info(f"Training labels[{len(gsc_labels)}]:\n{pp.pformat(gsc_labels)}")  # noqa: E501

    # The testing dataset.
    gsc_testing_dataset = GSCSSubsetSC(
        root=dataroot,
        url=gsc_url,
        download=True,
        subset="testing",
        transform=transforms,
        class_dict=class_dict,
    )
    logger.info(
        f"The testing data consists of {len(gsc_testing_dataset)} samples."
    )

    # The testing dataloader.
    gsc_testing_dataloader = torch.utils.data.DataLoader(
        gsc_testing_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
    )

    # Instantiate the model.
    model = RecurrentSpikingNetwork(device, bias, thr_func, network_size)
    model.to(device)

    # Test before training.
    test_acc_before_training = test(gsc_testing_dataloader, device, model)
    logger.info(f"{test_acc_before_training=}")

    # Prepare for training.
    base_params = (
        [
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
        if bias
        else [
            model.dense_1.dense.weight,
            model.rnn_1.dense.weight,
            model.rnn_1.recurrent.weight,
            model.dense_2.dense.weight,
        ]
    )

    optimizer_f = torch.optim.Adam(
        [
            {"params": base_params, "lr": learning_rate},
            {"params": model.thr, "lr": learning_rate * 0.01},
            {"params": model.dense_1.tau_m, "lr": learning_rate * 2},
            {"params": model.dense_2.tau_m, "lr": learning_rate * 2},
            {"params": model.rnn_1.tau_m, "lr": learning_rate * 2},
            {"params": model.dense_1.tau_adp, "lr": learning_rate * 2.0},
            #   {'}params': model.dense_2.tau_adp, 'lr': learning_rate * 10},
            {"params": model.rnn_1.tau_adp, "lr": learning_rate * 2.0},
        ],
        lr=learning_rate,
    )

    # scheduler_f = StepLR(optimizer_f, step_size=20, gamma=.5) # 20
    scheduler_f = StepLR(optimizer_f, step_size=10, gamma=0.1)  # 20
    # scheduler_f = LambdaLR(optimizer_f,lr_lambda=lambda epoch: 1-epoch/70)
    # scheduler_f = ExponentialLR(optimizer_f, gamma=0.85)

    # Training.
    train_acc_training_complete = train(
        gsc_training_dataloader,
        gsc_testing_dataloader,
        device,
        model,
        epochs,
        criterion_f,
        optimizer_f,
        scheduler_f,
    )
    logger.info(f"TRAINING COMPLETE: {train_acc_training_complete=}")

    # Testing.
    test_acc_after_training = test(gsc_testing_dataloader, device, model)
    logger.info(f"TESTING COMPLETE: {test_acc_after_training}")


if __name__ == "__main__":
    app()

# finis

# Local Variables:
# compile-command: "pyflakes srnn.py; pylint-3 -f parseable srnn.py" # NOQA, pylint: disable=C0301
# End:
