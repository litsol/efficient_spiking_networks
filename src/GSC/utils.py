# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0
# flake8: noqa
# pylint: skip-file
# type: ignore
# REUSE-IgnoreStart
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import torch
from matplotlib.gridspec import GridSpec


def txt2list(filename):
    """This function reads a file containing one filename per line
    and returns a list of lines.

    Could be replaced with:
    for fn in gen_find('*_list.txt', '/tmp/testdata/'):
        with open(fn) as fp:
            mylist = fp.read().splitlines()

    """
    lines_list = []
    with open(filename, "r") as txt:
        for line in txt:
            lines_list.append(line.rstrip("\n"))
    return lines_list


def plot_spk_rec(spk_rec, idx):

    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))
    gs = GridSpec(d, d)
    fig = plt.figure(figsize=(30, 20), dpi=150)
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(
            spk_rec[idx[i]].T,
            cmap=plt.cm.gray_r,
            origin="lower",
            aspect="auto",
        )
        if i == 0:
            plt.xlabel("Time")
            plt.ylabel("Units")


def plot_mem_rec(mem, idx):

    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))
    dim = (d, d)

    gs = GridSpec(*dim)
    plt.figure(figsize=(30, 20))
    dat = mem[idx]

    for i in range(nb_plt):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[i])


def get_random_noise(noise_files, size):

    noise_idx = np.random.choice(len(noise_files))
    fs, noise_wav = wav.read(noise_files[noise_idx])

    offset = np.random.randint(len(noise_wav) - size)
    noise_wav = noise_wav[offset : offset + size].astype(float)

    return noise_wav


def generate_random_silence_files(
    nb_files, noise_files, size, prefix, sr=16000
):

    for i in range(nb_files):

        silence_wav = get_random_noise(noise_files, size)
        wav.write(prefix + "_" + str(i) + ".wav", sr, silence_wav)


def split_wav(waveform, frame_size, split_hop_length):

    splitted_wav = []
    offset = 0

    while offset + frame_size < len(waveform):
        splitted_wav.append(waveform[offset : offset + frame_size])
        offset += split_hop_length

    return splitted_wav


# REUSE-IgnoreEnd
