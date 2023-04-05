# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
# SPDX-License-Identifier: MPL-2.0

"""
Utilities
"""

import numpy as np
import scipy.io.wavfile as wav

# from matplotlib.gridspec import GridSpec
# import matplotlib.pyplot as plt


def txt2list(filename):
    """This function reads a file containing one filename per line
    and returns a list of lines.

    Could be replaced with:
    for fn in gen_find('"*_list.txt', '/tmp/testdata/'):
        with open(fn) as fp:
            mylist = fp.read().splitlines()

    """
    lines_list = []
    with open(filename, "r") as txt:  # pylint: disable=W1514
        for line in txt:
            lines_list.append(line.rstrip("\n"))
    return lines_list


# def plot_spk_rec(spk_rec, idx):
#     nb_plt = len(idx)
#     d = int(np.sqrt(nb_plt))
#     gs = GridSpec(d, d)
#     fig = plt.figure(figsize=(30, 20), dpi=150)
#     for i in range(nb_plt):
#         plt.subplot(gs[i])
#         plt.imshow(
#             spk_rec[idx[i]].T,
#             cmap=plt.cm.gray_r,
#             origin="lower",
#             aspect="auto",
#         )
#         if i == 0:
#             plt.xlabel("Time")
#             plt.ylabel("Units")


# def plot_mem_rec(mem, idx):
#     nb_plt = len(idx)
#     d = int(np.sqrt(nb_plt))
#     dim = (d, d)

#     gs = GridSpec(*dim)
#     plt.figure(figsize=(30, 20))
#     dat = mem[idx]

#     for i in range(nb_plt):
#         if i == 0:
#             a0 = ax = plt.subplot(gs[i])
#         else:
#             ax = plt.subplot(gs[i], sharey=a0)
#         ax.plot(dat[i])


# The following two functions together generated random noise by
# randomly sampling a portion of sound from a randomly chozen
# background noise file. Unvortulately four of the six background
# noise files yield errors when read.


def get_random_noise(noise_files, size):  # pylint: disable=C0116
    noise_idx = np.random.choice(len(noise_files))
    fs, noise_wav = wav.read(noise_files[noise_idx])  # noqa: E501 pylint: disable=W0612,C0103,

    offset = np.random.randint(len(noise_wav) - size)
    noise_wav = noise_wav[offset: offset + size].astype(float)

    return noise_wav


def generate_random_silence_files(  # pylint: disable=C0116
    nb_files, noise_files, size, prefix, sr=16000  # pylint: disable=C0103
):
    for i in range(nb_files):
        silence_wav = get_random_noise(noise_files, size)
        wav.write(prefix + "_" + str(i) + ".wav", sr, silence_wav)


def generate_noise_files(
    nb_files, noise_file, output_folder, file_prefix, sr  # noqa: E501 pylint: disable=C0103
):
    """
    Generate many random noise files by taking random spans from a
    single noise file.
    """

    for i in range(nb_files):
        fs, noise_wav = wav.read(  # pylint: disable=C0103,W0612
            noise_file,
        )
        offset = np.random.randint(len(noise_wav) - sr)
        noise_wav = noise_wav[offset: offset + sr].astype(float)
        fn = output_folder / "".join(  # pylint: disable=C0103
            [file_prefix, f"{i}", ".wav"]
        )
        wav.write(fn, sr, noise_wav)


# def split_wav(waveform, frame_size, split_hop_length):
#     splitted_wav = []
#     offset = 0

#     while offset + frame_size < len(waveform):
#         splitted_wav.append(waveform[offset : offset + frame_size])
#         offset += split_hop_length

#     return splitted_wav


# finis

# Local Variables:
# compile-command: "pyflakes utils.py; pylint-3 -f parseable utils.py" # NOQA, pylint: disable=C0301
# End:
