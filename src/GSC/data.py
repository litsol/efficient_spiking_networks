# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
# SPDX-License-Identifier: MPL-2.0

"""
Classes that retrieve and manipualte input data.
"""

import os
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import scipy.io.wavfile as wav
import torch
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.utils import _load_waveform
from utils import txt2list


class GSCSSubsetSC(SPEECHCOMMANDS):
    """
    Our custom SPEECHCOMMANDS/dataset class that retrieves,
    segregates and transforms the GSC dataset.
    """

    def __init__(  # pylint: disable=R0913
        self,
        root: Union[str, Path],
        url: str = "speech_commands_v0.02",
        folder_in_archive: str = "SpeechCommands",
        download: bool = True,
        subset: Optional[str] = None,
        transform: Optional[str] = None,
        class_dict: dict = None,
    ) -> None:
        """
        Function Docstring
        """

        super().__init__(
            root, url=url, folder_in_archive="SpeechCommands", download=True
        )

        # two instance variables specific to this subclass
        self.transform = transform
        self.class_dict = class_dict

        def load_list(filename):
            """
            Function Docstring
            """

            filepath = os.path.join(self._path, filename)
            with open(filepath, mode="r", encoding="utf-8") as fileobj:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in fileobj
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt") + load_list(
                "silence_validation_list.txt"
            )
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            self._walker += load_list("silence_training_list.txt")
            excludes = (
                load_list("testing_list.txt")
                + load_list("validation_list.txt")
                + load_list("silence_validation_list.txt")
            )
            excludes = set(excludes)
            self._walker = [
                w
                for w in self._walker  # pylint: disable=C0103
                if w not in excludes  # pylint: disable=C0103
            ]  # noqa: E501 pylint: disable=C0103

    def __getitem__(self, n):
        """This iterator return a tuple consisting of a waveform and
        its numeric label provided by the classification
        dictionary.

        Here is where the pad, melspec, and rescale traansforms are applied.
        """

        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        maximum = torch.max(torch.abs(waveform))  # pylint: disable=E1101

        if maximum > 0:
            waveform /= maximum
        if self.transform is not None:
            waveform = self.transform(waveform.squeeze())
        return (
            waveform,
            self.class_dict[metadata[2]],
        )


class SpeechCommandsDataset(Dataset):
    """
    Class Docstring
    """

    def __init__(  # pylint: disable=R0912,R0913,R0914
        self, data_root, label_dct, mode, transform=None, max_nb_per_class=None
    ):
        """
        Function Docstring
        """

        assert mode in [
            "train",
            "valid",
            "test",
        ], 'mode should be "train", "valid" or "test"'

        self.filenames = []
        self.labels = []
        self.mode = mode
        self.transform = transform

        if (
            self.mode == "train"  # pylint: disable=R1714
            or self.mode == "valid"
        ):
            # Create lists of 'wav' files.
            testing_list = txt2list(
                os.path.join(data_root, "testing_list.txt")
            )
            validation_list = txt2list(
                os.path.join(data_root, "validation_list.txt")
            )
            # silence_validation_list.txt not in gsc dataset
            validation_list += txt2list(
                os.path.join(data_root, "silence_validation_list.txt")
            )
        else:
            testing_list = []
            validation_list = []

        for root, dirs, files in os.walk(data_root):  # pylint: disable=W0612
            if "_background_noise_" in root:
                continue
            for filename in files:
                if not filename.endswith(".wav"):
                    # Ignore files whose suffix is not 'wav'.
                    continue

                # Extract the cwd without a path.
                command = root.split("/")[-1]

                label = label_dct.get(command)
                if label is None:
                    print(f"ignored command: {command}")
                    break  # Out of here!
                partial_path = "/".join([command, filename])

                # These are Boolean values!
                testing_file = partial_path in testing_list
                validation_file = partial_path in validation_list
                training_file = not testing_file and not validation_file

                if (
                    (self.mode == "test")
                    or (self.mode == "train" and training_file)
                    or (self.mode == "valid" and validation_file)
                ):
                    full_name = os.path.join(root, filename)
                    self.filenames.append(full_name)
                    self.labels.append(label)

        if max_nb_per_class is not None:
            selected_idx = []
            for label in np.unique(self.labels):
                label_idx = [
                    i
                    for i, x in enumerate(self.labels)  # pylint: disable=C0103
                    if x == label  # noqa: E501 pylint: disable=C0103
                ]
                if len(label_idx) < max_nb_per_class:
                    selected_idx += label_idx
                else:
                    selected_idx += list(
                        np.random.choice(label_idx, max_nb_per_class)
                    )

            self.filenames = [self.filenames[idx] for idx in selected_idx]
            self.labels = [self.labels[idx] for idx in selected_idx]

        if self.mode == "train":
            label_weights = 1.0 / np.unique(self.labels, return_counts=True)[1]
            label_weights /= np.sum(label_weights)
            self.weights = torch.DoubleTensor(  # pylint: disable=E1101
                [label_weights[label] for label in self.labels]
            )

    def __len__(self):
        """
        Function Docstring
        """

        return len(self.labels)

    def __getitem__(self, idx):
        """
        Function Docstring
        """

        filename = self.filenames[idx]
        item = wav.read(filename)[1].astype(float)
        m = np.max(np.abs(item))  # pylint: disable=C0103
        if m > 0:
            item /= m
        if self.transform is not None:
            item = self.transform(item)

        label = self.labels[idx]

        return item, label


class Pad:  # pylint: disable=R0903
    """
    Pad class
    """

    def __init__(self, size: int):
        """
        Class constructor; size comes from the configuration file.
        """
        self.size = size

    def __call__(self, waveform):
        """
        Pad the waveform on the beginning and on the end such that the
        resulting array is the same length as the size the pad object
        was instantiated with.
        """

        wav_size = waveform.shape[0]
        pad_size = (self.size - wav_size) // 2
        padded_wav = np.pad(
            waveform,
            ((pad_size, self.size - wav_size - pad_size),),
            "constant",
            constant_values=(0, 0),
        )
        return padded_wav


# class RandomNoise:  # pylint: disable=R0903
#     """Class Docstring"""

#     def __init__(self, noise_files, size, coef):
#         """Function Docstring"""
#         self.size = size
#         self.noise_files = noise_files
#         self.coef = coef

#     def __call__(self, waveform):
#         """Function Docstring"""
#         if np.random.random() < 0.8:
#             noise_wav = get_random_noise(self.noise_files, self.size)
#             noise_power = (noise_wav**2).mean()
#             sig_power = (waveform**2).mean()

#             noisy_wav = waveform + self.coef * noise_wav * np.sqrt(
#                 sig_power / noise_power
#             )

#         else:
#             noisy_wav = waveform

#         return noisy_wav


# class RandomShift:  # pylint: disable=R0903
#     """Class Docstring"""

#     def __init__(self, min_shift, max_shift):
#         """Function Docstring"""
#         self.min_shift = min_shift
#         self.max_shift = max_shift

#     def __call__(self, waveform):
#         """Function Docstring"""
#         shift = np.random.randint(self.min_shift, self.max_shift + 1)
#         shifted_wav = np.roll(waveform, shift)

#         if shift > 0:
#             shifted_wav[:shift] = 0
#         elif shift < 0:
#             shifted_wav[shift:] = 0

#         return shifted_wav


class MelSpectrogram:  # pylint: disable=R0902,R0903
    """
    Mel Spectrogram Transformation
    """

    def __init__(  # pylint: disable=R0913
        self,
        sr,  # pylint: disable=C0103
        n_fft,
        hop_length,
        n_mels,
        fmin,
        fmax,
        delta_order=None,
        stack=True,
    ):
        """
        Class Constructor
        """

        self.sr = sr  # pylint: disable=C0103
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.delta_order = delta_order
        self.stack = stack

    def __call__(self, waveform):
        """
        Perform the Mel Spectrogram Transformation
        """

        spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=self.fmax,
            fmin=self.fmin,
        )

        maximum = np.max(np.abs(spectrogram))
        if maximum > 0:
            feat = np.log1p(spectrogram / maximum)
        else:
            feat = spectrogram

        if self.delta_order is not None and not self.stack:
            feat = librosa.feature.delta(feat, order=self.delta_order)
            return np.expand_dims(feat.T, 0)

        if self.delta_order is not None and self.stack:
            feat_list = [feat.T]
            for k in range(1, self.delta_order + 1):
                feat_list.append(librosa.feature.delta(feat, order=k).T)
            return np.stack(feat_list)

        return np.expand_dims(feat.T, 0)


class Rescale:  # pylint: disable=R0903
    """Rescale Class"""

    def __call__(self, data):
        """
        Function Docstring
        """

        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1

        return data / std


class Normalize:  # pylint: disable=R0903
    """
    Class Docstring
    """

    def __call__(self, data):
        """
        Function Docstring
        """

        data_ = (data > 0.1) * data
        std = np.std(data_, axis=1, keepdims=True)
        std[std == 0] = 1

        return input / std


# class WhiteNoise:  # pylint: disable=R0903
#     """Class Docstring"""

#     def __init__(self, size, coef_max):
#         """Function Docstring"""
#         self.size = size
#         self.coef_max = coef_max

#     def __call__(self, waveform):
#         """Function Docstring"""
#         noise_wav = np.random.normal(size=self.size)
#         noise_power = (noise_wav**2).mean()
#         sig_power = (waveform**2).mean()

#         coef = np.random.uniform(0.0, self.coef_max)

#         noisy_wav = waveform + coef * noise_wav * np.sqrt(
#             sig_power / noise_power
#         )

#         return noisy_wav

# finis

# Local Variables:
# compile-command: "pyflakes data.py; pylint-3 -d E0401 -f parseable data.py" # NOQA, pylint: disable=C0301
# End:
