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
import torch
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.utils import _load_waveform


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
            excludes = (
                load_list("testing_list.txt")
                + load_list("validation_list.txt")
                + load_list("silence_validation_list.txt")
            )
            excludes = set(excludes)
            self._walker = [
                w
                for w in self._walker  # pylint: disable=C0103
                if w not in excludes
            ]

            # debug: write our training list to the filesystem so we
            # can examine it. The validation and testing lists are
            # explicit.

            # with open("/tmp/training_list.txt",
            #     mode="wt",
            #     encoding="utf-8",
            # ) as fileobj:
            #     fileobj.write("\n".join(self._walker))

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


# finis

# Local Variables:
# compile-command: "pyflakes data.py; pylint-3 -d E0401 -f parseable data.py" # NOQA, pylint: disable=C0301
# End:
