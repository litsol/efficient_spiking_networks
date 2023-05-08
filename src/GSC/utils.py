# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
# SPDX-License-Identifier: MPL-2.0

"""
Utilities
"""

import numpy as np
import scipy.io.wavfile as wav


def generate_noise_files(
    nb_files,
    noise_file,
    output_folder,
    file_prefix,
    sr,  # noqa: E501 pylint: disable=C0103
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
        noise_wav = noise_wav[offset : offset + sr].astype(float)  # noqa: E203
        fn = output_folder / "".join(  # pylint: disable=C0103
            [file_prefix, f"{i}", ".wav"]
        )
        wav.write(fn, sr, noise_wav)


# finis

# Local Variables:
# compile-command: "pyflakes utils.py; pylint-3 -f parseable utils.py" # NOQA, pylint: disable=C0301
# End:
