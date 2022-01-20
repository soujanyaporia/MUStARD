#!/usr/bin/env python
# Re-implementation of the audio extraction script. It may not be exactly the same used in the paper, but should be
# fairly similar for reproducibility purposes.
import pickle
from pathlib import Path

import librosa
import numpy as np

AUDIOS_FOLDER = Path("data/audios/utterances_final")

WIDTH_SECONDS = 2
MARGIN_V = 10


def get_foreground_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if len(audio) / sample_rate > WIDTH_SECONDS:  # It'd fail otherwise.
        # From https://librosa.org/doc/main/auto_examples/plot_vocal_separation.html
        S_full, phase = librosa.magphase(librosa.stft(audio))
        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine",
                                               width=int(librosa.time_to_frames(WIDTH_SECONDS, sr=sample_rate)))
        S_filter = np.minimum(S_full, S_filter)
        mask_v = librosa.util.softmask(S_full - S_filter, MARGIN_V * S_filter, power=2)
        S_foreground = mask_v * S_full
        return librosa.istft(S_foreground * phase)
    else:
        return audio


def main() -> None:
    output = {}
    for path in AUDIOS_FOLDER.iterdir():
        audio, sample_rate = librosa.load(str(path))
        foreground_audio = get_foreground_audio(audio, sample_rate)

        hop_length = (len(audio) - 2048) // 11
        mfcc = librosa.feature.mfcc(foreground_audio, sr=sample_rate, n_mfcc=13, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        melspectrogram = librosa.feature.melspectrogram(foreground_audio, sr=sample_rate, hop_length=hop_length)
        melspectrogram_delta = librosa.feature.delta(melspectrogram)
        spectral_centroid = librosa.feature.spectral_centroid(foreground_audio, sr=sample_rate, hop_length=hop_length)

        id_ = path.name.split(".", maxsplit=1)[0]
        output[id_] = np.concatenate((mfcc, mfcc_delta, melspectrogram, melspectrogram_delta, spectral_centroid))

    with open("data/audio_features.p", "wb") as file:
        pickle.dump(output, file)


if __name__ == "__main__":
    main()
