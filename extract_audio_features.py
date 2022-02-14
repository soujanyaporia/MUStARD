#!/usr/bin/env python
import os
import pickle

import librosa
import numpy as np
from tqdm.auto import tqdm

AUDIOS_FOLDER = "data/audios/utterances_final"
AUDIO_FEATURES_PATH = "data/audio_features.p"


def get_librosa_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path)

    hop_length = 512  # Set the hop length; at 22050 Hz, 512 samples ~= 23ms

    # Remove vocals first
    D = librosa.stft(y, hop_length=hop_length)
    S_full, phase = librosa.magphase(D)

    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine",
                                           width=int(librosa.time_to_frames(0.2, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 2, 4
    power = 2
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    S_foreground = mask_v * S_full

    # Recreate vocal_removal y
    new_D = S_foreground * phase
    y = librosa.istft(new_D)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Compute MFCC features from the raw signal
    mfcc_delta = librosa.feature.delta(mfcc)  # And the first-order differences (delta features)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_delta = librosa.feature.delta(S)

    spectral_centroid = librosa.feature.spectral_centroid(S=S_full)

    audio_feature = np.vstack((mfcc, mfcc_delta, S, S_delta, spectral_centroid))  # combine features

    # binning data
    jump = int(audio_feature.shape[1] / 10)
    return librosa.util.sync(audio_feature, range(1, audio_feature.shape[1], jump))


def save_audio_features() -> None:
    audio_feature = {}
    for filename in tqdm(os.listdir(AUDIOS_FOLDER), desc="Computing the audio features"):
        id_ = filename.rsplit(".", maxsplit=1)[0]
        audio_feature[id_] = get_librosa_features(os.path.join(AUDIOS_FOLDER, filename))
        print(audio_feature[id_].shape)

    with open(AUDIO_FEATURES_PATH, "wb") as file:
        pickle.dump(audio_feature, file, protocol=2)


def get_audio_duration() -> None:
    filenames = os.listdir(AUDIOS_FOLDER)
    print(sum(librosa.core.get_duration(filename=os.path.join(AUDIOS_FOLDER, filename))
              for filename in tqdm(filenames, desc="Computing the average duration of the audios")) / len(filenames))


def main() -> None:
    get_audio_duration()

    # save_audio_features()
    #
    # with open(AUDIO_FEATURES_PATH, "rb") as file:
    #     pickle.load(file)


if __name__ == "__main__":
    main()
