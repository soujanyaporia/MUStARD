#!/usr/bin/env python
import argparse
import json
import os
from typing import Any, Iterable, Mapping, Tuple

import numpy as np
import sklearn
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import CONFIG_BY_KEY, Config
from data_loader import DataHelper, DataLoader

RESULT_FILE = "output/{}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-key", default="", choices=list(CONFIG_BY_KEY))
    return parser.parse_args()


def svm_train(config: Config, train_input: np.ndarray, train_output: np.ndarray) -> sklearn.base.BaseEstimator:
    clf = make_pipeline(
        StandardScaler() if config.svm_scale else FunctionTransformer(lambda x: x, validate=False),
        svm.SVC(C=config.svm_c, gamma="scale", kernel="rbf")
    )
    return clf.fit(train_input, np.argmax(train_output, axis=1))


def svm_test(clf: sklearn.base.BaseEstimator, test_input: np.ndarray,
             test_output: np.ndarray) -> Tuple[Mapping[str, Any], str]:
    probas = clf.predict(test_input)  # noqa
    y_pred = probas
    y_true = np.argmax(test_output, axis=1)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # To generate majority baseline
    # y_pred = [0] * len(y_pred)

    result_string = classification_report(y_true, y_pred, digits=3)
    print(confusion_matrix(y_true, y_pred))
    print(result_string)
    return classification_report(y_true, y_pred, output_dict=True, digits=3), result_string


def train_io(config: Config, data: DataLoader, train_index: Iterable[int],
             test_index: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_input, train_output = data.get_split(train_index)
    test_input, test_output = data.get_split(test_index)

    datahelper = DataHelper(train_input, train_output, test_input, test_output, config, data)

    train_input = np.empty((len(train_input), 0))
    test_input = np.empty((len(test_input), 0))

    if config.use_target_text:
        if config.use_bert:
            train_input = np.concatenate([train_input, datahelper.get_target_bert_feature(mode="train")], axis=1)
            test_input = np.concatenate([test_input, datahelper.get_target_bert_feature(mode="test")], axis=1)
        else:
            train_input = np.concatenate([train_input,
                                          np.array([datahelper.pool_text(utt)
                                                    for utt in datahelper.vectorize_utterance(mode="train")])], axis=1)
            test_input = np.concatenate([test_input,
                                         np.array([datahelper.pool_text(utt)
                                                   for utt in datahelper.vectorize_utterance(mode="test")])], axis=1)

    if config.use_target_video:
        train_input = np.concatenate([train_input, datahelper.get_target_video_pool(mode="train")], axis=1)
        test_input = np.concatenate([test_input, datahelper.get_target_video_pool(mode="test")], axis=1)

    if config.use_target_audio:
        train_input = np.concatenate([train_input, datahelper.get_target_audio_pool(mode="train")], axis=1)
        test_input = np.concatenate([test_input, datahelper.get_target_audio_pool(mode="test")], axis=1)

    if train_input.shape[1] == 0:
        raise ValueError("Invalid modalities")

    # Aux input

    if config.use_author:
        train_input_author = datahelper.get_author(mode="train")
        test_input_author = datahelper.get_author(mode="test")

        train_input = np.concatenate([train_input, train_input_author], axis=1)
        test_input = np.concatenate([test_input, test_input_author], axis=1)

    if config.use_context:
        if config.use_bert:
            train_input_context = datahelper.get_context_bert_features(mode="train")
            test_input_context = datahelper.get_context_bert_features(mode="test")
        else:
            train_input_context = datahelper.get_context_pool(mode="train")
            test_input_context = datahelper.get_context_pool(mode="test")

        train_input = np.concatenate([train_input, train_input_context], axis=1)
        test_input = np.concatenate([test_input, test_input_context], axis=1)

    train_output = datahelper.one_hot_output(mode="train", size=config.num_classes)
    test_output = datahelper.one_hot_output(mode="test", size=config.num_classes)

    return train_input, train_output, test_input, test_output


def train_speaker_independent(config: Config, data: DataLoader, model_name: str) -> None:  # noqa
    train_index, test_index = data.get_speaker_independent()
    train_input, train_output, test_input, test_output = train_io(config=config, data=data, train_index=train_index,
                                                                  test_index=test_index)

    clf = svm_train(config=config, train_input=train_input, train_output=train_output)
    svm_test(clf, test_input, test_output)


def train_speaker_dependent(config: Config, data: DataLoader, model_name: str) -> None:
    results = []
    for fold, (train_index, test_index) in enumerate(data.get_stratified_k_fold()):
        config.fold = fold + 1
        print("Present Fold:", config.fold)

        train_input, train_output, test_input, test_output = train_io(config=config, data=data, train_index=train_index,
                                                                      test_index=test_index)

        clf = svm_train(config=config, train_input=train_input, train_output=train_output)
        result_dict, result_str = svm_test(clf, test_input, test_output)

        results.append(result_dict)

    if not os.path.exists(os.path.dirname(RESULT_FILE)):
        os.makedirs(os.path.dirname(RESULT_FILE))

    with open(RESULT_FILE.format(model_name), "w") as file:
        json.dump(results, file)


def print_result(model_name: str) -> None:
    with open(RESULT_FILE.format(model_name)) as file:
        results = json.load(file)

    weighted_precision = []
    weighted_recall = []
    weighted_f_scores = []

    print("#" * 20)
    for fold, result in enumerate(results):
        weighted_f_scores.append(result["weighted avg"]["f1-score"])
        weighted_precision.append(result["weighted avg"]["precision"])
        weighted_recall.append(result["weighted avg"]["recall"])

        print(f"Fold {fold + 1}:")
        print(f"Weighted Precision: {result['weighted avg']['precision']}  "
              f"Weighted Recall: {result['weighted avg']['recall']}  "
              f"Weighted F score: {result['weighted avg']['f1-score']}")
    print("#" * 20)
    print("Avg :")
    print(f"Weighted Precision: {np.mean(weighted_precision):.3f}  "
          f"Weighted Recall: {np.mean(weighted_recall):.3f}  "
          f"Weighted F score: {np.mean(weighted_f_scores):.3f}")


def main() -> None:
    args = parse_args()
    print("Args:", args)

    config = CONFIG_BY_KEY[args.config_key]

    if config.speaker_independent:
        config.fold = "SI"
        data = DataLoader(config)
        train_speaker_independent(config=config, data=data, model_name=config.model)
    else:
        data = DataLoader(config)
        for _ in range(config.runs):
            train_speaker_dependent(config=config, data=data, model_name=config.model)
            print_result(model_name=config.model)


if __name__ == "__main__":
    main()
