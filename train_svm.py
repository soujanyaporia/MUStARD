import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import CONFIG_BY_KEY, Config
from data_loader import DataHelper
from data_loader import DataLoader

logger = logging.getLogger(__name__)


class Training:
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.data_loader = DataLoader(config)

        self.result_folder = Path("output")
        self.result_file_path = self.result_folder / f"{config.model}.json"

    def train_classifier(self, train_input, train_output) -> Pipeline:
        clf = make_pipeline(
            StandardScaler() if self.config.svm_scale else FunctionTransformer(lambda x: x, validate=False),
            svm.SVC(C=self.config.svm_c, gamma='scale', kernel='rbf'),
        )

        return clf.fit(train_input, np.argmax(train_output, axis=1))

    @staticmethod
    def test_classifier(clf: Pipeline, test_input, test_output) -> Tuple[Dict[str, ], str]:
        y_pred = clf.predict(test_input)
        y_true = np.argmax(test_output, axis=1)

        # y_pred = np.random.randint(2, size=len(y_pred))  # To generate random scores

        # y_pred = np.zeros_like(y_pred)  # To generate majority baseline

        print(confusion_matrix(y_true, y_pred))

        result_dict = classification_report(y_true, y_pred, output_dict=True, digits=3)
        result_string = classification_report(y_true, y_pred, digits=3)

        print(result_string)

        return result_dict, result_string

    def train_io(self, train_index, test_index):
        train_input, train_output = self.data_loader.get_split(train_index)
        test_input, test_output = self.data_loader.get_split(test_index)

        data_helper = DataHelper(train_input, train_output, test_input, test_output, self.config, self.data_loader)

        train_input = np.empty((len(train_input), 0))
        test_input = np.empty((len(test_input), 0))

        if self.config.use_target_text:
            if self.config.use_bert:
                train_input = np.concatenate([train_input, data_helper.get_target_bert_features(mode='train')], axis=1)
                test_input = np.concatenate([test_input, data_helper.get_target_bert_features(mode='test')], axis=1)
            else:
                train_input = np.concatenate([train_input,
                                              np.array([data_helper.pool_text(utt)
                                                        for utt in data_helper.vectorize_utterance(mode='train')])], axis=1)
                test_input = np.concatenate([test_input,
                                             np.array([data_helper.pool_text(utt)
                                                       for utt in data_helper.vectorize_utterance(mode='test')])], axis=1)

        if self.config.use_target_video:
            train_input = np.concatenate([train_input, data_helper.get_target_video_pool(mode='train')], axis=1)
            test_input = np.concatenate([test_input, data_helper.get_target_video_pool(mode='test')], axis=1)

        if self.config.use_target_audio:
            train_input = np.concatenate([train_input, data_helper.get_target_audio_pool(mode='train')], axis=1)
            test_input = np.concatenate([test_input, data_helper.get_target_audio_pool(mode='test')], axis=1)

        if train_input.shape[1] == 0:
            raise Exception("Invalid modalities")

        # Aux input

        if self.config.use_author:
            train_input_author = data_helper.get_speaker(mode="train")
            test_input_author = data_helper.get_speaker(mode="test")

            train_input = np.concatenate([train_input, train_input_author], axis=1)
            test_input = np.concatenate([test_input, test_input_author], axis=1)

        if self.config.use_context:
            if self.config.use_bert:
                train_input_context = data_helper.get_context_bert_features(mode="train")
                test_input_context = data_helper.get_context_bert_features(mode="test")
            else:
                train_input_context = data_helper.get_context_pool(mode="train")
                test_input_context = data_helper.get_context_pool(mode="test")

            train_input = np.concatenate([train_input, train_input_context], axis=1)
            test_input = np.concatenate([test_input, test_input_context], axis=1)

        train_output = data_helper.one_hot_output(mode="train", size=self.config.num_classes)
        test_output = data_helper.one_hot_output(mode="test", size=self.config.num_classes)

        return train_input, train_output, test_input, test_output

    def train_speaker_independent(self) -> None:
        self.config.fold = "SI"

        train_index, test_index = self.data_loader.get_speaker_independent_split_indices()
        train_input, train_output, test_input, test_output = self.train_io(train_index, test_index)

        clf = self.train_classifier(train_input, train_output)
        self.test_classifier(clf, test_input, test_output)

    def train_speaker_dependent(self) -> None:
        results = []
        for n_fold, (train_index, test_index) in enumerate(self.data_loader.get_split_indices()):
            self.config.fold = n_fold + 1
            logger.info(f"Present Fold: {self.config.fold}")

            train_input, train_output, test_input, test_output = self.train_io(train_index, test_index)

            clf = self.train_classifier(train_input, train_output)
            result_dict, result_str = self.test_classifier(clf, test_input, test_output)

            results.append(result_dict)

        os.makedirs(self.result_folder, exist_ok=True)
        with open(self.result_file_path, 'w') as file:
            json.dump(results, file)

    def print_result(self) -> None:
        with open(self.result_file_path) as file:
            results = json.load(file)

        weighted_avg_precision = np.mean([result["weighted avg"]["precision"]] for result in results)
        weighted_avg_recall = np.mean([result["weighted avg"]["recall"]] for result in results)
        weighted_avg_f_score = np.mean([result["weighted avg"]["f1-score"]] for result in results)

        print("#" * 20)
        for n_fold, result in enumerate(results):
            print(f"Fold {n_fold + 1}:")
            print(f"Weighted Precision: {result['weighted avg']['precision']}  "
                  f"Weighted Recall: {result['weighted avg']['recall']}  "
                  f"Weighted F score: {result['weighted avg']['f1-score']}")
        print("#" * 20)
        print("Avg :")
        print(f"Weighted Precision: {weighted_avg_precision:.3f}  "
              f"Weighted Recall: {weighted_avg_recall:.3f}  "
              f"Weighted F score: {weighted_avg_f_score:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-key', default='', choices=list(CONFIG_BY_KEY.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Args: {args}")

    config = CONFIG_BY_KEY[args.config_key]

    training = Training(config)

    if config.speaker_independent:
        training.train_speaker_independent()
    else:
        for _ in range(config.runs):
            training.train_speaker_dependent()
            training.print_result()


if __name__ == "__main__":
    main()
