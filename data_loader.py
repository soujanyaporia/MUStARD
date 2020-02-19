import json
import logging
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import h5py
import jsonlines
import nltk
import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import Config

DatasetType = Dict[str, Dict[str, Union[str, Iterable[str], bool]]]

PathType = Union[os.PathLike, str]

logger = logging.getLogger(__name__)


def pickle_loader(path: PathType) -> :
    with open(path, "rb") as file:
        return pickle.load(file, encoding="latin1")

@dataclass
class Instance:
    utterance
    speaker
    context
    context_speakers: Iterable[]
    audio
    video
    context_video
    text_bert
    context_bert
    show


class DataLoader:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_folder = Path("data")
        AUDIO_PICKLE = self.data_folder / "audio_features.p"
        INDICES_FILE = self.data_folder / "split_indices.p"
        GLOVE_DICT = self.data_folder / "glove_full_dict.p"
        BERT_CONTEXT_EMBEDDINGS = self.data_folder / "bert-output-context.jsonl"
        CLS_TOKEN_INDEX = 0
        UNK_TOKEN = "<UNK>"
        PAD_TOKEN = "<PAD>"

        with open(self.data_folder / "sarcasm_data.json") as file:
            dataset = json.load(file)

        if config.use_bert and config.use_target_text:
            text_bert_embeddings = []
            with jsonlines.open(self.data_folder / "bert-output.jsonl") as reader:
                for target_utterance in reader:
                    features = target_utterance["features"][self.CLS_TOKEN_INDEX]
                    bert_embedding_target = [np.array(features["layers"][i]["values"]) for i in range(4)]
                    bert_embedding_target = np.mean(bert_embedding_target, axis=0)
                    text_bert_embeddings.append(np.copy(bert_embedding_target))
        else:
            text_bert_embeddings = None

        context_bert_embeddings = self.load_context_bert(dataset) if config.use_context else None

        audio_features = pickle_loader(self.AUDIO_PICKLE) if config.use_target_audio else None

        if config.use_target_video:
            video_features_file = h5py.File(self.DATA_FOLDER / "features/utterances_final/resnet_pool5.hdf5")
            context_video_features_file = h5py.File(self.DATA_FOLDER / "features/context_final/resnet_pool5.hdf5")
        else:
            video_features_file = None
            context_video_features_file = None

        self.data_input = [(
            dataset[id_]["utterance"],
            dataset[id_]["speaker"],
            dataset[id_]["context"],
            dataset[id_]["context_speakers"],
            audio_features[id_] if audio_features else None,
            video_features_file[id_][()] if video_features_file else None,
            context_video_features_file[id_][()] if context_video_features_file else None,
            text_bert_embeddings[i] if text_bert_embeddings else None,
            context_bert_embeddings[i] if context_bert_embeddings else None,
            dataset[id_]["show"],
        ) for i, id_ in enumerate(dataset)]
        self.data_output = [int(dataset[id_]["sarcasm"]) for id_ in dataset]

        if config.use_target_video:
            video_features_file.close()
            context_video_features_file.close()

        self.setup_folds()
        self.setup_glove_dict()

        self.setup_speaker_independent_split()

    def load_context_bert(self, dataset: DatasetType) -> Iterable[Iterable[np.ndarray]]:
        length = np.asarray([len(instance["context"]) for instance in dataset.values()])

        with jsonlines.open(self.BERT_CONTEXT_EMBEDDINGS) as reader:
            context_utterance_embeddings = [
                np.array([np.array(context_utterance['features'][self.CLS_TOKEN_INDEX]["layers"][i]["values"])
                          for i in range(4)])
                for context_utterance in reader]

        assert len(context_utterance_embeddings) == length.sum()

        cumulative_length = length.cumsum()

        end_index = cumulative_length
        start_index = [0] + cumulative_length[:-1]

        return [[context_utterance_embeddings[i] for i in range(start, end)]
                for start, end in zip(start_index, end_index)]

    def setup_folds(self, n_splits: int = 5) -> None:
        """Prepares or loads (if existing) splits for cross-validation."""
        cross_validator = StratifiedKFold(n_splits=n_splits, shuffle=True)
        split_indices = [(train_index, test_index) for train_index, test_index in
                         cross_validator.split(self.data_input, self.data_output)]

        if not os.path.exists(self.INDICES_FILE):
            with open(self.INDICES_FILE, 'wb') as file:
                pickle.dump(split_indices, file, protocol=2)

    def get_split_indices(self):
        """Returns train/test indices for the folds."""
        self.split_indices = pickle_loader(self.INDICES_FILE)
        return self.split_indices

    def setup_speaker_independent_split(self) -> None:
        """Prepares split for speaker independent setting

        Train: Fr, TGG, Sa
        Test: TBBT
        """
        self.train_ind_si = [i for i, data in enumerate(self.data_input) if data[self.SHOW_ID == "FRIENDS"]]
        self.test_ind_si = [i for i, data in enumerate(self.data_input) if data[self.SHOW_ID != "FRIENDS"]]

    def get_speaker_independent_split_indices(self) -> Tuple[Iterable[int], Iterable[int]]:
        """Returns the split indices of the speaker independent setting."""
        return self.train_ind_si, self.test_ind_si

    def get_split(self, indices) -> Tuple[Iterable[], Iterable[]]:
        """Returns the split comprising of the indices"""
        data_input = [self.data_input[ind] for ind in indices]
        data_output = [self.data_output[ind] for ind in indices]
        return data_input, data_output

    def full_dataset_vocab(self) -> Dict[str, int]:
        """Return the full dataset's vocabulary to filter and cache GloVe embedding dictionary."""
        vocab = defaultdict(lambda: 0)
        utterances = [instance[self.UTT_ID] for instance in self.data_input]
        contexts = [instance[self.CONTEXT_ID] for instance in self.data_input]

        for utterance in utterances:
            clean_utt = DataHelper.clean_str(utterance)
            utt_words = nltk.word_tokenize(clean_utt)
            for word in utt_words:
                vocab[word.lower()] += 1

        for context in contexts:
            for c_utt in context:
                clean_utt = DataHelper.clean_str(c_utt)
                utt_words = nltk.word_tokenize(clean_utt)
                for word in utt_words:
                    vocab[word.lower()] += 1
        return vocab

    def setup_glove_dict(self) -> None:
        """Caches the GloVe dictionary based on all the words in the dataset.

        This cache is later used to create appropriate dictionaries for each fold's training vocabulary.
        """
        assert self.config.word_embedding_path is not None

        vocab = self.full_dataset_vocab()

        if os.path.exists(self.GLOVE_DICT):
            self.word_emb_dict = pickle_loader(self.GLOVE_DICT)
        else:
            self.word_emb_dict = {}
            with open(self.config.word_embedding_path) as file:
                for line in file:
                    split_line = line.split()
                    word = split_line[0]
                    try:
                        embedding = np.asarray([float(val) for val in split_line[1:]])

                        if word.lower() in vocab:
                            self.word_emb_dict[word.lower()] = embedding
                    except Exception:
                        logger.warn("Error word in glove file (skipped): ", word)
                        continue
            self.word_emb_dict[self.PAD_TOKEN] = np.zeros(self.config.embedding_dim)
            self.word_emb_dict[self.UNK_TOKEN] = np.random.uniform(-0.25, 0.25, self.config.embedding_dim)

            with open(self.GLOVE_DICT, "wb") as file:
                pickle.dump(self.word_emb_dict, file)


class DataHelper:
    UTT_ID = 0
    SPEAKER_ID = 1
    CONTEXT_ID = 2
    CONTEXT_SPEAKERS_ID = 3
    TARGET_AUDIO_ID = 4
    TARGET_VIDEO_ID = 5
    CONTEXT_VIDEO_ID = 6
    TEXT_BERT_ID = 7
    CONTEXT_BERT_ID = 8

    PAD_ID = 0
    UNK_ID = 1
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    GLOVE_MODELS = "data/temp/glove_dict_{}.p"
    GLOVE_MODELS_CONTEXT = "data/temp/glove_dict_context_{}.p"

    def __init__(self, train_input, train_output, test_input, test_output, config: Config,
                 data_loader: DataLoader) -> None:
        self.data_loader = data_loader
        self.config = config
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output

        self.create_vocab(config.use_context)
        logger.info(f"Vocabulary size: {len(self.vocab)}")

        self.load_glove_model_for_current_split(config.use_context)
        self.create_embedding_matrix()

    @staticmethod
    def clean_str(s: str) -> str:
        """Tokenization/string cleaning."""
        s = re.sub(r"[^A-Za-z0-9(),!?'`]", " ", s)
        s = re.sub(r"'s", " 's", s)
        s = re.sub(r"'ve", " 've", s)
        s = re.sub(r"n't", " n't", s)
        s = re.sub(r"'re", " 're", s)
        s = re.sub(r"'d", " 'd", s)
        s = re.sub(r"'ll", " 'll", s)
        s = re.sub(r",", " , ", s)
        s = re.sub(r"!", " ! ", s)
        s = re.sub(r'"', ' " ', s)
        s = re.sub(r"\(", " ( ", s)
        s = re.sub(r"\)", " ) ", s)
        s = re.sub(r"\?", " ? ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"\.", " . ", s)
        s = re.sub(r"., ", " , ", s)
        s = re.sub(r"\n", " ", s)
        return s.strip().lower()

    def get_data(self, id_, is_training: bool) -> Iterable[]:
        if is_training:
            return [instance[id_] for instance in self.train_input]
        else:
            return [instance[id_] for instance in self.test_input]

    def create_vocab(self, use_context: bool = False) -> None:
        self.vocab = vocab = defaultdict(lambda: 0)
        utterances = self.get_data(self.UTT_ID, mode="train")

        for utterance in utterances:
            clean_utt = self.clean_str(utterance)
            utt_words = nltk.word_tokenize(clean_utt)
            for word in utt_words:
                vocab[word.lower()] += 1

        # Add vocabulary from the context sentences of train split if the context is used.
        if use_context:
            context_utterances = self.get_data(self.CONTEXT_ID, mode="train")
            for context in context_utterances:
                for c_utt in context:
                    clean_utt = self.clean_str(c_utt)
                    utt_words = nltk.word_tokenize(clean_utt)
                    for word in utt_words:
                        vocab[word.lower()] += 1

    def load_glove_model_for_current_split(self, use_context: bool = False) -> None:
        """Loads the GloVe pre-trained model for the current split."""
        print("Loading GloVe model")

        # if model already exists:
        filename = self.GLOVE_MODELS_CONTEXT if use_context else self.GLOVE_MODELS
        filename = filename.format(self.config.fold)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if os.path.exists(filename):
            self.model = pickle_loader(filename)
            self.embed_dim = len(self.data_loader.wordemb_dict[self.PAD_TOKEN])
        else:
            self.model = model = {}
            self.embed_dim = 0

            # Further filter glove dict words to contain only train set vocab for current fold
            for word, embedding in self.data_loader.wordemb_dict.items():
                if word in self.vocab:
                    model[word.lower()] = embedding
                self.embed_dim = len(embedding)

            with open(filename, "wb") as file:
                pickle.dump(self.model, file, protocol=2)

    def create_embedding_matrix(self) -> None:
        """Get word matrix. W[i] is the vector for word indexed by i

        also creates word_idx_map : to map all words to proper index of i for associated embedding matrix W
        """
        vocab_size = len(self.model)  # length of filtered glove embedding words
        self.word_idx_map = word_idx_map = {}
        self.W = W = np.zeros(shape=(vocab_size + 2, self.embed_dim), dtype='float32')

        # Pad and Unknown
        W[self.PAD_ID] = self.data_loader.wordemb_dict[self.PAD_TOKEN]
        W[self.UNK_ID] = self.data_loader.wordemb_dict[self.UNK_TOKEN]
        word_idx_map[self.PAD_TOKEN] = self.PAD_ID
        word_idx_map[self.UNK_TOKEN] = self.UNK_ID

        # Other words
        i = 2
        for word in self.model:
            if word not in {self.PAD_TOKEN, self.UNK_TOKEN}:
                W[i] = np.copy(self.model[word])
                word_idx_map[word] = i
                i += 1

        # Make words not in glove as unknown
        for word in self.vocab:
            if word not in self.model:
                word_idx_map[word] = self.UNK_ID

    def get_embedding_matrix(self):
        return self.W

    def word_to_index(self, utterance: str):
        word_indices = [self.word_idx_map.get(word, self.UNK_ID) for word in
                        nltk.word_tokenize(self.clean_str(utterance))]

        # padding to max_sent_length
        word_indices = word_indices[:self.config.max_sent_length]
        word_indices = word_indices + [self.PAD_ID] * (self.config.max_sent_length - len(word_indices))
        assert len(word_indices) == self.config.max_sent_length
        return word_indices

    def get_target_bert_features(self, mode: str):
        return self.get_data(self.TEXT_BERT_ID, mode)

    def get_context_bert_features(self, mode: str) -> np.ndarray:
        return np.asarray([u.mean(axis=0) for u in self.get_data(self.CONTEXT_BERT_ID, mode)])

    def vectorize_utterance(self, mode: str) -> Iterable[]:
        return [self.word_to_index(u) for u in self.get_data(self.UTT_ID, mode)]

    def get_speaker(self, mode: str) -> Iterable[]:
        authors = self.get_data(self.SPEAKER_ID, mode)

        if mode == "train":
            author_list = set()
            author_list.add("PERSON")

            for author in authors:
                author = author.strip()
                if "PERSON" not in author:
                    author_list.add(author)

            self.author_ind = {author: ind for ind, author in enumerate(author_list)}
            self.UNK_AUTHOR_ID = self.author_ind["PERSON"]
            self.config.num_authors = len(self.author_ind)

        # Convert authors into author_ids
        authors = [self.author_ind.get(author.strip(), self.UNK_AUTHOR_ID) for author in authors]
        authors = self.to_one_hot(authors, len(self.author_ind))
        return authors

    def vectorize_context(self, mode: str):
        dummy_sent = [self.PAD_ID] * self.config.max_sent_length

        contexts = self.get_data(self.CONTEXT_ID, mode)

        vector_context = []
        for context in contexts:
            local_context = []
            for utterance in context[-self.config.max_context_length:]:  # taking latest (max_context_length) sentences
                # padding to max_sent_length
                word_indices = self.word_to_index(utterance)
                local_context.append(word_indices)
            for _ in range(self.config.max_context_length - len(local_context)):
                local_context.append(dummy_sent[:])
            local_context = np.array(local_context)
            vector_context.append(local_context)

        return np.array(vector_context)

    def pool_text(self, data: Iterable[int]) -> np.ndarray:
        return np.mean([self.W[i] for i in data if i != 0], axis=0)  # only pick up non pad words

    def get_context_pool(self, mode=None):
        contexts = self.get_data(self.CONTEXT_ID, mode)

        vector_context = []
        for context in contexts:
            local_context = []
            for utterance in context[-self.config.max_context_length:]:  # taking latest (max_context_length) sentences

                if utterance == "":
                    print(context)

                # padding to max_sent_length
                word_indices = self.word_to_index(utterance)
                word_avg = self.pool_text(word_indices)

                local_context.append(word_avg)

            local_context = np.array(local_context)
            vector_context.append(np.mean(local_context, axis=0))

        return np.array(vector_context)

    def one_hot_output(self, mode=None, size=None):
        """Returns a one-hot representation of the output."""
        if mode == "train":
            labels = self.to_one_hot(self.train_output, size)
        elif mode == "test":
            labels = self.to_one_hot(self.test_output, size)
        else:
            print("Set mode properly for toOneHot method() : mode = train/test")
            exit()
        return labels

    def to_one_hot(self, data, size=None):
        """
        Returns one hot label version of data
        """
        one_hot_data = np.zeros((len(data), size))
        one_hot_data[range(len(data)), data] = 1

        assert np.array_equal(data, np.argmax(one_hot_data, axis=1))
        return one_hot_data

    @staticmethod
    def get_audio_max_length(data: Iterable[np.ndarray]) -> int:
        return max(feature.shape[1] for feature in data)

    @staticmethod
    def pad_audio(data: List[np.ndarray], max_length: int) -> np.ndarray:
        for i, instance in enumerate(data):
            if instance.shape[1] < max_length:
                instance = np.concatenate([instance, np.zeros((instance.shape[0], (max_length - instance.shape[1])))],
                                          axis=1)
                data[i] = instance
            data[i] = data[i][:, :max_length]
            data[i] = data[i].transpose()
        return np.array(data)

    def get_target_audio(self, mode=None):
        audio = self.get_data(self.TARGET_AUDIO_ID, mode)

        if mode == "train":
            self.audio_max_length = self.get_audio_max_length(audio)

        audio = self.pad_audio(audio, self.audio_max_length)

        if mode == "train":
            self.config.audio_length = audio.shape[1]
            self.config.audio_embedding = audio.shape[2]

        return audio

    def get_target_audio_pool(self, mode=None) -> np.ndarray:
        audio = self.get_data(self.TARGET_AUDIO_ID, mode)
        return np.asarray([np.mean(feature_vector, axis=1) for feature_vector in audio])

    def get_target_video_pool(self, mode=None) -> np.ndarray:
        video = self.get_data(self.TARGET_VIDEO_ID, mode)
        return np.asarray([np.mean(feature_vector, axis=0) for feature_vector in video])
