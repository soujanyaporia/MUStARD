from dataclasses import dataclass


@dataclass
class Config:
    model: str = "SVM"
    runs: int = 1  # No. of runs of experiments

    # Training modes
    use_context: bool = False  # whether to use context information or not (default false)
    use_author: bool = False  # add author one-hot encoding in the input

    use_bert: bool = True  # if False, uses glove pooling

    use_target_text: bool = False
    use_target_audio: bool = False  # adds audio target utterance features.
    use_target_video: bool = False  # adds video target utterance features.

    speaker_independent: bool = False  # speaker independent experiments

    embedding_dim: int = 300  # GloVe embedding size
    word_embedding_path: str = "/home/sacastro/glove.840B.300d.txt"
    max_sent_length: int = 20
    max_context_length: int = 4  # Max sentence number to take from the context.
    num_classes: int = 2  # Binary classification of sarcasm
    epochs: int = 15
    batch_size: int = 16
    val_split: float = 0.1  # Percentage of data in validation set from training data

    svm_c: float = 10.0
    svm_scale: bool = True


@dataclass
class SpeakerDependentTConfig(Config):
    use_target_text: bool = True
    svm_c: float = 1.0


@dataclass
class SpeakerDependentAConfig(Config):
    use_target_audio: bool = True
    svm_c: float = 1.0


@dataclass
class SpeakerDependentVConfig(Config):
    use_target_video: bool = True
    svm_c: float = 1.0


@dataclass
class SpeakerDependentTAConfig(Config):
    use_target_text: bool = True
    use_target_audio: bool = True
    svm_c: float = 1.0


@dataclass
class SpeakerDependentTVConfig(Config):
    use_target_text: bool = True
    use_target_video: bool = True
    svm_c: float = 10.0


@dataclass
class SpeakerDependentAVConfig(Config):
    use_target_audio: bool = True
    use_target_video: bool = True
    svm_c: float = 30.0


@dataclass
class SpeakerDependentTAVConfig(Config):
    use_target_text: bool = True
    use_target_audio: bool = True
    use_target_video: bool = True
    svm_c: float = 10.0


@dataclass
class SpeakerDependentTPlusContext(SpeakerDependentTConfig):
    use_context: bool = True
    svm_c: float = 1.0


@dataclass
class SpeakerDependentTPlusAuthor(SpeakerDependentTConfig):
    use_author: bool = True
    svm_c: float = 10.0


@dataclass
class SpeakerDependentTVPlusContext(SpeakerDependentTVConfig):
    use_context: bool = True
    svm_c: float = 10.0


@dataclass
class SpeakerDependentTVPlusAuthor(SpeakerDependentTVConfig):
    use_author: bool = True
    svm_c: float = 10.0


@dataclass
class SpeakerIndependentTConfig(Config):
    svm_scale: bool = False
    use_target_text: bool = True
    svm_c: float = 10.0
    speaker_independent: bool = True


@dataclass
class SpeakerIndependentAConfig(Config):
    svm_scale: bool = False
    use_target_audio: bool = True
    svm_c: float = 1000.0
    speaker_independent: bool = True


@dataclass
class SpeakerIndependentVConfig(Config):
    svm_scale = False
    use_target_video: bool = True
    svm_c: float = 30.0
    speaker_independent: bool = True


@dataclass
class SpeakerIndependentTAConfig(Config):
    svm_scale: bool = False
    use_target_text: bool = True
    use_target_audio: bool = True
    svm_c: float = 500.0
    speaker_independent: bool = True


@dataclass
class SpeakerIndependentTVConfig(Config):
    svm_scale: bool = False
    use_target_text: bool = True
    use_target_video: bool = True
    svm_c: float = 10.0
    speaker_independent: bool = True


@dataclass
class SpeakerIndependentAVConfig(Config):
    svm_scale: bool = False
    use_target_audio: bool = True
    use_target_video: bool = True
    svm_c = 500.0
    speaker_independent: bool = True


@dataclass
class SpeakerIndependentTAVConfig(Config):
    svm_scale: bool = False
    use_target_text: bool = True
    use_target_audio: bool = True
    use_target_video: bool = True
    svm_c = 1000.0
    speaker_independent: bool = True


@dataclass
class SpeakerIndependentTPlusContext(SpeakerIndependentTConfig):
    use_context: bool = True
    svm_c: float = 10.0


@dataclass
class SpeakerIndependentTPlusAuthor(SpeakerIndependentTConfig):
    use_author: bool = True
    svm_c: float = 10.0


@dataclass
class SpeakerIndependentTAPlusContext(SpeakerIndependentTAConfig):
    use_context: bool = True
    svm_c: float = 1000.0


@dataclass
class SpeakerIndependentTAPlusAuthor(SpeakerIndependentTAConfig):
    use_author: bool = True
    svm_c: float = 1000.0


CONFIG_BY_KEY = {
    '': Config(),
    't': SpeakerDependentTConfig(),
    'a': SpeakerDependentAConfig(),
    'v': SpeakerDependentVConfig(),
    'ta': SpeakerDependentTAConfig(),
    'tv': SpeakerDependentTVConfig(),
    'av': SpeakerDependentAVConfig(),
    'tav': SpeakerDependentTAVConfig(),
    't-c': SpeakerDependentTPlusContext(),
    't-author': SpeakerDependentTPlusAuthor(),
    'tv-c': SpeakerDependentTVPlusContext(),
    'tv-author': SpeakerDependentTVPlusAuthor(),
    'i-t': SpeakerIndependentTConfig(),
    'i-a': SpeakerIndependentAConfig(),
    'i-v': SpeakerIndependentVConfig(),
    'i-ta': SpeakerIndependentTAConfig(),
    'i-tv': SpeakerIndependentTVConfig(),
    'i-av': SpeakerIndependentAVConfig(),
    'i-tav': SpeakerIndependentTAVConfig(),
    'i-t-c': SpeakerIndependentTPlusContext(),
    'i-t-author': SpeakerIndependentTPlusAuthor(),
    'i-ta-c': SpeakerIndependentTAPlusContext(),
    'i-ta-author': SpeakerIndependentTAPlusAuthor(),
}
