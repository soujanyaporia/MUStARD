class Config:

    model = "SVM"
    runs = 1  # No. of runs of experiments

    # Training modes
    use_context = False  # whether to use context information or not (default false)
    use_author = False  # add author one-hot encoding in the input

    use_target_text = True
    use_target_audio = False  # adds audio target utterance features.
    use_target_video = True  # adds video target utterance features.

    speaker_independent = False  # speaker independent experiments

    embedding_dim = 300  # GloVe embedding size
    word_embedding_path = "/home/sacastro/glove.840B.300d.txt"
    max_sent_length = 20
    max_context_length = 4  # Maximum sentences to take in context
    num_classes = 2  # Binary classification of sarcasm
    epochs = 15
    batch_size = 16
    val_split = 0.1  # Percentage of data in validation set from training data
    dropout_rate = 0.5

    filter_sizes = [8, 9, 10]  # for CNN model
    num_filters = 32
