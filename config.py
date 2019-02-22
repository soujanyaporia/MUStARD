class Config:

	embedding_dim = 300
	word_embedding_path = "/mnt/data/devamanyu/work/glove.840B.300d.txt"
	max_sent_length = 15
	max_context_length = 4 # Maximum sentences to take in context
	num_classes = 2 # Binary classification of sarcasm
	epochs = 50
	batch_size = 64
	val_split = 0.2 # Percentage of data in validation set from training data
	dropout_rate = 0.6

	filter_sizes = [5,6,7] # for CNN model
	num_filters = 64
	use_context = False # whether to use context information or not (default false)
	use_author = True # add author one-hot encoding in the input