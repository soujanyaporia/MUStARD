class Config:

	word_embedding_path = "/mnt/data/devamanyu/work/glove.840B.300d.txt"
	max_sent_length = 80
	num_classes = 2 # Binary classification of sarcasm
	epochs = 50
	batch_size = 32
	val_split = 0.1 # Percentage of data in validation set from training data