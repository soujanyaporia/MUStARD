import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras import optimizers
from keras.models import load_model
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional, Input
from keras.layers import Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout


class keras_basemodel:

	def train(self, train_x, train_y):

		ckpt_callback = ModelCheckpoint('keras_model', 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto')
		es_callback = EarlyStopping(monitor='val_loss', 
								  min_delta=0.01, 
								  patience=10, 
								  verbose=0, 
								  mode='auto', 
								  baseline=None, 
								  restore_best_weights=True)

		self.model.fit([train_x], train_y, 
          			   epochs = self.config.epochs, 
          			   batch_size = self.config.batch_size, 
          			   validation_split = self.config.val_split,
          			   callbacks = [ckpt_callback, es_callback])


	def test(self, test_x, test_y):

		probas = self.model.predict([test_x])
		y_pred = np.argmax(probas, axis=1)
		y_true = np.argmax(test_y, axis=1)
		result_string = classification_report(y_true, y_pred)
		print(confusion_matrix(y_true, y_pred))
		print(result_string)
		return classification_report(y_true, y_pred, output_dict=True), result_string





class text_GRU(keras_basemodel):

	def __init__(self, config):
		self.config = config

	def getModel(self, embeddingMatrix):

		
		input_utterance = Input(shape=(self.config.max_sent_length,), name="input")
		embeddingLayer = Embedding(input_dim=embeddingMatrix.shape[0], output_dim=embeddingMatrix.shape[1],
                               	   weights=[embeddingMatrix],
                               	   mask_zero=True, trainable=True)

		emb1 = embeddingLayer(input_utterance)
		rnn_out = Bidirectional(GRU(100, recurrent_dropout=0.5, dropout=0.5))(emb1)
		dense = Dense(64, activation='relu')(rnn_out)
		dense = Dense(32, activation='relu')(dense)
		dense2 = Dense(self.config.num_classes,activation='softmax')(dense)

		self.model = Model(inputs=[input_utterance], outputs=dense2)

		adam = optimizers.Adam(lr=0.0001)
		self.model.compile(loss = 'categorical_crossentropy', optimizer=adam,  metrics=['acc'])
		return self.model.summary()



class text_CNN(keras_basemodel):

	def __init__(self, config):
		self.config = config

	def getModel(self, embeddingMatrix):

		sentence_length = self.config.max_sent_length
		filter_sizes = self.config.filter_sizes
		embedding_dim = self.config.embedding_dim
		num_filters = self.config.num_filters


		input_utterance = Input(shape=(sentence_length,), name="input")
		embeddingLayer = Embedding(input_dim=embeddingMatrix.shape[0], output_dim=embeddingMatrix.shape[1],
                               	   weights=[embeddingMatrix],
                               	   trainable=True)

		emb1 = embeddingLayer(input_utterance)
		reshape = Reshape((sentence_length,self.config.embedding_dim,1))(emb1)
		conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', 
				 kernel_initializer='normal', activation='relu', data_format="channels_last")(reshape)
		conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', 
				 kernel_initializer='normal', activation='relu', data_format="channels_last")(reshape)
		conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', 
				 kernel_initializer='normal', activation='relu', data_format="channels_last")(reshape)

		maxpool_0 = MaxPool2D(pool_size=(sentence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
		maxpool_1 = MaxPool2D(pool_size=(sentence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
		maxpool_2 = MaxPool2D(pool_size=(sentence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

		concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
		flatten = Flatten()(concatenated_tensor)

		dropout = Dropout(self.config.dropout_rate)(flatten)
		dense = Dense(128, activation='tanh')(flatten)
		output_layer = Dense(self.config.num_classes,activation='softmax')(dense)

		self.model = Model(inputs=[input_utterance], outputs=output_layer)

		adam = optimizers.Adam(lr=0.0001)
		self.model.compile(loss = 'categorical_crossentropy', optimizer=adam,  metrics=['acc'])
		return self.model.summary()
	
