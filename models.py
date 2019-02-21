import numpy as np
from sklearn.metrics import classification_report

from keras import optimizers
from keras.models import load_model
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional, Input


class text_GRU:

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
		print(result_string)
		return classification_report(y_true, y_pred, output_dict=True), result_string
