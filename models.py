import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import keras.backend as K
from keras import optimizers
from keras.models import load_model
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional, Input, Lambda
from keras.layers import Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, BatchNormalization


class keras_basemodel:

    MODEL_PATH = "keras_model"

    def train(self, train_x, train_y):



        ckpt_callback = ModelCheckpoint(self.MODEL_PATH, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto')
        # es_callback = EarlyStopping(monitor='val_loss', 
        #                           min_delta=0.01, 
        #                           patience=10, 
        #                           verbose=0, 
        #                           mode='auto', 
        #                           restore_best_weights=True)

        
        self.model.fit(train_x, train_y, 
                       epochs = self.config.epochs, 
                       batch_size = self.config.batch_size, 
                       validation_split = self.config.val_split,
                       callbacks = [ckpt_callback])


    def test(self, test_x, test_y):

        model = load_model(self.MODEL_PATH)
        probas = model.predict(test_x)
        y_pred = np.argmax(probas, axis=1)
        y_true = np.argmax(test_y, axis=1)

        # To generate random scores
        # y_pred = np.random.randint(2, size=len(y_pred))
        
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

        adam = optimizers.Adam(lr=0.01)
        self.model.compile(loss = 'categorical_crossentropy', optimizer=adam,  metrics=['acc'])
        return self.model.summary()



class text_CNN(keras_basemodel):

    def __init__(self, config):
        self.config = config


    def getModel(self, embeddingMatrix):

        sentence_length = self.config.max_sent_length
        context_length = self.config.max_context_length
        filter_sizes = self.config.filter_sizes
        embedding_dim = self.config.embedding_dim
        num_filters = self.config.num_filters

            

        # Input layers
        input_utterance = Input(shape=(sentence_length,), name="input_utterance")

        if self.config.use_target_audio:
            audio_length = self.config.audio_length
            audio_dim = self.config.audio_embedding
            input_audio = Input(shape=(audio_length, audio_dim), name="input_audio")
            
        if self.config.use_context:
            input_context = Input(shape=(context_length, sentence_length), name="input_context")
        
        if self.config.use_author:
            num_authors = self.config.num_authors
            input_authors = Input(shape=(num_authors,), name="input_authors")
        


        # Layer functions
        embeddingLayer = Embedding(input_dim=embeddingMatrix.shape[0], output_dim=embeddingMatrix.shape[1],
                                   weights=[embeddingMatrix],
                                   trainable=False)

        def slicer(x, index):
            return x[:,K.constant(index, dtype='int32'),:]

        def slicer_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 3  # batch, seq_len, sent_len
            new_shape = (shape[0], shape[2])
            return new_shape

        def reshaper(x, axis):
            return K.expand_dims(x, axis=axis)
        
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', 
                 kernel_initializer='normal', activation='tanh', data_format="channels_last")
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', 
                 kernel_initializer='normal', activation='tanh', data_format="channels_last")
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', 
                 kernel_initializer='normal', activation='tanh', data_format="channels_last")
        maxpool_0 = MaxPool2D(pool_size=(sentence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')
        maxpool_1 = MaxPool2D(pool_size=(sentence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')
        maxpool_2 = MaxPool2D(pool_size=(sentence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')

        if self.config.use_target_audio:
            conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[0], audio_dim), padding='valid', 
                     kernel_initializer='normal', activation='tanh', data_format="channels_last")
            conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[1], audio_dim), padding='valid', 
                     kernel_initializer='normal', activation='tanh', data_format="channels_last")
            conv_5 = Conv2D(num_filters, kernel_size=(filter_sizes[2], audio_dim), padding='valid', 
                     kernel_initializer='normal', activation='tanh', data_format="channels_last")
            maxpool_3 = MaxPool2D(pool_size=(audio_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')
            maxpool_4 = MaxPool2D(pool_size=(audio_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')
            maxpool_5 = MaxPool2D(pool_size=(audio_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')

        dense_func = Dense(128, activation='relu', name="dense")
        batch_norm = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,)



        # Network graph

        ## for target utterance
        utt_emb = embeddingLayer(input_utterance)

        def convolution_operation(input_sent):
            reshaped = Lambda(reshaper, arguments={'axis':3})(input_sent)
            concatenated_tensor = Concatenate(axis=1)([maxpool_0(conv_0(reshaped)), maxpool_1(conv_1(reshaped)), maxpool_2(conv_2(reshaped))])
            return Flatten()(concatenated_tensor)

        def convolution_operation_audio(input_sent):
            reshaped = Lambda(reshaper, arguments={'axis':3})(input_sent)
            concatenated_tensor = Concatenate(axis=1)([maxpool_3(conv_3(reshaped)), maxpool_4(conv_4(reshaped)), maxpool_5(conv_5(reshaped))])
            concatenated_tensor = Dropout(self.config.dropout_rate)(concatenated_tensor)
            return Flatten()(concatenated_tensor)

        
        utt_conv_out = convolution_operation(utt_emb)

        if self.config.use_target_audio:
            # audio_vector = GRU(128)(input_audio)
            
            audio_vector = convolution_operation_audio(input_audio)


        ## for context utterances
        if self.config.use_context:
            cnn_output = []
            for ind in range(context_length):
                local_input = Lambda(slicer, output_shape=slicer_output_shape, arguments={"index":ind})(input_context) # Batch, word_indices
                local_utt_emb = embeddingLayer(local_input)
                local_utt_conv_out = convolution_operation(local_utt_emb)
                local_dense_output = dense_func(local_utt_conv_out)
                cnn_output.append(local_dense_output)
            
            def stack(x):
                return K.stack(x, axis=1)
            cnn_outputs = Lambda(stack)(cnn_output)

            def reduce_mean(x):
                return K.mean(x, axis=1)
        
            # context_vector = GRU(128)(cnn_outputs)
            context_vector= Lambda(reduce_mean)(cnn_outputs)
        
        if self.config.use_context:
            joint_input = Concatenate(axis=1)([utt_conv_out, context_vector])
        else:
            joint_input = utt_conv_out

        if self.config.use_author:
            joint_input = Concatenate(axis=1)([joint_input, input_authors])

        if self.config.use_target_audio:
            joint_input = Concatenate(axis=1)([joint_input, audio_vector])



        dropout = Dropout(self.config.dropout_rate)(joint_input)
        dropout = batch_norm(dropout)
        dense = Dense(128, activation='relu')(dropout)
        output_layer = Dense(self.config.num_classes,activation='softmax')(dense)

        # Define model input
        model_input = [input_utterance]
        if self.config.use_context:
            model_input.append(input_context)
        if self.config.use_author:
            model_input.append(input_authors)
        if self.config.use_target_audio:
            model_input.append(input_audio)


        self.model = Model(inputs=model_input, outputs=output_layer)

        adam = optimizers.Adam(lr=0.0001)
        self.model.compile(loss = 'categorical_crossentropy', optimizer=adam,  metrics=['acc'])
        return self.model.summary()
    
