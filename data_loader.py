
import os
import re
import json
import pickle

import nltk
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

class DataLoader:

    DATA_PATH_JSON = "./data/sarcasm_data.json"
    INDICES_FILE = "./data/split_indices.p"

    def __init__(self):
        
        dataset_json = json.load(open(self.DATA_PATH_JSON))
        self.parseData(dataset_json)
        self.StratifiedKFold()


    def parseData(self, json):
        '''
        Prepares json data into lists
        data_input = [ (utterance:string, speaker:string, context:list_of_strings, context_speakers:list_of_strings ) ]
        data_output = [ sarcasm_tag:int ]
        '''
        self.data_input, self.data_output = [], []
        
        for ID in json.keys():
            self.data_input.append( (json[ID]["utterance"], json[ID]["speaker"], json[ID]["context"], json[ID]["context_speakers"]) )
            self.data_output.append( int(json[ID]["sarcasm"]) )


    def StratifiedKFold(self, splits=5):
        '''
        Prepares or loads (if existing) splits for k-fold 
        '''
        skf = StratifiedKFold(n_splits=splits, shuffle=True)
        split_indices = [(train_index, test_index) for train_index, test_index in skf.split(self.data_input, self.data_output)]

        if not os.path.exists(self.INDICES_FILE):
            pickle.dump(split_indices, open(self.INDICES_FILE, 'wb'), protocol=2)
        

    def getStratifiedKFold(self):
        '''
        Returns train/test indices for k-folds
        '''
        self.split_indices = pickle.load(open(self.INDICES_FILE, 'rb'))
        return self.split_indices


    def getSplit(self, indices):
        '''
        Returns the split comprising of the indices
        '''
        data_input = [self.data_input[ind] for ind in indices]
        data_output = [self.data_output[ind] for ind in indices]
        return data_input, data_output

            


class DataHelper:

    UTT_ID = 0
    SPEAKER_ID = 1
    CONTEXT_ID = 2
    CONTEXT_SPEAKERS_ID = 3

    PAD_ID = 0
    UNK_ID = 1
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    GLOVE_MODELS = "./data/temp/glove_dict_{}.p"


    def __init__(self, train_input, train_output, test_input, test_output, config):
        self.config = config
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output

        self.createVocab()
        print("vocab size: " + str(len(self.vocab)))

        self.loadGloveModel(config.word_embedding_path)
        self.createEmbeddingMatrix()



    def getData(self, data, ID=None):
        return [instance[ID] for instance in data]



    def createVocab(self):

        self.vocab = vocab = defaultdict(lambda:0)
        utterances = self.getData(self.train_input, self.UTT_ID)

        for utterance in utterances:
            clean_utt = self.clean_str(utterance)
            utt_words = nltk.word_tokenize(clean_utt)
            for word in utt_words:
                vocab[word] += 1


    def loadGloveModel(self, gloveFile):
        '''
        Loads the Glove pre-trained model
        '''
        assert(gloveFile is not None)
        print("Loading glove model")

        # if model already exists:
        filename = self.GLOVE_MODELS.format(self.config.fold)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if os.path.exists(filename):
            self.model = pickle.load(open(filename, "rb"))
            self.embed_dim = len(self.model[self.PAD_TOKEN])
        else:
            self.model = model = {}
            self.embed_dim = 0
            for line in open(gloveFile,'r'):
                splitLine = line.split() 
                word = splitLine[0]
                try:
                    embedding = np.array([float(val) for val in splitLine[1:]])
                except:
                    print("Here", word)
                    continue
                if word in self.vocab: model[word] = embedding
                self.embed_dim = len(splitLine[1:])
            model[self.UNK_TOKEN] = np.zeros(self.embed_dim)
            model[self.PAD_TOKEN] = np.zeros(self.embed_dim)

            pickle.dump(self.model, open(filename, "wb"), protocol=2)


    def createEmbeddingMatrix(self):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        also creates word_idx_map : to map all words to proper index of i for associated
        embedding matrix W
        """

        vocab_size = len(self.model)
        self.word_idx_map = word_idx_map = dict()
        self.W = W = np.zeros(shape=(vocab_size+2, self.embed_dim), dtype='float32')            
        W[self.PAD_ID] = np.zeros(self.embed_dim, dtype='float32')
        W[self.UNK_ID] = np.random.uniform(-0.25,0.25,self.embed_dim) 
        i = 2
        for word in self.model:
            W[i] = self.model[word]
            word_idx_map[word] = i
            i += 1

        # Make words not in glove as unknown
        for word in self.vocab:
            if word not in self.model:
                word_idx_map[word] = self.UNK_ID

    def getEmbeddingMatrix(self):
        return self.W

    def clean_str(self, string):
        '''
        Tokenization/string cleaning.
        '''
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\"", " \" ", string) 
        string = re.sub(r"\(", " ( ", string) 
        string = re.sub(r"\)", " ) ", string) 
        string = re.sub(r"\?", " ? ", string) 
        string = re.sub(r"\s{2,}", " ", string) 
        string = re.sub(r"\.", " . ", string)    
        string = re.sub(r".\, ", " , ", string)  
        string = re.sub(r"\\n", " ", string)  
        return string.strip().lower()


    def vectorizeUtterance(self, mode=None):

        if mode == "train":
            utterances = self.getData(self.train_input, self.UTT_ID)
        elif mode == "test":
            utterances = self.getData(self.test_input, self.UTT_ID)
        else:
            print("Set mode properly for vectorizeUtterance method() : mode = train/test")
            exit()

        vector_utt = []
        for utterance in utterances:

            word_indices = [self.word_idx_map.get(word, self.UNK_ID) for word in utterance.split(" ")]
            
            #padding to max_sent_length
            word_indices = word_indices[:self.config.max_sent_length]
            word_indices = word_indices + [self.PAD_ID]*(self.config.max_sent_length - len(word_indices))
            assert(len(word_indices) == self.config.max_sent_length)
            vector_utt.append(word_indices)
        return vector_utt


    def toOneHot(self, mode=None):
        '''
        Returns one hot label version of [train/test]_output
        '''

        if mode == "train":
            labels = self.train_output
        elif mode == "test":
            labels = self.test_output
        else:
            print("Set mode properly for toOneHot method() : mode = train/test")
            exit()

        oneHotLabel = np.zeros((len(labels), self.config.num_classes))
        oneHotLabel[range(len(labels)),labels] = 1
        
        assert(np.array_equal(labels, np.argmax(oneHotLabel, axis=1)))
        return oneHotLabel




if __name__ == "__main__":
    dataLoader = DataLoader()