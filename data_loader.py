
import os
import sys
import re
import json
import pickle

import nltk
import numpy as np
from beeprint import pp
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold


def pickle_loader(filename):
    if sys.version_info[0] < 3:
        return pickle.load(open(filename, 'rb'))
    else:
        return pickle.load(open(filename, 'rb'), encoding="latin1")


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
        self.split_indices = pickle_loader(self.INDICES_FILE)
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
    GLOVE_MODELS_CONTEXT = "./data/temp/glove_dict_context_{}.p"


    def __init__(self, train_input, train_output, test_input, test_output, config):
        self.config = config
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output

        self.createVocab(config.use_context)
        print("vocab size: " + str(len(self.vocab)))

        self.loadGloveModel(config.word_embedding_path, config.use_context)
        self.createEmbeddingMatrix()



    def getData(self, ID=None, mode=None, error_message=None):

        if mode == "train":
            return [instance[ID] for instance in self.train_input]
        elif mode == "test":
            return [instance[ID] for instance in self.test_input]
        else:
            print(error_message)
            exit()



    def createVocab(self, use_context=False):

        self.vocab = vocab = defaultdict(lambda:0)
        utterances = self.getData(self.UTT_ID, mode="train")

        for utterance in utterances:
            clean_utt = self.clean_str(utterance)
            utt_words = nltk.word_tokenize(clean_utt)
            for word in utt_words:
                vocab[word.lower()] += 1

        if use_context:
            context_utterances = self.getData(self.CONTEXT_ID, mode="train")
            for context in context_utterances:
                for c_utt in context:
                    clean_utt = self.clean_str(c_utt)
                    utt_words = nltk.word_tokenize(clean_utt)
                    for word in utt_words:
                        vocab[word.lower()] += 1



    def loadGloveModel(self, gloveFile, use_context=False):
        '''
        Loads the Glove pre-trained model
        '''
        assert(gloveFile is not None)
        print("Loading glove model")

        # if model already exists:
        filename = self.GLOVE_MODELS_CONTEXT if use_context else self.GLOVE_MODELS
        filename = filename.format(self.config.fold)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        if os.path.exists(filename):
            self.model = pickle_loader(filename)
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
                if word in self.vocab: model[word.lower()] = embedding
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


    def wordToIndex(self, utterance):

        word_indices = [self.word_idx_map.get(word, self.UNK_ID) for word in nltk.word_tokenize(self.clean_str(utterance))]

        #padding to max_sent_length
        word_indices = word_indices[:self.config.max_sent_length]
        word_indices = word_indices + [self.PAD_ID]*(self.config.max_sent_length - len(word_indices))
        assert(len(word_indices) == self.config.max_sent_length)
        return word_indices

    def vectorizeUtterance(self, mode=None):

        
        utterances = self.getData(self.UTT_ID, mode, 
                                  "Set mode properly for vectorizeUtterance method() : mode = train/test")

        vector_utt = []
        for utterance in utterances:
            word_indices = self.wordToIndex(utterance)
            vector_utt.append(word_indices)

        return vector_utt

    def contextMask(self, mode=None):

        contexts = self.getData(self.CONTEXT_ID, mode, 
                                "Set mode properly for contextMask method() : mode = train/test")


    def getAuthor(self, mode=None):

        authors = self.getData(self.SPEAKER_ID, mode, 
                               "Set mode properly for contextMask method() : mode = train/test")

        # Create dictionary for speaker

        if mode=="train":
            author_list = set()
            author_list.add("PERSON")

            for author in authors:
                author = author.strip()
                if "PERSON" not in author:
                    author_list.add(author)

            self.author_ind={author:ind for ind, author in enumerate(author_list)}
            self.UNK_AUTHOR_ID = self.author_ind["PERSON"]
            self.config.num_authors = len(self.author_ind)
        
        # Convert authors into author_ids
        authors = [self.author_ind.get(author.strip(), self.UNK_AUTHOR_ID) for author in authors]
        authors = self.toOneHot(authors, len(self.author_ind))
        return authors
        

    def vectorizeContext(self, mode=None):

        dummy_sent = [self.PAD_ID]*self.config.max_sent_length

        contexts = self.getData(self.CONTEXT_ID, mode, 
                                "Set mode properly for vectorizeContext method() : mode = train/test")

        vector_context = []
        for context in contexts:
            local_context = []
            for utterance in context[-self.config.max_context_length:]: # taking latest (max_context_length) sentences
                #padding to max_sent_length
                word_indices = self.wordToIndex(utterance)
                local_context.append(word_indices)
            for _ in range(self.config.max_context_length - len(local_context)):
                local_context.append(dummy_sent[:])
            local_context = np.array(local_context)
            vector_context.append(local_context)

        return np.array(vector_context)


    def oneHotOutput(self, mode=None, size=None):
        '''
        Returns one hot encoding of the output
        '''
        if mode == "train":
            labels = self.toOneHot(self.train_output, size)
        elif mode == "test":
            labels = self.toOneHot(self.test_output, size)
        else:
            print("Set mode properly for toOneHot method() : mode = train/test")
            exit()
        return labels

    def toOneHot(self, data, size=None):
        '''
        Returns one hot label version of data
        '''
        oneHotData = np.zeros((len(data), size))
        oneHotData[range(len(data)),data] = 1
        
        assert(np.array_equal(data, np.argmax(oneHotData, axis=1)))
        return oneHotData




if __name__ == "__main__":
    dataLoader = DataLoader()
