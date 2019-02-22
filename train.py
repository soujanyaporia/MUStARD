import os
import pickle
import json


import numpy as np
from beeprint import pp

from config import Config
from data_loader import DataLoader
from data_loader import DataHelper
from models import text_GRU, text_CNN, text_CNN_context

# Desired graphics card selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

RESULT_FILE = "./output/{}.json"


def train(model_name=None):

    # Load config
    config = Config()
    
    # Load data
    data = DataLoader()


    # Iterating over each fold
    results=[]
    for fold, (train_index, test_index) in enumerate(data.getStratifiedKFold()):
        config.fold = fold+1
        print("Present Fold: {}".format(config.fold))

        # Prepare data
        train_input, train_output = data.getSplit(train_index)
        test_input, test_output = data.getSplit(test_index)
        datahelper = DataHelper(train_input, train_output, test_input, test_output, config)
        

        train_input = [datahelper.vectorizeUtterance(mode="train")]
        test_input = [datahelper.vectorizeUtterance(mode="test")]
        
        if config.use_context: # Add context to the input
            train_input.append(datahelper.vectorizeContext(mode="train"))
            test_input.append(datahelper.vectorizeContext(mode="test"))

        if config.use_author:
            train_input.append(datahelper.getAuthor(mode="train"))
            test_input.append(datahelper.getAuthor(mode="test"))

        
        train_output = datahelper.oneHotOutput(mode="train", size=config.num_classes)
        test_output = datahelper.oneHotOutput(mode="test", size=config.num_classes)

        if model_name == "text_GRU":
            model = text_GRU(config)
        elif model_name == "text_CNN":
            model = text_CNN_context(config)



        summary = model.getModel(datahelper.getEmbeddingMatrix())
        model.train(train_input, train_output)
        result_dict, result_str = model.test(test_input, test_output)
        results.append(result_dict)

    
    # Dumping result to output
    if not os.path.exists(os.path.dirname(RESULT_FILE)):
        os.makedirs(os.path.dirname(RESULT_FILE))
    json.dump(results, open(RESULT_FILE.format(model_name), "wb"))



def printResult(model_name=None):

    results = json.load(open(RESULT_FILE.format(model_name), "rb"))

    weighted_fscores, macro_fscores, micro_fscores = [], [], []
    print("#"*20)
    for fold, result in enumerate(results):
        weighted_fscores.append(result["weighted avg"]["f1-score"])
        macro_fscores.append(result["macro avg"]["f1-score"])
        micro_fscores.append(result["micro avg"]["f1-score"])

        print("Fold {}:".format(fold+1))
        print("Macro Fscore: {}  Micro Fscore: {}  Weighted Fscore: {}".format(result["weighted avg"]["f1-score"],
                                                                               result["macro avg"]["f1-score"],
                                                                               result["micro avg"]["f1-score"]))
    print("#"*20)
    print("Avg :")
    print("Macro Fscore: {}  Micro Fscore: {}  Weighted Fscore: {}".format(np.mean(macro_fscores),
                                                                           np.mean(micro_fscores),
                                                                           np.mean(weighted_fscores)))


if __name__ == "__main__":

    '''
    model_names:
    - text_GRU
    - text_CNN
    '''
    MODEL = "text_CNN"

    train(model_name=MODEL)
    printResult(model_name=MODEL)