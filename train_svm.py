import json
import os

import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

from config import Config
from data_loader import DataLoader
from data_loader import DataHelper

RESULT_FILE = "./output/{}.json"


# Load config
config = Config()

# Load data
data = DataLoader(config)



def svm_train(train_input, train_output):

    clf = svm.SVC(C=10.0, gamma='scale', kernel='rbf')

    clf.fit(train_input, np.argmax(train_output, axis=1))

    return clf

def svm_test(clf, test_input, test_output):

    probas = clf.predict(test_input)
    y_pred = probas
    y_true = np.argmax(test_output, axis=1)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # To generate majority baseline
    # y_pred = [0]*len(y_pred)
    
    result_string = classification_report(y_true, y_pred, digits=4)
    print(confusion_matrix(y_true, y_pred))
    print(result_string)
    return classification_report(y_true, y_pred, output_dict=True, digits=4), result_string



def trainIO(train_index, test_index):

    # Prepare data
    train_input, train_output = data.getSplit(train_index)
    test_input, test_output = data.getSplit(test_index)

    datahelper = DataHelper(train_input, train_output, test_input, test_output, config, data)

    train_input = np.empty((len(train_input), 0))
    test_input = np.empty((len(test_input), 0))

    if config.use_target_text:
        train_input = np.concatenate([train_input,
                                      np.array([datahelper.pool_text(utt)
                                                for utt in datahelper.vectorizeUtterance(mode='train')])], axis=1)
        test_input = np.concatenate([test_input,
                                     np.array([datahelper.pool_text(utt)
                                               for utt in datahelper.vectorizeUtterance(mode='test')])], axis=1)

    if config.use_target_video:
        train_input = np.concatenate([train_input, datahelper.getTargetVideoPool(mode='train')], axis=1)
        test_input = np.concatenate([test_input, datahelper.getTargetVideoPool(mode='test')], axis=1)

    if config.use_target_audio:
        train_input = np.concatenate([train_input, datahelper.getTargetAudioPool(mode='train')], axis=1)
        test_input = np.concatenate([test_input, datahelper.getTargetAudioPool(mode='test')], axis=1)

    if train_input.shape[1] == 0:
        print("Invalid modalities")
        exit(1)

    # Aux input

    if config.use_author:
        train_input_author = datahelper.getAuthor(mode="train")
        test_input_author =  datahelper.getAuthor(mode="test")

        train_input = np.concatenate([train_input, train_input_author], axis=1)
        test_input = np.concatenate([test_input, test_input_author], axis=1)

    if config.use_context:
        train_input_context = datahelper.getContextPool(mode="train")
        test_input_context =  datahelper.getContextPool(mode="test")

        train_input = np.concatenate([train_input, train_input_context], axis=1)
        test_input = np.concatenate([test_input, test_input_context], axis=1)

    
    train_output = datahelper.oneHotOutput(mode="train", size=config.num_classes)
    test_output = datahelper.oneHotOutput(mode="test", size=config.num_classes)

    return train_input, train_output, test_input, test_output



def trainSpeakerIndependent(model_name=None):

    config.fold = "SI"
    
    (train_index, test_index) = data.getSpeakerIndependent()
    train_input, train_output, test_input, test_output = trainIO(train_index, test_index)

    clf = svm_train(train_input, train_output)
    result_dict, result_str = svm_test(clf, test_input, test_output)



def trainSpeakerDependent(model_name=None):
    
    # Load data
    data = DataLoader(config)

    # Iterating over each fold
    results=[]
    for fold, (train_index, test_index) in enumerate(data.getStratifiedKFold()):

        # Present fold
        config.fold = fold+1
        print("Present Fold: {}".format(config.fold))

        train_input, train_output, test_input, test_output = trainIO(train_index, test_index)

        clf = svm_train(train_input, train_output)
        result_dict, result_str = svm_test(clf, test_input, test_output)

        results.append(result_dict)

    # Dumping result to output
    if not os.path.exists(os.path.dirname(RESULT_FILE)):
        os.makedirs(os.path.dirname(RESULT_FILE))
    with open(RESULT_FILE.format(model_name), 'w') as file:
        json.dump(results, file)


def printResult(model_name=None):

    results = json.load(open(RESULT_FILE.format(model_name), "rb"))

    weighted_precision, weighted_recall = [], []
    weighted_fscores, macro_fscores, micro_fscores = [], [], []

    print("#"*20)
    for fold, result in enumerate(results):
        micro_fscores.append(result["micro avg"]["f1-score"])
        macro_fscores.append(result["macro avg"]["f1-score"])
        weighted_fscores.append(result["weighted avg"]["f1-score"])
        weighted_precision.append(result["weighted avg"]["precision"])
        weighted_recall.append(result["weighted avg"]["recall"])
        

        print("Fold {}:".format(fold+1))
        print("Weighted Precision: {}  Weighted Recall: {}  Weighted Fscore: {}".format(result["weighted avg"]["precision"],
                                                                                 result["weighted avg"]["recall"],
                                                                                 result["weighted avg"]["f1-score"]))
    print("#"*20)
    print("Avg :")
    print("Weighted Precision: {}  Weighted Recall: {}  Weighted Fscore: {}".format(np.mean(weighted_precision),
                                                                             np.mean(weighted_recall),
                                                                             np.mean(weighted_fscores)))
 

if __name__ == "__main__":

    if config.speaker_independent:
        trainSpeakerIndependent(model_name=config.model)
    else:
        for _ in range(config.runs):
            trainSpeakerDependent(model_name=config.model)
            printResult(model_name=config.model)
