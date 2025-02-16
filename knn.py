import argparse
import numpy as np
import pandas as pd

class Knn(object):
    k = 0              # number of neighbors to use
    nFeatures = 0      # number of features seen in training
    nSamples = 0       # number of samples seen in training
    isFitted = False  # has train been called on a dataset?

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
       
        self.xFeat = xFeat
        self.y = y
        self.nSamples = xFeat.shape[0] # takes the number of rows from the 2D array
        self.nFeatures = xFeat.shape[1] # takes the number of columns from 2D array
        self.isFitted = True
        return self


    def predict(self, xFeat):
    
        m = xFeat.shape[0]
        yHat = np.zeros(m, dtype = int)
        for i in range(m):
            distances = np.sqrt(np.sum((self.xFeat - xFeat[i])**2, axis=1))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y[nearest_indices]
            label_counts = {}
            for label in nearest_labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
        
            majority_label = max(label_counts, key=label_counts.get)
            yHat[i] = majority_label

        return yHat


def accuracy(yHat, yTrue):

    count = 0
    m = len(yHat)
    for i in range(m):
        if(yHat[i] == yTrue[i]):
            count = count + 1

    return count/m


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="simxTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="simyTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="simxTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="simyTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    # assume the data is all numerical and 
    # no additional pre-processing is necessary
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain)
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest)
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()