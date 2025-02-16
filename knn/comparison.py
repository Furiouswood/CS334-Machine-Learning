
import argparse
import pandas as pd
from preprocess import standard_scale, minmax_range, add_irr_feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def evaluate_knn(xTrain, yTrain, xTest, yTest):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(xTrain, yTrain)
    y_pred = knn.predict(xTest)
    acc = accuracy_score(yTest, y_pred)
    return knn, acc
  

def evaluate_nb(xTrain, yTrain, xTest, yTest):

    nb = GaussianNB()
    nb.fit(xTrain, yTrain)
    y_pred = nb.predict(xTest)
    acc = accuracy_score(yTest, y_pred)
    return nb, acc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="space_trainx.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="space_trainy.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="space_testx.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="space_testy.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    # flatten to compress to 1-d rather than (m, 1)
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()

    preprocess_methods = [
        ("None", lambda x_tr, x_te: (x_tr, x_te)),  
        ("StandardScaler", standard_scale),
        ("MinMaxScaler", minmax_range),
        ("AddIrrelevantFeatures", add_irr_feature)
    ]
    knn_accuracies = []
    nb_accuracies = []
    for method_name, preprocess_func in preprocess_methods:
        x_train_processed, x_test_processed = preprocess_func(xTrain, xTest)
        knn_model, knn_acc = evaluate_knn(x_train_processed, yTrain, x_test_processed, yTest)
        knn_accuracies.append(knn_acc)
        nb_model, nb_acc = evaluate_nb(x_train_processed, yTrain, x_test_processed, yTest)
        nb_accuracies.append(nb_acc)
    print("Preprocessing Method  | K-NN Accuracy | Naive Bayes Accuracy")
    for i, (method_name, _) in enumerate(preprocess_methods):
        print(f"{method_name:<25} | {knn_accuracies[i]:.4f}        | {nb_accuracies[i]:.4f}")


if __name__ == "__main__":
    main()
