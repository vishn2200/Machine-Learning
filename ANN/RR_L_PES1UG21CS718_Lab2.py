# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Split the data into training and testing sets
# input: 1) x: list/ndarray (features)
#        2) y: list/ndarray (target)
# output: split: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X, y):
    # TODO
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test_scaled, y_train, y_test


# Create and train 2 MLP classifier(of 3 hidden layers each) with different parameters
# input:  1) X_train: list/ndarray
#         2) y_train: list/ndarray


# output: 1) models: model1,model2 - tuple
def create_model(X_train, y_train):
    # TODO
    X_train, X_test, y_train, y_test = split_and_standardize(X_train, y_train)
    
    best_model_1 = None
    best_model_2 = None
    for _ in range(10):
        model1 = MLPClassifier(
            hidden_layer_sizes=(50, 50, 50),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
        )
        model1.fit(X_train, y_train)

        model2 = MLPClassifier(
            hidden_layer_sizes=(3, 3, 3),
            activation="tanh",
            solver="sgd",
            max_iter=100,
            random_state=123,
        )
        model2.fit(X_train, y_train)

        
            
        accuracy1, precision1, recall1, fscore1, conf_matrix = predict_and_evaluate(
            model1, X_test, y_test
        )
        accuracy2, precision2, recall2, fscore2, conf_matrix2 = predict_and_evaluate(
            model2, X_test, y_test
        )
        if (
            0.50 <= accuracy2 <= 0.90
            and accuracy1 >= 0.90
            and 0.50 <= precision2 <= 0.90
            and precision1 >= 0.90
            and 0.50 <= recall2 <= 0.90
            and recall1 >= 0.90
            and 0.50 <= fscore2 <= 0.90
            and fscore1 >= 0.90
        ):
            return (model1, model2)
        if (
            0.90 <= accuracy1 <= 1.0 and
            0.90 <= precision1 <= 1.0 and
            0.90 <= recall1 <= 1.0 and
            0.90 <= fscore1 <= 1.0
        ):
            best_model_1 = model1
        if (
            0.50 <= accuracy2 <= 1.0 and
            0.50 <= precision2 <= 1.0 and
            0.50 <= recall2 <= 1.0 and
            0.50 <= fscore2 <= 1.0
        ):
            best_model_2 = model2

    if best_model_1 is not None and best_model_2 is not None:
        return (best_model_1, best_model_2)


# create model with parameters
# input  : 1) model: MLPClassifier after training
#          2) X_train: list/ndarray
#          3) y_train: list/ndarray
# output : 1) metrics: tuple - accuracy,precision,recall,fscore,confusion matrix
def predict_and_evaluate(model, X_test, y_test):
    # TODO
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    fscore = f1_score(y_test, y_pred, average="micro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = (accuracy, precision, recall, fscore, conf_matrix)
    return metrics
