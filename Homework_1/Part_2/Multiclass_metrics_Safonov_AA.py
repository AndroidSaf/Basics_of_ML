# На вход подаются массив y_true длины n, содержащий "метки" классов и
# матрица y_predict размером n на m, содержащая в каждой строке "1", означающую предсказание соответствующей
# метки класса и находящуюся на соответствующем метке индексе элемента этой строки
# Так как задание было некорректно сформулировано, то я реализовал percent следующим образом: percent - значение, 
# заключенное в полуинтервале (0, 1] и означающее часть выборки от генеральной, к которой и будут применены соответствующие метрики  
# На более конкретные вопросы готов ответить лично в Telegram
# Andrew Safonov

import numpy as np


def default_value():
    return 0.5


def matrix_to_vector(targets, percent):
    targets = targets[:int(targets.shape[0] * percent), :]
    return np.array([np.argmax(targets[i, :]) for i in range(targets.shape[0])])


def confusion_matrix(y_true, y_predict, n):
    matrix = np.array([np.argwhere((y_true == x) & (y_predict == y)).shape[0] for x in range(n) for y in range(n)])
    return np.reshape(matrix, (n, n))


def accuracy_score(y_true, y_predict, percent=default_value()):
    n_features = y_predict.shape[1]
    confusion_feature_matrix = confusion_matrix(y_true[:int(y_true.shape[0] * percent)],
                                                matrix_to_vector(y_predict, percent), n_features)
    result = np.trace(confusion_feature_matrix) / confusion_feature_matrix.sum()
    return result


def precision_score(y_true, y_predict, percent=default_value()):
    n_features = y_predict.shape[1]
    confusion_feature_matrix = confusion_matrix(y_true[:int(y_true.shape[0] * percent)],
                                                matrix_to_vector(y_predict, percent), n_features)
    result = np.array([(confusion_feature_matrix[feature, feature] /
                        confusion_feature_matrix[:, feature].sum()) for feature in range(n_features)])
    return result


def recall_score(y_true, y_predict, percent=default_value()):
    n_features = y_predict.shape[1]
    confusion_feature_matrix = confusion_matrix(y_true[:int(y_true.shape[0] * percent)],
                                                matrix_to_vector(y_predict, percent), n_features)
    result = np.array([(confusion_feature_matrix[feature, feature] /
                        confusion_feature_matrix[feature, :].sum()) for feature in range(n_features)])
    return result


def f1_score(y_true, y_predict, percent=default_value()):
    precision = precision_score(y_true, y_predict, percent)
    recall = recall_score(y_true, y_predict, percent)
    result = 2 * precision * recall / (precision + recall)
    return result


def lift_score(y_true, y_predict, percent=default_value()):
    n_features = y_predict.shape[1]
    confusion_feature_matrix = confusion_matrix(y_true[:int(y_true.shape[0] * percent)],
                                                matrix_to_vector(y_predict, percent), n_features)
    precision = precision_score(y_true, y_predict, percent)
    TP = confusion_feature_matrix.diagonal()
    FN = np.array([((np.delete(np.delete(confusion_feature_matrix, feature, axis=0), feature, axis=1)).sum() - 
                     (np.delete(np.delete(confusion_feature_matrix, feature, axis=0), feature, axis=1)).diagonal().sum()) 
                    for feature in range(n_features)])
    result = precision * int(y_true.shape[0] * percent) / (TP + FN)
    return result
