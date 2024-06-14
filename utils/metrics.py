import numpy as np
import pandas as pd


# positive == fake news, negative == true news
NEGATIVE_CLASS = 'true'


def get_accuracy(tp: int, fp: int, tn: int, fn: int):
    return (tp + tn) / (tp + tn + fp + fn)


def get_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)


def get_precision(tp: int, fp: int) -> float:
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def get_f1_score(tp: int, fp: int, fn: int) -> float:
    precision, recall = get_precision(tp=tp, fp=fp), get_recall(tp=tp, fn=fn)
    if precision + recall == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))


def get_fpr(fp: int, tn: int):
    if fp == 0:
        return 0
    if (fp + tn) == 0:
        return np.inf
    return fp / (fp + tn)


class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


def retrieve_confusion_matrices(predictions: np.array, classes: np.array) -> {str, ConfusionMatrix}:
    df = pd.DataFrame(data={'prediction': predictions, 'class': classes})

    classes = df['class'].unique()
    confusions = {c: ConfusionMatrix() for c in classes}
    confusions['total'] = ConfusionMatrix()

    for cl in classes:
        if cl != NEGATIVE_CLASS:
            # Here if cl is a fake news class
            fn_mask = (df['prediction'] == False) & (df['class'] == cl)
            tp_mask = (df['prediction'] == True) & (df['class'] == cl)
            fn = df[fn_mask].shape[0]
            tp = df[tp_mask].shape[0]

            # Number of 'tp' and 'fn' of the current class
            confusions[cl].fn = fn
            confusions[cl].tp = tp

            # Cumulative fn and tp
            confusions['total'].fn += fn
            confusions['total'].tp += tp
        else:
            # There is only one 'true news' class
            fp_mask = (df['prediction'] == True) & (df['class'] == NEGATIVE_CLASS)
            tn_mask = (df['prediction'] == False) & (df['class'] == NEGATIVE_CLASS)
            confusions[NEGATIVE_CLASS].fp = df[fp_mask].shape[0]
            confusions[NEGATIVE_CLASS].tn = df[tn_mask].shape[0]

    confusions['total'].fp = confusions[NEGATIVE_CLASS].fp if NEGATIVE_CLASS in confusions else 0
    confusions['total'].tn = confusions[NEGATIVE_CLASS].tn if NEGATIVE_CLASS in confusions else 0
    return confusions


def evaluatePerformance(outcome, evaluationLabels):
    # outcome: boolean      True    ->  TRUE NEWS
    #                       False   ->  FAKE NEWS
    # evaluationLabels:     original labels
    confusions = retrieve_confusion_matrices(outcome, evaluationLabels)
    formatted_print(confusions)


def formatted_print(confusions: {str, ConfusionMatrix}):
    TP = confusions['total'].tp
    TN = confusions['total'].tn
    FP = confusions['total'].fp
    FN = confusions['total'].fn

    accuracy = get_accuracy(tp=TP, fn=FN, tn=TN, fp=FP)
    recall = get_recall(tp=TP, fn=FN)
    precision = get_precision(tp=TP, fp=FP)
    f1 = get_f1_score(tp=TP, fn=FN, fp=FP)
    fpr = get_fpr(fp=FP, tn=TN)

    print('')
    print('Confusion matrix:')

    print('%42s' % ('prediction'))
    print('%22s %16s %14s' % ('|', 'TRUE (neg.) | ', 'FAKE (pos.)'))
    print('       --------------|---------------|---------------')
    print('%28s  %6d | FP = %9d' % ('TRUE (neg.) | TN = ', TN, FP))
    print('label  --------------|---------------|---------------')
    print('%28s  %6d | TP = %9d' % ('FAKE (pos.) | FN = ', FN, TP))
    print('       --------------|---------------|---------------')

    print('Metrics:')
    print('ACC = %5.3f  R = %5.3f  P = %5.3f  F1 score = %5.3f  FPR = %5.4f' % (accuracy, recall, precision, f1, fpr))


if __name__ == '__main__':
    labels = np.array(['true', 'fake', 'fake', 'true', 'fake'], dtype=np.str_)
    predictions = np.array([True, True, False, True, False], dtype=np.bool_)
    evaluatePerformance(predictions, labels)
