import pandas as pd

def Evaluate_Model(predictedresult,actualresult):
    predicted = predictedresult
    actual = actualresult
    #change pandas Dataframe to numpy array tien cho de so sanh
    predicted = predicted.values
    actual = actual.values
    a = 0

    for i in range (len(predicted)):
        if (predicted[i] == actual[i]):
            a = a + 1;

    truepositive = 0
    truenagative = 0
    falsepositive = 0
    falsenagative = 0

    for i in range (len(predicted)):
        #tinh true positive = 0
        if (predicted[i] == actual[i] and actual[i] == 0):
            truepositive = truepositive + 1
        #tinh true negative = 1
        elif (predicted[i] == actual[i] and actual[i] == 1):
            truenagative = truenagative + 1
        #tinh false positive
        elif (predicted[i] != actual[i] and actual[i] == 1):
            falsepositive = falsepositive + 1
        else:
            falsenagative = falsepositive + 1

    precision = truepositive * 100 / (truepositive + falsepositive)
    recall = truepositive * 100 / (truepositive + falsenagative)

    print('Accuracy: ' + str(a * 100 / len(predicted)))
    print('Precision : ' + str(precision))
    print('Recall: ' + str(recall))


re = pd.read_csv('ret.csv')
a = re['actual']
b = re['predicted']

Evaluate_Model(b,a)

