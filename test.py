import csv
import numpy as np

def opencsv(delta):
    d = delta
    output = []
    with open('testvalues.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            row = row[:-1]
            output.append(row)

    for i in range(len(output)):
        for j in range(len(output[0])):
            output[i][j] = eval(output[i][j])
    output = np.reshape(output, (256,4,2)).tolist()
    return output

def getvalues(start, end, steps):
    values = []
    deltas = [(start + (end-start) * i/steps) / 100 for i in range(steps)]
    for delta in deltas:
        values.append(opencsv(delta))

    return values

#print(getvalues(40,60,20))