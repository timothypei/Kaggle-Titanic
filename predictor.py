import numpy as np
import string
import csv

#reads csv file and outputs it as a list of dicts
def readCsv(f):
    l = []
    with open(f) as csvfile:
        reader = csv.DictReader(csvfile)
        fn = reader.fieldnames
        # ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'T   icket', 'Fare', 'Cabin', 'Embarked']
        for row in reader:
            l.append({fn[i]: row[fn[i]] for i in range(len(fn))})
        return l

trainset = readCsv("train.csv")
survived = []
