import numpy as np
import string
import csv
from sklearn.svm import LinearSVC
from sklearn import svm

#reads csv file and outputs it as a list of dicts
def readCsv(f):
    l = []
    with open(f) as csvfile:
        reader = csv.DictReader(csvfile)
        fn = reader.fieldnames
        # ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        for row in reader:
            l.append({fn[i]: row[fn[i]] for i in range(len(fn))})
        return l

#generates a subset from a set based on a fieldname and value
def genSet(s, field, val):
    l = [x for x in s if x[field] == val]
    return l

#generates an X set for the SVM
def genX(s):
    X = []
    for i in s:
        l = []
        l.append(float(i['Pclass']))
        if i['Sex'] == 'female':
            l.append(1)
        else:
            l.append(0)
        if i['Age']:
            l.append(1)
        else:
            l.append(0)
        l.append(float(i['SibSp']))
        l.append(float(i['Parch']))
        X.append(l)
    return X

trainset = readCsv("train.csv")
testset = readCsv("test.csv")

### Naive Bayes on male survival, female survival, and Age/no Age reported ###
# male = genSet(trainset, 'Sex', 'male')
# surMale = genSet(male, 'Survived', '1')
# surMaleRate = (len(surMale)*1.0) /len(male)
#
# fem = genSet(trainset, 'Sex', 'female')
# surFem = genSet(fem, 'Survived', '1')
# surFemRate = (len(surFem)*1.0) /len(fem)
#
# totalSurRate = ((len(surFem) + len(surMale)) * 1.0) / len(trainset)
#
# print "Male Survivor Rate: " + str(surMaleRate)
# print "Female Survivor Rate: " + str(surFemRate)
# print "Total Survivor Rate: " + str(totalSurRate)
#
# noAge = genSet(trainset, 'Age', '')
# noAgeSurvived = genSet(noAge, 'Survived','1')
# age = [x for x in trainset if x['Age']]
# ageSurvived = genSet(age, 'Survived', '1')
# print len(noAge)
# print (len(noAgeSurvived)* 1.0)/len(noAge)
# print len(age)
# print (len(ageSurvived)* 1.0)/len(age)
#
# children = [x for x in trainset if x['Age'] and float(x['Age']) < 6 ]
# childSurv = genSet(children, 'Survived', '1')
# print (len(childSurv)* 1.0)/len(children)


X_train = genX(trainset)
y_train = [x['Survived'] for x in trainset]

X_test = genX(testset)


clf = svm.SVC(C = 1)
clf.fit(X_train, y_train)
train_predictions = clf.predict(X_train)

#checked accuracy on the training set with different C values to prevent overfitting
accuracy = [int(a==b) for (a,b) in zip(train_predictions, y_train)]
accuracy = (sum(accuracy) * 1.0) / len(accuracy)
print accuracy

test_predictions = clf.predict(X_test)

#write predictions to file
with open("predictions_Titanic.txt", 'w') as pred_file:
    pred_file.write("PassengerId,Survived\n")
    for i in range(len(testset)):
        passID = testset[i]['PassengerId']
        prediction = test_predictions[i]
        pred_file.write("{0},{1}\n".format(passID, prediction))
