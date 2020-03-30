import csv
import Bagging
import Classify
import Boosting
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import Model_Selection

filename="data/letter-recognition.csv"
n_param=0
def readdataset(filename):
    with open(filename, 'r') as csvfile:
     csvreader = csv.reader(csvfile)
     label=[]
     dataset=[]
     for row in csvreader:
         label.append(ord(row[0])-65)
         row=[int(i) for i in row[1:]]
         #print(row)
         dataset.append(row)
    return dataset,label

def tuning_parameter():
    n_parameter=[20,50,80,100,150,200,250]
    param=0
    max_min=99
    for i in n_parameter:
        model=Model_Selection.Model_Selection(training_dataset,label_train)
        mean,std=model.k_folding_adaboost_parameter_tuning(5,i)
        #mean,std=model.k_folding_bagging_parameter_tuning(5,i)
        print("mean accuracy :",mean)
        print("standard deviation :",std)
        if max_min>mean:
            max_min=mean
            param=i
    print("best parameter selection at value :",param)
    return param


dataset,label=readdataset(filename)

training_dataset=dataset[:int(0.7*len(dataset))]
label_train=label[:int(0.7*len(dataset))]
test_dataset=dataset[int(0.7*len(dataset)):]
label_test=label[int(0.7*len(dataset)):]

n_param=tuning_parameter()

'''bagging=Bagging.Bagging(20)
bagging.BaggingClassifier(training_dataset,label_train,test_dataset,label_test,14000)
#print("final accuracy :",accuracy)

bagging=Bagging.Bagging(50)
bagging.BaggingClassifier(training_dataset,label_train,test_dataset,label_test,14000)

bagging=Bagging.Bagging(100)
bagging.BaggingClassifier(training_dataset,label_train,test_dataset,label_test,14000)

bagging=Bagging.Bagging(200)
bagging.BaggingClassifier(training_dataset,label_train,test_dataset,label_test,14000)'''
#n_param=100

#bagging=Bagging.Bagging(n_param)
#bagging.BaggingClassifier(training_dataset,label_train,training_dataset,label_train,14000)

boosting=Boosting.Boosting(n_param,training_dataset,label_train,test_dataset,label_test,14000)
boosting.fit()
accuracy = boosting.predict()
print("=====================================")
print("final accuracy :",accuracy)
'''accuracy=Classify.Gaussian_classify_NB(training_dataset,label_train,test_dataset,label_test)
print("======================================")
print("actual naive bayes accuracy :",accuracy)'''

'''clf=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_leaf_nodes=5,max_depth=2), n_estimators=150)
clf.fit(training_dataset,label_train)
print("original adaboost",clf.score(test_dataset,label_test))'''

