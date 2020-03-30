import Classify
import Boosting
import Bagging
import numpy as np

class Model_Selection:
    total_fold=[]
    label=[]
    
    def __init__(self,train_dataset,label_dataset):
        self.total_fold=train_dataset
        self.label=label_dataset

    def calculate5folding(self,i,total_fold,total_fold_label,part):
        if i==0:
            test_fold_dataset=total_fold[:part]
            test_fold_label=total_fold_label[:part]
            train_fold_dataset=total_fold[part:]
            train_fold_label=total_fold_label[part:]
        elif i==1:
            test_fold_dataset=total_fold[part:2*part]
            test_fold_label=total_fold_label[part:2*part]
            train_fold_dataset=total_fold[:part]+total_fold[2*part:]
            train_fold_label=total_fold_label[:part]+total_fold_label[2*part:]
        elif i==2:
            test_fold_dataset=total_fold[2*part:3*part]
            test_fold_label=total_fold_label[2*part:3*part]
            train_fold_dataset=total_fold[:2*part]+total_fold[3*part:]
            train_fold_label=total_fold_label[:2*part]+total_fold_label[3*part:]
        elif i==3:
            test_fold_dataset=total_fold[3*part:4*part]
            test_fold_label=total_fold_label[3*part:4*part]
            train_fold_dataset=total_fold[:3*part]+total_fold[4*part:]
            train_fold_label=total_fold_label[:3*part]+total_fold_label[4*part:]
        elif i==4:
            test_fold_dataset=total_fold[4*part:5*part]
            test_fold_label=total_fold_label[4*part:5*part]
            train_fold_dataset=total_fold[:4*part]
            train_fold_label=total_fold_label[:4*part]
        return test_fold_dataset,test_fold_label,train_fold_dataset,train_fold_label

    def k_folding_best_model(self,k):
        total_fold_size=len(self.total_fold)
        div=div=total_fold_size%k
        part=int(total_fold_size/k)
        max=-99
        accuracy_list=[]
        for i in range(0,k):
            #if i!=4:
             #   continue
            validation_fold_dataset,validation_fold_label,train_fold_dataset,train_fold_label=self.calculate5folding(i,self.total_fold,self.label,part)
            accuracy=Classify.Gaussian_classify_NB(train_fold_dataset,train_fold_label,validation_fold_dataset,validation_fold_label)
            print("Model accuracy",accuracy)
            accuracy_list.append(accuracy)
            if accuracy>max:
                best_model_dataset=train_fold_dataset+validation_fold_dataset
                best_model_lable=train_fold_label+validation_fold_label
        return best_model_dataset,best_model_lable,accuracy_list
    
    def k_folding_adaboost_parameter_tuning(self,k,tuning_param):
        total_fold_size=len(self.total_fold)
        div=div=total_fold_size%k
        part=int(total_fold_size/k)
        max=-99
        accuracy_list=[]
        for i in range(0,k):
            #if i!=4:
             #   continue
            validation_fold_dataset,validation_fold_label,train_fold_dataset,train_fold_label=self.calculate5folding(i,self.total_fold,self.label,part)
            boosting=Boosting.Boosting(tuning_param,train_fold_dataset,train_fold_label,validation_fold_dataset,validation_fold_label,len(train_fold_dataset))
            boosting.fit()
            accuracy=boosting.predict()
            print("Model accuracy",accuracy)
            accuracy_list.append(accuracy)
        return np.mean(np.asarray(accuracy_list)),np.std(np.asarray(accuracy_list))
    
    def k_folding_bagging_parameter_tuning(self,k,tuning_param):
        total_fold_size=len(self.total_fold)
        div=div=total_fold_size%k
        part=int(total_fold_size/k)
        max=-99
        accuracy_list=[]
        for i in range(0,k):
            #if i!=4:
             #   continue
            validation_fold_dataset,validation_fold_label,train_fold_dataset,train_fold_label=self.calculate5folding(i,self.total_fold,self.label,part)
            bagging=Bagging.Bagging(tuning_param)
            accuracy=bagging.BaggingClassifier_voting(train_fold_dataset,train_fold_label,validation_fold_dataset,validation_fold_label,14000)
            #boosting=Boosting.Boosting(tuning_param,train_fold_dataset,train_fold_label,validation_fold_dataset,validation_fold_label,len(train_fold_dataset))
            #boosting.fit()
            #accuracy=boosting.predict()
            print("Model accuracy",accuracy)
            accuracy_list.append(accuracy)
        return np.mean(np.asarray(accuracy_list)),np.std(np.asarray(accuracy_list))

