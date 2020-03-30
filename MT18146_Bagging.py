import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random

class Bagging:
    n_estimator=0

    def __init__(self,n_estimator):
        self.n_estimator=n_estimator

    def subsample(self,dataset,labelset,sample_size):

            #dataset=np.array(dataset)
            #labelset=np.array(labelset)
            #random_order=[]
            #for i in range(0,len(dataset)):
            #    random_order.append(i)
            index=np.random.randint(len(dataset),size=sample_size)
            data_sample=[]
            label_sample=[]
            for ind in index:
                data_sample.append(dataset[ind])
                label_sample.append(labelset[ind])
            #np.random.shuffle(random_order)
            #data_sample=np.array(dataset[random_order[:sample_size]])
            #label_sample=np.array(labelset[random_order[:sample_size]])
            return data_sample,label_sample
    
    
    def classifier(self,trainsample,labelsample,testset):
        classify=DecisionTreeClassifier(max_leaf_nodes=5,max_depth=2)
        classify.fit(trainsample,labelsample)
        #probab=classify.predict(testset)
        probab=classify.predict_proba(testset)
        return probab
    
    def classifier_predict(self,trainsample,labelsample,testset):
        classify=DecisionTreeClassifier(max_leaf_nodes=5,max_depth=2)
        classify.fit(trainsample,labelsample)
        #probab=classify.predict(testset)
        probab=classify.predict(testset)
        return probab

    def voting(self,predictor):
        final_predict=[]
        predictor=np.asarray(predictor)
        predictor=predictor.transpose()
        for i in range(0,len(predictor)):
            class_predict=list(predictor[i])
            #print("=========================")
            #print(class_predict)
            final_predict.append(max(class_predict,key=class_predict.count))
        return final_predict

    def min_max_scale(self,probaility_value):
        normalize_val=[]
        final_val=[]
        for elem in probaility_value:
            min=np.min(elem)
            max=np.max(elem)
            val=[]
            for x in elem:
                val.append((x-min)/max-min)
            final_val.append(val)
        return final_val

    def z_score_scale(self,probaility_value):
        normalize_val=[]
        final_val=[]
        for elem in probaility_value:
            mean=np.mean(elem)
            std=np.std(elem)
            val=[]
            for x in elem:
                val.append((x-mean)/std)
            final_val.append(val)
        return final_val

    def tanh_scale(self,probaility_value):
            normalize_val=[]
            final_val=[]
            for elem in probaility_value:
                mean=np.mean(elem)
                std=np.std(elem)
                val=[]
                for x in elem:
                    val.append(0.5*np.tanh(0.01*(x-mean)/std)+1)
                final_val.append(val)
            return final_val

    def predict(self,test_sample,train_model,operation,test_label):
        predict_sample=[]
        for x in test_sample:
            prob_array=[]
            for model in train_model:
                prob=model.predict_proba([x])
                temp=[]
                for p in prob:
                    temp.append(p)
                prob_array.append(temp)
            if operation==1:
                normalized_val=self.min_max_scale(prob_array)
            elif operation==2:
                normalized_val=self.z_score_scale(prob_array)
            elif operation==3:
                normalized_val=self.tanh_scale(prob_array)
            #print("norm :",normalized_val)
            sum=np.sum(normalized_val,axis=0)
            max=-999
            ind=0
            index=0
            for val in sum[0]:
                if max<val:
                    max=val
                    index=ind
                ind+=1
            predict_sample.append(index)
        return predict_sample


    def accuracy_predictor(self,predict_label,test_labelset):
        match=0
        #print(predict_label)
        #print(test_labelset)
        for i in range(0,len(test_labelset)):
            if predict_label[i]==test_labelset[i]:
                match+=1
        return (match/len(test_labelset))*100
    
    def BaggingClassifier(self,train_dataset,train_labelset,test_dataset,test_labelset,sample_size):
        model=[]
        for i in range(0,self.n_estimator):
            data_sample,label_sample=self.subsample(train_dataset,train_labelset,sample_size)
            #print("set length",len(set(label_sample)))
            classify=DecisionTreeClassifier(max_leaf_nodes=5,max_depth=2)
            clf=classify.fit(data_sample,label_sample)
            model.append(clf)

            '''match=0
            for i in range(0,len(test_labelset)):
                if p[i]==test_labelset[i]:
                    match+=1
            print("model accuracy :",(match/len(test_labelset))*100)'''
        predict_label=self.predict(test_dataset,model,1,test_labelset)
        accuracy=self.accuracy_predictor(predict_label,test_labelset)
        print("min_max accuracy: ",accuracy)
        predict_label=self.predict(test_dataset,model,2,test_labelset)
        accuracy=self.accuracy_predictor(predict_label,test_labelset)
        print("z score accuracy: ",accuracy)
        predict_label=self.predict(test_dataset,model,3,test_labelset)
        accuracy=self.accuracy_predictor(predict_label,test_labelset)
        print("tanh accuracy: ",accuracy)
        #model_val=self.min_max_scale(predictor_probab)
        #model_val=self.z_score_scale(predictor_probab)
        #print(model_val)
        #predict_label=self.voting(predictor_probab)
        #match_list=[i for i, j in zip(predict_label, test_labelset) if i == j]
        #predict_label=self.predict(model_val,test_labelset)
        
    def BaggingClassifier_voting(self,train_dataset,train_labelset,test_dataset,test_labelset,sample_size):
        predictor=[]
        for i in range(0,self.n_estimator):
            data_sample,label_sample=self.subsample(train_dataset,train_labelset,sample_size)
            pred=self.classifier_predict(data_sample,label_sample,test_dataset)
            predictor.append(pred)
        
        final_predict=self.voting(predictor)
        accuracy=self.accuracy_predictor(final_predict,test_labelset)
        return accuracy

