import numpy as np
from sklearn.tree import DecisionTreeClassifier
import copy

class Boosting:
    N=0
    train_dataset=[]
    train_label=[]
    test_dataset=[]
    test_label=[]
    aplhas_list=[]
    model_list=[]
    sample_size=[]
    def __init__(self,N,train_dataset,train_label,test_dataset,test_label,sample_size):
        self.N=N
        self.train_dataset=train_dataset
        self.train_label=train_label
        self.test_dataset=test_dataset
        self.test_label=test_label
        self.sample_size=sample_size

    def train_sample(self,weights,train_dataset,label_dataset,sample_size):
        #print(weights)
        index=np.arange(len(train_dataset))
        random_id=np.random.choice(index,sample_size,p=weights)
        train_sample=[]
        label_sample=[]
        for i in random_id:
            train_sample.append(train_dataset[i])
            label_sample.append(label_dataset[i])

        '''cp_train_dataset=copy.deepcopy(train_dataset)
        for i in range(0,len(cp_train_dataset)):
            cp_train_dataset[i].append(label_dataset[i])
        label_sample=[]
        train_sample=[x for _,x in sorted(zip(weights,cp_train_dataset),reverse=True)]
        tr_1=[]
        for tr in train_sample:
            label_sample.append(tr[-1])
            tr_1.append(tr[:len(tr)-1])
        train_sample=copy.deepcopy(tr_1)'''
        if len(set(weights))==1:
            dataset=np.array(train_dataset)
            labelset=np.array(label_dataset)
            random_order=np.arange(len(dataset))
            np.random.shuffle(random_order)
            data_sample=np.array(dataset[random_order[:sample_size]])
            label_sample=np.array(labelset[random_order[:sample_size]])
            return data_sample,label_sample
            #return train_dataset[:sample_size],label_dataset[:sample_size]
        else:
            #return train_sample,label_sample
            return train_dataset,label_dataset
        

    def classifier(self,trainsample,labelsample,testset):
        classify=DecisionTreeClassifier()
        classify.fit(trainsample,labelsample)
        label_predict=classify.predict(testset)
        return label_predict

    def calculate_alpha(self,error_rate):
        return 1/2*(np.log((1-error_rate)/error_rate)) + np.log(15)
    
    def calculate_error_rate(self,predict_label,train_dataset,train_label,weights):
        error=0
        for i in range(0,len(train_dataset)):
            #print(i)
            error+=weights[i]*np.where(predict_label[i]!=train_label[i],1,0)
        return error
        
    def fit(self):
        alpha=[]
        model=[]
        weight=[]
        wt=[]
        for i in range(0,len(self.train_dataset)):
            wt.append(1/(len(self.train_dataset)))
        weight.append(wt)
        for i in range(0,self.N):
            #print(weight[i])
            train_sample,train_label=self.train_sample(weight[i],self.train_dataset,self.train_label,self.sample_size)
            classify=DecisionTreeClassifier(max_leaf_nodes=5,max_depth=2)
            clf=classify.fit(train_sample,train_label,sample_weight=weight[i])
            model.append(clf)
            predict_label=clf.predict(self.train_dataset)
            error=(self.calculate_error_rate(predict_label,self.train_dataset,self.train_label,weight[i]))/np.sum(np.asarray(weight[i]))
            #print("error rate :",error)
            alpha_val=self.calculate_alpha(error)
            alpha.append(alpha_val)
            wt=[]
            sum=0
            for j in range(0,len(self.train_dataset)):
                val=weight[i][j]*np.exp(-alpha_val*np.where(predict_label[j]==self.train_label[j],1,-1))
                sum+=val
                wt.append(val)
            wt_1=[w/sum for w in wt]
            #print("sum weight :",np.sum(np.asarray(wt_1)))
            weight.append(wt_1)
            match=0
            for i in range(0,len(list(predict_label))):
                if predict_label[i]==self.train_label[i]:
                    match+=1
            #print("model accuracy",np.sum(np.where(predict_label[k]==self.train_label[k],1,0) for k in range(0,len(self.train_dataset)))/len(self.train_dataset)*100)
            #print("predicted accuracy",(1-error)*100)
        self.aplhas_list=alpha
        self.model_list=model

    '''def predict(self):
        prediction=[]

        for i in range(0,self.N):
            predict_label=self.model_list[i].predict(self.test_dataset)
            predict_label=[np.where(predict_label[j]==self.test_label[j],1,-1) for j in range(0,len(self.test_dataset))]
            predict_label=[label*self.aplhas_list[i] for label in predict_label]
            prediction.append(predict_label)
        predict_label=list(np.sum(np.array(prediction),axis=0))
        match=0
        for pr in predict_label:
            if pr>0:
                match+=1
        print("============================================")
        print("final accuracy :",(match/len(predict_label)*100))'''

    def predict(self):
            prediction=[]
            prediction_1=[]
            for i in range(0,len(self.test_dataset)):
                predict_dict=dict()
                for j in range(0,26):
                    predict_dict[j]=0
                prediction_1.append(predict_dict)
            #print(len(prediction_1))
            for i in range(0,self.N):
                predict_label=self.model_list[i].predict(self.test_dataset)
                for j in range(0,len(predict_label)):
                    prediction_1[j][predict_label[j]]+=self.aplhas_list[i]
                    #print("predict label :",prediction_1[j])

            final_predict_label=[]
            for i in range(0,len(prediction_1)):
                max=-999999
                max_class=0
                #print(prediction_1[i])
                for key in prediction_1[i].keys():
                    if max<prediction_1[i][key]:
                        max=prediction_1[i][key]
                        max_class=key
                #print(max_class)
                final_predict_label.append(max_class)
                #predict_label=[np.where(predict_label[j]==self.test_label[j],1,-1) for j in range(0,len(self.test_dataset))]
                #predict_label=[label*self.aplhas_list[i] for label in predict_label]
                #prediction.append(predict_label)
            #predict_label=list(np.sum(np.array(prediction),axis=0))
            #print(final_predict_label)
            match=0
            for i in range(0,len(self.test_label)):
                if final_predict_label[i]==self.test_label[i]:
                    match+=1
            #print("============================================")
            #print("final accuracy :",(match/len(self.test_label)*100))
            #print(self.aplhas_list)
            return 100-(match/len(self.test_label)*100)
