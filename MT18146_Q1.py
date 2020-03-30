import cv2
import os
import random
import PCA
from sklearn.naive_bayes import GaussianNB
import Model_Selection
import Classify
import numpy as np
import LDA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

datapath="G:/SML/ASSIGNMENT/Assignment_3/Face_data"
data_path_cifar="G:/SML/ASSIGNMENT/Assignment_3/cifar-10-batches-py"

def readdata(datapath):
    directory = os.listdir(datapath)
    dataset=[]
    label_dataset=[]
    for folder in directory:
        files_list = os.listdir(datapath+'/'+folder)
        for file in files_list:
            img = cv2.imread(datapath+'/'+folder+'/'+file,0)
            dim=(32,32)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            dataset.append(resized)
            label_dataset.append(int(os.path.basename(folder)))
    return dataset,label_dataset

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def readdata_cifar_10(datapath):
    train_dic_1 = unpickle(data_path_cifar+'/'+'data_batch_1')
    train_dic_2 = unpickle(data_path_cifar+'/'+'data_batch_2')
    train_dic_3 = unpickle(data_path_cifar+'/'+'data_batch_3')
    train_dic_4 = unpickle(data_path_cifar+'/'+'data_batch_4')
    train_dic_5 = unpickle(data_path_cifar+'/'+'data_batch_5')
    test_dic = unpickle(data_path_cifar+'/'+'test_batch')
    train_dataset=list(train_dic_1[b'data']) + list(train_dic_2[b'data']) + list(train_dic_3[b'data']) + list(train_dic_4[b'data']) + list(train_dic_5[b'data'])
    train_labelset= train_dic_1[b'labels'] + train_dic_2[b'labels'] + train_dic_3[b'labels'] + train_dic_4[b'labels'] + train_dic_5[b'labels']
    test_dataset=test_dic[b'data']
    test_labelset=test_dic[b'labels']
    train_dataset=[0.3*x[:1024]+0.59*x[1024:2048]+0.11*x[2048:] for x in train_dataset]
    test_dataset=[0.3*x[:1024]+0.59*x[1024:2048]+0.11*x[2048:] for x in test_dataset]
    return train_dataset,train_labelset,test_dataset,test_labelset

def extract_feature(dataset):
    data_set=[]
    for img in dataset:
        val=[]
        for i in range(0,32):
            for j in range(0,32):
                val.append(img[i][j])
        data_set.append(val)
    return data_set

'''def seggregate_data(dataset,label_dataset,train_percent):
    length_train=train_percent*len(dataset)
    random_set=[]
    random_set=random.sample(range(len(dataset)),int(length_train))
    train_dataset=[]
    train_labelset=[]
    test_dataset=[]
    test_labelset=[]
    for i in range(0,len(dataset)):
        if i in random_set:
            train_dataset.append(dataset[i])
            train_labelset.append(label_dataset[i])
        else:
            test_dataset.append(dataset[i])
            test_labelset.append(label_dataset[i])
    return train_dataset,train_labelset,test_dataset,test_labelset'''

def tsne_plot(query_vector,label):
    labels=['modified query','relevance vector','non relevance vector']
    tokens=[]
    i=0
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(query_vector)
    x=dict()
    y=dict()
    c=0
    for value in new_values:
        if label[c] not in x.keys():
            temp_x=[]
            temp_y=[]
            temp_x.append(value[0])
            temp_y.append(value[1])
            x[label[c]]=temp_x
            y[label[c]]=temp_y
        else:
            x[label[c]].append(value[0])
            y[label[c]].append(value[1])
        c+=1
    for i in x.keys():
        plt.scatter(x[i],y[i],label="class "+str(i))
    plt.xlabel("feature x1")
    plt.ylabel("feature x2")
    plt.legend()
    plt.axis('equal')
    plt.show()

def seggregate_data(dataset,label_dataset,train_percent):
    length_train=train_percent*len(dataset)
    data=np.array(dataset)
    label=np.array(label_dataset)
    random_order=np.arange(len(dataset))
    np.random.shuffle(random_order)
    train_dataset=np.array(data[random_order[:int(length_train)]])
    train_labelset=np.array(label[random_order[:int(length_train)]])
    test_dataset=np.array(data[random_order[int(length_train):]])
    test_labelset=np.array(label[random_order[int(length_train):]])
    return list(train_dataset),list(train_labelset),list(test_dataset),list(test_labelset)

def perform_pca(train_dataset,test_dataset,energy):
    pca = PCA.PCA(train_dataset,energy)
    projection_matrix,projected_train_data=pca.fit()
    projected_test_data=pca.test_fit(projection_matrix,test_dataset)
    #pca.project_eigenfaces(projection_matrix)
    return projected_train_data,projected_test_data

def perform_lda(train_dataset,train_labelset,test_dataset):
    lda = LDA.LDA(train_dataset,train_labelset)
    projection_matrix,projected_train_data=lda.fit()
    print(np.shape(projection_matrix),np.shape(np.shape(test_dataset)))
    projected_test_data=lda.test_fit(projection_matrix,test_dataset)
    return projected_train_data,projected_test_data

#dataset,label_dataset=readdata(datapath)
#data_set=extract_feature(dataset)
#train_dataset,train_labelset,test_dataset,test_labelset=seggregate_data(data_set,label_dataset,0.7)


train_dataset,train_labelset,test_dataset,test_labelset=readdata_cifar_10(data_path_cifar)
print(set(test_labelset))
accuracy=Classify.Gaussian_classify_NB(train_dataset,train_labelset,test_dataset,test_labelset)
print("Initial accuracy: ",accuracy)
print(train_dataset[0])
print(train_dataset[len(train_dataset)-1])



############################PCA OVER LDA################
#######################CHANGE C-1 FOR LDA###############
print("====================================")
print("pca over lda")
projected_train_data,projected_test_data=perform_lda(train_dataset,train_labelset,test_dataset)
projected_train_data,projected_test_data=perform_pca(projected_train_data,projected_test_data,0.99)

############################LDA OVER PCA################
#######################CHANGE C-1 FOR LDA###############
'''print("====================================")
print("lda over pca")
projected_train_data,projected_test_data=perform_pca(train_dataset,test_dataset,0.95)
projected_train_data,projected_test_data=perform_lda(projected_train_data,train_labelset,projected_test_data)'''

###########################PCA COMPARISON##########################
#projected_train_data,projected_test_data=perform_pca(train_dataset,test_dataset,0.95)
#accuracy=Classify.Gaussian_classify_NB(projected_train_data,train_labelset,projected_test_data,test_labelset)
#print("final accuracy: ",accuracy)
#tsne_plot(projected_test_data,test_labelset)
'''projected_train_data,projected_test_data=perform_pca(train_dataset,test_dataset,0.90)
accuracy=Classify.Gaussian_classify_NB(projected_train_data,train_labelset,projected_test_data,test_labelset)
print("final accuracy: ",accuracy)
projected_train_data,projected_test_data=perform_pca(train_dataset,test_dataset,0.99)
accuracy=Classify.Gaussian_classify_NB(projected_train_data,train_labelset,projected_test_data,test_labelset)
print("final accuracy: ",accuracy)'''

############################ONLY PCA################################
#projected_train_data,projected_test_data=perform_pca(train_dataset,test_dataset,0.95)

###########################ONLY LDA################################
##########################CHANGE C-1 FOR LDA######################
#projected_train_data,projected_test_data=perform_lda(train_dataset,train_labelset,test_dataset)
#tsne_plot(projected_test_data,test_labelset)
#model=Model_Selection.Model_Selection(list(projected_train_data),train_labelset)
#train_dataset_1,train_labelset,accuracy_list=model.k_folding_best_model(5)
#print("mean across 5 folds: ",np.mean(accuracy_list))
#print("====================================")
#print("standar deviation of across 5 folds: ",np.std(accuracy_list))
#print("====================================")

accuracy=Classify.Gaussian_classify_NB(projected_train_data,train_labelset,projected_test_data,test_labelset)
#Classify.calculta_confusion_matrix()

#Classify.draw_ROC()
print("final accuracy: ",accuracy)









############################ONLY PCA################################
#projected_train_data,projected_test_data=perform_pca(train_dataset,test_dataset,0.95)

###########################ONLY LDA################################
'''projected_train_data,projected_test_data=perform_lda(train_dataset,train_labelset,test_dataset)'''

############################LDA OVER PCA################
print("====================================")
print("lda over pca")
projected_train_data,projected_test_data=perform_pca(train_dataset,test_dataset,0.99)
projected_train_data,projected_test_data=perform_lda(projected_train_data,train_labelset,projected_test_data)

#model=Model_Selection.Model_Selection(list(projected_train_data),train_labelset)
#train_dataset,train_labelset,accuracy_list=model.k_folding_best_model(5)
#print("mean across 5 folds: ",np.mean(accuracy_list))
#print("====================================")
#print("standar deviation of across 5 folds: ",np.std(accuracy_list))
#print("====================================")

accuracy=Classify.Gaussian_classify_NB(projected_train_data,train_labelset,projected_test_data,test_labelset)
#Classify.calculta_confusion_matrix()

#Classify.draw_ROC()
print("final accuracy: ",accuracy)
