import numpy as np
from numpy import linalg as LA
import copy

class LDA:
    dataset=[]
    label=[]

    def __init__(self,dataset,label):
        self.dataset=dataset
        self.label=label

    def preprocess_dataset(self):
        data_dict=dict()
        labels_set=set(self.label)
        for l in labels_set:
            val=[]
            for i in range(0,len(self.dataset)):
                if self.label[i]==l:
                    val.append(self.dataset[i])
            data_dict[l]=val
        return data_dict

    def calculate_SW(self,data_dict):
        Sw=0
        for key in  data_dict.keys():
            data=data_dict[key]
            transpose_data=np.asarray(data).transpose()
            Sw+=np.round(np.cov(transpose_data),2)
        return Sw

    def calculate_total_mean_vector(self,data_dict):
        total_mean=[]
        class_mean=dict()
        for key in data_dict.keys():
            data=data_dict[key]
            mean_vec=np.mean(data,axis=0)
            '''for v in mean_vec:
                class_mean_val
            +=v'''
            class_mean[key]=mean_vec/len(data)
            if len(total_mean)==0:
                total_mean=mean_vec
            else:
                total_mean+=mean_vec
        total_mean=total_mean/(len(self.dataset))
        return total_mean,class_mean

    def calculate_SB(self,data_dict,total_mean,class_mean):
        Sb=[]
        for key in class_mean.keys():
            diff_val=np.asarray(class_mean[key])-np.asarray(total_mean)
            copy_diff_val=copy.deepcopy(diff_val)
            transpose_diff_val=np.matrix(copy_diff_val).transpose()
            #print(len(diff_val),len(diff_val[0]))
            #print(len(transpose_diff_val),len(transpose_diff_val[0]))
            temp=np.matmul(transpose_diff_val,np.matrix(diff_val))
            #print(temp)
            if len(Sb)==0:
                Sb=temp
            else:
                Sb+=temp
        #print(len(Sb))
        return Sb

    def fit(self):
        data_dict=self.preprocess_dataset()
        Sw=self.calculate_SW(data_dict)
        total_mean,class_mean=self.calculate_total_mean_vector(data_dict)
        Sb=self.calculate_SB(data_dict,total_mean,class_mean)
        #print(len(Sw),len(Sw[0]))
        #print(len(Sb),len(Sb[0]))
        final_matrix=np.matmul(np.linalg.inv(np.asarray(Sw)),Sb)
        final_matrix=np.round(final_matrix,2)
        eigen_value,eigen_vector=np.linalg.eigh(final_matrix)
        eigen_vector=np.round(eigen_vector,2)
        #print(eigen_vector)
        p=np.asarray(self.dataset).transpose()
        #print(len(p))
        #eigen_vector.sort()
        eigen_vector=eigen_vector[:,:9].transpose()
        print(np.shape(eigen_vector),np.shape(p))
        final_data=np.matmul(eigen_vector,p)
        #print(len(final_data))
        return eigen_vector,final_data.transpose()

    def test_fit(self,projection_matrix,test_dataset):
        transpose_data=np.asarray(test_dataset).transpose()
        return np.matmul(projection_matrix,transpose_data).transpose()
