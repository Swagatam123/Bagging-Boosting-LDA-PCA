import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
class PCA:
    dataset=[]
    def __init__(self,dataset,energy):
        self.dataset=dataset
        self.energy=energy

    def project_eigenfaces(self,eigen_vec):
        #eigen_vec=np.round(eigen_vec,2)
        eigen_vec=np.asarray(eigen_vec).transpose()
        print(len(eigen_vec[0]))
        #fig=plt.figure(figsize=(12,20))
        for i in range(0,26):
            plt.subplot(7,5,i+1)
            vec=eigen_vec[:,i].reshape(32,32)
            plt.imshow(vec,interpolation="bilinear",cmap="gray")
        plt.show()

    def fit(self):
        dataarray=np.asarray(self.dataset)
        #print(len(dataarray))
        '''for i in range(0,len(dataarray)):
            print(i)
            print(len(dataarray[i]))
        #dataarray=[[1,2,3],[4,5,6],[7,8,9],[1,2,9]]'''
        transpose_dataarray=dataarray.transpose()
        #print(len(transpose_dataarray))
        '''for i in range(0,len(transpose_dataarray)):
            print(i)
            print(len(transpose_dataarray[i]))'''
        #print(len(transpose_dataarray))
        #print(len(transpose_dataarray[0]))
        cov_matrix=np.cov(transpose_dataarray)
        cov_matrix=np.round(cov_matrix,2)
        eigen_value,eigen_vector=LA.eig(cov_matrix)
        sort_ordr=eigen_value.argsort()[::-1]
        eigen_value=eigen_value[sort_ordr]
        eigen_vector=eigen_vector[:,sort_ordr]
        eigen_vector=eigen_vector.transpose()
        sum=0
        for val in eigen_value:
            sum+=val
        sum_1=0
        count=0
        for eig in eigen_value:
            sum_1+=eig
            count+=1
            if sum_1/sum>=self.energy:
                break
        #print(count)
        eig_vec=[]
        for i in range(0,len(eigen_vector)):
            if i<=count:
                eig_vec.append(eigen_vector[i])
            else:
                break
        #eig_vec=np.asarray(eig_vec).transpose()
        final_data_pts=np.matmul(eig_vec,transpose_dataarray)
        #print(len(final_data_pts[0]))
        #print(len(final_data_pts))
        return eig_vec,final_data_pts.transpose()
    
    def test_fit(self,projection_matrix,test_dataset):
        transpose_data=np.asarray(test_dataset).transpose()
        return np.matmul(projection_matrix,transpose_data).transpose()
