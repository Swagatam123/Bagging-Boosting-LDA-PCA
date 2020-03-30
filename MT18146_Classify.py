from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import copy
import numpy
import seaborn as sns

dataset=[]
posteriors=[]
labelset=[]
predict_label=[]
def Gaussian_classify_NB(train_dataset,train_labelset,test_dataset,test_labelset):
    global labelset
    labelset=test_labelset
    global dataset
    dataset=test_dataset
    clf = GaussianNB()
    clf.fit(train_dataset, train_labelset)
    match=0
    global predict_label
    label=clf.predict(test_dataset)
    #print("helloo")
    predict_label=label
    global posteriors
    val=clf.predict_proba(test_dataset)
    #print(val)
    #print(val.transpose())
    posteriors=val.transpose()
    for i in range(0,len(test_dataset)):
        if label[i]==test_labelset[i]:
            match+=1
    #print(match)
    #print("accuracy :",match/len(test_dataset)*100)
    return match/len(test_dataset)*100


def calculta_confusion_matrix():
    #confusion_matix = [[0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0]]
    confusion_matix = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
    #print(len(confusion_matix),len(confusion_matix[0]))
    posterior=posteriors.transpose()
    for i in range(0,len(posterior)):
        #print(labelset[i],predict_label[i])
        confusion_matix[labelset[i]-1][predict_label[i]-1]+=1
    ax=sns.heatmap(confusion_matix)
    plt.show()

def ROC_curve_points(l,discrimanting_fnc_list,labelset,count_one,count_two):
    posterior = copy.copy(discrimanting_fnc_list)
    #discrimanting_fnc_list.sort()
    x_points=[]
    y_points=[]
    threshold_val=numpy.linspace(-0.1,1.1,100,endpoint=False)
    #print(threshold_val)
    for threshold in threshold_val:
    #for threshold in range(0,len(discrimanting_fnc_list)):
        fp=0
        tp=0
        for i in range(0,len(dataset)):
            #feature = binarized_test_dataset[i]
            #likelihood_trouser = calculate_likelihood(mean_trouser,feature,variance_trouser)
            #posterior_probability_trouser = calculate_posterior_probability(likelihood_trouser,actual_test_trouser/(actual_test_pullover+actual_test_trouser))
            if posterior[i] > threshold :
                if labelset[i]==l:
                    tp+=1
                else:
                    fp+=1
        #print(tp,fp)
        y_points.append(tp/count_one)
        x_points.append(fp/count_two)
        #print(threshold,fp/count_two,tp/count_one)
    #print(x_points,y_points)
    return x_points,y_points

def count_class_dataset(class_val):
    #print(set(labelset))
    #print("count :",labelset.count(class_val))
    return labelset.count(class_val)

def draw_ROC():
    roc_matrix=[]
    print(numpy.shape(posteriors))
    for i in range(0,len(posteriors)):
        class_pos=posteriors[i]
        #x,y=ROC_curve_points(i+1,class_pos,labelset,count_class_dataset(i+1),len(labelset)-count_class_dataset(i+1))
        x,y=ROC_curve_points(i,class_pos,labelset,count_class_dataset(i),len(labelset)-count_class_dataset(i))
        val=[]
        val.append(x)
        val.append(y)
        roc_matrix.append(val)
        #break

    #x,y=ROC_curve_points(1,posteriors[:,[0]],labelset,count_class_dataset(1),len(labelset)-count_class_dataset(1))
    #print(x)
    #print(y)
    #plt.plot(x,y,label="1 vs all")
    #plt.plot(x_1,y_1, label=str(i+1)+"as positive class")
    for i in range(0,len(roc_matrix)):
        x=roc_matrix[i][0]
        y=roc_matrix[i][1]
        #print(roc_matrix[i][0],roc_matrix[i][1])
        plt.plot(x,y, label=str(i)+"as positive class")
        #plt.show()
        #plt.plot(roc_matrix[i][0],roc_matrix[i][1], label=str(i+1)+"as positive class")'''
    plt.legend(loc="lower right")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()
