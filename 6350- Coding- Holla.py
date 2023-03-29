#### 6350- FINAL PROJECT-
"""
Created on Sun Dec  4 20:13:41 2022

Submission by Sheethal Holla
"""

#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from scipy import stats as st
from sklearn.metrics import confusion_matrix
import random as random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



### Data preprocessing
#loading dataset
path='C:/Users/Vedhus/Desktop/MSDS/6350 Stat learning & data mining/Take home project/v1/BridgeData.csv'
datas = pd.read_csv(path)
datas = pd.DataFrame(datas)

#checking missing values
print(datas.isnull().sum())

#preliminary data description
datas.describe()

#box plots of features
plt.figure(figsize = [15,21]) 
for i in range(1,len(datas.columns)):
    plt.subplot(6,2,i) 
    sns.boxplot(datas.iloc[:,i])
    plt.xlabel(datas.columns[i])
    i=i+1
        
#density plot
datas.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False, fontsize=1)
plt.show()

#correlation 
plt.figure(figsize=(15,8))
correlation= '%.2f'%datas.corr()
sns.heatmap('%.2f'%datas.corr(), annot = True)
plt.show() 

### Class Rebalacning
#checking for preliminary classes
print(datas['Deck_Condition'].value_counts())

#converting 10 class to 4 class classification
mapping={0:0,1:0,2:0,3:0,4:0,5:0,6:1,7:2,8:3,9:3}
datas['trueclass']= datas['Deck_Condition'].map(lambda x: mapping[x])
datas['trueclass'].value_counts()
datas['Deck_Condition'].value_counts()

#creating true class
random.seed(1)
Y = datas['trueclass']
datas.drop(['trueclass','Deck_Condition'],axis = 1,inplace=True)


#standardising data
scaler = StandardScaler()
data1 = scaler.fit_transform(datas)
data = pd.DataFrame(data1, columns=datas.columns)
print(data)

#Creating Train & test class
X_train, X_test, y_train, y_test  = train_test_split(data, Y, test_size=0.15)

#defining rebalanced classes
old_class_num = y_train.value_counts()
test_class_num = y_test.value_counts()
new_class_dict = {0:4000, 3: 5000, 1:old_class_num[1], 2:old_class_num[2]}
test_class_dict = {0:1000, 3: 1250, 1:test_class_num[1], 2:test_class_num[2]}

#Applying smoting
sm = SMOTE(sampling_strategy = new_class_dict, random_state=42)
smtest = SMOTE(sampling_strategy = test_class_dict, random_state=42)

#smoting results
X_sm_train, y_sm_train = sm.fit_resample(X_train, y_train)
X_sm_test, y_sm_test = smtest.fit_resample(X_test, y_test)

#class before smoting
y_train.value_counts()
#new class after smoting
y_sm_train.value_counts()



### Random Forest
#Defining RF parameters
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=400, max_depth=None,
     min_samples_split=3, class_weight='balanced_subsample',random_state=0, oob_score = True)

clf.fit(X_sm_train, y_sm_train)

#without sub sampling
y_pred=clf.predict(X_sm_test) 
#print("Accuracy:",metrics.accuracy_score(y_sm_test, y_pred))
#(confusion_matrix(y_sm_test, y_pred,normalize = 'true'))

#creating sub samples
majority_class = 2
X_sm_train_ss = X_sm_train[y_sm_train!=majority_class]
y_sm_train_ss = y_sm_train[y_sm_train!=majority_class]

X_sm_train_class2 = X_sm_train[y_sm_train==majority_class]
y_sm_train_class2 = y_sm_train[y_sm_train==majority_class]

#Implementation of RF
random.seed(0)
num_of_bags = 10
sampleSize = 6000
y_pred = np.zeros([np.size(y_sm_test), num_of_bags])
y_trainpred = np.zeros([np.size(y_sm_train), num_of_bags])
oobs = np.zeros(num_of_bags)
TestAcc = np.zeros(num_of_bags)
TrainAcc=np.zeros(num_of_bags)
for i in range(num_of_bags):
    subsetX = X_sm_train_class2.sample(sampleSize)
    X_ss = pd.concat([X_sm_train_ss,subsetX ])
    y_ss = pd.concat([y_sm_train_ss, y_sm_train_class2[subsetX.index]])
    clf.fit(X_ss, y_ss)
    y_pred[:,i] = clf.predict(X_sm_test) 
    y_trainpred[:,i]=clf.predict(X_sm_train)
    oobs[i] = 1-clf.oob_score_
    TrainAcc[i]= metrics.accuracy_score(y_sm_train, y_trainpred[:,i])
    TestAcc[i] = metrics.accuracy_score(y_sm_test, y_pred[:,i])
    
#applying bagged RFs to obtain Test Acc 
y_bag_test = np.squeeze(st.mode(y_pred,axis = 1,keepdims = True)[0])
OOB_error= 1-clf.oob_score_
y_bag_train = np.squeeze(st.mode(y_trainpred,axis = 1,keepdims = True)[0])

#Output after bagging
print("TestAcc:",metrics.accuracy_score(y_sm_test, y_bag_test))
print("TrainAcc:",metrics.accuracy_score(y_sm_train, y_bag_train))
(confusion_matrix(y_sm_test, y_bag_test,normalize = 'true'))
(confusion_matrix(y_sm_train, y_bag_train,normalize = 'true'))

#Feature importance 
feature_imp = pd.Series(clf.feature_importances_,index=data.columns).sort_values(ascending=False)
feature_imp

#Creating a bar plot of feature importance
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score',fontsize=6)
plt.ylabel('Features',fontsize=6)
plt.title("Visualizing Important Features",fontsize=6)
plt.tight_layout()
plt.legend()
plt.tick_params(axis='both',labelsize=6)
plt.savefig('Featureplot.png',dpi=400)


#creating plots of trees & accuracy
Trees=[100,250,400,500]
TrainAcc=[89.2, 88.8, 88.6,	88.4]
TestAcc=[70.2,70.2,72.3,71.4]
plt.plot(Trees,TrainAcc,label='TrainAcc',linestyle="--")
plt.plot(Trees,TestAcc,label='TestAcc',linestyle="-.")
plt.title('Trees Vs. Accuracy')
plt.legend()
plt.xlabel('Trees')
plt.ylabel('Accuracy%')
plt.savefig('TreesVs accuracy.png',dpi=400)


#creating plots of min_samples_split & accuracy
split=[2,3,4]
TrainAcc=[88.6,	88.4,88.2]
TestAcc=[72.3,72.7,72]
plt.plot(split,TrainAcc,label='TrainAcc',linestyle=":",color='red')
plt.plot(split,TestAcc,label='TestAcc',linestyle="-.",color='green')
plt.legend()
plt.title('Splits Vs. Accuracy')
plt.xlabel('split')
plt.ylabel('Accuracy%')
plt.savefig('Split Vs accuracy.png',dpi=400)


#creating plots of min_samples_impurity & accuracy
impurity=[0.00,0.02,0.03]
TrainAcc=[88.6,27.5,32.1]
TestAcc=[72.3,33.2,26.4]
plt.plot(impurity,TrainAcc,label='TrainAcc',linestyle=":",color='red')
plt.plot(impurity,TestAcc,label='TestAcc',linestyle="-.",color='green')
plt.legend()
plt.title('Impurity Vs. Accuracy')
plt.xlabel('Impurity')
plt.ylabel('Accuracy%')
plt.savefig('Impurity Vs accuracy.png',dpi=400)



############        SVM  ##################################################################
#creating dataset for SVM
svm_data=pd.concat([datas[datas['trueclass']==1], datas[datas['trueclass']==2]],axis=0)
svm_pre=svm_data.drop(['Deck_Condition','trueclass'],axis=1)

#Standardizing data
scaler = StandardScaler()
svm_array = scaler.fit_transform(svm_pre)
svm=pd.DataFrame(svm_array,columns=svm_pre.columns)

#Creating Train & test class
random.seed(1)
Y = svm_data['trueclass']
Y.to_csv('trueclass.csv', index= False)
X_train, X_test, y_train, y_test  = train_test_split(svm, Y, test_size=0.15)

#defining rebalanced classes
old_class_num = y_train.value_counts()
print(old_class_num)
test_class_num = y_test.value_counts()
print(test_class_num)
new_class_dict = {1:15000, 2:old_class_num[2]}
test_class_dict = {1:2600, 2:test_class_num[2]}

#Applying smoting
sm = SMOTE(sampling_strategy = new_class_dict, random_state=123)
smtest = SMOTE(sampling_strategy = test_class_dict, random_state=133)

#smoting results
X_svm_train, y_svm_train = sm.fit_resample(X_train, y_train)
X_svm_test, y_svm_test = smtest.fit_resample(X_test, y_test)

#Defining RF parameters
Cost=[0.001,0.01, 0.1, 1,5,10,50,100,500,1000] #cost coefficient
GammaList=[(1/10/len(X_svm_train)),'scale',(1/len(X_svm_train)),(10/len(X_svm_train)),(100/len(X_svm_train))]
x=0
result = np.zeros(shape=(50,6))

#Implementation of SVM
from sklearn.svm import SVC
random.seed(123)
for i in Cost:
    for j in GammaList:
        RadialSVM = SVC(kernel='rbf',C=i,gamma=j)
        RadialSVM.fit(X_svm_train, y_svm_train)
        Y_train_pred=RadialSVM.predict(X_svm_train)
        Y_test_pred=RadialSVM.predict(X_svm_test)
        trainAcc=round(sum(Y_train_pred==y_svm_train )/len(y_svm_train)*100,2)
        testAcc=round(sum(Y_test_pred==y_svm_test )/len(y_svm_test)*100,2)
        Ratio=round((testAcc/trainAcc)*100,2)
        Support_vectors=round((len(RadialSVM.support_vectors_)/len(X_svm_train)*100),2)
        result[x,0]= i
        result[x,1]= j
        result[x,2]=trainAcc
        result[x,3]=testAcc
        result[x,4]=Ratio
        result[x,5]=Support_vectors
        x=x+1

result=pd.DataFrame(result)
result.columns=['cost','gamma','trainAcc','testAcc','Ratio','%Support Vector']
result    

#Grid validation
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
C_range =(1,10,100)
gamma_range = [100/len(X_svm_train),0.03613496,10000/len(X_svm_train)]
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_svm_train, y_svm_train)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

#confusion matrix for selected parameters #default gamma + cost=10
Radialbest = SVC(kernel='rbf',C=10)
Radialbest.fit(X_svm_train, y_svm_train)
Y_train_pred_best=Radialbest.predict(X_svm_train)
Y_test_pred_best=Radialbest.predict(X_svm_test)
trainAcc_best=round(sum(Y_train_pred_best==y_svm_train )/len(y_svm_train)*100,2)
testAcc_best=round(sum(Y_test_pred_best==y_svm_test )/len(y_svm_test)*100,2)
Ratio_best=round((testAcc_best/trainAcc_best)*100,2)
Support_vectors_best=round((len(Radialbest.support_vectors_)/len(X_svm_train)*100),2)
Radialbest._gamma
trainAcc_best
testAcc_best
Ratio_best
Support_vectors_best
(confusion_matrix(y_svm_test, Y_test_pred_best,normalize = 'true'))
SVEC=Radialbest.support_vectors_


#cross validation
from sklearn.model_selection import cross_val_score
svc=SVC(kernel='rbf',C=10)
scores = cross_val_score(svc, X_svm_train,  y_svm_train, cv=10, scoring='accuracy')
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


scores1 = cross_val_score(svc, X_svm_test,  y_svm_test, cv=10, scoring='accuracy')
print(scores1)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores1.mean(), scores1.std()))

#Plot of cross validation scores
K=[1,2,3,4,5,6,7,8,9,10]
Scores=[78.1,	76.5,	78.2,	77.8,81.2,81.7,82.1,	81.5,81.3,	80.3]
plt.plot(K,Scores,label='Validation scores',linestyle="--",color='green')
plt.xlabel('K')
plt.ylabel('Validation score%')


#creating plots of gamma & accuracy  #at cost=10
gamma=[0.000003,0.00003,0.0003,0.003,0.03,0.3]
TrainAcc=[66.71, 70.14, 71.09, 74.03, 84.61, 97.6]
TestAcc=[66,70.3, 70.4, 73.58,76.44,71.82]
supportvect=[91, 75.9, 70.01, 63.24, 53.3, 56.35]
plt.plot(gamma,TrainAcc,label='TrainAcc',linestyle="--",color='green')
plt.plot(gamma,TestAcc,label='TestAcc',linestyle="-.",color='blue')
plt.plot(gamma,supportvect,label='SupportVector',linestyle=":", color='red')
plt.title('Gamma Vs. Accuracy')
plt.legend( )
plt.xlabel('Gamma')
plt.ylabel('Accuracy%')


#creating plots of cost & accuracy  #gamma=0.03
cost=[0.1,1, 10,100]
TrainAcc=[74.14,78.57, 84.61, 91.33]
TestAcc=[73.5, 75.1, 76.44, 75.35]
supportvect=[68.3, 60.37, 53.3, 45.33]
plt.plot(cost,TrainAcc,label='TrainAcc',linestyle="--",color='green')
plt.plot(cost,TestAcc,label='TestAcc',linestyle="-.",color='blue')
plt.plot(cost,supportvect,label='SupportVector',linestyle=":", color='red')
plt.title('Cost Vs. Accuracy')
plt.legend()
plt.xlabel('Cost')
plt.ylabel('Accuracy%')

#Grid test
C_2d_range = [1,10,100]
gamma_2d_range = [100/len(X_svm_train),0.03613496,10000/len(X_svm_train)]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_svm_test,y_svm_test )
        classifiers.append((C, gamma, clf))


#Visualization
scores = grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range)) 
from matplotlib.colors import Normalize
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

gamma_range = [0.003,0.03,0.3]
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(
    scores,
    interpolation="nearest",
    cmap=plt.cm.hot,
    norm=MidpointNormalize(vmin=0.2, midpoint=0.15),)
plt.xlabel("gamma")

plt.ylabel("C")
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title("Validation accuracy")
plt.show()

#################  linear SVM ####################################################

Radiallinear = SVC(kernel='linear',C=10)
Radiallinear.fit(X_svm_train, y_svm_train)
Y_train_pred_lin=Radiallinear.predict(X_svm_train)
Y_test_pred_lin=Radiallinear.predict(X_svm_test)
trainAcc_lin=round(sum(Y_train_pred_lin==y_svm_train )/len(y_svm_train)*100,2)
testAcc_lin=round(sum(Y_test_pred_lin==y_svm_test )/len(y_svm_test)*100,2)
Support_vectors_lin=round((len(Radiallinear.support_vectors_)/len(X_svm_train)*100),2)
trainAcc_lin
testAcc_lin
Support_vectors_lin
(confusion_matrix(y_svm_test, Y_test_pred_lin,normalize = 'true'))


#plot
plt.figure(figsize=(10, 8))
# Plotting our two-features-space
sns.scatterplot(x=X_svm_train.iloc[:,0], #age
                y=X_svm_train.iloc[:,7], #deck area
                hue=y_svm_train,
                palette=['blue', 'orange'],
                s=8)
# Constructing a hyperplane using a formula.
w = Radiallinear.coef_[0]           # w consists of 2 elements
b = Radiallinear.intercept_[0]      # b consists of 1 element
x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
# Plotting a red hyperplane
plt.plot(x_points, y_points, c='r')
# Encircle support vectors
plt.scatter(Radiallinear.support_vectors_[:, 0],
            Radiallinear.support_vectors_[:, 1], 
            s=8, 
            facecolors='green') 




        
