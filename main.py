# GRP-02 PROJECT- HANDLING CLASS IMBALANCE IN FRAUD DETECTION 



import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve,confusion_matrix
import warnings
warnings.filterwarnings("ignore")
from yellowbrick.classifier import ClassificationReport






def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X[y==l, 0],X[y==l, 1],c=c, label=l, marker=m)
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

def generate_model_report(y_actual, y_predicted):
    print("Accuracy = " , accuracy_score(y_actual, y_predicted))
    print("Precision = " ,precision_score(y_actual, y_predicted))
    print("Recall = " ,recall_score(y_actual, y_predicted))
    print("F1 Score = " ,f1_score(y_actual, y_predicted))
    pass

df = pd.read_csv('creditcard.csv')
df.head()
df.describe()
df.isnull().sum().max()
df.columns

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()


from sklearn.preprocessing import StandardScaler, RobustScaler


std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

df.head()


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values


train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

df = df.sample(frac=1)

fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()


print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))



# sns.countplot('Class', data=new_df, palette=colors)
# plt.title('Equally Distributed Classes', fontsize=14)
# plt.show()


# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V10 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print('----' * 44)

# -----> V12 removing outliers from fraud transactions
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))
outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
print('----' * 44)


# Removing outliers V10 Feature
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('V10 outliers: {}'.format(outliers))
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))


f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))

colors = ['#B3F9C5', '#f9c5b3']
# Boxplots with outliers removed
# Feature V14
sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1, palette=colors)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)
ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),arrowprops=dict(facecolor='black'),fontsize=14)

# Feature 12
sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)
ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),arrowprops=dict(facecolor='black'),fontsize=14)

# Feature V10
sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)
ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),arrowprops=dict(facecolor='black'),fontsize=14)


plt.show()


X = new_df.drop('Class', axis=1)
y = new_df['Class']





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values



classifiers = {"LogisiticRegression": LogisticRegression(),"KNearest": KNeighborsClassifier(),"Support Vector Classifier": SVC(),"RandomForest": RandomForestClassifier()}

from sklearn.model_selection import cross_val_score


for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")


from sklearn.model_selection import GridSearchCV


log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
# KNears best estimator
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)

# SVC best estimator
svc = grid_svc.best_estimator_

# RandomForestClassifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(RandomForestClassifier(), tree_params)
grid_tree.fit(X_train, y_train)

# tree best estimator
tree_clf = grid_tree.best_estimator_



log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')


knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('RandomForest Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')




from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,method="decision_function")

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)




from sklearn.metrics import roc_auc_score

print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
print('RandomForest Classifier: ', roc_auc_score(y_train, tree_pred))





log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)


def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='RandomForestClassifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),arrowprops=dict(facecolor='#6E726D', shrink=0.05),)
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
plt.show()


# def logistic_roc_curve(log_fpr, log_tpr):
#     plt.figure(figsize=(12,8))
#     plt.title('Logistic Regression ROC Curve', fontsize=16)
#     plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlabel('False Positive Rate', fontsize=16)
#     plt.ylabel('True Positive Rate', fontsize=16)
#     plt.axis([-0.01,1,0,1])
    
    
# logistic_roc_curve(log_fpr, log_tpr)
# plt.show()


# We will undersample during cross validating
undersample_X = df.drop('Class', axis=1)
undersample_y = df['Class']

for train_index, test_index in sss.split(undersample_X, undersample_y):
    print("Train:", train_index, "Test:", test_index)
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values 

undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []



# Nearmiss Logistic Regession
def Nm_LR(X_train,Y_train,X_test,Y_test):
    import matplotlib.pyplot as plt
    X, y = NearMiss().fit_resample(undersample_X.values, undersample_y.values)
    dataframe=pd.DataFrame(y, columns=['target']) 
    target_count = dataframe.target.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    zero=target_count[0]
    one=target_count[1]
    left = [1, 2]  
    height = [zero,one] 
    tick_label = ['Not Fraud', 'Fraud'] 
    plt.bar(left, height, tick_label = tick_label, width = 0.8, color = ['red', 'green']) 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.title('Nearmiss Logistic Regression') 
    plt.show()
    lg= LogisticRegression().fit(X, y)

    Y_Test_Pred = lg.predict(original_Xtest)

    #confusion matrix
    matrix =confusion_matrix(original_ytest, Y_Test_Pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1)
    #plt.title('nearmiss Svm', fontsize=8)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    
    #Classification Report
    generate_model_report(pd.Series(Y_Test_Pred), pd.Series(original_ytest))
    target_names = ['NO', 'YES']

    prediction=lg.predict(original_Xtest)
    print(classification_report(original_ytest, prediction, target_names=target_names))
    classes = ["NO", "YES"]
    visualizer = ClassificationReport(lg, classes=classes, support=True)
    visualizer.fit(X, y)  
    visualizer.score(original_Xtest, original_ytest)  
    g = visualizer.poof()
    

    y_score = lg.predict_proba(original_Xtest)

    y_score=y_score[:,1]
    from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)

    from sklearn.metrics import average_precision_score

    undersample_average_precision = average_precision_score(original_ytest, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(undersample_average_precision))

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,6))

    precision, recall, _ = precision_recall_curve(original_ytest, y_score)

    plt.step(recall, precision, color='#004a93', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='#48a6ff')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('UnderSampling using NearMiss LR \n Average Precision-Recall Score ={0:0.2f}'.format(undersample_average_precision), fontsize=16)

    plt.show()
    
Nm_LR(undersample_Xtrain, undersample_ytrain,undersample_Xtest, undersample_ytest)






#Random under smapling using LR

def RUS_LR(X_train,Y_train,X_test,Y_test):
    import matplotlib.pyplot as plt
    X, y = RandomUnderSampler().fit_resample(undersample_X.values, undersample_y.values)
    dataframe=pd.DataFrame(y, columns=['target']) 
    target_count = dataframe.target.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    zero=target_count[0]
    one=target_count[1]
    left = [1, 2]  
    height = [zero,one] 
    tick_label = ['Not Fraud', 'Fraud'] 
    plt.bar(left, height, tick_label = tick_label, width = 0.8, color = ['red', 'green']) 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.title('Random Under Smapling Logistic Regression') 
    plt.show()
    lg= LogisticRegression().fit(X, y)

    Y_Test_Pred = lg.predict(original_Xtest)

    #confusion matrix
    matrix =confusion_matrix(original_ytest, Y_Test_Pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    
    #Classification Report
    generate_model_report(pd.Series(Y_Test_Pred), pd.Series(original_ytest))
    target_names = ['NO', 'YES']

    prediction=lg.predict(original_Xtest)
    print(classification_report(original_ytest, prediction, target_names=target_names))
    classes = ["NO", "YES"]
    visualizer = ClassificationReport(lg, classes=classes, support=True)
    visualizer.fit(X, y)  
    visualizer.score(original_Xtest, original_ytest)  
    g = visualizer.poof()
    

    y_score = lg.predict_proba(original_Xtest)

    y_score=y_score[:,1]

    from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)
    from sklearn.metrics import average_precision_score

    undersample_average_precision1 = average_precision_score(original_ytest, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(undersample_average_precision1))

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,6))
    precision, recall, _ = precision_recall_curve(original_ytest, y_score)
    plt.step(recall, precision, color='#004a93', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='#48a6ff')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('UnderSampling using Undersampler LR \n Average Precision-Recall Score ={0:0.2f}'.format(undersample_average_precision1), fontsize=16)
    plt.show()
    

RUS_LR(undersample_Xtrain, undersample_ytrain,undersample_Xtest, undersample_ytest)







#SMOTE using LR
def SM_LR(X_train,Y_train,X_test,Y_test):
    print("-------------------------------------------SMOTE with Logistic Regression-------------------------------------------------------")
    print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
    print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))

    X, y = SMOTE(sampling_strategy='minority').fit_resample(original_Xtrain, original_ytrain)
    lg= LogisticRegression().fit(X, y)

    dataframe=pd.DataFrame(y, columns=['target']) 
    target_count = dataframe.target.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    zero=target_count[0]
    one=target_count[1]
    left = [1, 2]  
    height = [zero,one] 
    tick_label = ['Not Fraud', 'Fraud'] 
    plt.bar(left, height, tick_label = tick_label, width = 0.8, color = ['red', 'green']) 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.title('SMOTE Logistic Regression') 
    plt.show()


    Y_Test_Pred = lg.predict(original_Xtest)

    #confusion matrix
    matrix =confusion_matrix(original_ytest, Y_Test_Pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    
    #Classification Report
    generate_model_report(pd.Series(Y_Test_Pred), pd.Series(original_ytest))
    target_names = ['NO', 'YES']

    prediction=lg.predict(original_Xtest)
    print(classification_report(original_ytest, prediction, target_names=target_names))
    classes = ["NO", "YES"]
    visualizer = ClassificationReport(lg, classes=classes, support=True)
    visualizer.fit(X, y)  
    visualizer.score(original_Xtest, original_ytest)  
    g = visualizer.poof()
    

    y_score = lg.predict_proba(original_Xtest)

    y_score=y_score[:,1]

    #PR curve
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(original_ytest, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    fig = plt.figure(figsize=(12,6))
    precision, recall, _ = precision_recall_curve(original_ytest, y_score)
    plt.step(recall, precision, color='r', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='#F59B00')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('OverSampling using Smote LR \n Average Precision-Recall Score ={0:0.2f}'.format(average_precision), fontsize=16)
    plt.show()

SM_LR(original_Xtrain, original_ytrain,original_Xtest,original_ytest)






#ADASYN using LR
def ADA_LR(X_train,Y_train,X_test,Y_test):

    print("-------------------------------------------ADASYN with Logistic Regression-------------------------------------------------------")
    X, y = ADASYN(sampling_strategy='minority').fit_resample(original_Xtrain, original_ytrain)
    lg= LogisticRegression().fit(X, y)
    dataframe=pd.DataFrame(y, columns=['target']) 
    target_count = dataframe.target.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    zero=target_count[0]
    one=target_count[1]
    left = [1, 2]  
    height = [zero,one] 
    tick_label = ['Not Fraud', 'Fraud'] 
    plt.bar(left, height, tick_label = tick_label, width = 0.8, color = ['red', 'green']) 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.title('ADASYN Logistic Regression') 
    plt.show()
    Y_Test_Pred = lg.predict(original_Xtest)

    #confusion matrix
    matrix =confusion_matrix(original_ytest, Y_Test_Pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    #Classification Report
    generate_model_report(pd.Series(Y_Test_Pred), pd.Series(original_ytest))
    from sklearn.metrics import classification_report
    target_names = ['NO', 'YES']
    prediction=lg.predict(original_Xtest)
    print(classification_report(original_ytest, prediction, target_names=target_names))
    classes = ["NO", "YES"]
    visualizer = ClassificationReport(lg, classes=classes, support=True)
    visualizer.fit(X, y)  
    visualizer.score(original_Xtest, original_ytest)  
    g = visualizer.poof()

    #PR curve
    y_score = lg.predict_proba(original_Xtest)
    y_score=y_score[:,1]

    average_precision1 = average_precision_score(original_ytest, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision1))
    fig = plt.figure(figsize=(12,6))
    precision, recall, _ = precision_recall_curve(original_ytest, y_score)
    plt.step(recall, precision, color='r', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='#F59B00')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('OverSampling Using Adasyn LR \n Average Precision-Recall Score ={0:0.2f}'.format(average_precision1), fontsize=16)
    plt.show()
    

ADA_LR(original_Xtrain, original_ytrain,original_Xtest,original_ytest)


#SMOTE using RF
def SM_RF(X_train,Y_train,X_test,Y_test):
    print("-------------------------------------------SMOTE with RandomForest-------------------------------------------------------")
    print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
    print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))

    X, y = SMOTE(sampling_strategy='minority').fit_resample(original_Xtrain, original_ytrain)
    dt= RandomForestClassifier().fit(X, y)

    dataframe=pd.DataFrame(y, columns=['target']) 
    target_count = dataframe.target.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    zero=target_count[0]
    one=target_count[1]
    left = [1, 2]  
    height = [zero,one] 
    tick_label = ['Not Fraud', 'Fraud'] 
    plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = ['red', 'green']) 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.title('SMOTE Random Forest') 
    plt.show()


    Y_Test_Pred = dt.predict(original_Xtest)

    #confusion matrix
    matrix =confusion_matrix(original_ytest, Y_Test_Pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    #Classification Report
    generate_model_report(pd.Series(Y_Test_Pred), pd.Series(original_ytest))
    from sklearn.metrics import classification_report
    target_names = ['NO', 'YES']
    prediction=dt.predict(original_Xtest)
    print(classification_report(original_ytest, prediction, target_names=target_names))
    classes = ["NO", "YES"]
    visualizer = ClassificationReport(dt, classes=classes, support=True)
    visualizer.fit(X, y)  
    visualizer.score(original_Xtest, original_ytest)  
    g = visualizer.poof()

    y_score = dt.predict_proba(original_Xtest)

    y_score=y_score[:,1]

    #PRCurve
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(original_ytest, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
    fig = plt.figure(figsize=(12,6))
    precision, recall, _ = precision_recall_curve(original_ytest, y_score)
    plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#F59B00')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('OverSampling using Smote RF \n Average Precision-Recall Score ={0:0.2f}'.format(
          average_precision), fontsize=16)
    plt.show()

SM_RF(original_Xtrain, original_ytrain,original_Xtest,original_ytest)



# Nearmiss RF
def Nm_RF(X_train,Y_train,X_test,Y_test):
    
    print("-------------------------------------------NearMiss with RandomForest-------------------------------------------------------")
    print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
    print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))
    
    import matplotlib.pyplot as plt
    X, y = NearMiss().fit_resample(undersample_X.values, undersample_y.values)
    dataframe=pd.DataFrame(y, columns=['target']) 
    target_count = dataframe.target.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    zero=target_count[0]
    one=target_count[1]
    left = [1, 2]  
    height = [zero,one] 
    tick_label = ['Not Fraud', 'Fraud']  
    plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = ['red', 'green']) 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.title('Nearmiss Random Forest') 
    plt.show()
    dt= RandomForestClassifier().fit(X, y)

    Y_Test_Pred = dt.predict(original_Xtest)

    #confusion matrix
    matrix =confusion_matrix(original_ytest, Y_Test_Pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix ', y=1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    
    #Classification Report
    generate_model_report(pd.Series(Y_Test_Pred), pd.Series(original_ytest))
    from sklearn.metrics import classification_report
    target_names = ['NO', 'YES']
    prediction=dt.predict(original_Xtest)
    print(classification_report(original_ytest, prediction, target_names=target_names))
    classes = ["NO", "YES"]
    visualizer = ClassificationReport(dt, classes=classes, support=True)
    visualizer.fit(X, y)  
    visualizer.score(original_Xtest, original_ytest)  
    g = visualizer.poof()

    y_score = dt.predict_proba(original_Xtest)

    y_score=y_score[:,1]
    from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)
    #PRCurve

    from sklearn.metrics import average_precision_score

    undersample_average_precision = average_precision_score(original_ytest, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
      undersample_average_precision))

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,6))

    precision, recall, _ = precision_recall_curve(original_ytest, y_score)

    plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('UnderSampling using NearMiss RF \n Average Precision-Recall Score ={0:0.2f}'.format(
          undersample_average_precision), fontsize=16)

    plt.show()
    
Nm_RF(undersample_Xtrain, undersample_ytrain,undersample_Xtest, undersample_ytest)



#Random under smapling using RF

def RUS_RF(X_train,Y_train,X_test,Y_test):
    
    print("-------------------------------------------RUS with RandomForest-------------------------------------------------------")
    print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
    print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))
    
    import matplotlib.pyplot as plt
    X, y = RandomUnderSampler().fit_resample(undersample_X.values, undersample_y.values)
    dataframe=pd.DataFrame(y, columns=['target']) 
    target_count = dataframe.target.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    zero=target_count[0]
    one=target_count[1]
    left = [1, 2]  
    height = [zero,one] 
    tick_label = ['Not Fraud', 'Fraud'] 
    plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = ['red', 'green']) 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.title('Random Under Smapling Random Forest') 
    plt.show()

    dt= RandomForestClassifier().fit(X, y)

    Y_Test_Pred = dt.predict(original_Xtest)

    #confusion matrix
    matrix =confusion_matrix(original_ytest, Y_Test_Pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    
    #Classification Report
    generate_model_report(pd.Series(Y_Test_Pred), pd.Series(original_ytest))
    from sklearn.metrics import classification_report
    target_names = ['NO', 'YES']
    prediction=dt.predict(original_Xtest)
    print(classification_report(original_ytest, prediction, target_names=target_names))
    classes = ["NO", "YES"]
    visualizer = ClassificationReport(dt, classes=classes, support=True)
    visualizer.fit(X, y)  
    visualizer.score(original_Xtest, original_ytest)  
    g = visualizer.poof()

    y_score = dt.predict_proba(original_Xtest)
    from sklearn.metrics import precision_recall_curve
    y_score=y_score[:,1]
    precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)
    #PRCurve
    undersample_average_precision1 = average_precision_score(original_ytest, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
      undersample_average_precision1))

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12,6))
    precision, recall, _ = precision_recall_curve(original_ytest,y_score)
    plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('UnderSampling using Undersampler RF \n Average Precision-Recall Score ={0:0.2f}'.format(
          undersample_average_precision1), fontsize=16)
    plt.show()
    

RUS_RF(undersample_Xtrain, undersample_ytrain,undersample_Xtest, undersample_ytest)



#ADASYN using RF
def ADA_RF(X_train,Y_train,X_test,Y_test):
    print("-------------------------------------------ADASYN with RandomForest-------------------------------------------------------")
    print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
    print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))
    
    X, y = SMOTE(sampling_strategy='minority').fit_resample(original_Xtrain, original_ytrain)
    dt= RandomForestClassifier().fit(X, y)

    dataframe=pd.DataFrame(y, columns=['target']) 
    target_count = dataframe.target.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    zero=target_count[0]
    one=target_count[1]
    left = [1, 2]  
    height = [zero,one] 
    tick_label = ['Not Fraud', 'Fraud'] 
    plt.bar(left, height, tick_label = tick_label, 
                width = 0.8, color = ['red', 'green']) 
    plt.xlabel('x - axis') 
    plt.ylabel('y - axis') 
    plt.title('ADASYN Random Forest') 
    plt.show()

    Y_Test_Pred = dt.predict(original_Xtest)

    #confusion matrix
    matrix =confusion_matrix(original_ytest, Y_Test_Pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    
    #Classification Report
    generate_model_report(pd.Series(Y_Test_Pred), pd.Series(original_ytest))
    from sklearn.metrics import classification_report
    target_names = ['NO', 'YES']
    prediction=dt.predict(original_Xtest)
    print(classification_report(original_ytest, prediction, target_names=target_names))
    classes = ["NO", "YES"]
    visualizer = ClassificationReport(dt, classes=classes, support=True)
    visualizer.fit(X, y)  
    visualizer.score(original_Xtest, original_ytest)  
    g = visualizer.poof()

    y_score = dt.predict_proba(original_Xtest)

    y_score=y_score[:,1]

    average_precision1 = average_precision_score(original_ytest, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision1))
    fig = plt.figure(figsize=(12,6))
    precision, recall, _ = precision_recall_curve(original_ytest, y_score)
    plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#F59B00')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('OverSampling Using Adasyn RF \n Average Precision-Recall Score ={0:0.2f}'.format(
          average_precision1), fontsize=16)
    plt.show()
    #plot_2d_space(X, y, 'ADASYN SVM')

ADA_RF(original_Xtrain, original_ytrain,original_Xtest,original_ytest)




