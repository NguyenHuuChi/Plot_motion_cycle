import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from keras.layers import LSTM
from keras.utils import to_categorical
from special_function import read_text_file, calculate_divisions, read_injury_person, extract_different_distance,extract_the_similarity
tf.config.run_functions_eagerly(True)
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Path_0="D:\\Code\\Python\\summer_research_2\\Classifycation\\4543435"

        
def analyze_database(database,featua_to_read):
    column=database.columns
    Value=[]
    
    A=["25","35","45"]
    for i in range(len(A)):
        values1 = []
        a=A[i]
        for feat in featua_to_read:

            for  col in column:
                
                if feat in col and a in col:
                    if col=="PercGcycle" :continue
                    values= database[col]
                    max_value = max(values)
                    min_value = min(values)
                    mean_value = sum(values) / len(values)
                    values1.append(max_value)
                    values1.append(mean_value)
                    values1.append(min_value)
                    break
        
        Value.append(values1) 
    return Value

def process_file(folder_path):
    files_in_path= os.listdir(folder_path)
    List_processed_file=[]
    for folder in files_in_path :
        if len(folder)>13 and folder[-13:] == "processed.txt" :
            List_processed_file.append(folder)
    X=[]
    Y=[]
    k=0
    feature_new=[]
    Addition_infor=read_injury_person(folder_path)
    inju=Addition_infor[0]
    age=Addition_infor[2]
    BMI1 = Addition_infor [3]
    gender = Addition_infor [4]
    feature_read=[ "RhipAngX",	"RhipAngY",	"RhipAngZ",	"RkneeAngX",	"RkneeAngY",	"RkneeAngZ",	"RankleAngX",	"RankleAngY",	"RankleAngZ","RgrfX",	"RgrfY",	"RgrfZ"]
    for i,processed_file in enumerate(List_processed_file) :
        
        path_processed =os.path.join(folder_path, processed_file)
        Dataframe= read_text_file(path_processed)
        # read the name of feature
        if k==0 :
            for colu in feature_read :
                if colu=="PercGcycle" :continue
                feature_new.append("max"+colu)
                feature_new.append("mean"+colu)
                feature_new.append("min"+colu)
                k+=1
        value = analyze_database(Dataframe,feature_read)

        similartity_data= extract_the_similarity(Dataframe)
        for j in range(len(value)):
            value[j].extend([age[j], BMI1[j],gender[j]])
        for j in range(len(similartity_data)):
            similartity_data[j].extend([age[j], BMI1[j],gender[j]])    
        X.extend(similartity_data)
        if (inju[i]=="Yes" or inju[i]=="yes"):
            Y.extend([1]*len(similartity_data))
        elif (inju[i]=="No" or inju[i]=="no"):   
            Y.extend([0]*len(similartity_data))
    feature_new.extend(['age',"BMI1","gender"])
    return X,Y,feature_new
        
X,Y,features=process_file(Path_0)

# feature for general 
feature=[]
for i in range(len(X[0])):
    a=(f"Fe %d",i+1)
    feature.append(a)
X=pd.DataFrame(X,columns= feature)
seed=42

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.15,random_state= seed)


def remove_contain_Null_value(X):
    columns=X.columns
    for col in columns :
        if X[col].isnull().any() :
            X=X.drop([col], axis=1)
    return X

X_train= remove_contain_Null_value(X_train)

X_test= remove_contain_Null_value(X_test)

print(X_train)

scaler=StandardScaler()
transform=scaler.fit(X_train)
X_train=transform.transform(X_train)
X_test=transform.transform(X_test)
#Knnnn
# print("---------------knn-----------")
# knn = KNN(n_neighbors = 4)
# knn.fit(X_train, Y_train)
# y_pred = knn.predict(X_test)



#decision tree

# print("-----------Decision tree -------------")
# clf=DecisionTreeClassifier()
# clf.fit(X_train,Y_train)
# importance=clf.feature_importances_

# columns_to_drop = [] 
# for i in range(len(importance)):
#     if abs(importance[i]) <= 0.1:
#         columns_to_drop.append(i)

# X_test_filtered = np.delete(X_test, columns_to_drop, axis=1)
# X_train_filtered =np.delete(X_train, columns_to_drop, axis=1)

# clf1=DecisionTreeClassifier()
# clf1.fit(X_train_filtered,Y_train)
# y_pred = clf1.predict(X_test_filtered)

# random forest 
print("------------Random forest-------------")
clf= RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print(classification_report(Y_test, y_pred))
importance = clf.feature_importances_
columns_to_drop = [] 
for i in range(len(importance)):
    if abs(importance[i]) <= (max(importance)-min(importance))/10:
        columns_to_drop.append(i)

X_test_filtered = np.delete(X_test, columns_to_drop, axis=1)
X_train_filtered =np.delete(X_train, columns_to_drop, axis=1)
print("--------------after filter in Random forest-----------")
clf1=RandomForestClassifier(n_estimators=100,random_state=42)
clf1.fit(X_train_filtered,Y_train)
y_pred=clf1.predict(X_test_filtered)



# print("-----------------Boosted Gradient--------------")
# gbc = GradientBoostingClassifier(n_estimators=300,
#                                  learning_rate=0.05,
#                                  random_state=100,
#                                  max_features=5 )
# # Fit to training set
# gbc.fit(X_train,Y_train)
 
# # Predict on test set
# y_pred = gbc.predict(X_test)
 




#naive Bayes classifyier --> it is so inconvenient it is suitable for word
# print("------------naive Bayes classifyier-------------")
# clf=GaussianNB()
# clf.fit(X_train,Y_train)
# y_pred=clf.predict(X_test)

# Support vetor machine ---for multiple class classification using one- one or one-rest (one one will create n*(N-1)/2 
#  binary classifiers are created. Each classifier focuses on distinguishing between two classes, and the class that 
# receives the most votes from all the binary classifiers is chosen as the final prediction ... one vs rest will create
# . If there are N classes, N binary classifiers are created. Each classifier focuses on distinguishing one class from 
# all the other classes combined . During prediction, each classifier produces a confidence score for its class, and the 
# class with the highest confidence score is selected as the final prediction.  )

# print("-----------------Support vetor machine one one ------------")
# clf=SVC(kernel="linear",C=1.0)
# clf.fit(X_train,Y_train)
# y_pred=clf.predict(X_test)


print(classification_report(Y_test, y_pred))





