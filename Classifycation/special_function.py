import pandas as pd
import matplotlib.pyplot as plt
import csv
import linecache
import numpy as np
import seaborn as sns
import os
import random
def read_text_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error occurred while reading the file: {e}")
        return None  
def calculate_divisions(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Input arrays must have the same length")

    divisions = []
    for i in range(len(array1)):
        if array2[i] == 0:
            raise ValueError("Cannot divide by zero")
        division_result = array1[i] / ((array2[i]/100)**2)
        divisions.append(division_result)

    return divisions

def read_injury_person(Folder_path):
    files_in_path= os.listdir(Folder_path)
    for folder in files_in_path :
        if len(folder) >13 and folder[-13:] =="RBDSinfo.xlsx" :
            path_contain_injury=os.path.join(Folder_path,folder)
            data=pd.read_excel(path_contain_injury)
            subject=data["Subject"]
            Injuri=data["Injury"]
            Injuri_loc=data["InjuryLoc"]
            Age=data["Age"]
            Heigh=data["Height"]
            Mass=data["Mass"]
            
            Gender=data["Gender"]
            for i in range(len(Gender)):
                if(Gender[i]=="M"):
                    Gender[i]=0
                else:
                    Gender[i]=1
            inju=[]
            inju_loc=[]
            age,heigh,mass,gender=[],[],[],[]
            k =1
            for i,sub in enumerate(subject) :
                if sub ==k :
                    k+=1
                    inju.append(Injuri[i])
                    inju_loc.append(Injuri_loc[i])
                    age.append(Age[i])
                    heigh.append(Heigh[i])
                    mass.append(Mass[i])
                    gender.append(Gender[i])
            BMI1=calculate_divisions(mass,heigh)
            return [inju, inju_loc,age,BMI1,gender]
#----------------function create more data------------
def  create_more_data(data_frame,numer_more,shift=True,random_change=False): #in here data_frame have 3 dimensions dataframe(batch_size,feature, time_Step)
    data_frame=np.array(data_frame)
    Data_more=[]
    for i in range(len(data_frame[0])):
        Data_more_=[]
        for j in range(len(data_frame)):
            Data_more_.append(data_frame[j][i])
        average=np.mean(Data_more_,axis=0)
        std_dev=np.std(Data_more_,axis=0)
        Random_for_one_feature=[]   # (batch_size, time_step)

        #-------First way to agument is add a random number of in the std deviation range to the average value
        if random_change :
            for i in range(numer_more):
                randomX=[]
                for j in range(len(average)):
                    randomX.append(average[j]+np.random.uniform(-std_dev[j],std_dev[j]))
                Random_for_one_feature.append(randomX)

        # ---second way to agument is shift all data to above or below position----
        max=np.max(average)
        min=np.min(average)
        rangee= (max-min)/20
        if shift :
            for i in range(numer_more):
                randomX=[]
                shift= np.random.uniform(-rangee,+rangee)
                for j in range(len(average)):
                    randomX.append(average[j]+shift)
                Random_for_one_feature.append(randomX)

        if len(Data_more) ==0 :
            Data_more=[[a] for a in Random_for_one_feature]
        else :
            for j in range(len(Random_for_one_feature)):
                Data_more[j].append(Random_for_one_feature[j]) # have a shape of ( batch_size, feature, time_step)
    # Data_more=np.array(Data_more)

    return Data_more


def plot_data(X,Y,name_column): # X(batchsize ,features, timestep)
    a=np.arange(0,101,1)
    n=len(name_column)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(15, 5 * n))
    
    for i, ax_roww in enumerate(axes):
        for j in range(len(X)):
            if np.array_equal(Y[j],[1.,0.,0.]) :
                ax_roww[0].plot(a,X[j][i])
            elif np.array_equal(Y[j],[0.,1.,0.]) :
                ax_roww[1].plot(a,X[j][i]) 
            elif np.array_equal(Y[j],[0.,0.,1.]) :
                ax_roww[2].plot(a,X[j][i]) 
        ax_roww[0].set_title(name_column[i]+"25")
        ax_roww[1].set_title(name_column[i]+"35")
        ax_roww[2].set_title(name_column[i]+"45")
    plt.show()

def extract_features(feature_names, dataframe):
    extracted_data35 = []
    extracted_data25 = []
    extracted_data45 = []
    for feature_name in feature_names:
        for column in dataframe.columns :
            if feature_name in column and "35" in column:
                extracted_data35.append(dataframe[column])
            elif feature_name in column and "25" in column:
                extracted_data25.append(dataframe[column])
            elif feature_name in column and "45" in column:
                extracted_data45.append(dataframe[column])
    A=[]
    if len(extracted_data25)!=0 :
        A.append(extracted_data25)   
    if len(extracted_data35)!=0 :
        A.append(extracted_data35)  
    if len(extracted_data45)!=0 :
        A.append(extracted_data45)                
    return A
def extract_different_distance(DataFrame):
    extracted_data35 = []
    extracted_data25 = []
    extracted_data45 = []
    columnss=DataFrame.columns
    for column in columnss :
        if column[0] != "R" :
            continue
        for colum in columnss :
            if colum[0] !="L":
                continue
            if column[1:] == colum[1:] :
                if column[-2:] =="25":
                    extracted_data25.append(DataFrame[column]-DataFrame[colum])
                if column[-2:] =="35":
                    extracted_data35.append(DataFrame[column]-DataFrame[colum])
                if column[-2:] =="45":
                    extracted_data45.append(DataFrame[column]-DataFrame[colum])
    A=[]
    if len(extracted_data25)!=0 :
        A.append(extracted_data25)   
    if len(extracted_data35)!=0 :
        A.append(extracted_data35)  
    if len(extracted_data45)!=0 :
        A.append(extracted_data45)                
    return A


def cos_similarity(vector_a,vector_b):
    # Calculate the dot product of A and B
    dot_product = np.dot(vector_a, vector_b)

    # Calculate the Euclidean norms (magnitudes) of A and B
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (norm_a * norm_b)
    return cosine_similarity
def l2_distance(vector1, vector2):
    # Ensure that both vectors have the same length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length for L2 distance calculation.")
    
    # Calculate the L2 (Euclidean) distance
    distance = np.linalg.norm(np.array(vector1) - np.array(vector2))
    return distance

def dot_product(vector1, vector2):
    # Ensure that both vectors have the same length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length for dot product calculation.")
    
    # Calculate the dot product
    result = np.dot(np.array(vector1), np.array(vector2))
    return result
def extract_the_similarity(DataFrame):
    extracted_data35 = []
    extracted_data25 = []
    extracted_data45 = []
    columnss=DataFrame.columns
    for column in columnss :
        if column[0] != "R" :
            continue
        for colum in columnss :
            if colum[0] !="L":
                continue
            if column[1:] == colum[1:] :
                first_distance= DataFrame[column][0] - DataFrame[colum][0]
                shifted_left= DataFrame[colum] +first_distance
                if column[-2:] =="25":
                    extracted_data25.append(cos_similarity(DataFrame[column], shifted_left))
                if column[-2:] =="35":
                    extracted_data35.append(cos_similarity(DataFrame[column], shifted_left))
                if column[-2:] =="45":
                    extracted_data45.append(cos_similarity(DataFrame[column], shifted_left))
    A=[]
    # if len(extracted_data25)!=0 :
    #     A.append(extracted_data25)   
    if len(extracted_data35)!=0 :
        A.append(extracted_data35)  
    # if len(extracted_data45)!=0 :
    #     A.append(extracted_data45)                
    return A