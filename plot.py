import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import linecache
import numpy as np
import seaborn as sns
from tkinter import filedialog
import os

# Function read the file contain specific time ; start and end is the start and the end of row want to read
def read_file_time(path, start=1 ,end=4) :
    if(path[-4:] == "xlsx"):
        new_path=path[:len(path)-4]+"csv"
        read_file=pd.read_excel(path)
        read_file.to_csv(new_path, index= None, header=True)
        with open(new_path) as file_obj :
            read_obj= csv.reader(file_obj)
            interestingrows=[row for idx, row in enumerate(read_obj) if idx in range(start,end)]
            return interestingrows
    elif (path[-3:]=='csv') :
        with open(path) as file_obj:
            read_obj=csv.reader(file_obj)
            interestingrows=[row for idx, row in enumerate(read_obj) if idx in range(start,end)]
            return interestingrows

# data= read_file_time("2023-06-14-11-02_Treadmill_9-11kph.xlsx",1,11) # read from line 2 to line 11

#----------- Find where contain time in file-------------
def read_time(data):
    index=0
    while(index < len(data[0])-1) :
        if len(data[0][index])==0:   # check empty 
            index+=1
            continue
        elif data[0][index][-1]== "Z" and data[0][index][0]=="2" :
            break
        index +=1
    standard_time=[]
    for i in range(len(data)):
        standard_time.append(data[i][index])
    return standard_time
# a=read_time(data)

# ---------------calculate the time difference (array contain the time difference ) --------------------------
def calculate_time_differences(timestamps):
    first_timestamp = datetime.fromisoformat(timestamps[0].replace("Z", ""))
    time_diffs = []

    for timestamp in timestamps[2:]:
        if timestamp:
            current_timestamp = datetime.fromisoformat(timestamp.replace("Z", ""))
            time_diff = int((current_timestamp - first_timestamp).total_seconds()*100)
            time_diffs.append(time_diff)

    return time_diffs

def calculate_the_different_time(path,len):
    data=read_file_time(path,1,len)
    list_time=read_time(data)
    different_time= calculate_time_differences(list_time)
    return different_time
# different_time=calculate_the_different_time("2023-06-14-11-02_Treadmill_9-11kph.xlsx",11)
# print("time difference",different_time)

#----------------Read data angle ------------------------------------------

# Check if it can transfer to interger
def check_number(string):
    try:
        number = float(string)
        return True
    except:
        return False


#------------function read specific row in csv file----------
def read_specific_rows(filename, row_indices):

    with open(filename, 'r') as file:
        row = linecache.getline(filename, row_indices)
        row.strip()
        row=row.split(',')
    return row
def read_csv_from_frame(csv_file_path):
    try:
        with open(csv_file_path, 'r') as file:
            # Find the row index that starts with "Frame"
            frame_row_index = None
            for i, line in enumerate(file):
                if line.strip().startswith("Frame"):
                    frame_row_index = i
                    break

            if frame_row_index is None:
                print("No row starting with 'Frame' found in the CSV.")
                return None

        # Read the CSV file starting from the "Frame" row using pandas
        df = pd.read_csv(csv_file_path, skiprows=frame_row_index)

        return df,frame_row_index
    except Exception as e:
        print("Error occurred during CSV reading:", e)
        return None

class Read_data_runing:
    def __init__(self,file_path):
        self.Model_output=[] # ---------first element is index in column, second is start frame , third is the end frame---------------------
        self.Trajectory=[]
        self.file_path=file_path
        
        self.estimate_the_start_frame()

    #-------------------- Read the data and Find the frame where to start-------------------------
    def estimate_the_start_frame(self,skiprow=3):

        self.data_frame, index_from_frame = read_csv_from_frame(self.file_path) 
        self.Header1=read_specific_rows(self.file_path,index_from_frame)  
        Frame=self.data_frame["Frame"]
        self.Model_output.append(1)
        self.Model_output.append(int(Frame[1]))
        
        #Find where Model_output finish
        try :
            i=1
            while  i<len(Frame) : 
                if not check_number(Frame[i]):
                    break
                i+=1
            self.Model_output.append(int(Frame[i-1]))
            while i<len(Frame):
                if(Frame[i]=="Frame"):
                    break
                i+=1
            self.Trajectory.append(i+2)
            self.Trajectory.append(int(Frame[i+2]))
            self.Trajectory.append(int(Frame[len(Frame)-1]))
        except :
            pass
        self.Column_name=self.data_frame.columns

    def read_specific_columns(self,name_column):
        columna= self.data_frame[name_column]
        
        # ----------------------------Split part Model output and Trajectory------------------------
        Model_ouput_co=columna[self.Model_output[0]:self.Model_output[2]-self.Model_output[1]+self.Model_output[0]+1]
        try :
            Trajectory_co=columna[self.Trajectory[0]:self.Trajectory[2]-self.Trajectory[1]+self.Trajectory[0]+1]
        except :
            pass
        return Model_ouput_co
    def read_multiple_columns(self,list_name_columns):
        Data_Frame_model={}
        for name_columns in list_name_columns:
            left_leg="L"+name_columns+"Angles"
            right_leg="R"+name_columns +"Angles"

            for i , header in enumerate(self.Header1):
                if left_leg in header :
                    flex     = self.Column_name[i]
                    ab       = self.Column_name[i+1]
                    rotation = self.Column_name[i+2]

                    flex_model= self.data_frame[flex][self.Model_output[0]:self.Model_output[2]-self.Model_output[1]+self.Model_output[0]+1]
                    ab_model  =self.data_frame[ab][self.Model_output[0]:self.Model_output[2]-self.Model_output[1]+self.Model_output[0]+1]
                    rotation_model = self .data_frame[rotation][self.Model_output[0]:self.Model_output[2]-self.Model_output[1]+self.Model_output[0]+1]

                    Data_Frame_model[left_leg] =[flex_model,ab_model,rotation_model] 
               
                if right_leg in header :
                    flex     = self.Column_name[i]
                    ab       = self.Column_name[i+1]
                    rotation = self.Column_name[i+2]

                    flex_model= self.data_frame[flex][self.Model_output[0]:self.Model_output[2]-self.Model_output[1]+self.Model_output[0]+1]
                    ab_model  =self.data_frame[ab][self.Model_output[0]:self.Model_output[2]-self.Model_output[1]+self.Model_output[0]+1]
                    rotation_model = self .data_frame[rotation][self.Model_output[0]:self.Model_output[2]-self.Model_output[1]+self.Model_output[0]+1]

                    Data_Frame_model[right_leg] =[flex_model , ab_model , rotation_model ]

        return Data_Frame_model

def read_cycle_left_right(start_frame,end_frame,path_left,path_right,check_left =True):

    data_run_left=pd.read_csv(path_left)
    left_start=data_run_left["Left start"]
    left_end=data_run_left["Left end"]
    percent_stance_left=data_run_left['percent_stance_left']
    Percent_initial_left=data_run_left["Percent initial left"]
    Percent_Mid_swing_left =data_run_left["Percent mid swing left "]
    Percent_Termial_left =data_run_left["Percent termial left"]

    data_run_right=pd.read_csv(path_right)
    right_start=data_run_right["Right start"]
    right_end=data_run_right["Right end"]
    percent_stance_right=data_run_right["percent_stance_right"]
    Percent_initial_right=data_run_right["Percent initial right"]
    Percent_mid_swing_right =data_run_right["Percent mid swing right "]
    Percent_termial_right=data_run_right["Percent termial right"]

    information_cycle=[]
    phase_split_line=[]
    # print("start - end frame", start_frame, end_frame)
    if check_left :
        cycle=0
        while(cycle < len(left_start)):
            if (left_start[cycle] > start_frame):
                begin_cycle=cycle
                break
            cycle+=1
        while (cycle < len(left_end)):
            if(left_end[cycle]> end_frame):
                final_cycle=cycle
                break
            cycle+=1
        # print("begin - final cycle",begin_cycle, final_cycle)
        for cycle in range(begin_cycle,final_cycle):
            split_line=[]
            information=[]
            information.append(left_start[cycle]-start_frame)
            information.append(left_end[cycle]-start_frame)

            split_line.append(percent_stance_left[cycle]*100)
            split_line.append(Percent_initial_left[cycle]*100+split_line[-1]) # the index of the phase must add with the previous point
            split_line.append(Percent_Mid_swing_left[cycle]*100+split_line[-1])
            
            information_cycle.append(information)
            phase_split_line.append(split_line)
            # phase_split_line.append(split_line)
        return information_cycle , phase_split_line
    else :   # right
        cycle=0
        while(cycle < len(right_start)):
            if (right_start[cycle] > start_frame):
                begin_cycle=cycle
                break
            cycle+=1
        while (cycle < len(right_end)):
            if(right_end[cycle]> end_frame):
                final_cycle=cycle
                break
            cycle+=1
        for cycle in range(begin_cycle,final_cycle):
            split_line=[]
            information=[]
            information.append(right_start[cycle]-start_frame)
            information.append(right_end[cycle]-start_frame)

            split_line.append(percent_stance_right[cycle]*100)
            split_line.append(Percent_initial_right[cycle]*100+split_line[-1]) # the index of the phase must add with the previous point
            split_line.append(Percent_mid_swing_right[cycle]*100+split_line[-1])
            
            information_cycle.append(information)
            phase_split_line.append(split_line)
        return information_cycle , phase_split_line

# '''--------- split all cycle in all file and return them in list data_cycle_model and data_cycle_trajectory --------------------
#             This function also return phase_split_line list which contain information of phases splitting in cycles
#             The returning order : data_cycle_model , data_cycle_trajectory , phase_split_line
# '''

def read_data(different_time_,list_file,List_column_to_read,path_left,path_right):
    
    All_data_frame={}
    print(different_time_, list_file)
    for index,file_path in enumerate(list_file):
        
        data_running= Read_data_runing(file_path)

        data_frame_file=data_running.read_multiple_columns(List_column_to_read) #{"as0" :[[flex],[ab],[rotation]]}

        
        # Find the range of Model and Traject in the file
        Range_model= data_running.Model_output
        Range_traj = data_running.Trajectory

        start_frame_model=different_time_[index]+Range_model[1]
        end_frame_model  =different_time_[index]+Range_model[2]

     
        #determine the correspond begin and final cycle


        for key in data_frame_file.keys() :
            data_cycle_flex     =[]
            data_cycle_ab       =[]
            data_cycle_rotation =[]

            # Check if the column was alreadly in the All_data_frame 
            if key in All_data_frame : # {"column" : [[[model],[1,3],[3,4]],[split_line]]}

                if key[0]=='L' :
                    information_cycl, split_lin=read_cycle_left_right(start_frame=start_frame_model,end_frame=end_frame_model,path_left=path_left,path_right=path_right,check_left=True)
                    flex     = data_frame_file[key][0]
                    ab       = data_frame_file[key][1]
                    rotation = data_frame_file[key][2]
                    for cycle in information_cycl:
                        data_flex     =flex[cycle[0]:cycle[1]+1]  # Split data corresponding to each cycle
                        data_ab       =ab[cycle[0]:cycle[1]+1]
                        data_rotation =rotation[cycle[0]:cycle[1]+1]
                        if len(data_flex)==0 :
                            continue
                        data_cycle_flex.append(data_flex)
                        data_cycle_ab.append(data_ab)
                        data_cycle_rotation.append(data_rotation)
                    All_data_frame[key][0].extend(data_cycle_flex)
                    All_data_frame[key][1].extend(data_cycle_ab)
                    All_data_frame[key][2].extend(data_cycle_rotation)
                    All_data_frame[key][3].extend(split_lin)

                else:
                    information_cycl, split_lin=read_cycle_left_right(start_frame=start_frame_model,end_frame=end_frame_model,path_left=path_left,path_right=path_right,check_left=False)
                    flex     = data_frame_file[key][0]
                    ab       = data_frame_file[key][1]
                    rotation = data_frame_file[key][2]
                    for cycle in information_cycl:
                        data_flex     =flex[cycle[0]:cycle[1]+1]  # Split data corresponding to each cycle
                        data_ab       =ab[cycle[0]:cycle[1]+1]
                        data_rotation =rotation[cycle[0]:cycle[1]+1]
                        if len(data_flex)==0 :
                            continue
                        data_cycle_flex.append(data_flex)
                        data_cycle_ab.append(data_ab)
                        data_cycle_rotation.append(data_rotation)
                    All_data_frame[key][0].extend(data_cycle_flex)
                    All_data_frame[key][1].extend(data_cycle_ab)
                    All_data_frame[key][2].extend(data_cycle_rotation)
                    All_data_frame[key][3].extend(split_lin)
            if key not in All_data_frame :

                if key[0]=='L' :
                    information_cycl, split_lin=read_cycle_left_right(start_frame=start_frame_model,end_frame=end_frame_model,path_left=path_left,path_right=path_right,check_left=True)
                    flex     = data_frame_file[key][0]
                    ab       = data_frame_file[key][1]
                    rotation = data_frame_file[key][2]
                    for cycle in information_cycl:
                        data_flex     =flex[cycle[0]:cycle[1]+1]  # Split data corresponding to each cycle
                        data_ab       =ab[cycle[0]:cycle[1]+1]
                        data_rotation =rotation[cycle[0]:cycle[1]+1]
                        if len(data_flex)==0 :
                            continue
                        data_cycle_flex.append(data_flex)
                        data_cycle_ab.append(data_ab)
                        data_cycle_rotation.append(data_rotation)
                    All_data_frame[key]=[data_cycle_flex,data_cycle_ab,data_cycle_rotation,split_lin]

                else:
                    information_cycl, split_lin=read_cycle_left_right(start_frame=start_frame_model,end_frame=end_frame_model,path_left=path_left,path_right=path_right,check_left=False)
                    flex     = data_frame_file[key][0]
                    ab       = data_frame_file[key][1]
                    rotation = data_frame_file[key][2]
                    for cycle in information_cycl:
                        data_flex     =flex[cycle[0]:cycle[1]+1]  # Split data corresponding to each cycle
                        data_ab       =ab[cycle[0]:cycle[1]+1]
                        data_rotation =rotation[cycle[0]:cycle[1]+1]
                        if len(data_flex)==0 :
                            continue
                        data_cycle_flex.append(data_flex)
                        data_cycle_ab.append(data_ab)
                        data_cycle_rotation.append(data_rotation)
                    All_data_frame[key]=[data_cycle_flex,data_cycle_ab,data_cycle_rotation,split_lin]
               
    return All_data_frame

# #-----------------------interpolate data--> 100% :-----------------------
# ----Transfer elements of an array from string to float----------------
def change_to_number(arr):
    arr_number=[float(element) for element in arr]  
    return arr_number
#----------------scalse and interpolate --------------------------
def interp_sampling(arr, total_frames):
    return np.interp(
        np.linspace(0,1.0,total_frames), 
        np.linspace(0,1.0,len(arr)),
        arr
    )
def interp_shampe_many_data(list_data_cycle):
    inter_data_cycle=[]
    for data_cycle in list_data_cycle :
        number_cy= change_to_number(data_cycle) 
        inter_sham=interp_sampling(number_cy,100)
        inter_data_cycle.append(inter_sham)
    return inter_data_cycle



# interp_data_cycle
def plot_average_with_error(interp_data_cycle,split_cycle,title1,save_folder):
    #calculate the average of period
    average_split_cycle_left = np.mean(split_cycle[0], axis=0)
    average_split_cycle_right = np.mean(split_cycle[1], axis=0)

    # Calculate the average and standard error
    average_left = np.mean(interp_data_cycle[0], axis=0)
    average_right = np.mean(interp_data_cycle[1], axis=0)
    
    phase_labels =["Stance phase","Initial swing","Mid swing", "Termial swing"]
    # Create x-axis values
    x_left = np.arange(len(average_left))
    x_right = np.arange(len(average_right))

    # Plot the average with shaded standard error using Seaborn
    plt.figure(figsize=(10, 6))

    for i, a in enumerate(average_split_cycle_left):
        plt.axvline(x=a , color='red', linestyle='-')
    for i, a in enumerate(average_split_cycle_right):
        plt.axvline(x=a , color='blue', linestyle='-')
    # Add phase labels to the graph
    plt.axhline(y=0,color='red',linestyle='-')
    bottom = min(0,np.min(interp_data_cycle[0]),np.min(interp_data_cycle[1]))
    peak=max(np.max(interp_data_cycle[0]),np.max(interp_data_cycle[1]))
    if bottom==0 :
        y_offset=bottom-(peak-bottom)/25
    else :
        y_offset=bottom-(peak-bottom)/25
    for i, label in enumerate(phase_labels):
        if i > 0:
            plt.text(average_split_cycle_left[i - 1], y_offset, label, color='red', ha='left')
        else:
            plt.text(0, y_offset, label, color='red', ha='left')
    # Plot the split mode
    custom_xticks = np.arange(0, 101)
    custom_xticklabels=[]
    for i in range (10):
        custom_xticklabels.append(10*i)
        custom_xticklabels.extend(['']*9)
        # custom_xticklabels.
    custom_xticklabels.append(100)
    plt.xticks(custom_xticks, custom_xticklabels)

    plt.fill_between(x_left, np.min(interp_data_cycle[0], axis=0), np.max(interp_data_cycle[0], axis=0), color='pink', alpha=0.2)
    sns.lineplot(x=x_left, y=average_left, color='red', label='Average left')
    plt.fill_between(x_right, np.min(interp_data_cycle[1], axis=0), np.max(interp_data_cycle[1], axis=0), color='lightblue', alpha=0.2)
    sns.lineplot(x=x_right, y=average_right, color='blue', label='Average right')
    # plt.fill_between(x, average - std_devi, average + std_devi, color='lightblue', alpha=0.4, label='Standard deviation')
    plt.xlabel('Nomalized (percent)',size=16)
    plt.ylabel('Angle (degree)',size=16, labelpad=25)
    plt.title(title1,size=20)
    plt.legend()
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Construct the full file path
    filename = os.path.join(save_folder, title1.replace("/", "_") + ".png")
    
    # Save the plot as an image
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the plot to free up resources
    # plt.show()
def plot_all_data(All_data,folder):
    index=0
    list_key=list(All_data.keys())
    subfolder_name = "GraphXXXXXXXXX" 
    subfolder_path = os.path.join(folder, subfolder_name)
    while index < len(list_key) -1 :
        key_left=list_key[index]
        key_right= list_key[index+1]
        if key_left[1:]== "PelvisAngles" :
            for i in range(len(All_data[list_key[index]])-1) :
                if i==0:
                    title= key_left[1:] +"_tilt"
                elif i ==1 :
                    title=key_left[1:]+"_obliquity"
                elif i==2 :
                    title=key_left[1:]+ "_rotation"
                left_leg= interp_shampe_many_data(All_data[key_left][i])
                right_leg=interp_shampe_many_data(All_data[key_right][i])

                plot_average_with_error([left_leg,right_leg],[All_data[key_left][-1],All_data[key_right][-1]],title,subfolder_path)
        else :

            for i in range(len(All_data[list_key[index]])-1) :
                if i==0:
                    title= key_left[1:] +"_flex/ext"
                elif i ==1 :
                    title=key_left[1:]+"_ab/add"
                elif i==2 :
                    title=key_left[1:]+ "_rotation"
                left_leg= interp_shampe_many_data(All_data[key_left][i])
                right_leg=interp_shampe_many_data(All_data[key_right][i])

                plot_average_with_error([left_leg,right_leg],[All_data[key_left][-1],All_data[key_right][-1]],title,subfolder_path)
        index+=2


# This function will take the mean value of angle of each type of angle and transfer them to the csv file
def export_data(All_data,list_file,folder_path): #type of All_data is dic which out put of read_data function
    data_aframe={}
    data_start_of_phase={}
    path_data = os.path.join(folder_path, list_file[0][:-6] + "_averge_std.csv")
    path_data_start = os.path.join(folder_path, list_file[0][:-6] + "start_of_phase.csv")
    for key in All_data.keys():
        for i in range(len(All_data[key])-1) :
            data_key = interp_shampe_many_data(All_data[key][i])
            average = np.mean(data_key, axis=0)
            std_devi = np.std(data_key, axis=0) 
            if i ==0 :
                flex="X" 
                std_flex="std_X"
                data_aframe[key]={flex:average, std_flex :std_devi}
            elif i==1 :
                ab="Y"
                std_ab="std_Y"
                data_aframe[key].update({ab : average ,std_ab : std_devi})
            elif i==2 :
                rotation ="Z"
                std_rotation = "std_Z"
                data_aframe[key].update({rotation : average , std_rotation : std_devi})
        data_key=All_data[key][3]
        average = np.mean(data_key, axis=0)
        std_devi = np.std(data_key, axis=0) 
        data_aframe[key].update({"Split line" : average})
    
    with open(path_data, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the first header line
        row_header1=[]
        for keyss in data_aframe.keys() :
            row_header1.extend([keyss , "", "" ,"" ,"" ,"" ,""  ])
        writer.writerow(row_header1)

        # Write the second header line
        writer.writerow(["X","std_X" ,"Y","std_Y", "Z", "std_Z","Split_line"] * len(data_aframe.keys()))
        
        for i in range(100) :
            row1=[]
            for keys in data_aframe.keys() :
                a=data_aframe[keys]
                for keyss in a.keys():
                    if i < len(data_aframe[keys][keyss]) :
                        row1.append(data_aframe[keys][keyss][i])
                    else :
                        row1.append("")
            writer.writerow(row1)

    for key in All_data.keys():
        for i in range(len(All_data[key])-1) :
            initial=int(data_aframe[key]["Split line"][0])
            mid    =int(data_aframe[key]["Split line"][1])
            Termin =int(data_aframe[key]["Split line"][2])
            if i ==0 :
                flex="X" 
                std_flex="std_X"
                
                start_phase=    [data_aframe[key][flex][0],    data_aframe[key][flex][initial],     data_aframe[key][flex][mid],     data_aframe[key][flex][Termin]]
                std_start_phase=[data_aframe[key][std_flex][0],data_aframe[key][std_flex][initial], data_aframe[key][std_flex][mid], data_aframe[key][std_flex][Termin]]

                data_start_of_phase[key]={flex:start_phase, std_flex :std_start_phase}
            elif i==1 :
                ab="Y"
                std_ab="std_Y"

                start_phase=    [data_aframe[key][ab][0],    data_aframe[key][ab][initial],     data_aframe[key][ab][mid],     data_aframe[key][ab][Termin]]
                std_start_phase=[data_aframe[key][std_ab][0],data_aframe[key][std_ab][initial], data_aframe[key][std_ab][mid], data_aframe[key][std_ab][Termin]]

                data_start_of_phase[key].update({ab : start_phase ,std_ab : std_start_phase})
            elif i==2 :
                rotation ="Z"
                std_rotation = "std_Z"

                start_phase=    [data_aframe[key][rotation][0],    data_aframe[key][rotation][initial],     data_aframe[key][rotation][mid],     data_aframe[key][rotation][Termin]]
                std_start_phase=[data_aframe[key][std_rotation][0],data_aframe[key][std_rotation][initial], data_aframe[key][std_rotation][mid], data_aframe[key][std_rotation][Termin]]

                data_start_of_phase[key].update({rotation : start_phase , std_rotation : std_start_phase})
    # print(data_start_of_phase)

    with open(path_data_start, mode="w", newline="") as file:
        writer = csv.writer(file)

        header_row1 = ["Degree at start of phase", "Stance", "", "", "", "Initial swing", "", "", "", "Mid swing", "","","", "Terminal swing","","",""]
        header_row2 = ["", "Left", "", "Right", "", "Left", "", "Right", "", "Left", "", "Right", ""]
        header_row3= ["","Mean","STD","Mean","STD","Mean","STD","Mean","STD","Mean","STD","Mean","STD","Mean","STD","Mean","STD",] 
        writer.writerow(header_row1)
        writer.writerow(header_row2)
        writer.writerow(header_row3)

        index_key=0
        keys = list(data_start_of_phase.keys())

        while index_key < len(keys) - 1:
            keyL = keys[index_key]
            keyR = keys[index_key + 1]

            keyss =keyL[1:-6]
            # print(keyR, keyL,"keyyyyyyyyyyyy")

            row_flex= [keyss+" flex/ext", data_start_of_phase[keyL]["X"][0] , data_start_of_phase[keyL]["std_X"][0],data_start_of_phase[keyR]["X"][0] , data_start_of_phase[keyR]["std_X"][0],
                                        data_start_of_phase[keyL]["X"][1] , data_start_of_phase[keyL]["std_X"][1],data_start_of_phase[keyR]["X"][1] , data_start_of_phase[keyR]["std_X"][1] ,
                                        data_start_of_phase[keyL]["X"][2] , data_start_of_phase[keyL]["std_X"][2],data_start_of_phase[keyR]["X"][2] , data_start_of_phase[keyR]["std_X"][2] ,
                                        data_start_of_phase[keyL]["X"][3] , data_start_of_phase[keyL]["std_X"][3],data_start_of_phase[keyR]["X"][3] , data_start_of_phase[keyR]["std_X"][3] ,
                       ]
            row_ab= [keyss+" ab/add",    data_start_of_phase[keyL]["Y"][0] , data_start_of_phase[keyL]["std_Y"][0], data_start_of_phase[keyR]["Y"][0] , data_start_of_phase[keyR]["std_Y"][0],
                                        data_start_of_phase[keyL]["Y"][1] , data_start_of_phase[keyL]["std_Y"][1], data_start_of_phase[keyR]["Y"][1] , data_start_of_phase[keyR]["std_Y"][1] ,
                                        data_start_of_phase[keyL]["Y"][2] , data_start_of_phase[keyL]["std_Y"][2], data_start_of_phase[keyR]["Y"][2] , data_start_of_phase[keyR]["std_Y"][2] ,
                                        data_start_of_phase[keyL]["Y"][3] , data_start_of_phase[keyL]["std_Y"][3], data_start_of_phase[keyR]["Y"][3] , data_start_of_phase[keyR]["std_Y"][3] ,
                       ]
            row_Rota=[keyss+" rotation", data_start_of_phase[keyL]["Z"][0] , data_start_of_phase[keyL]["std_Z"][0], data_start_of_phase[keyR]["Z"][0] , data_start_of_phase[keyR]["std_Z"][0],
                                        data_start_of_phase[keyL]["Z"][1] , data_start_of_phase[keyL]["std_Z"][1], data_start_of_phase[keyR]["Z"][1] , data_start_of_phase[keyR]["std_Z"][1] ,
                                        data_start_of_phase[keyL]["Z"][2] , data_start_of_phase[keyL]["std_Z"][2], data_start_of_phase[keyR]["Z"][2] , data_start_of_phase[keyR]["std_Z"][2] ,
                                        data_start_of_phase[keyL]["Z"][3] , data_start_of_phase[keyL]["std_Z"][3], data_start_of_phase[keyR]["Z"][3] , data_start_of_phase[keyR]["std_Z"][3] ,
                       ]
            writer.writerow(row_flex)
            writer.writerow(row_ab)
            writer.writerow(row_Rota)

            index_key+=2

    return data_aframe

# different_time_from03=different_time[3:5]
# # different_time_from03.append(different_time[6])
# # list_file=["TuanAnhRunning03.csv","TuanAnhRunning04.csv","TuanAnhRunning05.csv","TuanAnhRunning06.csv","TuanAnhRunning07.csv"]
# list_file=["TuanAnhRunning03.csv","TuanAnhRunning04.csv"]
# List_to_read=["Hip","Knee"]
# All_data= read_data(different_time_from03,list_file=list_file,List_column_to_read=List_to_read,path_left="data_run__left_leg_processed__.csv",path_right="data_run__right_leg_processed__.csv")
# # print(All_data)

# # data_aframe= export_data(All_data,list_file)
# # print(data_aframe)

# plot_all_data(All_data)

# [137, 1621, 2341, 3279, 4155, 3279] ['TuanAnhRunning03.csv', 'TuanAnhRunning04.csv', 'TuanAnhRunning05.csv', 'TuanAnhRunning06.csv', 'TuanAnhRunning07.csv']

