import numpy as np
import pandas as pd
import os
import csv
import tkinter as tk
from tkinter import filedialog

# def open_file():
#     file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
#     mass = mass_entry.get()
#     process_file(file_path,mass)
def find_zero(ind_start,arr,mass) :
    i=ind_start
    while arr[i+5]!= 0 :
        if (arr[i+5] >= mass *2 *10):
            while arr[i+5] >= mass *2 *10 :
                i+=5
            range= i -ind_start
            i =ind_start+ range*2
            if (arr[i] ==0):
                while arr[i]==0 :
                    i-=1
                return i+1
        else :
            i+=5
    while arr[i]!=0 :
        i+=1
    return i


# -----------return the index where the force of one of two leg is not equal to 0-------------
def find_not_zero(ind_start, arr):
    i=ind_start
    while(arr[i+4] ==0):
        i+=4
    while arr [i]==0 :
        i+=1
    return i

#---------------Function process data ----------------------------
def process_file(file_path,mass):
    
    # mass=60
    data = pd.read_csv(file_path,skiprows=3)
    columns=data.columns
    left_column=columns[3]
    right_column=columns[4]
    left_force=data[left_column]
    right_force=data[right_column]


    left_circle=[]
    left_start=[]
    left_end=[]
    stance_left=[]
    percent_stance_left=[]
    Initial_swing_left=[]
    percent_Initial_swing_left=[]
    Mid_swing_left=[]
    percent_mid_swing_left=[]
    Termial_swing_left=[]
    percent_termial_swing_left=[]

    right_circle=[]
    right_start=[]
    right_end=[]
    stance_right=[]
    percent_stance_right=[]
    Initial_swing_right=[]
    percent_Initial_swing_right=[]
    Mid_swing_right=[]
    percent_mid_swing_right=[]
    Termial_swing_right=[]
    percent_termial_swing_right=[]

    left_number_cycle=0
    right_number_cycle =0
    #---------------- Find the frame that start the circle----------------

    number_frame =1
    check_start_left_right =-1 # left 0 ;  right 1
    while left_force[number_frame] == 0 and right_force[number_frame] ==0 :
        number_frame +=1
        if left_force[number_frame]!=0 :
            check_start_left_right=0
        elif right_force[number_frame]!= 0 :
            check_start_left_right =1

    #------------------------ split part of data ---------------------

    # Left

    if check_start_left_right ==0 :
        while number_frame<= len(left_force)-100 : #each while  loop will be a cycle of both right and left leg
            left_start.append(number_frame)
            stance_left.append(number_frame)
            left_number_cycle+=1
            left_circle.append(left_number_cycle)                            # add number of cycle        
            

            # finish stance phase and start initial swing in left leg - termial swing right
            zero_left=find_zero(number_frame, left_force,mass)
            Initial_swing_left.append(zero_left)
            if(right_number_cycle!=0):
                Termial_swing_right.append(zero_left)                       # Initial left = terminal right
            number_frame =zero_left                                     # change the number of frame                               

            # Finish initial swing (left) start mid swing left - stance right
            start_right= find_not_zero(number_frame, right_force)       # move to find on right side
            number_frame=start_right
            Mid_swing_left.append(number_frame)
            stance_right.append(number_frame)
            right_start.append(number_frame)
            if(right_number_cycle !=0 ):
                right_end.append (number_frame-1)
            right_number_cycle+=1
            right_circle.append(right_number_cycle)

            #Finish the Midswing (left)- finish the stance (right) -start the  termial swing  (left)  and initial swing (right)
            start_terminal_swing = find_zero(number_frame, right_force,mass)
            number_frame= start_terminal_swing
            Termial_swing_left.append(start_terminal_swing)                 
            Initial_swing_right.append(start_terminal_swing)            # Termial left = Initial right

            #Finish the terminal swing (left) - start the stance left and mid swing right
            start_new_circle= find_not_zero(number_frame,left_force)
            number_frame=  start_new_circle
            Mid_swing_right.append(start_new_circle)
            left_end.append(number_frame-1)
        right_circle.pop()
        right_start.pop()
        stance_right.pop()
        Initial_swing_right.pop()
        Mid_swing_right.pop()
        for i in range(len(left_circle)):
            total_frame=left_end[i]-left_start[i]+1
            percent_stance_left.append((Initial_swing_left[i]-stance_left[i])/total_frame)
            percent_Initial_swing_left.append((Mid_swing_left[i]-Initial_swing_left[i])/total_frame)
            percent_mid_swing_left.append((Termial_swing_left[i]-Mid_swing_left[i])/total_frame)
            percent_termial_swing_left.append((left_end[i]-Termial_swing_left[i]+1)/total_frame)
        for i in range(len(right_circle)):
            total_frame=right_end[i]-right_start[i]+1
            percent_stance_right.append((Initial_swing_right[i]-stance_right[i])/total_frame)
            percent_Initial_swing_right.append((Mid_swing_right[i]-Initial_swing_right[i])/total_frame)
            percent_mid_swing_right.append((Termial_swing_right[i]-Mid_swing_right[i])/total_frame)
            percent_termial_swing_right.append((right_end[i]-Termial_swing_right[i]+1)/total_frame)

    # right
    elif check_start_left_right==1 : 
        while number_frame<= len(left_force)-100 : #each while  loop will be a cycle of both right and left leg
            right_start.append(number_frame)
            stance_right.append(number_frame)
            right_number_cycle+=1
            right_circle.append(right_number_cycle)                            # add number of cycle        
            

            # finish stance phase and start initial swing in left leg - termial swing right
            zero_right=find_zero(number_frame, right_force,mass)
            Initial_swing_right.append(zero_right)
            if (left_number_cycle !=0 ):
                Termial_swing_left.append(zero_right)                       # Initial left = terminal right
            number_frame =zero_right                                        # change the number of frame                               

            # Finish initial swing (left) start mid swing left - stance right
            start_left= find_not_zero(number_frame, left_force)       # move to find on right side
            number_frame=start_left
            Mid_swing_right.append(number_frame)
            stance_left.append(number_frame)
            left_start.append(number_frame)
            if(left_number_cycle !=0 ):
                left_end.append (number_frame-1)
            left_number_cycle+=1
            left_circle.append(left_number_cycle)
            

            #Finish the Midswing (left)- finish the stance (right) -start the  termial swing  (left)  and initial swing (right)
            start_terminal_swing = find_zero(number_frame, left_force,mass)
            number_frame= start_terminal_swing
            Termial_swing_right.append(start_terminal_swing)                 
            Initial_swing_left.append(start_terminal_swing)            # Termial left = Initial right

            #Finish the terminal swing (left) - start the stance left and mid swing right
            start_new_circle= find_not_zero(number_frame,right_force)
            number_frame=  start_new_circle
            Mid_swing_left.append(start_new_circle)
            right_end.append(number_frame-1)    

        left_circle.pop()
        left_start.pop()
        stance_left.pop()
        Initial_swing_left.pop()
        Mid_swing_left.pop()


        for i in range(len(left_circle)):
            total_frame=left_end[i]-left_start[i]+1
            percent_stance_left.append((Initial_swing_left[i]-stance_left[i])/total_frame)
            percent_Initial_swing_left.append((Mid_swing_left[i]-Initial_swing_left[i])/total_frame)
            percent_mid_swing_left.append((Termial_swing_left[i]-Mid_swing_left[i])/total_frame)
            percent_termial_swing_left.append((left_end[i]-Termial_swing_left[i]+1)/total_frame)
        for i in range(len(right_circle)):
            total_frame=right_end[i]-right_start[i]+1
            percent_stance_right.append((Initial_swing_right[i]-stance_right[i])/total_frame)
            percent_Initial_swing_right.append((Mid_swing_right[i]-Initial_swing_right[i])/total_frame)
            percent_mid_swing_right.append((Termial_swing_right[i]-Mid_swing_right[i])/total_frame)
            percent_termial_swing_right.append((right_end[i]-Termial_swing_right[i]+1)/total_frame)
        
    data_out_right={"Right cycle":right_circle, "Right start":right_start , "Right end" : right_end, 
            "Stance right " :stance_right,"percent_stance_right":percent_stance_right  ,
            "Initial swing right" : Initial_swing_right,"Percent initial right" :percent_Initial_swing_right ,
            "Mid swing right": Mid_swing_right, "Percent mid swing right ":percent_mid_swing_right,
            "Terminal right" : Termial_swing_right , "Percent termial right": percent_termial_swing_right,}
    data_out_left={
            "Left cycle" : left_circle , "Left start" : left_start , "Left end" : left_end,
            "Stance left": stance_left, "percent_stance_left":percent_stance_left,
            "Initial swing left" : Initial_swing_left,"Percent initial left" :percent_Initial_swing_left ,
            "Mid swing left": Mid_swing_left, "Percent mid swing left ":percent_mid_swing_left,
            "Terminal left" : Termial_swing_left , "Percent termial left": percent_termial_swing_left}
    
    left_processed_data_path=file_path[:-4]+"__left_leg_processed__.csv"
    df=pd.DataFrame(data_out_left)
    df.to_csv(left_processed_data_path, index=False)   

    right_processed_data_path=file_path[:-4]+"__right_leg_processed__.csv"
    df=pd.DataFrame(data_out_right)
    df.to_csv(right_processed_data_path, index=False) 





# # Create the main window
# window = tk.Tk()
# window.geometry("400x300")

# # Create a label and entry for user input
# mass_label = tk.Label(window, text="Enter mass (kg):")
# mass_label.pack()
# mass_entry = tk.Entry(window)
# mass_entry.pack(anchor="center",pady=10)

# # Create a button to open the file dialog
# button = tk.Button(window, text="Select CSV File", command=open_file)
# button.pack(anchor="center")

# # Start the main event loop
# window.mainloop()








