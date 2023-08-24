import tkinter as tk 
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import os

from plot import read_data ,plot_all_data, export_data, calculate_the_different_time
from Classify_period import process_file
import time

def xlsx_to_csv():
    try:
        # Ask the user to select an XLSX file using the file dialog
        xlsx_file_path = filedialog.askopenfilename(filetypes=[("XLSX files", "*.xlsx")])

        # Check if a file was selected
        if not xlsx_file_path:
            print("No XLSX file selected.")
            return

        # Read the XLSX file using pandas
        df = pd.read_excel(xlsx_file_path)

        # Get the name of the XLSX file without the extension
        file_name = os.path.splitext(os.path.basename(xlsx_file_path))[0]

        # Create the output CSV file path by replacing the extension with ".csv"
        csv_file_path = os.path.join(os.path.dirname(xlsx_file_path), file_name + ".csv")

        # Write the data to the CSV file
        df.to_csv(csv_file_path, index=False)

        print(f"XLSX file converted to CSV: {csv_file_path}")
    except Exception as e:
        print("Error occurred during conversion:", e)

# def open_file():
#     file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("XLSX files", "*.xlsx")])
#     mass = int(mass_entry.get())
#     process_file(file_path,mass)



    
List_file_run = []
path_right = ""
path_left = ""
folder_path=''
def find_files_with_text(directory, text):
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if text in file:
                matching_files.append(os.path.join(root, file))
    return matching_files
def plot_data():
    
    a= len(List_file_run)
    All_data= read_data(different_time[:a],List_file_run,["Hip","Knee","Pelvis","Ankle"],path_left,path_right)
    plot_all_data(All_data,folder_path)

def export_data_to_csvfile():
    # different_time= calculate_the_different_time(file_different_time,100)
    a= len(List_file_run)
    All_data= read_data(different_time[:a],List_file_run,["Hip","Knee","Pelvis","Ankle"],path_left,path_right)
    export_data(All_data,List_file_run,folder_path)
file_different_time = None
def main_function():
    global List_file_run, path_right, path_left,different_time ,folder_path # Use the global variables

    # Step 1: Choose a directory using file dialog
    folder_path = filedialog.askdirectory()

    # Step 2: Find the file with "MR3" and get its directory
    
    for file in os.listdir(folder_path):
        if "MR3" in file and file.endswith(".csv"):
            file_different_time = os.path.join(folder_path, file)
            break
    
    if file_different_time:
        # Step 3: Process the initial MR3 file
        mass = mass_entry.get()
        if mass:
            mass=int(mass)
            process_file(file_different_time, mass)
        else:
            process_file(file_different_time)
        different_time= calculate_the_different_time(file_different_time,100)
       
        # Step 4: Find the processed right and left leg files
        file_prefix = os.path.splitext(file_different_time)[0]
        print("File Prefix:", file_prefix)
        processed_right_leg_files = find_files_with_text(folder_path, "__right_leg_processed__")
        print("Processed Right Leg Files:", processed_right_leg_files)
        if processed_right_leg_files:
            path_right = processed_right_leg_files[0]
        else:
            print("No processed right leg file found.")

        processed_left_leg_files = find_files_with_text(folder_path, "__left_leg_processed__")
        print("Processed Left Leg Files:", processed_left_leg_files)
        if processed_left_leg_files:
            path_left = processed_left_leg_files[0]
        else:
            print("No processed left leg file found.")
        
        # Step 5: Find and append Nexus files to the list
        List_file_run = find_files_with_text(folder_path, "Nexus")
        print(List_file_run)
    else:
        print("No matching MR3 file found.")
    plot_data()
    export_data_to_csvfile()
# Create the main window

# Create the main window
window = tk.Tk()
window.geometry("400x300")

# Create tab control
tab_control = ttk.Notebook(window)


# Create widgets for "Classify Period" tab
button_select_file = tk.Button( text="Transfer to csv file", command=xlsx_to_csv)
button_select_file.pack()
mass_label = tk.Label( text="Enter mass (kg):")
mass_label.pack()
mass_entry = tk.Entry()
mass_entry.pack(anchor="center", pady=10)
button_select_file = tk.Button( text="Select The Folder", command=main_function)
button_select_file.pack()

# Pack the tab control
tab_control.pack(expand=True, fill=tk.BOTH)

# Start the main event loop
window.mainloop()

