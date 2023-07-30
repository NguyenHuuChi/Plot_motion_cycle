import tkinter as tk 
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import os

from plot import read_data ,plot_all_data, export_data, calculate_the_different_time
from Classify_period import process_file

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

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("XLSX files", "*.xlsx")])
    mass = int(mass_entry.get())
    process_file(file_path,mass)



path_left =""
def read_path_left():
    global path_left
    path_left=filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])

path_right=''
def read_path_right():
    global path_right
    path_right=filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])


def Select_processed_left_right_file():
    read_path_left()
    read_path_right()

List_file_run =[]
def open_multiple_file():
    global List_file_run
    list_pat = filedialog.askopenfilenames(filetypes=[("CSV files","*.csv")])
    List_file_run= list(list_pat)
    print(List_file_run)

file_different_time=''
def open_single_file():
    global file_different_time
    file_different_time= filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
    

def plot_data():
    different_time= calculate_the_different_time(file_different_time,100)
    a= len(List_file_run)
    All_data= read_data(different_time[:a],List_file_run,["Hip","Knee","Pelvis"],path_left,path_right)
    plot_all_data(All_data)

def export_data_to_csvfile():
    different_time= calculate_the_different_time(file_different_time,100)
    a= len(List_file_run)
    All_data= read_data(different_time[:a],List_file_run,["Hip","Knee"],path_left,path_right)
    export_data(All_data,List_file_run)
# Create the main window

# Create the main window
window = tk.Tk()
window.geometry("400x300")

# Create tab control
tab_control = ttk.Notebook(window)

# Create "Classify Period" tab
classify_period_frame = ttk.Frame(tab_control)
tab_control.add(classify_period_frame, text="Classify Period")

# Create widgets for "Classify Period" tab
button_select_file = tk.Button(classify_period_frame, text="Transfer to csv file", command=xlsx_to_csv)
button_select_file.pack()
mass_label = tk.Label(classify_period_frame, text="Enter mass (kg):")
mass_label.pack()
mass_entry = tk.Entry(classify_period_frame)
mass_entry.pack(anchor="center", pady=10)
button_select_file = tk.Button(classify_period_frame, text="Select CSV File", command=open_file)
button_select_file.pack(anchor="center")

# Create "Export and Plot Data" tab
export_plot_frame = ttk.Frame(tab_control)
tab_control.add(export_plot_frame, text="Export and Plot Data")

# Create widgets for "Export and Plot Data" tab
button_select_processed_file = tk.Button(export_plot_frame, text="Select Processed Left/Right File", command=Select_processed_left_right_file)
button_select_processed_file.pack()
button_select_single_file = tk.Button(export_plot_frame, text="Select File Containing Different Time", command=open_single_file)
button_select_single_file.pack()
button_select_multiple_files = tk.Button(export_plot_frame, text="Select List of Running Files", command=open_multiple_file)
button_select_multiple_files.pack()
button_plot_data = tk.Button(export_plot_frame, text="Plot", command=plot_data)
button_plot_data.pack()
button_export_data = tk.Button(export_plot_frame, text="Export Data", command=export_data_to_csvfile)
button_export_data.pack()

# Pack the tab control
tab_control.pack(expand=True, fill=tk.BOTH)

# Start the main event loop
window.mainloop()

