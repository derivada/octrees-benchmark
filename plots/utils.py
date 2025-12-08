# Common utilities for plot generators
import os
from typing import Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import matplotlib as mpl
import glob
from datetime import datetime
import csv

OUTPUT_DPI = 300

def set_default_style(font_size=14, axes_labelsize=14, xtick_labelsize=14, 
                      ytick_labelsize=14, legend_fontsize=14, legend_title_fontsize=16):
    plt.style.use(["science", "grid"])
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': axes_labelsize,
        'xtick.labelsize': xtick_labelsize,
        'ytick.labelsize': ytick_labelsize,
        'legend.fontsize': legend_fontsize,
        'legend.title_fontsize': legend_title_fontsize,  
        
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.linestyle': '-',
        'grid.linewidth': 0.4,
        'grid.color': '#CCCCCC',
        
        'axes.axisbelow': True,
        'figure.figsize': (6, 4),
        'figure.dpi': 100, # poner a 300 para calidad final
        
        'lines.linewidth': 1.0,
        'lines.markersize': 4,

        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
    })
    mpl.rcParams['text.usetex'] = True

# reads a locality data file with name <DATASET_NAME>-<ENCODER>-locality.csv 
def get_locality_file(data_path: str, cloud_name: str, encoder: str = "none") -> pd.DataFrame:
    file_name = cloud_name + "-" + encoder + "-locality.csv"
    file_path = os.path.join(data_path, cloud_name, file_name)
    print(f"Loading results file: {file_path}")
    df = pd.read_csv(file_path)
    return df.sort_values("distance")

# reads all locality files for given encoders and puts them into a dict
def get_all_locality_files(data_path: str, cloud_name: str, encoders: Tuple[str, ...] = ("none", "mort", "hilb")) -> Dict[str, pd.DataFrame]:    
    print(data_path)
    datasets = {}
    for enc in encoders:
        datasets[enc] = get_locality_file(data_path, cloud_name, enc)
    return datasets

def get_dataset_file(data_path, cloud_name, timestamp="latest"):
    csv_folder = os.path.join(data_path, cloud_name)
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    df = None

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the folder: {csv_folder}")

    if timestamp == "latest":
        latest_file = None
        latest_time = None

        for file in csv_files:
            try:
                filename = os.path.basename(file)
                ts_str = '-'.join(filename.split('-')[1:]).replace('.csv', '')
                ts = datetime.strptime(ts_str, "%Y-%m-%d-%H:%M:%S")
                if latest_time is None or ts > latest_time:
                    latest_time = ts
                    latest_file = file
            except Exception as e:
                print(f"Skipping file with invalid timestamp format: {filename} ({e})")
                continue

        if latest_file:
            print(f"Loading results file: {latest_file}")
            df = pd.read_csv(latest_file)
        else:
            raise FileNotFoundError(f"No valid timestamped files found in: {csv_folder}")

    else:
        for file in csv_files:
            try:
                filename = os.path.basename(file)
                ts_str = '-'.join(filename.split('-')[1:]).replace('.csv', '')
                if timestamp == ts_str:
                    print(f"Loading file: {file}")
                    df = pd.read_csv(file)
                    break
            except Exception as e:
                print(f"Skipping file with invalid format: {filename} ({e})")
                continue

        if df is None:
            raise FileNotFoundError(f"File with date '{timestamp}' not found in folder: {csv_folder}")

    return df


def read_multiple_datasets(data_path, clouds_datasets):
    dfs = {}
    for cloud, dataset in clouds_datasets.items():
        dfs[cloud] = get_dataset_file(data_path, cloud)
    return dfs

# Read point cloud from a CSV file
def read_points(data_path: str, cloud_name: str):
    file_path = os.path.join(data_path, cloud_name + ".csv")
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        points = np.array([list(map(float, row)) for row in reader])
    return points

# Output a figure
def output_fig(fig: plt.Figure, output_folder: str, filename: str) -> None: # type: ignore
    # Create folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Save the figure
    filepath = os.path.join(output_folder, filename + ".pdf")
    fig.savefig(filepath, dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close(fig)



    