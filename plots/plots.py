# Functions for generating the plots

import os
from typing import Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scienceplots
import matplotlib as mpl
from utils import *
import scipy.stats as stats

def plot_locality_hist(
    datasets: Dict[str, pd.DataFrame],
    bin_size: float,
    min_dist: float,
    max_dist: float,
    use_log: bool
) -> plt.Figure: # type: ignore
    fig, ax = plt.subplots(figsize=(6,4))
    cmap = plt.get_cmap("tab10")

    for i, (encoder, df) in enumerate(datasets.items()):
        encoder_str = r"$H_{" + encoder + r"}$"
        ax.bar(
            df["distance"],
            df["count"],
            alpha=0.6,
            width=bin_size,
            align="edge",
            label=encoder_str,
            color=cmap(i % 10),
            rasterized=True
        )

    ax.set_xlabel(rf"Index distance $|i-j|$")
    ax.set_xlim([min_dist, max_dist]) # type: ignore
    
    if use_log:
        ax.set_yscale("log") # log scale for the y-axis

    ax.legend(
        loc="upper right",
        framealpha=0.9,
        borderpad=0.4,
        handletextpad=0.4,
        labelspacing=0.25,
        title=None 
    )
    fig.tight_layout()

    return fig

def plot_locality_kde(
    datasets: Dict[str, pd.DataFrame],
    min_dist: float,
    max_dist: float
) -> None:
    plt.figure(figsize=(10,6))
    cmap = plt.get_cmap("tab10")

    for i, (encoder, df) in enumerate(datasets.items()):
        encoder_str = r"$H_{" + encoder + r"}$"
        sns.kdeplot(
            x=df["distance"],
            weights=df["count"],
            label=encoder_str,
            color=cmap(i % 10),
            fill=True,
            alpha=0.5,
            clip=(0, None)
        )

    plt.xlabel(rf"Index distance $|i-j|$")
    plt.ylabel("Density")
    plt.xlim([min_dist, max_dist])    
    plt.legend()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_runtime_comparison(data_path, cloud, operations, types_info,
                                encoder = "all", kernel="all", 
                                show_warmup_time=False, cols = 1, figsz = (7,10)):
    df = get_dataset_file(data_path, cloud)
    
    # Filter the dataset for the specified operation
    df = df[df['operation'].isin(operations)]
    
    # Get unique radii
    radii = sorted(df['radius'].unique())
    
    # Filter by specific kernel argument if provided
    if kernel != "all":
        if isinstance(kernel, str):
            kernel = [kernel]
        df = df[df["kernel"].isin(kernel)]

    # --- FIX START: Rename variable to unique_kernels ---
    unique_kernels = df['kernel'].unique()
    # ----------------------------------------------------

    if encoder != "all":
        df = df[df["encoder"] == encoder]

    # Create subplot grid with reduced vertical spacing
    fig, axes = plt.subplots(int(np.ceil(len(radii)/cols)), cols, figsize=figsz,
                                gridspec_kw={'hspace': 0.3}, squeeze = False)

    # Bar spacing parameters
    bar_width = 0.15
    group_width = bar_width * len(types_info["available_types"])
    group_gap = 0.2
    legend_handles, legend_labels = [], []
    curr_row, curr_col = 0, 0
    
    # First loop through each radius
    for radius_idx, radius in enumerate(radii):
        ax = axes[curr_row][curr_col]
        radius_data = df[df['radius'] == radius]
        
        kernel_labels = []
        
        # --- FIX START: Iterate over unique_kernels using a new variable name (k_name) ---
        for i, k_name in enumerate(unique_kernels):
            kernel_data = radius_data[radius_data['kernel'] == k_name]
            
            if kernel_data.empty:
                kernel_labels.append(f'No data')
                continue
            
            avg_total = kernel_data['avg_result_size'].iloc[0]
            # Updated label to use k_name
            kernel_labels.append(f"$\\mathcal{{N}}_{{{k_name}}}$\n$\\mu = {avg_total:.0f}$")
            
            # Iterate through octree implementations
            for j, (_, params) in enumerate(types_info["available_types"].iterrows()):
                key = tuple(params[col] for col in types_info["type_parameters"])
                
                # Filter dynamically
                octree_data = kernel_data[
                    (kernel_data[types_info["type_parameters"]] == pd.Series(key, index=types_info["type_parameters"])).all(axis=1)
                ]
                
                if octree_data.empty:
                    continue
                    
                means = octree_data['mean'].values
                stdevs = octree_data['stdev'].values
                warmup_times = octree_data['warmup_time'].values
                
                x_pos = i * (group_width + group_gap) + j * bar_width
                
                # Main execution time bar
                bar = ax.bar(x_pos, means[0], bar_width,
                                color=types_info["palette"][key])
                
                # Warmup time bar
                if show_warmup_time:
                    ax.bar(x_pos, warmup_times[0], bar_width, 
                            color="none", 
                            edgecolor='black', alpha = 0.5,
                            zorder=-2)
                
                ax.errorbar(x_pos, means[0], stdevs[0],
                            color='gray', capsize=3, capthick=1,
                            fmt='none', elinewidth=1)
                            
                formatted_label = types_info["display_name"][key]
                if radius_idx == 0 and formatted_label not in legend_labels:
                    legend_handles.append(bar)
                    legend_labels.append(formatted_label)
        
        kernel_group_centers = [i * (group_width + group_gap) + group_width/2 for i in range(len(unique_kernels))]
        ax.set_xticks(kernel_group_centers)
        ax.set_xticklabels(kernel_labels)
        
        alignment_x = 0
        ax.text(alignment_x, 1.1, f'$r = {radius}~m$',
                transform=ax.transAxes,
                fontsize=16,
                va='top', ha='left')
        
        ax.set_ylabel(f"Total runtime (s)")
        ax.tick_params(axis="x", which="both", length=0)
        
        curr_col += 1
        if curr_col == cols:
            curr_col = 0
            curr_row+=1
        
    return fig


def plot_result_sizes_runtime_log(data_path, clouds_datasets, operations,
                              types_info, encoder = "all", kernel="all", fsz=(6,6)):
    dfs = read_multiple_datasets(data_path, clouds_datasets)
    fig, ax = plt.subplots(figsize=fsz)
    legend_handles, legend_labels = [], []
    
    for j, (_, params) in enumerate(types_info["available_types"].iterrows()):
        key = tuple(params[col] for col in types_info["type_parameters"])
        avg_sizes, runtimes = [], []
        for df_name, df in dfs.items():
            if kernel == "all":
                df = df[(df['operation'].isin(operations))]
            else:
                df = df[(df['kernel'] == kernel) & (df['operation'].isin(operations))]
            if encoder != "all":
                df = df[df["encoder"] == encoder]
            if df.empty:
                continue
            octree_data = df[
                (df[types_info["type_parameters"]] == pd.Series(key, index=types_info["type_parameters"])).all(axis=1)
            ]
            if octree_data.empty:
                continue
            avg_sizes.extend(octree_data['avg_result_size'].tolist())
            runtimes.extend(octree_data['mean'].tolist())

        if avg_sizes and runtimes:
            scatter_kwargs = {'s': 20}
            if "palette" in types_info:
                scatter_kwargs['color'] = types_info["palette"][key]
            if "markertype" in types_info:
                scatter_kwargs['s'] = 35
                scatter_kwargs['marker'] = types_info["markertype"][key]

            scatter = ax.scatter(avg_sizes, runtimes, **scatter_kwargs) # type: ignore
            legend_handles.append(scatter)
            legend_labels.append(types_info["display_name"][key])
            # linear regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(avg_sizes), np.log(runtimes))
            regression_line = np.exp(intercept + slope * np.log(np.array(avg_sizes)))
            ax.plot(avg_sizes, regression_line, color=types_info["palette"][key], linestyle='dashed', linewidth=1)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    ax.set_xlabel(r"$\mu$", fontsize=18)
    ax.set_ylabel(f"Total runtime (s)", fontsize=18)

    return fig


def plot_octree_parallelization(data_path, cloud, algo, annotated=False, encoder="HilbertEncoder3D"):
    df = get_dataset_file(data_path, cloud, "latest")
    df = df[(df["operation"] == algo) & (df["encoder"] == encoder)][["num_searches", "repeats", "npoints", "radius", "mean", "openmp_threads"]]
    # Extract ntreads=1 baseline
    baseline = df[df["openmp_threads"] == 1].set_index("radius")["mean"]
    # Merge it on the df
    df = df.merge(baseline.rename("T1"), on="radius")
    # Compute the efficiency as (time 1 thread) / (time n threads * n)
    df["efficiency"] = df["T1"] / (df["openmp_threads"] * df["mean"])
    # Pivot and get the efficiency matrix
    efficiency_matrix = df.pivot(index="radius", columns="openmp_threads", values="efficiency")
    figsize = (7, 2.5)
    fig, ax = plt.subplots(figsize=figsize, gridspec_kw={'top': 0.75})
    heatmap = sns.heatmap(efficiency_matrix, cmap="viridis", annot=annotated, fmt=".2f", linewidths=0, 
                vmin=0, vmax=1, 
                cbar_kws={'label': 'Efficiency'}, # Add the shrink parameter
                annot_kws={"size": 11},
                ax=ax)    
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel('Efficiency', fontsize=15)
    plt.subplots_adjust(bottom=0.05)
    # Labels and title
    ax.set_xlabel("Number of threads", fontsize=15)
    ax.set_ylabel(r"$r$", fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(axis="both", which="both", length=0)
    cbar.ax.tick_params(axis="y", which="both", length=0)
    return fig



def plot_knn_comparison(
    data_path,
    cloud,
    types_info,
    struct_whitelist=None,
    low_limit=0,
    high_limit=1e9,
    label_low_limit=0,
    show_legend=True,
    fsz=(6, 6)
):
    # Load dataset
    df = get_dataset_file(data_path, cloud)
    df = df[df["kernel"] == "KNN"]
    df = df[[
        "octree", "encoder", "npoints", "operation",
        "num_searches", "sequential_searches", "mean", "avg_result_size"
    ]]
    assert low_limit <= high_limit
    if low_limit > 0:
        df = df[df["avg_result_size"] >= low_limit]
    if high_limit < 1e9:
        df = df[df["avg_result_size"] <= high_limit]
    if struct_whitelist is not None:
        df = df[df["octree"].isin(struct_whitelist)]
        
    # Create figure and axes
    fig, ax = plt.subplots(figsize=fsz)
    legend_handles, legend_labels = [], []
    all_y_values = []
    xticks_set = set()

    # Loop through available octree/KNN implementations from TYPES_INFO
    for _, params in types_info["available_types"].iterrows():
        key = tuple(params[col] for col in types_info["type_parameters"])
        octree_type = key[0]
        print(octree_type)
        # Filter data for this octree implementation
        octree_data = df[df["octree"] == octree_type]
        if octree_data.empty:
            continue

        # Group by k (avg_result_size)
        grouped = octree_data.groupby("avg_result_size")["mean"].mean().reset_index()
        grouped = grouped.sort_values("avg_result_size")

        # Determine color, marker, label
        color = types_info["palette"][key[0]]
        label = types_info["display_name"][key[0]]
        marker = types_info["markertype"].get(key[0], 'o')  # fallback to circle

        # Plot line for this implementation
        line, = ax.plot(
            grouped["avg_result_size"],
            grouped["mean"],
            marker=marker,
            color=color,
            label=label,
            linewidth=1.3,
            markersize=7,
        )
        all_y_values.extend(grouped["mean"].values)
        xticks_set.update(grouped["avg_result_size"].unique())

        legend_handles.append(line)
        legend_labels.append(label)

    # --- Axes setup ---
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(f"Total Runtime (s)")
    ax.grid(True, linestyle="--", alpha=0.5)

    # --- Filter xticks ---
    xticks = sorted(x for x in xticks_set if x >= label_low_limit)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [str(int(x)) if x == int(x) else f"{x:.1f}" for x in xticks],
        rotation=45
    )

    if show_legend:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            framealpha=0.9,
            borderpad=0.4,
            handletextpad=0.4,
            labelspacing=0.25,
            title=None 
        )

    return fig
