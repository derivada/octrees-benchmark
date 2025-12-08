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

def filter_by_params(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Helper to filter DF based on a dictionary of column:value pairs."""
    mask = pd.Series(True, index=df.index)
    for col, val in params.items():
        if col in df.columns:
            mask &= (df[col] == val)
    return df[mask]

def plot_runtime_comparison(data_path, cloud, viz_config, encoder="all", kernel="all", 
                            show_warmup_time=False, cols=1, figsz=(7, 10)):
    
    df = get_dataset_file(data_path, cloud)
    
    # Pre-filter
    radii = sorted(df['radius'].unique())
    
    if kernel != "all":
        if isinstance(kernel, str):
            kernel = [kernel]
        df = df[df["kernel"].isin(kernel)]
    
    unique_kernels = df['kernel'].unique()

    if encoder != "all":
        df = df[df["encoder"] == encoder]

    # Plot setup
    fig, axes = plt.subplots(int(np.ceil(len(radii)/cols)), cols, figsize=figsz,
                               gridspec_kw={'hspace': 0.3}, squeeze=False)
    
    bar_width = 0.15
    # Calculate group width based on number of configured types
    group_width = bar_width * len(viz_config)
    group_gap = 0.2
    
    legend_handles, legend_labels = [], []
    curr_row, curr_col = 0, 0
    
    for radius_idx, radius in enumerate(radii):
        ax = axes[curr_row][curr_col]
        radius_data = df[df['radius'] == radius]
        
        kernel_labels = []
        
        for i, k_name in enumerate(unique_kernels):
            kernel_data = radius_data[radius_data['kernel'] == k_name]
            
            if kernel_data.empty:
                kernel_labels.append('No data')
                continue
            
            avg_total = kernel_data['avg_result_size'].iloc[0]
            kernel_labels.append(f"$\\mathcal{{N}}_{{{k_name}}}$\n$\\mu = {avg_total:.0f}$")
            
            # Iterate through the config LIST
            for j, config in enumerate(viz_config):
                
                # Filter using the helper and the 'params' key from config
                octree_data = filter_by_params(kernel_data, config["params"])
                
                if octree_data.empty:
                    continue
                    
                means = octree_data['mean'].values
                stdevs = octree_data['stdev'].values
                warmup_times = octree_data['warmup_time'].values
                
                x_pos = i * (group_width + group_gap) + j * bar_width
                
                # Main execution time bar
                bar = ax.bar(x_pos, means[0], bar_width,
                             color=config["style"]["color"])
                
                # Warmup time bar
                if show_warmup_time:
                    ax.bar(x_pos, warmup_times[0], bar_width, 
                           color="none", 
                           edgecolor='black', alpha=0.5,
                           zorder=-2)
                
                ax.errorbar(x_pos, means[0], stdevs[0],
                            color='gray', capsize=3, capthick=1,
                            fmt='none', elinewidth=1)
                            
                formatted_label = config["display_name"]
                
                if radius_idx == 0 and formatted_label not in legend_labels:
                    legend_handles.append(bar)
                    legend_labels.append(formatted_label)
        
        kernel_group_centers = [i * (group_width + group_gap) + group_width/2 for i in range(len(unique_kernels))]
        ax.set_xticks(kernel_group_centers)
        ax.set_xticklabels(kernel_labels)
        
        ax.text(0, 1.1, f'$r = {radius}~m$', transform=ax.transAxes,
                fontsize=16, va='top', ha='left')
        
        ax.set_ylabel("Total runtime (s)")
        ax.tick_params(axis="x", which="both", length=0)
        
        curr_col += 1
        if curr_col == cols:
            curr_col = 0
            curr_row += 1
            
    fig.legend(legend_handles, legend_labels, loc="upper center",
               bbox_to_anchor=(0.5, 1), ncol=len(legend_labels),
               framealpha=0.9, borderpad=0.4, handletextpad=0.4,
               labelspacing=0.25, title=None)
    return fig


def plot_result_sizes_runtime_log(data_path, clouds_datasets, viz_config, 
                                  encoder="all", kernel="all", fsz=(7,7), lin_reg=False):
    dfs = read_multiple_datasets(data_path, clouds_datasets)
    fig, ax = plt.subplots(figsize=fsz)
    
    # List to store plot info for sorting later: {'handle': obj, 'label': str, 'sort_val': float}
    legend_items = []
    
    # Iterate through the config list
    for config in viz_config:
        avg_sizes, runtimes = [], []
        
        for df_name, df in dfs.items():
            if kernel != "all":
                df = df[df['kernel'] == kernel]
            if encoder != "all":
                df = df[df["encoder"] == encoder]
                
            if df.empty: continue
            
            # Specific config filtering
            octree_data = filter_by_params(df, config["params"])
            
            if octree_data.empty: continue
            
            avg_sizes.extend(octree_data['avg_result_size'].tolist())
            runtimes.extend(octree_data['mean'].tolist())

        if avg_sizes and runtimes:
            # Convert to numpy for math and sorting
            X = np.array(avg_sizes)
            Y = np.array(runtimes)

            # Prepare scatter arguments
            scatter_kwargs = {'s': 30}
            scatter_kwargs.update(config["style"])

            # Plot Scatter
            scatter = ax.scatter(X, Y, **scatter_kwargs)
            
            # --- Linear Regression ---
            if lin_reg:
                slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(X), np.log(Y))
                
                # Sort X values to ensure the line plots smoothly
                sort_idx = np.argsort(X)
                X_sorted = X[sort_idx]
                
                regression_line = np.exp(intercept + slope * np.log(X_sorted))
                
                line_color = config["style"].get("color", "black")
                ax.plot(X_sorted, regression_line, 
                        color=line_color, 
                        linestyle='dashed', 
                        linewidth=1)

            # --- Calculate Sorting Metric ---
            # Find the runtime (Y) at the maximum result size (X)
            # This represents the "rightmost point"
            max_x_idx = np.argmax(X)
            rightmost_y_val = Y[max_x_idx]

            legend_items.append({
                "handle": scatter,
                "label": config["display_name"],
                "sort_val": rightmost_y_val
            })

    # --- Sort Legend ---
    # Sort descending (High Runtime -> Low Runtime)
    # This places Slowest (Top of visual graph) at Top of Legend
    # and Fastest (Bottom of visual graph) at Bottom of Legend.
    legend_items.sort(key=lambda x: x["sort_val"], reverse=True)

    sorted_handles = [x["handle"] for x in legend_items]
    sorted_labels = [x["label"] for x in legend_items]

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel(r"$\mu$", fontsize=18)
    ax.set_ylabel("Total runtime (s)", fontsize=18)
    
    ax.legend(
        sorted_handles,
        sorted_labels,
        loc="upper left",
        framealpha=0.9,
        borderpad=0.4,
        handletextpad=0.4,
        labelspacing=0.25,
        title=None
    )
    
    return fig


def plot_knn_comparison(data_path, cloud, viz_config,
                        struct_whitelist=None, low_limit=0, high_limit=1e9,
                        label_low_limit=0, fsz=(6, 6)):
    
    df = get_dataset_file(data_path, cloud)
    df = df[df["kernel"] == "KNN"]
    df = df[["octree", "encoder", "npoints", "operation",
             "num_searches", "sequential_searches", "mean", "avg_result_size"]]
    
    assert low_limit <= high_limit
    if low_limit > 0:
        df = df[df["avg_result_size"] >= low_limit]
    if high_limit < 1e9:
        df = df[df["avg_result_size"] <= high_limit]
    if struct_whitelist is not None:
        df = df[df["octree"].isin(struct_whitelist)]
        
    fig, ax = plt.subplots(figsize=fsz)
    legend_handles, legend_labels = [], []
    xticks_set = set()

    # Iterate through Config List
    for config in viz_config:
        
        # Filter data for this specific configuration
        octree_data = filter_by_params(df, config["params"])
        if octree_data.empty:
            continue

        grouped = octree_data.groupby("avg_result_size")["mean"].mean().reset_index()
        grouped = grouped.sort_values("avg_result_size")

        # Extract styles safely
        style = config.get("style", {})
        color = style.get("color", "black")
        marker = style.get("marker", "o")
        label = config["display_name"]

        line, = ax.plot(
            grouped["avg_result_size"],
            grouped["mean"],
            marker=marker,
            color=color,
            label=label,
            linewidth=1.3,
            markersize=7,
        )
        xticks_set.update(grouped["avg_result_size"].unique())

        legend_handles.append(line)
        legend_labels.append(label)

    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Total Runtime (s)")
    ax.grid(True, linestyle="--", alpha=0.5)

    xticks = sorted(x for x in xticks_set if x >= label_low_limit)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [str(int(x)) if x == int(x) else f"{x:.1f}" for x in xticks],
        rotation=45
    )

    ax.legend(legend_handles, legend_labels, loc="upper left",
              framealpha=0.9, borderpad=0.4, handletextpad=0.4,
              labelspacing=0.25, title=None)

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