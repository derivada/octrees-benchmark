# Functions for generating the plots
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import *
from constants import *
import scipy.stats as stats
import matplotlib.lines as mlines

def plot_locality_hist(
    datasets: Dict[str, pd.DataFrame],
    bin_size: float,
    min_dist: float,
    max_dist: float,
    k: int,
    use_log: bool,
) -> plt.Figure: # type: ignore
    fig, ax = plt.subplots(figsize=(6,4))
    cmap = plt.get_cmap("tab10")

    for i, (encoder, df) in enumerate(datasets.items()):
        encoder_str = r"$H_{" + str(k) + r"}^{" + encoder + r"}$"
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
                            show_warmup_time=False, cols=1, figsz=(7, 10), algos="all"):
    
    df = get_dataset_file(data_path, cloud)
    if algos != "all":
        df = df[df["operation"].isin(algos)]
    
    if kernel != "all":
        if isinstance(kernel, str):
            kernel = [kernel]
        df = df[df["kernel"].isin(kernel)]
    if encoder != "all":
        df = df[df["encoder"] == encoder]
    
    radii = sorted(df['radius'].unique())
    unique_kernels = df['kernel'].unique()

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
    
    # Changed: List to store plot info for sorting later
    legend_items = [] 
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

        # --- New Sorting Logic ---
        # Since 'grouped' is already sorted by avg_result_size (X axis),
        # the last row contains the rightmost value.
        rightmost_y_val = grouped.iloc[-1]["mean"]

        legend_items.append({
            "handle": line,
            "label": label,
            "sort_val": rightmost_y_val
        })

    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Total Runtime (s)")
    ax.grid(True, linestyle="--", alpha=0.5)

    xticks = sorted(x for x in xticks_set if x >= label_low_limit)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [str(int(x)) if x == int(x) else f"{x:.1f}" for x in xticks],
        rotation=45
    )

    # --- Apply Legend Sorting ---
    # Sort descending based on the rightmost Y value (highest runtime at top)
    legend_items.sort(key=lambda x: x["sort_val"], reverse=True)

    sorted_handles = [x["handle"] for x in legend_items]
    sorted_labels = [x["label"] for x in legend_items]

    ax.legend(sorted_handles, sorted_labels, loc="upper left",
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

def table_speedup_vs_baseline(data_path, clouds_datasets, kernel, viz_config, encoder="all"):
    
    dfs = read_multiple_datasets(data_path, clouds_datasets)
    all_results = []

    baseline_params = {"octree": POINTER_OCTREE, "operation": NEIGHBOURS_PTR}

    # Define result size bins: [(min, max), ...]
    bins = [(10**i, 10**(i+2)) for i in range(1, 7, 2)]

    for bin_min, bin_max in bins:
        bin_results = []
        
        # 1. Collect Data for all configs in this Bin
        for config in viz_config:
            runtimes = []

            for df_name, df in dfs.items():
                if df.empty: continue
                
                # General Filters
                if kernel != "all":
                    df = df[(df['kernel'] == kernel)]
                
                if encoder != "all":
                    df = df[df["encoder"] == encoder]

                # Specific Config Filter (Using the helper from previous steps)
                struct_data = filter_by_params(df, config["params"])

                if struct_data.empty: continue

                # Bin Filter
                struct_data = struct_data[
                    (struct_data["avg_result_size"] >= bin_min) & 
                    (struct_data["avg_result_size"] < bin_max)
                ]
                
                if struct_data.empty: continue

                runtimes.extend(struct_data["mean"].tolist())

            avg_runtime = np.mean(runtimes) if runtimes else np.nan
            
            bin_results.append({
                "config": config,
                "avg_runtime": avg_runtime
            })

        # 2. Identify Baseline Runtime for this Bin
        # We look for the entry in bin_results where the params match baseline_params
        baseline_runtime = np.nan
        for res in bin_results:
            if res["config"]["params"] == baseline_params:
                baseline_runtime = res["avg_runtime"]
                break

        # 3. Calculate Speedups and Build Rows
        for res in bin_results:
            runtime = res["avg_runtime"]
            config = res["config"]
            
            # Calculate Speedup (Baseline / Current)
            if np.isnan(runtime) or np.isnan(baseline_runtime) or runtime == 0:
                speedup = np.nan
            else:
                speedup = baseline_runtime / runtime

            label = config["display_name"]
            
            # Store raw numbers for sorting, formatted strings for display
            all_results.append({
                "Size Range": f"{bin_min:.0e}–{bin_max:.0e}",
                "Structure": label,
                "_sort_val": speedup if not np.isnan(speedup) else -1.0, # Hidden col for sorting
                "Average Runtime (ms)": f"{runtime:.4f}" if not np.isnan(runtime) else "N/A",
                "Speedup": f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A"
            })

    # Create DataFrame
    df_res = pd.DataFrame(all_results)
    
    if not df_res.empty:
        # Sort using the numeric value to ensure "10.0x" > "2.0x"
        df_res = df_res.sort_values(by=["Size Range", "_sort_val"], ascending=[True, False])
        # Remove the hidden helper column
        df_res = df_res.drop(columns=["_sort_val"])
        
    return df_res


def plot_encoder_improvement(data_path, clouds_datasets, viz_config, 
                             base_encoder, improved_encoder, 
                             kernel="all", outlier_threshold=80,  # Added parameter
                             fsz=(7, 7), fs_legend=14):
    
    dfs = read_multiple_datasets(data_path, clouds_datasets)
    
    all_data = pd.DataFrame()
    for ds_name, df in dfs.items():
        df = df.copy()
        df['cloud'] = ds_name
        all_data = pd.concat([all_data, df], ignore_index=True)

    fig, ax = plt.subplots(figsize=fsz)
    structure_handles = []
    
    for config in viz_config:
        struct_data = filter_by_params(all_data, config["params"])
        
        if kernel != "all":
            struct_data = struct_data[struct_data['kernel'] == kernel]
            
        if struct_data.empty:
            continue

        df_enc1 = struct_data[struct_data["encoder"] == base_encoder]
        df_enc2 = struct_data[struct_data["encoder"] == improved_encoder]
        
        if df_enc1.empty or df_enc2.empty:
            print(f"Skipping {config['display_name']}: Missing data for one of the encoders.")
            continue

        merge_cols = ['cloud', 'radius', 'kernel']
            
        merged_df = pd.merge(
            df_enc1, 
            df_enc2, 
            on=merge_cols, 
            suffixes=('_base', '_target'),
            how='inner'
        )
        
        if merged_df.empty:
            print(f"Skipping {config['display_name']}: No matching {merge_cols} found.")
            continue

        # compute relative improvement from enc1 to enc2
        t1 = merged_df['mean_base']
        t2 = merged_df['mean_target']
        rel_improvement = ((t1 - t2) / t1) * 100

        if outlier_threshold is not None:
            mask = rel_improvement.abs() <= outlier_threshold
            rel_improvement = rel_improvement[mask]
            merged_df = merged_df[mask]
            
            if rel_improvement.empty:
                print(f"Skipping {config['display_name']}: All data filtered out by threshold {outlier_threshold}.")
                continue

        print(config["params"]["operation"], rel_improvement.mean())
        
        base_color = config["style"].get("color", "black")
        base_marker = config["style"].get("marker", "o")
        scatter_kwargs = {'s': 30}
        scatter_kwargs.update(config["style"])
        
        ax.scatter(merged_df['avg_result_size_base'], rel_improvement, **scatter_kwargs)
        
        structure_handles.append(
            mlines.Line2D([], [], color=base_color, marker=base_marker, 
                          linestyle='None', markersize=8, label=config["display_name"])
        )

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mu$", fontsize=16)
    ax.set_ylabel(r"Relative Improvement ($\%$)", fontsize=14)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    ax.legend(handles=structure_handles, loc="best", fontsize=fs_legend, framealpha=0.5)
    
    return fig