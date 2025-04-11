##

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

def plot_event_distribution(df, df_histogram, start, end, xaxis='time', window_size=100, filename=None, show=True):
    """
    Plots the event distribution over time or frame for a single cluster.

    Parameters:
    df (pd.DataFrame): Not directly used here but included for compatibility
    df_histogram (pd.DataFrame): DataFrame with 'timestamps_sec', 'frames', and 'count' columns
    start (float): Start of the cluster (in seconds or frames)
    end (float): End of the cluster (in seconds or frames)
    xaxis (str): 'time' or 'frame'
    window_size (int): Window size for moving average
    filename (str): Optional file name to save the plot
    show (bool): Whether to display the plot
    """
    if xaxis == 'time':
        x_column = 'timestamps_sec'
        margin = 0.002   #in seconds
        xlabel = "Time [s]"
    else:
        x_column = 'frames'
        margin = 2
        xlabel = "Frame"

    # Filter the histogram by the current cluster's range
    df_filtered = df_histogram[
        (df_histogram[x_column] >= start) &
        (df_histogram[x_column] <= end)
    ].dropna(subset=[x_column, 'count'])

    # Sort for clean plotting
    df_filtered = df_filtered.sort_values(by=x_column)

    if df_filtered.empty:
        print(f"Warning: No data in {x_column} between {start} and {end}. Skipping plot.")
        return 0

    total_events = df_filtered['count'].sum()
    avg_count = df_filtered['count'].mean()
    moving_avg = df_filtered['count'].rolling(window=window_size).mean()

    xdata = df_filtered[x_column]

    # Begin plotting
    plt.figure(figsize=(15, 5))
    plt.plot(xdata, df_filtered['count'], marker='o', linestyle='-', label='Event Count', color='lightsteelblue')
    plt.plot(xdata, moving_avg, color='steelblue', linestyle='-', linewidth=2, label='Moving Average')
    plt.axhline(y=avg_count, color='darkblue', linestyle='--', linewidth=2, label="Average")

    plt.axvspan(start, end, color='red', alpha=0.2)
    plt.xlabel(xlabel, fontsize=14)
    plt.xlim(start - margin, end + margin)
    plt.ylabel("Event Count")
    plt.title(f"Event Distribution from {start:.2f} to {end:.2f} ({xaxis}). Total events: {total_events}")
    plt.grid(True)
    plt.legend()

    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved as {filename}.pdf")

    if show:
        plt.show()
    else:
        plt.close()

    return total_events


def plot_event_distribution2(df, df_histogram, start, end, xaxis='time', window_size=100, filename=None, show=True):
    """
    Plots the event distribution over time or frame for a single cluster.

    Parameters:
    df (pd.DataFrame): Not directly used here but included for compatibility
    df_histogram (pd.DataFrame): DataFrame with 'timestamps_sec', 'frames', and 'count' columns
    start (float): Start of the cluster (in seconds or frames)
    end (float): End of the cluster (in seconds or frames)
    xaxis (str): 'time' or 'frame'
    window_size (int): Window size for moving average
    filename (str): Optional file name to save the plot
    show (bool): Whether to display the plot
    """
    if xaxis == 'time':
        x_column = 'timestamps_sec'
        margin = 0.02   # in seconds
        xlabel = "Time [s]"
    else:
        x_column = 'frames'
        margin = 2
        xlabel = "Frame"
    
    # First filter based on x_column (e.g., 'timestamps_sec')
    df_plot = df_histogram[
    (df_histogram[x_column] >= start - margin) &
    (df_histogram[x_column] <= end + margin)
    ].dropna(subset=[x_column, 'count']).sort_values(by=x_column)

    # Now filter based on x being within 50 pixels of a reference point
    reference_x = df_plot['x'].mean()  # <-- define this based on your use case
    pixel_tolerance = 30

    df_plot = df_plot[
    (df_plot['x'] >= reference_x - pixel_tolerance) &
    (df_plot['x'] <= reference_x + pixel_tolerance)
    ]

    print(df_plot)

    # Data for statistics (excludes margin)
    df_stats = df_histogram[
        (df_histogram[x_column] >= start) &
        (df_histogram[x_column] <= end)
    ].dropna(subset=[x_column, 'count'])

    total_events = df_stats['count'].sum()
    max_event = df_stats['count'].max()
    avg_count = df_histogram['count'].mean()

    # Create a rolling average on the plotting data (use available data)
    moving_avg = df_plot['count'].rolling(window=window_size).mean()
    xdata = df_plot[x_column]

    # Begin plotting
    plt.figure(figsize=(15, 5))
    plt.plot(xdata, df_plot['count'], marker='o', linestyle='-', label='Event Count', color='lightsteelblue')
    plt.plot(xdata, moving_avg, color='steelblue', linestyle='-', linewidth=2, label='Moving Average')
    plt.axhline(y=avg_count, color='darkblue', linestyle='--', linewidth=2, label="Average (within range)")

    plt.axvspan(start, end, color='red', alpha=0.2)
    plt.xlabel(xlabel, fontsize=14)
    plt.xlim(start - margin, end + margin)
    plt.ylabel("Event Count")
    plt.title(f"Event Distribution from {start:.2f} to {end:.2f} ({xaxis}). Total events: {total_events}")
    plt.grid(True)
    plt.legend()

    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved as {filename}.pdf")

    if show:
        plt.show()
    else:
        plt.close()

    return total_events, max_event



def plot_cluster_locations(df, title="Cluster Locations", filename=None, show=True):
    """
    Plots merged cluster locations using already-processed DataFrame.

    Parameters:
    - df: DataFrame from filter_and_merge_clusters() with 'x', 'y', and 'mean time'.
    - title: Title of the plot.
    - filename: If given, saves the plot to this path.
    - show: Whether to display the plot.
    """

    num_clusters = len(df)
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    plt.figure(figsize=(8, 6))

    for i in range(num_clusters):
        x_vals = df['x'][i]
        y_vals = df['y'][i]
        mean_time = df['mean time'][i]

        plt.scatter(x_vals, y_vals,
                    color=colors[i],
                    s=25,
                    alpha=0.7,
                    label=f"Lightning @ {mean_time:.2f}s")

    plt.title(title)
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.xlim(0,346)
    plt.ylim(0.260)
    
    # Limit to first 15 legend entries if there are too many
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 15:
        handles = handles[:15]
        labels = labels[:15]
        plt.legend(handles, labels, fontsize='small')
    else:
        plt.legend()

    plt.grid(True)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
        print(f"✅ Plot saved as {filename}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_event_count(df, x_res=None, y_res=None, vmax=None, filename=None, show=True):
    """
    Fast version of event count heatmap using numpy.histogram2d.
    """

    
    #x = np.concatenate(df['x'])
    #y = np.concatenate(df['y'])

    
    if x_res is None:
        x_res = df['x'].max() + 1
    if y_res is None:
        y_res = df['y'].max() + 1
        
    x_res = int(x_res)
    y_res = int(y_res)


    # 2D histogram (note: swap x and y to match image axes)
    event_count_map, xedges, yedges = np.histogram2d(df['y'], df['x'], bins=[y_res, x_res])

    plt.figure(figsize=(8, 6))
    plt.imshow(event_count_map, cmap='hot', vmax=vmax)
    plt.gca().invert_yaxis()  # Reverse the y-axis here
    plt.colorbar(label='Event count per pixel')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved as {filename}.pdf")

    if show:
        plt.show()
    else:
        plt.close()

def plot_event_timeline(df_histogram, 
                        df=None, 
                        window_size=100,
                        xaxis='time',
                        filename=None,
                        show=True): 
    """
    Plots the event distribution over time from a dataframe.
    Adds average and moving average lines. Saves the plot as a PDF.
    Adds an inset zoom to the region around the maximum event count.

    Parameters:
    df_histogram (pd.DataFrame): DataFrame containing 'time[s]' and 'count' columns
    cluster_data (pd.DataFrame or list): DataFrame or list with cluster start and end times
    window_size (int): Window size for calculating the moving average
    filename (str): The name of the PDF file to save the plot
    """
    
    x_data = df_histogram['timestamps_sec'] if xaxis == 'time' else df_histogram['frames']

    # Set default font size globally
    plt.rcParams.update({'font.size': 15})

    # Calculate average and moving average
    avg_count = df_histogram['count'].mean()
    moving_avg = df_histogram['count'].rolling(window=window_size).mean()

    # Create main plot
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot event count
    ax.plot(
        x_data,
        df_histogram['count'],
        marker='o',
        linestyle='-',
        label='Event Count',
        color='lightsteelblue')

    # Plot moving average
    ax.plot(
        x_data,
        moving_avg,
        color='steelblue',
        linestyle='-',
        linewidth=2,
        label='Moving Average')
    
    # Plot average line
    ax.axhline(y=avg_count, color='darkblue', linestyle='--', linewidth=2, label="Average")

    # Labels and title with font sizes
    ax.set_xlabel("Time [s]" if xaxis == 'time' else "Frame", fontsize=14)
    ax.set_ylabel("Event Count", fontsize=14)

    # Tick label size
    ax.tick_params(axis='both', labelsize=14)

    # Inset zoom implementation
    # Find the timestamp where the max event occurs
    max_event_row = df_histogram.loc[df_histogram['count'].idxmax()]
    max_x = max_event_row['timestamps_sec'] if xaxis == 'time' else max_event_row['frames']

    # Define zoom area (x-axis and y-axis limits)
    x1 = max_x - (0.05 if xaxis == 'time' else 2)  # adjust zoom window
    x2 = max_x + (0.05 if xaxis == 'time' else 2)

    # Get the subset to determine y-limits
    subset = df_histogram[(x_data >= x1) & (x_data <= x2)]
    
    y1 = 0
    y2 = subset['count'].max()

    # Create inset axes
    axins = ax.inset_axes([0.65, 0.66, 0.13, 0.3], xlim=(x1, x2), ylim=(y1, y2))

    # Plot on inset
    axins.plot(
        df_histogram['timestamps_sec'],
        df_histogram['count'],
        marker='o',
        linestyle='-',
        color='lightsteelblue')

    axins.plot(
        df_histogram['timestamps_sec'],
        moving_avg,
        color='steelblue',
        linestyle='-',
        linewidth=2)

    # Highlight clusters if cluster_data is provided
    if df is not None:
        is_dataframe = isinstance(df, pd.DataFrame)

        if is_dataframe:
            df['start'] = pd.to_numeric(df['start'], errors='coerce')
            df['end'] = pd.to_numeric(df['end'], errors='coerce')

            # Iterate over DataFrame rows
            for idx, row in df.iterrows():
                if xaxis == 'time':
                    cluster_start = row['start']
                    cluster_end = row['end']
                else:
                    cluster_start = row['start frame']
                    cluster_end = row['end frame']

                # Add label only on first iteration
                label = 'Event Interval' if idx == 0 else None

                ax.axvspan(cluster_start, cluster_end, color='red', alpha=0.2, label=label)
                axins.axvspan(cluster_start, cluster_end, color='red', alpha=0.2)

        else:
            pass
            # Assume it's a list of lists or tuples
            #for idx, cluster in enumerate(df):
             #   cluster_start = cluster[0]  # Assuming 'start' is the first element
              #  cluster_end = cluster[1]    # Assuming 'end' is the second element


                # Add label only on first iteration
                #label = 'Event Interval' if idx == 0 else None

                #ax.axvspan(cluster_start, cluster_end, color='red', alpha=0.3, label=label)
                #axins.axvspan(cluster_start, cluster_end, color='red', alpha=0.3)

    # Hide tick labels on inset
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    # Indicate zoom area on main plot
    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=4)

    # Grid and legend
    ax.grid(True)

    # Set legend font size
    ax.legend(fontsize=14, loc='upper right')

    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved as {filename}.pdf")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_variable(df, xvariable, yvariable, xlabel, ylabel, title, filename=None, show=True):
    """
    Plots a desired variable

    Parameters:
    - output_df: DataFrame with temporal spread data.
    - std_col: Name of the column for standard deviation (default: 'std (time) [ms]').
    - mean_col: Name of the column for mean cluster time (default: 'mean time').
    - filename: Optional file path to save the plot (e.g., 'spread_plot.png').
    - show: If True, the plot will be shown.
    """
    
    x = df[xvariable]
    y = df[yvariable]

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='o', linestyle='-', color='darkorange')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved as {filename}.pdf")

    if show:
        plt.show()
    else:
        plt.close()



