import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

def plot_event_distribution(df_histogram, window_size=100, start_time=None, end_time=None, df=None, filename=None, show=True):
    """
    Plots the event distribution over time from a dataframe within a specific time interval.
    Adds average and moving average lines. Saves the plot as a PDF.

    Parameters:
    df_histogram (pd.DataFrame): DataFrame containing 'timestamps_sec' and 'count' columns
    window_size (int): Window size for calculating the moving average
    filename (str): The name of the PDF file to save the plot
    start_time (float, optional): Start time of the interval to filter the data
    end_time (float, optional): End time of the interval to filter the data
    """

    # Filter the data based on the specified time interval
    if start_time is not None and end_time is not None:
        df_filtered = df_histogram[(df_histogram['timestamps_sec'] >= start_time) & (df_histogram['timestamps_sec'] <= end_time)]
    elif start_time is not None:
        df_filtered = df_histogram[df_histogram['timestamps_sec'] >= start_time]
    elif end_time is not None:
        df_filtered = df_histogram[df_histogram['timestamps_sec'] <= end_time]
    else:
        df_filtered = df_histogram

    # Ensure both columns are filtered together
    df_filtered = df_filtered.dropna(subset=['timestamps_sec', 'count'])
    
    total_events = df_filtered['count'].sum()

    # Calculate average and moving average
    avg_count = df_filtered['count'].mean()
    moving_avg = df_filtered['count'].rolling(window=window_size).mean()

    # Plotting
    plt.figure(figsize=(15, 5))

    # Event count
    plt.plot(
        df_filtered['timestamps_sec'],
        df_filtered['count'],
        marker='o',
        linestyle='-',
        label='Event Count',
        color='lightsteelblue')

    # Moving average line
    plt.plot(
        df_filtered['timestamps_sec'],
        moving_avg,
        color='steelblue',
        linestyle='-',
        linewidth=2,
        label='Moving Average')
    
    # Average line
    plt.axhline(y=avg_count, color='darkblue', linestyle='--', linewidth=2, label="Average")

    # Shaded cluster period
    plt.axvspan(start_time + 0.02, end_time - 0.02, color='red', alpha=0.2)

    # Labels and title
    plt.xlabel("Time [s]")
    plt.ylabel("Event Count")
    
    # Add a title
    if start_time is not None and end_time is not None:
        plt.title(f"Event Distribution from {start_time:.2f}s to {end_time:.2f}s. Total events: {total_events}")
    else:
        plt.title("Event Distribution")

    # Grid and legend
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

def plot_cluster_locations(df, title="Cluster Plot", filename=None, show=True):
    """
    Plot the locations of all merged clusters as a scatter plot.

    Args:
    - merged_clusters (pd.Series): A Series where each entry is a list of indices for each cluster.
    - x (list of np.ndarray): List of x-coordinates for each event in each cluster.
    - y (list of np.ndarray): List of y-coordinates for each event in each cluster.
    - merged_mean_time (list): List of mean times for each merged cluster.
    - title (str): Title of the plot.

    Returns:
    - None
    """
    
    merged_clusters = df['clusters']
    x = df['x']
    y = df['y']
    merged_mean_time = df['mean time']
    
    num_clusters = len(merged_clusters)
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))  # auto-generate distinct colors

    
    # Flatten the coordinates into one long array
    x_coords_valid = np.concatenate(x)
    y_coords_valid = np.concatenate(y)
    
    # Plot the locations for each cluster with a unique color
    plt.figure(figsize=(8, 6))
    
    for i, cluster_group in enumerate(merged_clusters):
        # Get the color for this cluster from the colormap
        color = colors[i]
        
        # Get the indices for the current cluster group
        cluster_indices = np.array(cluster_group)
        
        # Plot each cluster with its respective color and label
        plt.scatter(x_coords_valid[cluster_indices], y_coords_valid[cluster_indices], 
                    color=color, label=f'Lightning @ {merged_mean_time[i]:.2f}s', s=10)

    # Set limits for the plot
    plt.xlim(0, 350)
    plt.ylim(0, 260)

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)

    # Show legend
    plt.legend()
    
    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Plot saved as {filename}.pdf")

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

    # Set default font size globally
    plt.rcParams.update({'font.size': 15})

    # Calculate average and moving average
    avg_count = df_histogram['count'].mean()
    moving_avg = df_histogram['count'].rolling(window=window_size).mean()

    # Create main plot
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot event count
    ax.plot(
        df_histogram['timestamps_sec'],
        df_histogram['count'],
        marker='o',
        linestyle='-',
        label='Event Count',
        color='lightsteelblue')

    # Plot moving average
    ax.plot(
        df_histogram['timestamps_sec'],
        moving_avg,
        color='steelblue',
        linestyle='-',
        linewidth=2,
        label='Moving Average')
    
    # Plot average line
    ax.axhline(y=avg_count, color='darkblue', linestyle='--', linewidth=2, label="Average")

    # Labels and title with font sizes
    ax.set_xlabel("Time [s]", fontsize=14)
    ax.set_ylabel("Event Count", fontsize=14)

    # Tick label size
    ax.tick_params(axis='both', labelsize=14)

    # Inset zoom implementation
    # Find the timestamp where the max event occurs
    max_event_row = df_histogram.loc[df_histogram['count'].idxmax()]
    max_timestamp = max_event_row['timestamps_sec']

    # Define zoom area (x-axis and y-axis limits)
    zoom_window = 0.05  # seconds before/after
    x1 = max_timestamp - zoom_window
    x2 = max_timestamp + zoom_window

    # Get the subset to determine y-limits
    subset = df_histogram[
        (df_histogram['timestamps_sec'] >= x1) & 
        (df_histogram['timestamps_sec'] <= x2)
    ]
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
                cluster_start = row['start']
                cluster_end = row['end']


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



