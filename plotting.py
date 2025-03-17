import matplotlib.pyplot as plt
import numpy as np

def plot_event_distribution(df_histogram, window_size=100, filename='event_distribution.pdf', start_time=None, end_time=None):
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


    # Optional: Show the plot
    plt.show()
    
    return total_events


def plot_cluster_locations(merged_clusters, x, y, merged_mean_time, title="Cluster Plot"):
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
    # Flatten the coordinates into one long array
    x_coords_valid = np.concatenate(x)
    y_coords_valid = np.concatenate(y)
    
    # Generate a colormap
    cmap = plt.get_cmap("tab20", len(merged_clusters))  # Use the new method to get the colormap
    
    # Plot the locations for each cluster with a unique color
    plt.figure(figsize=(8, 6))
    
    for i, cluster_group in enumerate(merged_clusters):
        # Get the color for this cluster from the colormap
        color = cmap(i)
        
        # Get the indices for the current cluster group
        cluster_indices = np.array(cluster_group)
        
        # Plot each cluster with its respective color and label
        plt.scatter(x_coords_valid[cluster_indices], y_coords_valid[cluster_indices], 
                    color=color, label=f'Lightning @ {merged_mean_time[i]:.2f}s', s=10)

    # Set limits for the plot
    plt.xlim(0, max(x_coords_valid))
    plt.ylim(0, max(y_coords_valid))

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()

