from plotting import plot_event_distribution
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def calculate_total_events(merged_clusters, timestamps_valid, first_timestamp, df_histogram):
    """
    Calculate the total events for each merged cluster and store the relevant data.
    
    Args:
    - merged_clusters (list): A list of merged clusters, where each cluster is a list of event indices.
    - timestamps_valid (np.ndarray): Array of timestamps for all valid events.
    - first_timestamp (float): The first timestamp (in seconds) for adjusting the timestamps of clusters.
    - df_histogram (DataFrame): The histogram DataFrame used for plotting event distributions.
    
    Returns:
    - cluster_data (list): List of data for each cluster, including period, start, end, and total events.
    - total_events_list (list): List of the total events for each merged cluster.
    """
    cluster_data = []
    total_events_list = []

    for cluster_idx, merged_cluster_indices in enumerate(merged_clusters):
        # Get the timestamps for the merged cluster
        merged_cluster_timestamps = timestamps_valid[merged_cluster_indices] - first_timestamp
        
        # Calculate the period for this merged cluster
        cluster_period = max(merged_cluster_timestamps) - min(merged_cluster_timestamps)
        cluster_start = min(merged_cluster_timestamps)
        cluster_end = max(merged_cluster_timestamps)

        # Calculate the total events and plot for this merged cluster
        total_events = plot_event_distribution(df_histogram, start_time=cluster_start - 0.02, end_time=cluster_end + 0.02)
        total_events_list.append(total_events)
        
        # Store the cluster data with rounded values
        cluster_data.append([round(cluster_period * 1000, 3),  # cluster_period in ms
                             round(cluster_start, 3),          # cluster_start in seconds
                             round(cluster_end, 3),            # cluster_end in seconds
                             round(total_events, 3)])          # total_events

    return cluster_data, total_events_list

def calculate_cluster_size(cluster_indices, x_coords, y_coords):
    """
    Calculate the size of the cluster as the maximum distance between any two points in the cluster.
    
    Args:
    - cluster_indices (list): Indices of the events in the cluster.
    - x_coords (np.ndarray): The x-coordinates of all valid events.
    - y_coords (np.ndarray): The y-coordinates of all valid events.
    
    Returns:
    - max_distance (float): The maximum distance between any two points in the cluster.
    """
    # Get the coordinates of the events in the cluster
    cluster_x = x_coords[cluster_indices]
    cluster_y = y_coords[cluster_indices]
    
    # Stack the coordinates into a 2D array for distance computation
    coordinates = np.stack((cluster_x, cluster_y), axis=1)
    
    # Calculate pairwise distances between all points in the cluster
    distances = cdist(coordinates, coordinates)
    
    # Get the maximum distance (size of the cluster)
    max_distance = np.max(distances)
    return max_distance

def compute_cluster_df(merged_clusters, x_coords_valid, y_coords_valid, merged_mean_time, df_histogram, first_timestamp, cluster_data):
    """
    Compute the final DataFrame containing cluster data, sizes, and event rates.
    
    Args:
    - merged_clusters (list): A list of merged clusters, each represented by a list of event indices.
    - x_coords_valid (np.ndarray): The x-coordinates of the valid events.
    - y_coords_valid (np.ndarray): The y-coordinates of the valid events.
    - merged_mean_time (list): A list of mean times for each merged cluster.
    - df_histogram (DataFrame): The histogram DataFrame used for plotting event distributions.
    - first_timestamp (float): The first timestamp (in seconds) of the event data.
    - cluster_data (list): The cluster data containing [cluster_period, cluster_start, cluster_end, total_events] for each cluster.
    
    Returns:
    - output_df (DataFrame): The final DataFrame containing the cluster data with sizes and event rates.
    """
    # Calculate the size for each merged cluster
    cluster_sizes = []
    
    for cluster_idx, merged_cluster_indices in enumerate(merged_clusters):
        size = calculate_cluster_size(merged_cluster_indices, x_coords_valid, y_coords_valid)
        cluster_sizes.append(size)
    
    # Create a DataFrame from the cluster data
    output_df = pd.DataFrame(cluster_data, columns=['Period [ms]', 'Start [s]', 'End [s]', 'Total Events'])
    
    # Add the cluster sizes to the DataFrame
    output_df['Size [px]'] = [round(size, 3) for size in cluster_sizes]
    
    # Calculate the event rate (Total Events / Period [s]) and add it to the DataFrame
    output_df['Event Rate [events/ms]'] = output_df['Total Events'] / (output_df['Period [ms]'])
    
    # Round the event rate to 3 decimal places
    output_df['Event Rate [events/ms]'] = output_df['Event Rate [events/ms]'].round(3)
    
    return output_df
