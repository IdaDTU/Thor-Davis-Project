from plotting import plot_event_distribution
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def calculate_total_events(merged_clusters, start, end, df_histogram):
    """
    Calculate the total events for each merged cluster and store the relevant data.
    
    Args:
    - merged_clusters (list): A list of merged clusters, where each cluster is a list of event indices.
    - start (list or np.ndarray): Start times for each merged cluster.
    - end (list or np.ndarray): End times for each merged cluster.
    - df_histogram (DataFrame): The histogram DataFrame used for plotting event distributions.
    
    Returns:
    - total_events_list (list): List of the total events for each merged cluster.
    """
    total_events_list = []

    # Iterate over merged clusters
    for i, merged_cluster_indices in enumerate(merged_clusters):
        # Calculate the total events and plot for this merged cluster
        total_events = plot_event_distribution(df_histogram, start_time=start[i] - 0.02, end_time=end[i] + 0.02)
        total_events_list.append(total_events)

    return total_events_list


def calculate_cluster_sizes(merged_clusters, x, y):
    """
    Calculate the size (maximum distance between any two points) for each merged cluster.
    
    Args:
    - merged_clusters (list): A list of merged clusters, where each cluster is a list of event indices.
    - x_coords_valid (np.ndarray): The x-coordinates of the valid events.
    - y_coords_valid (np.ndarray): The y-coordinates of the valid events.
    
    Returns:
    - cluster_sizes (list): A list of sizes for each cluster, where each size is the maximum distance between any two points.
    """
    
    # Flatten the coordinates into one long array
    x_coords_valid = np.concatenate(x)
    y_coords_valid = np.concatenate(y)
    # Initialize an empty list to store the sizes of each cluster
    cluster_sizes = []
    
    # Loop over the merged clusters and calculate the size for each
    for cluster_idx, merged_cluster_indices in enumerate(merged_clusters):
        # Get the coordinates of the events in the cluster
        cluster_x = x_coords_valid[merged_cluster_indices]
        cluster_y = y_coords_valid[merged_cluster_indices]
        
        # Stack the coordinates into a 2D array for distance computation
        coordinates = np.stack((cluster_x, cluster_y), axis=1)
        
        # Calculate pairwise distances between all points in the cluster
        distances = cdist(coordinates, coordinates)
        
        # Get the maximum distance (size of the cluster)
        max_distance = np.max(distances)
        
        # Append the size of the cluster to the list
        cluster_sizes.append(max_distance)
    
    return cluster_sizes

def calculate_event_rate(total_events, periods):
    """
    Calculate the event rate as Total Events / Period [s], given total events and periods in ms.
    
    Args:
    - total_events (list or np.ndarray): List or array containing the total events for each cluster.
    - periods (list or np.ndarray): List or array containing the period in milliseconds for each cluster.
    
    Returns:
    - event_rates (list): List of event rates, rounded to 3 decimal places, for each cluster.
    """
    # Calculate the event rate (Total Events / Period [s])
    event_rates = total_events / periods  # Convert periods from ms to seconds
    
    # Round the event rates to 3 decimal places
    event_rates = np.round(event_rates, 3)
    
    return event_rates

def create_and_save_df(start, end, period, size, total_events, event_rate, file_path, decimals=3):
    # Create a dictionary of the input data
    data = {
        'start': start,
        'end': end,
        'period [ms]': period,
        'size [px]': size,
        'total events': total_events,
        'event rate [event/ms]': event_rate 
    }

    # Create DataFrame from the data
    df = pd.DataFrame(data)
    
    # Round the numeric columns to the specified number of decimal places
    df['start'] = df['start'].round(decimals)
    df['end'] = df['end'].round(decimals)
    df['period [ms]'] = df['period [ms]'].round(decimals)
    df['size [px]'] = df['size [px]'].round(decimals)
    df['total events'] = df['total events'].round(decimals)
    df['event rate [event/ms]'] = df['event rate [event/ms]'].round(decimals)

    # Convert numeric columns to strings and apply left alignment
    for col in ['start', 'end','period [ms]', 'size [px]', 'total events', 'event rate [event/ms]']:
        df[col] = df[col].apply(lambda x: f'{x:<{max(df[col].astype(str).apply(len))}}')

    # Save the DataFrame to the specified CSV file
    #df.to_csv(file_path, index=False)
    #print(f"DataFrame has been saved successfully to {file_path}")
    
    return df
