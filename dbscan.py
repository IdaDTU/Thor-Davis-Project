from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

def dbscan(event_array, eps=3, min_samples=100):
    """
    Apply DBSCAN clustering on the spatial and temporal data (x, y, timestamp).
    
    Args:
    - event_array (np.ndarray): A NumPy array with columns [x, y, timestamp] for the events.
    - eps (float): The maximum distance between two samples to be considered as in the same neighborhood.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    - labels (np.ndarray): Cluster labels for each event, -1 indicates noise.
    - x_coords (np.ndarray): The x-coordinates of the events.
    - y_coords (np.ndarray): The y-coordinates of the events.
    - timestamps (np.ndarray): The timestamps of the events.
    - period_sec (float): The period (time span) in seconds of the event batch.
    """
    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(event_array)
    
    # Extract coordinates and timestamps from the event_array
    x_coords = event_array[:, 0]
    y_coords = event_array[:, 1]
    timestamps = event_array[:, 2]

    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'timestamp': timestamps,
        'labels': labels
    })
    
    return df


def filter_and_merge_clusters(event_array, labels, first_timestamp, min_clusters=10, max_duration=0.5, time_tolerance=0.01):
    """
    Filter clusters based on the number of events and duration, and merge clusters with similar mean times.
    
    Args:
    - event_array (np.ndarray): A NumPy array with [x, y, timestamp] for each event.
    - labels (np.ndarray): DBSCAN labels for each event, where -1 indicates noise.
    - first_timestamp (float): The first timestamp (in seconds) of the event data.
    - min_events (int): Minimum number of events to be considered a valid cluster.
    - max_duration (float): Maximum allowed duration of a cluster in seconds (clusters lasting longer than this will be excluded).
    - time_tolerance (float): The tolerance for merging clusters based on their mean times.
    
    Returns:
    - merged_clusters (list): A list of merged clusters with event indices.
    - merged_mean_time (list): A list of mean times for the merged clusters.
    """
    # Extract coordinates and timestamps
    x_coords = event_array[:, 0]
    y_coords = event_array[:, 1]
    timestamps = event_array[:, 2]

    # Exclude the noise label (-1) by filtering the data
    valid_data_mask = labels != -1
    x_coords_valid = x_coords[valid_data_mask]
    y_coords_valid = y_coords[valid_data_mask]
    timestamps_valid = timestamps[valid_data_mask] / 1000  # Convert to seconds
    labels_valid = labels[valid_data_mask]

    # Find unique cluster labels
    unique_labels = np.unique(labels_valid)

    # Filter clusters with more than 'min_events' events
    valid_clusters = [label for label in unique_labels if np.sum(labels_valid == label) > min_clusters]

    # Initialize lists for short clusters and their mean times
    short_clusters = []
    short_mean_time = []

    # Filter clusters that last less than 'max_duration' seconds
    for cluster in valid_clusters:
        cluster_indices = np.where(labels_valid == cluster)[0]  # Get the event indices for this cluster
        cluster_timestamps = timestamps_valid[cluster_indices] - first_timestamp  # Adjust timestamps
        cluster_period = max(cluster_timestamps) - min(cluster_timestamps)
        
        if cluster_period < max_duration:  # Filter clusters with less than max_duration duration
            mean_time = np.mean(cluster_timestamps)
            short_clusters.append(cluster)
            short_mean_time.append(mean_time)
            

    print(f"Clusters lasting less than {max_duration} seconds:", len(short_clusters))

    # Step 1: Merging clusters with similar mean times
    merged_clusters = []  # List to store merged clusters
    merged_mean_time = []  # List to store mean time of merged clusters

    for idx, mean_time in enumerate(short_mean_time):
        cluster = short_clusters[idx]
        cluster_indices = np.where(labels_valid == cluster)[0]  # Get the event indices for this cluster
        found = False

        # Check if we already have a cluster with this mean time
        for i, existing_mean_time in enumerate(merged_mean_time):
            if abs(mean_time - existing_mean_time) < time_tolerance:  # Check if the mean times are close enough
                # If mean times are similar, merge clusters by adding the current cluster's indices
                merged_clusters[i].extend(cluster_indices)  # Merge the event indices
                found = True
                break

        if not found:
            # If no similar mean time was found, create a new entry
            merged_clusters.append(list(cluster_indices))  # Start a new group with this cluster's indices
            merged_mean_time.append(mean_time)  # Store the mean time for this group

    print("Clusters after merging:", len(merged_clusters))

    # Now create the DataFrame with the required variables
    cluster_data = {
        'clusters': merged_clusters,
        'mean time': merged_mean_time,
        'timestamps': [timestamps_valid[indices] for indices in merged_clusters],
        'x': [x_coords_valid[indices] for indices in merged_clusters],
        'y': [y_coords_valid[indices] for indices in merged_clusters],
        'start': [min(timestamps_valid[indices])-first_timestamp for indices in merged_clusters],
        'end': [max(timestamps_valid[indices])-first_timestamp for indices in merged_clusters],
        'period': [(max(timestamps_valid[indices]) - min(timestamps_valid[indices]))*1000 for indices in merged_clusters]
    }

    # Create the DataFrame from the cluster data
    filtered_df = pd.DataFrame(cluster_data)

    # Return the final DataFrame
    return filtered_df
