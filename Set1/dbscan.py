from sklearn.cluster import DBSCAN 
import numpy as np
import pandas as pd

def dbscan(event_array, first_timestamp, eps=3, min_samples=100):
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

def filter_and_merge_clusters(df, first_timestamp, min_clusters=1, max_duration=0.5, time_tolerance=0.01):
    """
    Filter clusters based on the number of events and duration, and merge clusters with similar mean times.
    
    Returns:
    - pd.DataFrame: With time fields relative to first_timestamp (in seconds)
    """
    # Extract coordinates and timestamps (don't shift here yet)
    x_coords = df['x']
    y_coords = df['y']
    timestamps = df['timestamp'] / 1000
    labels = df['labels']

    # Exclude noise
    valid_mask = labels != -1
    x_valid = x_coords[valid_mask]
    y_valid = y_coords[valid_mask]
    t_valid = timestamps[valid_mask]
    labels_valid = labels[valid_mask]

    # Filter by cluster size
    unique_labels = np.unique(labels_valid)
    valid_clusters = [label for label in unique_labels if np.sum(labels_valid == label) > min_clusters]
    print('Amount of clusters:', len(valid_clusters))

    short_clusters = []
    short_mean_times = []

    # Filter clusters by duration
    for cluster in valid_clusters:
        indices = np.where(labels_valid == cluster)[0]
        cluster_times = t_valid.iloc[indices]
        duration = cluster_times.max() - cluster_times.min()

        if duration < max_duration:
            mean_time = cluster_times.mean()
            short_clusters.append(cluster)
            short_mean_times.append(mean_time)

    print(f"Clusters lasting less than {max_duration} seconds:", len(short_clusters))

    # Merge close-in-time clusters
    merged_clusters = []
    merged_mean_times = []

    for i, mean_time in enumerate(short_mean_times):
        indices = np.where(labels_valid == short_clusters[i])[0]
        found = False
        for j, existing_time in enumerate(merged_mean_times):
            if abs(mean_time - existing_time) < time_tolerance:
                merged_clusters[j].extend(indices)
                found = True
                break
        if not found:
            merged_clusters.append(list(indices))
            merged_mean_times.append(mean_time)

    print("Clusters after merging:", len(merged_clusters))

    # Build result DataFrame (apply - first_timestamp here!)
    cluster_data = {
        'clusters': merged_clusters,
        'mean time': [mean_time - first_timestamp for mean_time in merged_mean_times],
        'timestamps': [t_valid.iloc[inds] - first_timestamp for inds in merged_clusters],
        'x': [x_valid.iloc[inds] for inds in merged_clusters],
        'y': [y_valid.iloc[inds] for inds in merged_clusters],
        'start': [t_valid.iloc[inds].min() - first_timestamp for inds in merged_clusters],
        'end': [t_valid.iloc[inds].max() - first_timestamp for inds in merged_clusters],
        'period': [(t_valid.iloc[inds].max() - t_valid.iloc[inds].min()) for inds in merged_clusters],
        'std_time': [t_valid.iloc[indices].std() for indices in merged_clusters],
        'std_size': [np.sqrt(x_valid.iloc[inds].std()**2 + y_valid.iloc[inds].std()**2) for inds in merged_clusters]
    }

    return pd.DataFrame(cluster_data)



