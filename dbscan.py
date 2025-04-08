##

from sklearn.cluster import DBSCAN 
import numpy as np
import pandas as pd
import sys

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
    
    # Check if any clusters (labels other than -1) are found
    if np.all(labels == -1):
        print("Exiting: No clusters found (all labels are -1).")
        sys.exit(1)
    
    # Extract coordinates and timestamps from the event_array
    x_coords = event_array[:, 0]
    y_coords = event_array[:, 1]
    timestamps = event_array[:, 2]

    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'timestamp': timestamps/5000 - first_timestamp,    #change according to input time
        'labels': labels
    })
    
    return df

def filter_and_merge_clusters(df, min_clusters=1, max_duration=0.5, time_tolerance=0.01, frame_rate=40):
    """
    Filter clusters based on the number of events and duration, and merge clusters with similar mean times.
    
    Returns:
    - pd.DataFrame: 
    """
    # Extract coordinates and timestamps (don't shift here yet)
    x_coords = df['x']
    y_coords = df['y']
    timestamps = df['timestamp']
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
            if abs(mean_time - existing_time) <= time_tolerance:
                merged_clusters[j].extend(indices)
                found = True
                break
        if not found:
            merged_clusters.append(list(indices))
            merged_mean_times.append(mean_time)

    print("Clusters after merging:", len(merged_clusters))
    
    # Calculate frames and add it to the frame
    frames_valid = np.floor(t_valid * frame_rate).astype(int) #makes sure they are not rounded

    # Build result DataFrame
    cluster_data = {
        'clusters': merged_clusters,
        'mean time': [mean_time for mean_time in merged_mean_times],
        'timestamps': [t_valid.iloc[inds] for inds in merged_clusters],
        'x': [x_valid.iloc[inds] for inds in merged_clusters],
        'y': [y_valid.iloc[inds] for inds in merged_clusters],
        'start': [t_valid.iloc[inds].min() for inds in merged_clusters],
        'end': [t_valid.iloc[inds].max() for inds in merged_clusters],
        'period': [(t_valid.iloc[inds].max() - t_valid.iloc[inds].min()) for inds in merged_clusters],
        'std_time': [t_valid.iloc[indices].std() for indices in merged_clusters],
        'std_size': [np.sqrt(x_valid.iloc[inds].std()**2 + y_valid.iloc[inds].std()**2) for inds in merged_clusters],
        'labels': [labels_valid.iloc[inds].tolist() for inds in merged_clusters],
        'frames': [frames_valid.iloc[inds].tolist() for inds in merged_clusters],
        'start frame': [frames_valid.iloc[inds].min() for inds in merged_clusters],
        'end frame': [frames_valid.iloc[inds].max() for inds in merged_clusters],

    }
    result_df = pd.DataFrame(cluster_data)
    result_df['frame period'] = [max(frames_valid.iloc[inds]) - min(frames_valid.iloc[inds]) for inds in merged_clusters] # The frame periods are so small that they are 0, i. e. all events are from the same frame

    # Use if you want to see which frame each timestamp corresponds to
    #for t, f in zip(t_valid.iloc[:20], frames_valid.iloc[:20]):
       # print(f"{t:.6f} sec → frame {f}")

    return result_df



