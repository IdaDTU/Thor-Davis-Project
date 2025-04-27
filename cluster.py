import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import dv_processing as dvp
import glob
from video_information import create_dataframe
from plotting import plot_event_distribution
import pandas as pd

# Define the path to your Aedat file
FIL = glob.glob("C:/Users/user/OneDrive/Desktop/ThorDavis/dvSave-2023_12_30_20_04_46.aedat4", recursive=True)[0]

# Open the file
capture = dvp.io.MonoCameraRecording(FIL)

# Specify how many event batches you want to process
num_batches = 1000  #41500 is all batches, 1000 is 10 sec

df_histogram = create_dataframe(FIL, 
                                num_bins=10*num_batches,
                                batches=num_batches)

# Initialize empty lists for storing events and timestamps
on_events_list = []
first_run = True

# Loop over the specified number of event batches
for _ in range(num_batches):
    events = capture.getNextEventBatch()
    
    if first_run:
        first_timestamp = events[0].timestamp() / 1000000  # Define the first timestamp and make it into sec
        first_run = False
    
    # Keep only ON polarities
    for e in events:
        if e.polarity() == True:
            time = e.timestamp() / 1000  # Convert to milliseconds
            x = e.x()
            y = e.y()
            on_events_list.append((x, y, time))

print(f"Number of 'on' events collected: {len(on_events_list)}")

# Find clusters

event_array = np.array(on_events_list)

# Apply DBSCAN to cluster spatial and temporal data
eps = 3  # Max distance between events in the feature space
min_samples = 100  # Minimum number of events to form a cluster

# Perform DBSCAN clustering
db = DBSCAN(eps=eps, min_samples=min_samples)   
labels = db.fit_predict(event_array)
print(labels)

# Filter clusters

x_coords = event_array[:, 0]
y_coords = event_array[:, 1]
timestamps = event_array[:, 2]

print("Period in sec:", (max(timestamps) - min(timestamps)) / 1000)

# Exclude the noise label (-1) by filtering the data
valid_data_mask = labels != -1
x_coords_valid = x_coords[valid_data_mask]
y_coords_valid = y_coords[valid_data_mask]
timestamps_valid = timestamps[valid_data_mask] / 1000  # Convert to seconds
labels_valid = labels[valid_data_mask]

# Find unique cluster labels
unique_labels = np.unique(labels_valid)

# Filter clusters with more than 10 events
valid_clusters = [label for label in unique_labels if np.sum(labels_valid == label) > 10]

# Initialize lists for short clusters and their mean times
short_clusters = []
short_mean_time = []

# Filter clusters that last less than 500ms
for cluster in valid_clusters:
    cluster_indices = np.where(labels_valid == cluster)[0]  # Get the event indices for this cluster
    cluster_timestamps = timestamps_valid[cluster_indices] - first_timestamp  # Adjust timestamps
    cluster_period = max(cluster_timestamps) - min(cluster_timestamps)
    
    if cluster_period < 0.5:  # Filter clusters with less than 500ms duration
        mean_time = np.mean(cluster_timestamps)
        short_clusters.append(cluster)
        short_mean_time.append(mean_time)

print("Clusters lasting less than 500ms:", len(short_clusters))

# Step 1: Merging clusters first
merged_clusters = []  # List to store merged clusters
merged_mean_time = []  # List to store mean time of merged clusters

# Tolerance for merging clusters with similar mean times
time_tolerance = 0.01  # Adjust this tolerance as needed

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

# Step 2: Now calculate the total_events for each merged cluster
cluster_data = []
total_events_list = []

for cluster_idx, merged_cluster_indices in enumerate(merged_clusters):
    # Get the timestamps for the merged cluster
    merged_cluster_timestamps = timestamps_valid[merged_cluster_indices] - first_timestamp
    
    # Calculate the period for this merged cluster
    cluster_period = max(merged_cluster_timestamps) - min(merged_cluster_timestamps)
    cluster_start = min(merged_cluster_timestamps)
    cluster_end = max(merged_cluster_timestamps)

    # Now, calculate the total events and plot for this merged cluster
    total_events = plot_event_distribution(df_histogram, start_time=cluster_start - 0.02, end_time=cluster_end + 0.02)
    total_events_list.append(total_events)
    
    cluster_data.append([cluster_period, cluster_start, cluster_end, total_events])

    #print(f"Merged Cluster {cluster_idx} period: {min(merged_cluster_timestamps):.3f} - {max(merged_cluster_timestamps):.3f}")

# Print the final list of total events for the merged clusters
print("Total events for each merged clusters:", total_events_list)


#%% Plot clusters location with shown time
for i, cluster_group in enumerate(merged_clusters):
    # Merge all the event indices for the current cluster group
    all_indices = np.array(cluster_group)

    # Plot the events for this merged cluster
    plt.scatter(x_coords_valid[all_indices], y_coords_valid[all_indices], 
                label=f'Lightning @ time: {merged_mean_time[i]:.2f} s', s=10)
    
plt.xlim(0, max(x_coords_valid))
plt.ylim(0, max(y_coords_valid))

# Add labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Clusters with More Than 10 "100 Events" lasting less than 500 ms')

# Show legend
plt.legend()

# Show the plot
plt.show()

#%% Create df with relevant cluster data

# Convert the list to a pandas DataFrame
cluster_df = pd.DataFrame(cluster_data, columns=['cluster_period', 'cluster_start', 'cluster_end', 'total_events'])

# Print the DataFrame
print(cluster_df)











