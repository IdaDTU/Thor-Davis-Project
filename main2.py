import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import dv_processing as dvp
import glob
from video_information import create_dataframe, collect_on_events
from plotting import plot_event_distribution, plot_cluster_locations
import pandas as pd
from dbscan import dbscan, filter_and_merge_clusters
from output_df import calculate_total_events, calculate_cluster_size

# Define the path to your Aedat file
FIL = glob.glob("/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ThorDavis/Data/dvSave-2023_12_30_20_04_46.aedat4", recursive=True)[0]

# Open the file
capture = dvp.io.MonoCameraRecording(FIL)

# Specify how many event batches you want to process
num_batches = 1000  #41500 is all batches, 1000 is 10 sec

df_histogram = create_dataframe(FIL, 
                                num_bins=10*num_batches,
                                batches=num_batches)

event_array, first_timestamp = collect_on_events(capture, num_batches)

labels, x_coords, y_coords, timestamps = dbscan(event_array, eps=3, min_samples=100)

merged_clusters, merged_mean_time, timestamps_valid, x_coords_valid, y_coords_valid= filter_and_merge_clusters(event_array, labels, first_timestamp, min_events=10, max_duration=0.5, time_tolerance=0.01)

cluster_data, total_events_list = calculate_total_events(merged_clusters, timestamps_valid, first_timestamp, df_histogram)

plot_cluster_locations(merged_clusters, x_coords_valid, y_coords_valid, merged_mean_time, title="Clusters with More Than 10 '100 Events' lasting less than 500 ms")

size = calculate_cluster_size(merged_cluster_indices, x_coords_valid, y_coords_valid)

