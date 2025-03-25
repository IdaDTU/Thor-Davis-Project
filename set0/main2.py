import dv_processing as dvp
import glob
from video_information import create_dataframe, collect_on_events
from plotting import plot_cluster_locations
import pandas as pd
from dbscan import dbscan, filter_and_merge_clusters
from output_df import calculate_total_events, calculate_cluster_sizes, calculate_event_rate, create_and_save_df

# Define the path to your Aedat file
FIL = glob.glob("/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ThorDavis/Data/dvSave-2023_12_30_20_04_46.aedat4", recursive=True)[0]

# Open the file
capture = dvp.io.MonoCameraRecording(FIL)

# Specify how many event batches you want to process
num_batches = 2000  #41500 is all batches, 1000 is 10 sec

df_histogram = create_dataframe(FIL, 
                                num_bins=10*num_batches,
                                batches=num_batches)

event_array, first_timestamp = collect_on_events(capture,
                                                 num_batches)

df = dbscan(event_array, eps=3, min_samples=100)
#labels, x_coords, y_coords, timestamps

filtered_df = filter_and_merge_clusters(event_array,
                                        df['labels'],
                                        first_timestamp,
                                        min_clusters=10,
                                        max_duration=0.5,
                                        time_tolerance=0.01)

#%% plot of the clusters

plot_cluster_locations(filtered_df['clusters'],
                        filtered_df['x'], 
                        filtered_df['y'], 
                        filtered_df['mean time'], 
                        title="Clusters with More Than 10 '100 Events' lasting less than 500 ms")

#%% creating the output df

total_events = calculate_total_events(filtered_df['clusters'],
                                      filtered_df['start'],
                                      filtered_df['end'],
                                      df_histogram)

size = calculate_cluster_sizes(filtered_df['clusters'], 
                               filtered_df['x'], 
                               filtered_df['y'])

rate = calculate_event_rate(total_events, 
                            filtered_df['period'])

#%%

df = create_and_save_df(filtered_df['start'], 
                       filtered_df['end'], 
                       filtered_df['period'], 
                       size, 
                       total_events, 
                       rate,
                       file_path = '/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ThorDavis/output/lightning_data9.csv',
                       decimals = 3)


# Set display options for pandas to show more rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Prevent wrapping of long lines
pd.set_option('display.max_colwidth', None)  # Allow columns to expand fully
pd.set_option('display.float_format', '{:.3f}'.format)  # Format floats to 3 decimal places

print(df)

