import dv_processing as dvp
import glob
import math
import numpy as np
from video_information import create_dataframe, collect_on_events
from plotting import plot_cluster_locations, plot_event_count, plot_event_timeline, plot_variable
from dbscan import dbscan, filter_and_merge_clusters
from output_df import calculate_total_events, calculate_cluster_sizes, calculate_event_rate, create_and_save_df

# Define the path to your Aedat file
FIL = glob.glob("/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ThorDavis/Data/dvSave-2023_12_30_20_04_46.aedat4", recursive=True)[0]

# Open the file
capture = dvp.io.MonoCameraRecording(FIL)

# Choose what time interval to study. 
time1 = 0         # seconds
time2 = 10        # seconds
frame_rate = 60   # Davis 346: 40fps, Quickreader: 60fps (no periods before 1000)

num_batches = math.ceil(100 * time2)

df_histogram = create_dataframe(FIL, 
                                num_bins=10*num_batches,
                                frame_rate=frame_rate,
                                batches=num_batches)
df_histogram = df_histogram[(df_histogram['timestamps_sec'] >= time1) & (df_histogram['timestamps_sec'] <= time2)]

event_array, first_timestamp = collect_on_events(capture, num_batches, df_histogram)

#%% Find clusters
df = dbscan(event_array,      # input time is in 10 microseconds
             first_timestamp, # return time is in seconds
             eps=5, 
             min_samples=14)

#%% Filter and merge clusters
filtered_df = filter_and_merge_clusters(df,
                                        min_clusters=10,         # amount of clusters to form a lightning
                                        max_duration=0.01,       # 10 ms
                                        time_tolerance=0.0005,   # 0.5 ms
                                        frame_rate=frame_rate)      

#%% Plot of the clusters location. Only the first 15 legends are shown
plot_cluster_locations(df=filtered_df,
                        title="Location of Lightnings Found with DBSCAN",
                        filename=None,   # use strings and no format: fx: 'plot' or None
                        show=False)       # either True or False

#%% Creating the output df and cluster histograms
total_events = calculate_total_events(df_histogram=df_histogram,
                                      df=filtered_df,
                                      filename=None,   # use strings and no format: fx: 'plot' or None
                                      show=False)      # either True or False

size = calculate_cluster_sizes(filtered_df)

rate = calculate_event_rate(total_events, 
                            filtered_df['period'])

#%% Print/save output df
output_df = create_and_save_df(filtered_df,
                               size,
                               total_events, 
                               rate,
                               filename = None,    # use strings and no format: fx: 'data' or None
                               decimals = 3)
#print(output_df)

#%% Plot location of ON events
plot_event_count(df, 
                 x_res=350, 
                 y_res=260, 
                 vmax=100, 
                 filename=None,  # use strings and no format: fx: 'plot' or None
                 show=False)     # either True or False

#%% Plot timeline with events and clusters
plot_event_timeline(df_histogram, 
                    output_df, 
                    window_size=100, 
                    xaxis='frames',   # either 'time' or 'frames'
                    filename=None,    # use strings and no format: fx: 'plot' or None
                    show=True)        # either True or False

#%% Plot a variable in 2D
plot_variable(df=output_df, 
              xvariable='mean time', 
              yvariable='std (time) [ms]', 
              xlabel='Mean Time [s]', 
              ylabel='std (time) [ms]', 
              title='Temporal Standard Deviation of Clusters Over Time', 
              filename=None,    # use strings and no format: fx: 'plot' or None
              show=False)       # either True or False


