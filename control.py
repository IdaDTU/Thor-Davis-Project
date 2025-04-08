##

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% LYNDATA
# Load the data
LYN = np.load("/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ThorDavis/Data/Nikon_2023_12_30_20_04_46.mov.LYN.npy", allow_pickle=True)

# Extract relevant fields
LON = [entry['location']['coordinates'][0] for entry in LYN]
LAT = [entry['location']['coordinates'][1] for entry in LYN]
CUR = [entry['signalStrengthKA'] for entry in LYN]
TIME = [entry['time'] for entry in LYN]
CLOUD = [entry['cloud'] for entry in LYN]
MULTI = [entry['multiplicity'] if 'multiplicity' in entry else None for entry in LYN]

# Create DataFrame
df = pd.DataFrame({
    'Longitude': LON,
    'Latitude': LAT,
    'SignalStrengthKA': CUR,
    'time': TIME,
    'cloud': CLOUD,
    'multiplicity':MULTI
})

# Convert time to datetime and compute fractional minutes
df['datetime'] = pd.to_datetime(df['time'])
df['minutes'] = df['datetime'].dt.minute + df['datetime'].dt.second / 60 + df['datetime'].dt.microsecond / 1e6 / 60

# Temporal filter
df = df[(df['minutes'] >= 4.46) & (df['minutes'] <= 6.56)]

# Compute seconds since the start for timeline plotting
start_time = df['datetime'].min()
df['seconds'] = (df['datetime'] - start_time).dt.total_seconds()

# Spatial filter
lon_min, lon_max = 21, 34
lat_min, lat_max = -32, -24.5


filt_df = df[
    (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
    (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]

print('Finished filtering!')

#%% Cluster
import dv_processing as dvp
import glob
import math
import numpy as np
import pandas as pd
from video_information import create_dataframe, collect_on_events
from plotting import plot_cluster_locations, plot_event_count, plot_event_timeline, plot_variable
from dbscan import dbscan, filter_and_merge_clusters
from output_df import calculate_total_events, calculate_cluster_sizes, calculate_event_rate, create_and_save_df, create_and_save_df2, calculate_cluster_sizes2

# Define the path to your Aedat file
FIL = glob.glob("/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ThorDavis/Data/dvSave-2023_12_30_20_04_46.aedat4", recursive=True)[0]

# Open the file
capture = dvp.io.MonoCameraRecording(FIL)

# Choose what time interval to study. 
time1 = 0         # seconds
time2 = 130        # seconds
frame_rate = 40   # Davis 346: 40fps, Quickreader: 60fps (no periods before 1000)

num_batches = min(math.ceil(1000 * time2), 41500)

df_histogram = create_dataframe(FIL, 
                                num_bins=10*num_batches,    #50*num_batches
                                frame_rate=frame_rate,
                                batches=num_batches)
df_histogram = df_histogram[(df_histogram['timestamps_sec'] >= time1) & (df_histogram['timestamps_sec'] <= time2)]

event_array, first_timestamp = collect_on_events(capture, num_batches, df_histogram)
#%%
eps = 7
min_samples = 120
min_clusters = 1400

# Find clusters
df = dbscan(event_array,      # input time is in 200 microseconds
             first_timestamp, # return time is in seconds
             eps=eps, #7
             min_samples=min_samples) #120

# Filter and merge clusters
filtered_df = filter_and_merge_clusters(df,
                                        min_clusters=min_clusters,    #1425     # amount of clusters to form a lightning
                                        max_duration=0.01,         # 10 ms
                                        time_tolerance=0.0005,     # 0.5 ms
                                        frame_rate=frame_rate)    
#print(filtered_df.columns)

# Creating the output df and cluster histograms
total_events = calculate_total_events(df=filtered_df,
                                      df_histogram=df_histogram,
                                      xaxis='time',    # either 'time' or 'frames'
                                      filename=None,   # use strings and no format: fx: 'plot' or None
                                      show=False)       # either True or False

size = calculate_cluster_sizes2(filtered_df)

rate = calculate_event_rate(total_events, 
                            filtered_df['period'])
#%%
print("Timestamps sample:", filtered_df['timestamps'].iloc[0])
print("Counts sample:", df_histogram['count'].iloc[1])


# Print/save output df
output_df = create_and_save_df2(filtered_df,
                               df_histogram,
                               size,
                               total_events, 
                               rate,
                               filename = None,    # use strings and no format: fx: 'data' or None
                               decimals = 3)
# Temporarily expand display to show full DataFrame
#with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
   # print(output_df)

#%% Histogram samlet
# --- Cluster data setup ---
output_df['second_bin'] = output_df['total events'].astype(int)
bin_range = np.arange(0, 130, 0.5)

cluster_binned = (
    output_df.groupby('second_bin')['total events']
    .sum()
    .reindex(bin_range, fill_value=0)
    .reset_index()
)

# --- Signal strength setup ---
filt_df['second_bin'] = filt_df['seconds'].astype(int)

signal_binned = (
    filt_df.groupby('second_bin')['SignalStrengthKA']
    .sum()
    .reindex(bin_range, fill_value=0)
    .reset_index()
)

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(14, 5))

# Bar for signal strength (on primary y-axis)
ax1.bar(
    signal_binned['second_bin'],
    np.abs(signal_binned['SignalStrengthKA']),
    width=0.9,
    color='skyblue',
    alpha=0.6,
    label='Summed Signal Strength (kA)'
)
ax1.set_xlabel('Time Since Start (s)')
ax1.set_ylabel('Summed Signal Strength (kA)', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# Line for total events (on secondary y-axis)
ax2 = ax1.twinx()
ax2.bar(
    cluster_binned['second_bin'],
    cluster_binned['total events'],
    color='darkorange',
    alpha=0.5,
    linewidth=2,
    label='Clustered Event Count'
)
ax2.set_ylabel('Clustered Lightning Events', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

# X-ticks every 5 seconds
plt.xticks(
    ticks=np.arange(0, 131, 10),
    labels=[f"{s}s" for s in np.arange(0, 131, 10)],
    rotation=45
)

# Titles and layout
plt.title('Lightning Activity: Signal Strength vs Clustered Events')
fig.tight_layout()
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

# Optional: add legend
fig.legend(loc='upper right', bbox_to_anchor=(0.92, 0.92))

plt.show()

#%% Cross-correlation to find delay
import numpy as np
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt

strength = np.abs(signal_binned['SignalStrengthKA']).to_numpy()
events = cluster_binned['total events'].to_numpy()
# Now find cross-correlation
#correlation = correlate(filtered_ms - np.mean(filtered_ms), output_ms - np.mean(output_ms), mode='full')
correlation = correlate(strength, events, mode='full')
lags = correlation_lags(len(strength), len(events), mode='full')
print(f"Max correlation value: {np.max(correlation)}")
print(f"Min correlation value: {np.min(correlation)}")

best_lag = lags[np.argmax(correlation)]

print(f"Best lag: {best_lag}")

# Plot cross-correlation
plt.figure(figsize=(10,4))
plt.plot(lags, correlation)
plt.title("Cross-correlation")
plt.xlabel("Lag")
plt.ylabel("Correlation coefficient")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Shifted histogram
import matplotlib.pyplot as plt
import numpy as np

# Calculate new x-axis for shifted output_ms
new_axis = np.arange(len(events))
shifted_axis = new_axis + best_lag  # shift by lag

# Create full x-range
min_index = min(0, np.min(shifted_axis))
max_index = max(len(strength), np.max(shifted_axis))

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(14, 5))

# Bar for signal strength (on primary y-axis)
ax1.bar(
    np.arange(len(strength)),
    strength,
    width=0.9,
    color='skyblue',
    alpha=0.9,
    label='Summed Signal Strength (kA)'
)
ax1.set_xlabel('Time Since Start (s)')
ax1.set_ylabel('Summed Signal Strength (kA)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Line for total events (on secondary y-axis)
ax2 = ax1.twinx()
ax2.bar(
    shifted_axis,
    events,
    color='darkorange',
    alpha=0.5,
    linewidth=2,
    label='Clustered Event Count'
)
ax2.set_ylabel('Clustered Lightning Events', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

# X-ticks every 5 seconds
plt.xticks(
    ticks=np.arange(0, 261+best_lag, 15),
    labels=[f"{s}s" for s in np.arange(0, 261+best_lag, 15)],
    rotation=45
)

# Titles and layout
plt.title(f'eps: {eps}, min_samples: {min_samples}, min_clusters: {min_clusters}, max correlation: {np.max(correlation)}')
fig.tight_layout()
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

# Optional: add legend
fig.legend(loc='upper right', bbox_to_anchor=(0.92, 0.92))

plt.show()


