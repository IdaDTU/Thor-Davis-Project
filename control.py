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

df = pd.DataFrame({
    'Longitude': LON,
    'Latitude': LAT,
    'SignalStrengthKA_abs': np.abs(CUR),
    'SignalStrengthKA': CUR,
    'time': TIME,
    'cloud': CLOUD,
    'multiplicity': MULTI
})


# Convert time to datetime and compute fractional minutes
df['datetime'] = pd.to_datetime(df['time'])
df['minutes'] = df['datetime'].dt.minute + df['datetime'].dt.second / 60 + df['datetime'].dt.microsecond / 1e6 / 60

# Convert time to datetime
df['datetime'] = pd.to_datetime(df['time'])

# Define your datetime range
start = pd.Timestamp("2023-12-30 20:04:46.440000+00:00")
end = pd.Timestamp("2023-12-30 20:06:56.000000+00:00")

# Filter rows within the range
df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]

# Sort (optional)
df = df.sort_values(by='datetime')

# Show result
print(df[['time', 'datetime']])

# Spatial filter
lon_min, lon_max = 21, 34
lat_min, lat_max = -32, -24.5


filt_df = df[
    (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
    (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]

from geopy.distance import geodesic
from skyfield.api import load, EarthSatellite

# TLE for ISS on December 30, 2023
line1 = "1 25544U 98067A   23363.69831336  .00019743  00000-0  35458-3 0  9998"
line2 = "2 25544  51.6430  80.0897 0003230 327.3849 132.1384 15.49874161432116"

# Load timescale and satellite
ts = load.timescale()
sat = EarthSatellite(line1, line2, "ISS", ts)

# Time window centered at TLE epoch
minutes = np.arange(0, 7)  # 7 values
times = ts.utc(2023, 12, 30, 20, 4 + minutes)
# Satellite subpoint positions
geocentric = sat.at(times)
subpoint = geocentric.subpoint()
iss_lats = subpoint.latitude.degrees
iss_lons = subpoint.longitude.degrees

# Precompute ISS track as list of (lat, lon) pairs
iss_track = list(zip(iss_lats, iss_lons))

# Function to find the minimum geodesic distance from one point to the ISS track
def min_distance_to_track(lat, lon, track):
    return min([geodesic((lat, lon), (t_lat, t_lon)).km for t_lat, t_lon in track])

# Apply to each lightning strike in filtered_df
filt_df['distance_to_iss_km'] = [
    min_distance_to_track(lat, lon, iss_track)
    for lat, lon in zip(filt_df['Latitude'], filt_df['Longitude'])
]

# Define weighting function — can tweak epsilon and exponent for smoother or sharper effects
power = 2  # increase to reduce impact of distant strikes more aggressively
filt_df['weighted_strength'] = filt_df['SignalStrengthKA_abs'] / (filt_df['distance_to_iss_km'])**power * 1000

print(filt_df['weighted_strength'].describe())

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

print('1')
# Open the file
capture = dvp.io.MonoCameraRecording(FIL)
print('2')
# Choose what time interval to study. 
time1 = 0         # seconds
time2 = 150        # seconds
frame_rate = 40   # Davis 346: 40fps, Quickreader: 60fps (no periods before 1000)

num_batches = min(math.ceil(1000 * time2), 41500)
print('3')
df_histogram = create_dataframe(FIL, 
                                num_bins=10*num_batches,    #50*num_batches
                                frame_rate=frame_rate,
                                batches=num_batches)
df_histogram = df_histogram[(df_histogram['timestamps_sec'] >= time1) & (df_histogram['timestamps_sec'] <= time2)]
print('4')
event_array, first_timestamp = collect_on_events(capture, num_batches, df_histogram)
#%%
eps = 5
min_samples = 4
min_clusters = 2300

# Find clusters
df = dbscan(event_array,      # input time is in 250 microseconds
             first_timestamp, # return time is in seconds
             eps=eps, #7
             min_samples=min_samples) #120

# Filter and merge clusters
filtered_df = filter_and_merge_clusters(df,
                                        min_clusters=min_clusters,    #1425     # amount of clusters to form a lightning
                                        max_duration=0.05,         # seconds
                                        time_tolerance=0.05,     # seconds
                                        frame_rate=frame_rate)    
#print(filtered_df.columns)

# Creating the output df and cluster histograms
total_events, max_event = calculate_total_events(df=filtered_df,
                                      df_histogram=df_histogram,
                                      xaxis='time',    # either 'time' or 'frames'
                                      filename=None,   # use strings and no format: fx: 'plot' or None
                                      show=False)       # either True or False

size = calculate_cluster_sizes2(filtered_df)

rate = calculate_event_rate(total_events, 
                            filtered_df['period'])

#%%
# Print/save output df
output_df = create_and_save_df2(filtered_df,
                               df_histogram,
                               size,
                               total_events, 
                               max_event,
                               rate,
                               filename = None,    # use strings and no format: fx: 'data' or None
                               decimals = 3)
# Temporarily expand display to show full DataFrame
#with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
 #   print(output_df)
   
#%% Histogram samlet
# --- Cluster data setup ---
# Bin size in seconds — 
bin_size = 0.001  # seconds

event_type = 'max event'
#event_type = 'total events'
#event_type = 'norm size'
#strength_type = 'weighted_strength'
strength_type = 'SignalStrengthKA_abs'

# Compute bin index and range
output_df['second_bin'] = (output_df['max event time'] // bin_size) * bin_size
bin_range = np.arange(0, 131, bin_size).round(3)  # round to avoid float imprecision

# Bin the data
cluster_binned = (
    output_df.groupby('second_bin')[event_type]
    .sum()
    .reindex(bin_range, fill_value=0)
    .reset_index()
)

# --- Signal strength setup ---
#filt_df['second_bin'] = filt_df['seconds'].astype(int)
filt_df.loc[:, 'second_bin'] = filt_df['seconds'].astype(int)


signal_binned = (
    filt_df.groupby('second_bin')[strength_type]
    .sum()
    .reindex(bin_range, fill_value=0)
    .reset_index()
)
norm = np.ones_like(signal_binned['second_bin'])
# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(14, 5))

# Bar for signal strength (on primary y-axis)
ax1.bar(
    #signal_binned['second_bin'],
    norm,
    np.abs(signal_binned[strength_type]),
    width=0.25,
    color='blue',
    alpha=0.5,
    label='Signal Strength (kA)'
)
ax1.set_xlabel('Time Since Start (s)')
ax1.set_ylabel('Signal Strength (kA)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Line for total events (on secondary y-axis)
ax2 = ax1.twinx()
ax2.bar(
    cluster_binned['second_bin'],
    cluster_binned[event_type],
    width = 0.25,
    color='red',
    alpha=0.5,
    label='Max Event Count'
)
ax2.set_ylabel('Clustered Lightning Events', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# X-ticks every 5 seconds
plt.xticks(
    ticks=np.arange(0, 131, 10),
    labels=[f"{s}s" for s in np.arange(0, 131, 10)],
    rotation=45
)

# Titles and layout
plt.title('Strength vs Max Events')
fig.tight_layout()
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

# Optional: add legend
fig.legend(loc='upper right', bbox_to_anchor=(0.92, 0.92))

plt.show()

#%% Cross-correlation to find delay
import numpy as np
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt

strength = np.abs(signal_binned[strength_type]).to_numpy()
events = cluster_binned[event_type].to_numpy()
# Now find cross-correlation
#correlation = correlate(filtered_ms - np.mean(filtered_ms), output_ms - np.mean(output_ms), mode='full')
correlation = correlate(strength, events, mode='full')
lags = correlation_lags(len(strength), len(events), mode='full')
# Limit lags to ±100
max_lag = 100
lag_mask = np.abs(lags) <= max_lag
limited_correlation = correlation[lag_mask]
limited_lags = lags[lag_mask]
print(f"Max correlation value: {np.max(correlation)}")
print(f"Min correlation value: {np.min(correlation)}")

best_lag = lags[np.argmax(correlation)]

print("Best lag:", best_lag*bin_size, "seconds (",best_lag,' bins )')

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

# Calculate x-axis in real time
#signal_x = np.arange(len(strength)) * bin_size
signal_x = np.arange(len(norm)) * bin_size
event_x = (np.arange(len(events)) + best_lag) * bin_size

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(14, 5))

# Bar for signal strength (on primary y-axis)
ax1.bar(
    signal_x,
    strength,
    width=0.25,
    color='blue',
    alpha=0.5,
    label='Signal Strength [kA]'
)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Signal Strength [kA]', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Bar for clustered lightning events (on secondary y-axis)
ax2 = ax1.twinx()
ax2.bar(
    event_x,
    events,
    width=0.25,
    color='red',
    alpha=0.5,
    label='Clustered Event Count'
)
ax2.set_ylabel('Clustered Lightning Events', color='red')

# X-ticks every 10 seconds (independent of bin count)
max_time = max(np.max(signal_x), np.max(event_x))
xtick_vals = np.arange(0, max_time + 10, 10)
plt.xticks(
    ticks=xtick_vals,
    labels=[f"{int(s)}s" for s in xtick_vals],
    rotation=45
)

# Titles and layout
plt.title(f'eps: {eps}, min_samples: {min_samples}, min_clusters: {min_clusters}, max correlation: {np.max(correlation):.0f}')
fig.tight_layout()
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
fig.legend(loc='upper right', bbox_to_anchor=(0.92, 0.92))

plt.show()



