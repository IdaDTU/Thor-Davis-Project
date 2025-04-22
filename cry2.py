##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% LYNDATA
# Load the data
LYN = np.load("/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ThorDavis/Data/Nikon_2023_12_30_20_04_46.mov.LYN.npy", allow_pickle=True)
#LYN = np.load("/zhome/57/6/168999/Desktop/ThorDavis/new3/Nikon_2023_12_30_20_04_46.mov.LYN.npy", allow_pickle=True)

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

# Construct DataFrame
df = pd.DataFrame({
    'Longitude': [entry['location']['coordinates'][0] for entry in LYN],
    'Latitude': [entry['location']['coordinates'][1] for entry in LYN],
    'SignalStrengthKA': [entry['signalStrengthKA'] for entry in LYN],
    'SignalStrengthKA_abs': [abs(entry['signalStrengthKA']) for entry in LYN],
    'datetime': [entry['time'] for entry in LYN],
    'cloud': [entry['cloud'] for entry in LYN],
    'multiplicity': [entry.get('multiplicity') for entry in LYN]})

# Convert to datetime before filtering
df['datetime'] = pd.to_datetime(df['datetime'])

df.loc[(df['multiplicity'] > 0) & (df['cloud'] == 1), 'type'] = 'MS + IC'
df.loc[(df['multiplicity'] > 0) & (df['cloud'] == 0), 'type'] = 'MS + CG'
df.loc[(df['multiplicity'] > 0) & (pd.isna(df['cloud'])), 'type'] = 'MS'
df.loc[(pd.isna(df['multiplicity'])) & (df['cloud'] == 1), 'type'] = 'IC'
df.loc[(pd.isna(df['multiplicity'])) & (df['cloud'] == 0), 'type'] = 'CG'
df.loc[pd.isna(df['multiplicity']) & pd.isna(df['cloud']), 'type'] = 'Unknown'

# Calculate time delta from the defined start timestamp
start = pd.Timestamp("2023-12-30 20:04:28.000000+00:00")
end = pd.Timestamp("2023-12-30 20:06:56.000000+00:00")

# Time range filtering first
filt_df = df[(df['datetime'] >= start) & (df['datetime'] <= end)].sort_values(by='datetime')

# Now calculate seconds from 'start', not from first timestamp
filt_df['delta'] = filt_df['datetime'] - start
filt_df['seconds_from_start'] = filt_df['delta'].dt.total_seconds()
filt_df['minutes_from_start'] = filt_df['seconds_from_start'] / 60


# Spatial filtering
lon_min, lon_max = 21, 26
lat_min, lat_max = -27, -24.5
filt_df = filt_df[(filt_df['Latitude'] >= lat_min) & (filt_df['Latitude'] <= lat_max) &
                  (filt_df['Longitude'] >= lon_min) & (filt_df['Longitude'] <= lon_max)]

# Filter strikes that happened within the first 130 seconds
first_130_sec_filt_df = filt_df[filt_df['seconds_from_start'] <= 130]
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
filt_df.loc[:, 'weighted_strength'] = filt_df['SignalStrengthKA_abs'] / (filt_df['distance_to_iss_km'])**power * 1000  # Weighted strength
#filt_df.loc[:, 'weighted_strength'] = 1

print('Finished filtering!')
print(filt_df['seconds_from_start'])

#%% Cluster
import dv_processing as dvp
import glob
import math
import numpy as np
import pandas as pd
from video_information import create_dataframe, collect_on_events
from plotting import plot_cluster_locations, plot_event_count, plot_event_timeline, plot_variable
from dbscan import dbscan, Hdbscan, filter_and_merge_clusters
from output_df import calculate_total_events, calculate_cluster_sizes, calculate_event_rate, create_and_save_df, create_and_save_df2, calculate_cluster_sizes2
#%%
# Define the path to your Aedat file
FIL = glob.glob("/Users/josephine/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/ThorDavis/Data/dvSave-2023_12_30_20_04_46.aedat4", recursive=True)[0]
#FIL = glob.glob("/zhome/57/6/168999/Desktop/ThorDavis/dvSave-2023_12_30_20_04_46.aedat4", recursive=True)[0]

print('1')
# Open the file
capture = dvp.io.MonoCameraRecording(FIL)
print('2')
# Choose what time interval to study. 
time1 = 0         # seconds
time2 = 130       # seconds
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
min_samples = 4 # determines how sensitive it is to noise
min_clusters = 1500
time_tolerance = 1
#name = 'OldTime_NewArea_500msBins'
name = 'test'
bin_size = 0.001
print(name)
# Find clusters
df = dbscan(event_array,      # input time is in 200 microseconds
             first_timestamp, # return time is in seconds
             eps=eps, #7
             min_samples=min_samples) #120

#df = Hdbscan(event_array, 
 #            first_timestamp, 
  #           min_cluster_size=min_clusters,
   #          min_samples=min_samples)
#%%
print(df['labels'].value_counts())
print(len(df['labels']))

cluster_counts = df['labels'].value_counts()
cluster_counts[cluster_counts != -1].describe()


#%%
# Filter and merge clusters
filtered_df = filter_and_merge_clusters(df,
                                        min_clusters=min_clusters,    #1425     # amount of clusters to form a lightning
                                        max_duration=1,         # seconds
                                        time_tolerance=time_tolerance,     # seconds
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
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(output_df)
 
 #%%
 
print('Max event times:', len(output_df['max event time']))
#print(output_df['max event time'])
print('Max events:', len(output_df['max event']))
#print(output_df['max event'])
print('GLD times:', len(filt_df['seconds_from_start']))
print(filt_df['seconds_from_start'])
print('GLD strength:', len(filt_df['weighted_strength']))
#print(filt_df['weighted_strength'])


#%% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Extract data
max_event_times = output_df['max event time'] #+ 28
#max_events = output_df['max event']
max_events = np.ones(len(max_event_times))


gld_times = filt_df['seconds_from_start'] # leap seconds
#gld_strength = filt_df['weighted_strength']
gld_strength = np.ones(len(gld_times))


#%%  ------------- prik-before -------------------
import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
plt.figure(figsize=(15, 5))

# Plot Max Events as blue bars +0.352
plt.bar(max_event_times+0.2, max_events, width=0.3, color='tab:orange', label='Max Events')

type_colors = {
	'MS + IC': '#FF7F0E',
	'MS + CG': '#1F77B4',
	'MS': '#2CA02C',
	'IC': '#D62728',
	'CG': '#9467BD',
	'Unknown': '#A9A9A9'
	}

for t, color in type_colors.items():
 	subset = filt_df[filt_df['type'] == t]
 	plt.scatter(
 		subset['seconds_from_start'],
 		np.ones(len(subset['seconds_from_start'])),
 		s=10,
 		color=color,
 		alpha=1,
 		label=t
 		)

# Plot GLD times as red dots
#plt.scatter(gld_times, gld_strength, color='blue', s=10, label='GLD Detections', zorder=5)

# Labels and formatting
plt.xlabel('Time (s)')
plt.ylabel('Event Presence')
plt.title('Event Times (bars) and GLD Times (dots)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.locator_params(axis='x', nbins=10)  # Reduce number of ticks
plt.tight_layout()
#plt.savefig(f'/zhome/57/6/168999/Desktop/ThorDavis/hdbscan/final/{min_clusters}_{min_samples}_{time_tolerance}_prikbefore_{name}.png', dpi=300) 
plt.show()


#%% prik-correlation -----------------
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Assume these are your arrays of event and GLD times in seconds
event_times = np.array(max_event_times)        # from bars (orange)
gld_times = np.array(filt_df['seconds_from_start'])           # from dots (colored)

# Create a common time axis
start_time = min(event_times.min(), gld_times.min())
end_time = max(event_times.max(), gld_times.max())
dt = 0.1  # resolution in seconds
time_axis = np.arange(start_time, end_time, dt)

# Convert event and gld times into impulses on the common time axis
event_series = np.zeros_like(time_axis)
gld_series = np.zeros_like(time_axis)

# For each event/GLD time, find the closest index in time_axis and set to 1
event_indices = np.searchsorted(time_axis, event_times)
gld_indices = np.searchsorted(time_axis, gld_times)

event_indices = event_indices[event_indices < len(event_series)]
gld_indices = gld_indices[gld_indices < len(gld_series)]

event_series[event_indices] = 1
gld_series[gld_indices] = 1

# Normalize
event_series = (event_series - np.mean(event_series)) / np.std(event_series)
gld_series = (gld_series - np.mean(gld_series)) / np.std(gld_series)

# Cross-correlation
corr = correlate(event_series, gld_series, mode='full')
lags = np.arange(-len(event_series) + 1, len(event_series)) * dt

# Get indices of the 5 highest correlation values
top_5_indices = np.argsort(corr)[-5:][::-1]  # descending order

# Get the corresponding lags and correlation values
top_5_lags = lags[top_5_indices]
top_5_corrs = corr[top_5_indices]

# Print results
for lag, value in zip(top_5_lags, top_5_corrs):
    print(f"Lag: {lag}, Correlation: {value}")

# Find lag with maximum correlation
best_lag_index = np.argmin(np.abs(top_5_lags))
best_lag = top_5_lags[best_lag_index]
max_corr_value = top_5_corrs[best_lag_index]

# Print result
print(f'Best lag: {best_lag} seconds)')
print(f'Max correlation: {max_corr_value:.3f}')

# Plot
plt.figure(figsize=(12, 5))
plt.scatter(lags, corr, color='purple')
plt.scatter(best_lag, max_corr_value, s=20, color='pink', label=f'Best lag: {best_lag}s')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Cross-Correlation between Event Times and GLD Times")
plt.xlabel("Lag (seconds)")
plt.ylabel("Cross-correlation")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig(f'/zhome/57/6/168999/Desktop/ThorDavis/hdbscan/final/{min_clusters}_{min_samples}_{time_tolerance}_prikcorr_{name}.png', dpi=300) 
plt.show()



#%% # prik-after ----------------

plt.figure(figsize=(15, 5))

shifted_event = max_event_times.copy()
shifted_event = shifted_event + best_lag  # or best_lag_seconds if you already calculated it

# Plot Max Events as blue bars
plt.bar(shifted_event, np.ones(len(shifted_event)), width=0.3, color='tab:orange', label='Max Events')

for t, color in type_colors.items():
 	subset = filt_df[filt_df['type'] == t]
 	plt.scatter(
 		subset['seconds_from_start'],
 		np.ones(len(subset['seconds_from_start'])),
 		s=10,
 		color=color,
 		alpha=1,
 		label=t
 		)

# Labels and formatting
plt.xlabel('Time (s)')
plt.ylabel('Event Presence')
plt.title('Event Times (bars) and GLD Times (dots)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.locator_params(axis='x', nbins=10)  # Reduce number of ticks
plt.tight_layout()
#plt.savefig(f'/zhome/57/6/168999/Desktop/ThorDavis/hdbscan/final/{min_clusters}_{min_samples}_{time_tolerance}_prikafter_{name}.png', dpi=300) 
plt.show()





#%% binned plots -------------------

# Create bins from 0 to max time using variable bin_size
max_time = max(max_event_times.max(), gld_times.max())
bins = np.arange(0, np.ceil(max_time) + bin_size, bin_size)

# Bin max events
event_binned = pd.DataFrame({'time': max_event_times, 'event': max_events})
event_binned = event_binned.groupby(pd.cut(event_binned['time'], bins), observed=False).sum()
event_binned.index = event_binned.index.map(lambda x: x.left)

# Bin GLD strength
# Create bin labels
gld_binned = pd.DataFrame({
    'time': gld_times,
    'strength': gld_strength,
    'type': filt_df['type']
})
gld_binned['bin'] = pd.cut(gld_binned['time'], bins, labels=bins[:-1], include_lowest=True)

# Now group by bin and type
gld_grouped = gld_binned.groupby(['bin', 'type'], observed=False)['strength'].sum().reset_index()
gld_grouped['bin'] = gld_grouped['bin'].astype(float)  # convert Interval index to float bin start

# Create aligned time axis
all_times = np.arange(0, np.ceil(max_time), bin_size)
event_binned = event_binned.reindex(all_times, fill_value=0)
gld_binned = gld_binned.reindex(all_times, fill_value=0)

# Plot
fig, ax1 = plt.subplots(figsize=(15, 6))
bar_width = 0.3  # scale bar width with bin size

# Plot max events (orange)
#ax1.bar(all_times - bar_width/2, event_binned['event'], width=bar_width,
       # color='tab:orange', label='Max Events', alpha=0.7)
ax1.bar(event_binned['time'] - bar_width/2, event_binned['event'], width=bar_width,
        color='tab:orange', label='Max Events', alpha=0.7)
ax1.set_ylabel('Max Events', color='tab:orange')
ax1.tick_params(axis='y', labelcolor='tab:orange')

# Continue plotting
ax2 = ax1.twinx()

for t, color in type_colors.items():
    subset = gld_grouped[gld_grouped['type'] == t]
    subset = subset[subset['strength'] > 0]
    ax2.scatter(
        subset['bin'],
        subset['strength'],
        s=10,
        color=color,
        alpha=1,
        label=t
    )
    
ax2.set_ylabel('GLD Strength', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(title='Lightning Type', loc='upper left', bbox_to_anchor=(1.05,1))

# Auto-adjust x-axis ticks based on bin size
tick_interval = max(1, 10, 10)  # adjust how frequent ticks should be
xticks = np.arange(0, np.ceil(max_time), tick_interval)
ax1.set_xticks(xticks)
ax1.set_xlim(0, np.ceil(max_time))  # ensure all bars are visible

# Optional: rotate x-tick labels if very dense
if bin_size < 0.5:
    plt.xticks(rotation=45)

# Final formatting
ax1.set_xlabel('Time (s)')
plt.title(f'Binned Max Events and GLD Strength ({bin_size}s bins)')
plt.tight_layout()
#plt.savefig(f'/zhome/57/6/168999/Desktop/ThorDavis/hdbscan/final/{min_clusters}_{min_samples}_{time_tolerance}_before_{name}.png', dpi=300) 
plt.show()
plt.close()
#%%
# Find overlapping non-zero bins
overlap_mask = (event_binned['event'] > 0) & (gld_binned['strength'] > 0)
num_overlapping_bins = overlap_mask.sum()

print(f"Number of overlapping bins before shift: {num_overlapping_bins}")
bin_interval = 500
# Convert GLD presence to a binary array
gld_binary = (gld_binned['strength'] > 0).astype(int)

# Create a rolling sum over ±10 bins (21-bin window centered)
gld_window = gld_binary.rolling(window=bin_interval*2+1, center=True, min_periods=1).sum()

# Any event bin with gld_window > 0 is within ±10 bins of a GLD
fuzzy_overlap_mask = (event_binned['event'] > 0) & (gld_window > 0)
fuzzy_overlap_count = fuzzy_overlap_mask.sum()

print(f"Number of fuzzy-overlapping bins (within ± {bin_interval*bin_size}s): {fuzzy_overlap_count}")

#%%
from scipy.signal import correlate
import numpy as np
import matplotlib.pyplot as plt

# Get the binned arrays
event_series = event_binned['event'].values
gld_series = gld_binned['strength'].values

# Normalize both series (zero-mean)
event_series = (event_series - np.mean(event_series)) / np.std(event_series)
gld_series = (gld_series - np.mean(gld_series)) / np.std(gld_series)

# Compute cross-correlation
corr = correlate(event_series, gld_series, mode='full')
lags = np.arange(-len(event_series) + 1, len(event_series))
#lags = np.arange(-25, 25)

# Get indices of the 5 highest correlation values
top_5_indices = np.argsort(corr)[-15:][::-1]  # descending order

# Get the corresponding lags and correlation values
top_5_lags = lags[top_5_indices]
top_5_corrs = corr[top_5_indices]

# Print results
for lag, value in zip(top_5_lags, top_5_corrs):
    print(f"Lag: {lag*bin_size}, Correlation: {value}")

# Find lag with maximum correlation
best_lag_index = np.argmin(np.abs(top_5_lags))
best_lag = top_5_lags[best_lag_index]
max_corr_value = top_5_corrs[best_lag_index]

# Convert lag to seconds
best_lag_seconds = best_lag * bin_size

# Print result
print(f'Best lag: {best_lag} bins ({best_lag_seconds:.2f} seconds)')
print(f'Max correlation: {max_corr_value:.3f}')

# Plot correlation vs lag
plt.figure(figsize=(12, 5))
plt.scatter(lags*bin_size, corr, s=0.7, color='purple', label='Cross-correlations')
plt.scatter(best_lag_seconds,max_corr_value, s=20, color='pink', label=f'Best lag: {best_lag_seconds:.2f}s')
#plt.axvline(best_lag_seconds, color='r', linestyle='--', label=f'Best lag = {best_lag_seconds:.2f}s')
plt.xlabel('Lag (seconds)')
plt.ylabel('Cross-correlation')
plt.title('Cross-Correlation between Event and GLD Strength Time Series')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig(f'/zhome/57/6/168999/Desktop/ThorDavis/hdbscan/final/{min_clusters}_{min_samples}_{time_tolerance}_corr_{name}.png', dpi=300) 
plt.show()
plt.close()

 #%%
# --- Shift event_binned by best_lag ---
shifted_event = event_binned['event'].shift(best_lag, fill_value=0)

# Round all indexes consistently
event_binned.index = np.round(event_binned.index, 10)
gld_binned.index = np.round(gld_binned.index, 10)

# Determine shifted event min and max
shifted_event_min = np.round(event_binned.index.min() + best_lag * bin_size, 10)
shifted_event_max = np.round(event_binned.index.max() + best_lag * bin_size, 10)

# Get full time range covering both datasets (plus padding)
total_min = min(shifted_event_min, gld_binned.index.min())
total_max = max(shifted_event_max, gld_binned.index.max()) + 2 * bin_size

# Generate extended time axis (and round it)
extended_times = np.round(np.arange(total_min, total_max + bin_size, bin_size), 10)

# Create empty extended series (float dtype)
shifted_event_extended = pd.Series(0.0, index=extended_times)
gld_extended = pd.Series(0.0, index=extended_times)

# --- Assign shifted max events ---
shifted_indices = np.round(event_binned.index + best_lag * bin_size, 10)
valid_mask = shifted_indices.isin(shifted_event_extended.index)
valid_indices = shifted_indices[valid_mask]
valid_values = event_binned['event'].values[valid_mask]
shifted_event_extended.loc[valid_indices] = valid_values

# --- Assign GLD strength safely using index intersection ---
matching_gld_index = gld_binned.index.intersection(gld_extended.index)
gld_extended.loc[matching_gld_index] = gld_binned.loc[matching_gld_index, 'strength'].values

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(15, 6))
bar_width = 0.3

# Plot shifted events (orange bars)
ax1.bar(extended_times - bar_width/2, shifted_event_extended.values, width=bar_width,
        color='tab:orange', label='Max Events (lagged)', alpha=0.7)
#ax1.bar(shifted_indices - bar_width/2, shifted_event_extended.values, width=bar_width,
 #       color='tab:orange', label='Max Events (lagged)', alpha=0.7)
ax1.set_ylabel('Max Events (lagged)', color='tab:orange')
ax1.tick_params(axis='y', labelcolor='tab:orange')

# Plot GLD strength (blue bars)
ax2 = ax1.twinx()
for t, color in type_colors.items():
    subset = gld_grouped[gld_grouped['type'] == t]
    subset = subset[subset['strength'] > 0]
    ax2.scatter(
        subset['bin'],
        subset['strength'],
        s=10,
        color=color,
        alpha=1,
        label=t
    )
ax2.set_ylabel('GLD Strength', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(title='Lightning Type', loc='upper left', bbox_to_anchor=(1.05,1))


# Labels and formatting
ax1.set_xlabel('Time (s)')
plt.title(f'All Values: Shifted Max Events vs. GLD Strength (Lag = {best_lag * bin_size:.2f} s)')
ax1.set_xlim(extended_times.min(), extended_times.max())
plt.tight_layout()
#plt.savefig(f'/zhome/57/6/168999/Desktop/ThorDavis/hdbscan/final/{min_clusters}_{min_samples}_{time_tolerance}_after_{name}.png', dpi=300) 
plt.show()
plt.close()

#%%
# Find overlapping non-zero bins
overlap_mask = (shifted_event_extended > 0) & (gld_extended > 0)
num_overlapping_bins = overlap_mask.sum()

print(f"Number of overlapping bins after shift: {num_overlapping_bins}")
 
bin_interval = 50
# Convert GLD presence to a binary array
gld_binary = (gld_extended > 0).astype(int)

# Create a rolling sum over ±10 bins (21-bin window centered)
gld_window = gld_binary.rolling(window=bin_interval*2+1, center=True, min_periods=1).sum()

# Any event bin with gld_window > 0 is within ±10 bins of a GLD
fuzzy_overlap_mask = (shifted_event_extended > 0) & (gld_window > 0)
fuzzy_overlap_count = fuzzy_overlap_mask.sum()

print(f"Number of fuzzy-overlapping bins (within ± {bin_interval*bin_size}s): {fuzzy_overlap_count}")

 

 
 
 
 
 
 
 
 
 