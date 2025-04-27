##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global font settings
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',  # Computer Modern
})

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

df.loc[(df['multiplicity'] > 0) & (df['cloud'] == 1), 'type'] = 'MS + CC'
df.loc[(df['multiplicity'] > 0) & (df['cloud'] == 0), 'type'] = 'MS + CG'
#df.loc[(df['multiplicity'] > 0) & (pd.isna(df['cloud'])), 'type'] = 'MS'
df.loc[(pd.isna(df['multiplicity'])) & (df['cloud'] == 1), 'type'] = 'CC'
df.loc[(pd.isna(df['multiplicity'])) & (df['cloud'] == 0), 'type'] = 'CG'
df.loc[pd.isna(df['multiplicity']) & pd.isna(df['cloud']), 'type'] = 'Unknown'

# Calculate time delta from the defined start timestamp
start = pd.Timestamp("2023-12-30 20:04:28.000000+00:00")   #4:28
end = pd.Timestamp("2023-12-30 20:06:18.000000+00:00")     #6:56

# Time range filtering first
filt_df = df[(df['datetime'] >= start) & (df['datetime'] <= end)].sort_values(by='datetime')

# Now calculate seconds from 'start', not from first timestamp
filt_df['delta'] = filt_df['datetime'] - start
filt_df['seconds_from_start'] = filt_df['delta'].dt.total_seconds()
filt_df['minutes_from_start'] = filt_df['seconds_from_start'] / 60
filt_df['type'] = filt_df['type'].fillna('Unknown')



# Spatial filtering
lon_min, lon_max = 21, 26
lat_min, lat_max = -27, -24.5
filt_df = filt_df[(filt_df['Latitude'] >= lat_min) & (filt_df['Latitude'] <= lat_max) &
                  (filt_df['Longitude'] >= lon_min) & (filt_df['Longitude'] <= lon_max)]

# Filter strikes that happened within the first 130 seconds
first_130_sec_filt_df = filt_df[filt_df['seconds_from_start'] <= 110]
print(len(first_130_sec_filt_df))
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
print(len(filt_df['seconds_from_start']))

#%% Cluster
import dv_processing as dvp
import glob
import math
import numpy as np
import pandas as pd
from video_information import create_dataframe, collect_on_events
from plotting import plot_cluster_locations, plot_event_count, plot_event_timeline, plot_variable
from dbscan import dbscan, Hdbscan, filter_and_merge_clusters, filter_and_merge_clusters2
from output_df import calculate_total_events, calculate_cluster_sizes, calculate_event_rate, create_and_save_df, create_and_save_df2, calculate_cluster_sizes2, calculate_cluster_sizes3
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
time2 = 110       # seconds
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
min_clusters = 700
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
filtered_df = filter_and_merge_clusters2(df,
                                        min_clusters=min_clusters,    #1425     # amount of clusters to form a lightning
                                        max_duration=1,         # seconds
                                        time_tolerance=time_tolerance,     # seconds
                                        frame_rate=frame_rate)    
print(len(filtered_df))
#%%
# Creating the output df and cluster histograms
total_events, max_event = calculate_total_events(df=filtered_df,
                                      df_histogram=df_histogram,
                                      xaxis='time',    # either 'time' or 'frames'
                                      filename='FINAL',   # use strings and no format: fx: 'plot' or None
                                      show=False)       # either True or False

#%%
size = calculate_cluster_sizes3(filtered_df)
#%%
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
                               filename = 'final.csv',    # use strings and no format: fx: 'data' or None
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
#print(filt_df['seconds_from_start'])
print('GLD strength:', len(filt_df['weighted_strength']))
#print(filt_df['weighted_strength'])

#%%  ------------- prik-before -------------------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

max_event_times = output_df['max event time'] #+ 28
max_events = np.ones(len(max_event_times))
mean_time = output_df['mean time']
start_times = np.array(output_df['start'])
period = output_df['period [ms]']/1000

gld_times = filt_df['seconds_from_start'] # leap seconds
gld_strength = 0.7 * np.ones(len(gld_times))



# Set up the figure
plt.figure(figsize=(15, 5))

# Plot Max Events as blue bars +0.352
#plt.bar(max_event_times, max_events, width=0.3, color='tab:orange', label='Max Events')

visible_widths = np.maximum(period, 0.15)  # e.g., force at least 0.2 seconds wide
#plt.bar(start_time, max_events, width=visible_widths, align='edge', color='#e89e42', label='Max Events')
bar_handle = plt.bar(
    start_times,
    max_events,
    width=visible_widths,
    align='edge',
    color='#e89e42',
    label='DBSCAN',
    zorder=1
)

type_colors = {
	'MS + CC': '#66a386',
	'MS + CG': '#1d468b',
	#'MS': '#9370DB',
	'CC': '#c0281b',
	'CG': '#6fbcbc',
	'Unknown': '#9370DB'
	}

for t, color in type_colors.items():
    subset = filt_df[filt_df['type'] == t]
    plt.scatter(
        subset['seconds_from_start'],
        0.9 * np.ones(len(subset['seconds_from_start'])),
        s=30,
        color=color,
        alpha=1,
        label=t
    )

# Labels and formatting
plt.xlabel("Time [s]", fontsize=14, family='serif')
#plt.title('Temporal Distribution of DBSCAN Clusters and GLD Lightnings')

# First legend: for the bar plot
bar_legend = plt.legend(
    handles=[bar_handle[0]],  # just one bar needed for the legend symbol
    labels=['Clusters'],
    title='    DBSCAN     ',
    loc='upper right',
    prop={'family': 'serif', 'size': 14},
    title_fontproperties=FontProperties(family='serif', size=14),
    bbox_to_anchor=(1, 0.65)
)
plt.gca().add_artist(bar_legend)  # prevent it from being overridden


# Second legend: for the GLD types
scatter_legend_dots = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=color,markeredgecolor='none',  markersize=6, label=label)
    for label, color in type_colors.items()
]
plt.legend(
    handles=scatter_legend_dots,
    title='GLD Types',
    loc='lower right',
    prop={'family': 'serif', 'size': 14},
    title_fontproperties=FontProperties(family='serif', size=14)
)

plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0,110)
plt.locator_params(axis='x', nbins=25)  # Reduce number of ticks
plt.gca().get_yaxis().set_visible(False)
plt.xticks(fontsize=14, family='serif')
plt.tight_layout()
plt.savefig('/Users/josephine/Desktop/before.pdf', dpi=300, bbox_inches='tight', format='pdf') 
plt.show()


#%% prik-correlation -----------------
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Assume these are your arrays of event and GLD times in seconds
#event_times = np.array(max_event_times)        # from bars (orange)
event_times = np.array(start_times) 
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
top_5_indices = np.argsort(corr)[-1:][::-1]  # descending order

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
plt.scatter(lags, corr, s=70, alpha=0.7, color='#e89e42', label='Cross-Correlation')
plt.scatter(best_lag, max_corr_value, s=100, color='#e89e42', edgecolor='black',linewidth=1.5, label=f'Best lag: {best_lag}s')
#plt.title("Cross-Correlation between Clusters and GLD Lightnings")
plt.xlabel("Lag [s]")
plt.ylabel("Cross-Correlation Value")
plt.legend()
plt.grid(True, zorder=0)
plt.tight_layout()
#plt.savefig(f'/zhome/57/6/168999/Desktop/ThorDavis/hdbscan/final/{min_clusters}_{min_samples}_{time_tolerance}_prikcorr_{name}.png', dpi=300) 
plt.show()

#%% # prik-after ----------------
print(start_times)
# Set up the figure
plt.figure(figsize=(15, 5))
#plt.figure(figsize=(8, 5))

# Plot Max Events as blue bars +0.352
#plt.bar(max_event_times, max_events, width=0.3, color='tab:orange', label='Max Events')

#plt.bar(start_time, max_events, width=visible_widths, align='edge', color='#e89e42', label='Max Events')
bar_handle = plt.bar(
    start_times + np.abs(best_lag),
    max_events,
    #width=visible_widths,
    width=period,
    align='edge',
    color='#e89e42',
    label='DBSCAN',
    zorder=1
)

type_colors = {
	'MS + CC': '#66a386',
	'MS + CG': '#1d468b',
	#'MS': '#9370DB',
	'CC': '#c0281b',
	'CG': '#6fbcbc',
	'Unknown': '#9370DB'
	}

for t, color in type_colors.items():
 	subset = filt_df[filt_df['type'] == t]
 	plt.scatter(
 		subset['seconds_from_start'],
 		0.9 * np.ones(len(subset['seconds_from_start'])),
 		s=30,
 		color=color,
 		alpha=1,
 		label=t
 		)

# Plot GLD times as red dots
#plt.scatter(gld_times, gld_strength, color='blue', s=10, label='GLD Detections', zorder=5)

# Labels and formatting
plt.xlabel("Time [s]", fontsize=16, family='serif')

#plt.title('Temporal Distribution of DBSCAN Clusters and GLD Lightnings')

# First legend: for the bar plot
bar_legend = plt.legend(
    handles=[bar_handle[0]],  # just one bar needed for the legend symbol
    labels=['Clusters'],
    title='    DBSCAN     ',
    loc='upper right',
    prop={'family': 'serif', 'size': 14},
    title_fontproperties=FontProperties(family='serif', size=14),
    bbox_to_anchor=(1, 0.65)
)
plt.gca().add_artist(bar_legend)  # prevent it from being overridden


# Second legend: for the GLD types
scatter_legend_dots = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=color,markeredgecolor='none',  markersize=6, label=label)
    for label, color in type_colors.items()
]

plt.legend(
    handles=scatter_legend_dots,
    title='GLD Types',
    loc='lower right',
    prop={'family': 'serif', 'size': 14},
    title_fontproperties=FontProperties(family='serif', size=14)
)



plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0,110)
#plt.xlim(98.3,99.92)
plt.ylim(0,1)
plt.xticks(fontsize=14, family='serif')
plt.locator_params(axis='x', nbins=25)  # Reduce number of ticks
#plt.locator_params(axis='x', nbins=10)
plt.gca().get_yaxis().set_visible(False)
plt.tight_layout()
plt.savefig('/Users/josephine/Desktop/after.pdf', dpi=300, bbox_inches='tight', format='pdf') 
plt.show()

