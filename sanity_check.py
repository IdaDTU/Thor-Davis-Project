import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from skyfield.api import load, EarthSatellite



# Load the data
LYN = np.load('C:/Users/user/OneDrive/Desktop/ThorDavis/data/Nikon_2023_12_30_20_04_46.mov.LYN.npy', allow_pickle=True)

# Extract relevant fields
LON = [entry['location']['coordinates'][0] for entry in LYN]
LAT = [entry['location']['coordinates'][1] for entry in LYN]
CUR = [entry['signalStrengthKA'] for entry in LYN]
TIME = [entry['time'] for entry in LYN]
CLOUD = [entry['cloud'] for entry in LYN]
MULTI = [entry['multiplicity'] if 'multiplicity' in entry else None for entry in LYN]
PULSE = [entry['numCloudPulses'] if 'numCloudPulses' in entry else None for entry in LYN]

#%% Create DataFrame
import numpy as np

df = pd.DataFrame({
    'Longitude': LON,
    'Latitude': LAT,
    'SignalStrengthKA_abs': np.abs(CUR),
    'SignalStrengthKA': CUR,
    'datetime': TIME,
    'cloud': CLOUD,
    'multiplicity': MULTI,
    'numCloudPulses':PULSE
})


#%%
# Convert time to datetime and compute fractional minutes

# Convert to datetime *before* filtering
df['datetime'] = pd.to_datetime(df['datetime'])

# Time range filtering
start = pd.Timestamp("2023-12-30 20:04:28.000000+00:00")
end = pd.Timestamp("2023-12-30 20:06:56.000000+00:00")
df = df[(df['datetime'] >= start) & (df['datetime'] <= end)].sort_values(by='datetime')

# Calculate time delta from the first timestamp
df['delta'] = df['datetime'] - df['datetime'].iloc[0]

# Calculate seconds and minutes from start
df['seconds_from_start'] = df['delta'].dt.total_seconds()
df['minutes_from_start'] = df['seconds_from_start'] / 60

#%%
# Compute seconds since the start for timeline plotting
start_time = df['datetime'].min()
df['seconds'] = (df['datetime'] - start_time).dt.total_seconds()

# Spatial filter
lon_min, lon_max = 21, 26
lat_min, lat_max = -27, -24.5

filtered_df = df[
    (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
    (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]

print('Finished filtering!')

# Classify lightning
# Make a copy to avoid SettingWithCopyWarning
filtered_df = filtered_df.copy()

filtered_df.loc[(filtered_df['multiplicity'] > 0) & (filtered_df['cloud'] > 0), 'type'] = 'Multi-Strike and Inter-Cloud'
filtered_df.loc[(filtered_df['multiplicity'] == 0) & (filtered_df['cloud'] == 0), 'type'] = 'Neither'
filtered_df.loc[(filtered_df['multiplicity'] > 0) & (filtered_df['cloud'] == 0), 'type'] = 'Multi-Strike'
filtered_df.loc[(filtered_df['multiplicity'] == 0) & (filtered_df['cloud'] == True), 'type'] = 'Inter-Cloud'
filtered_df.loc[pd.isna(filtered_df['multiplicity']) & pd.isna(filtered_df['cloud']), 'type'] = 'Not Classified'

print('Finished classification!')

# Filter strikes that happened within the first 20 seconds
first_130_sec_filtered_df = filtered_df[filtered_df['seconds'] <= 110]
first_130_sec_df = df[df['seconds'] <= 110]

print("...")
print("Number of lightning strikes within the first 130 seconds after filtering:", len(first_130_sec_filtered_df))
print("Number of lightning strikes within the entire dataframe after filtering:", len(filtered_df))
print("...")
print("Number of lightning strikes within the first 130 seconds before filtering:", len(first_130_sec_df))
print("Number of lightning strikes within the entire dataframe before filtering:", len(df))


#%% Plot in interval


# Define DTU red and white
dtu_red = '#990000'
white = '#ffffff'
dtu_navy = '#030F4F'

# Spatial filter for plot
lon_min_plot, lon_max_plot = 15, 28 #-> 20,35
lat_min_plot, lat_max_plot = -30, -14 # - > -35,-20

# Create linear colormap from DTU red to white
dtu_reds = LinearSegmentedColormap.from_list("dtu_reds", [dtu_navy,dtu_red])

# Set up figure and map
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Load and convert background image to grayscale
img = mpimg.imread("C:/Users/user/OneDrive/Desktop/NASA_Dark_Marble.jpg")
gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale

# Show grayscale background
ax.imshow(gray, origin='upper', extent=[-180, 180, -90, 90],
          cmap='gray', transform=ccrs.PlateCarree(), zorder=0)

# Zoom to region of interest
ax.set_extent([lon_min_plot, lon_max_plot, lat_min_plot, lat_max_plot], crs=ccrs.PlateCarree())

# Map features
ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.5, zorder=2)

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Scatter lightning strikes
sc = ax.scatter(filtered_df['Longitude'],
                filtered_df['Latitude'],
                c=filtered_df['seconds'],
                cmap='Oranges',
                s=30,
                alpha=0.9,
                edgecolor='w',
                linewidth=0.3,
                transform=ccrs.PlateCarree(),
                zorder=3)

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

# Plot ISS track as a line
ax.plot(iss_lons, iss_lats, color='#fdc18d', linewidth=2, alpha=0.7, linestyle='--', label='ISS Track', zorder=4)

# Find the index for the time 2 min and 10 sec
target_time = ts.utc(2023, 12, 30, 20, 4 + 2 + 10 / 60)  # Time is 2 min 10 sec after the epoch
target_index = np.abs(times - target_time).argmin()  # Find the closest index

# Get corresponding lat and lon for the target time
target_lat = iss_lats[target_index]
target_lon = iss_lons[target_index]

# Plot a dot at that specific point
change_of_angle_dot = ax.scatter(target_lon, target_lat, color='#c2501b', s=60, marker='x', label='Change of Angle', zorder=5)

# Colorbar
# cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=1,
#                     label='Observation Time [s]', fraction=0.2, pad=0.04)
# cbar.ax.tick_params(labelsize=14)

# Legend with scatter plot and other labels
ax.legend(loc='lower left', handles=[sc, ax.lines[-1], change_of_angle_dot], 
          labels=['Lightning Strikes', 'ISS Track', 'Change of Angle'])

from matplotlib.patches import Rectangle

# Define zoom window size (in degrees)
zoom_width = 2
zoom_height = 2

# Set new zoom center closer to the lightning cluster (you can fine-tune these)
zoom_center_lon = 23.9
zoom_center_lat = -25.5

x0, x1 = zoom_center_lon - zoom_width / 2, zoom_center_lon + zoom_width / 2
y0, y1 = zoom_center_lat - zoom_height / 2, zoom_center_lat + zoom_height / 2

# # Adjusted position for the inset: [left, bottom, width, height]
# inset_pos = [0.48, 0.60, 0.3, 0.3]  # [left, bottom, width, height] in ax coords


# # Create inset axes with PlateCarree projection
# axins = fig.add_axes(inset_pos, projection=ccrs.PlateCarree(), transform=ax.transAxes)

# # Set white background for the inset
# axins.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

# # Replot in the inset
# axins.imshow(gray, origin='upper', extent=[-180, 180, -90, 90],
#              cmap='gray', transform=ccrs.PlateCarree(), zorder=0)
# axins.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.5, zorder=2)

# # Lightning strikes in inset
# axins.scatter(filtered_df['Longitude'],
#               filtered_df['Latitude'],
#               c=filtered_df['seconds'],
#               cmap='Oranges',
#               s=35,
#               alpha=0.9,
#               edgecolor='w',
#               linewidth=0.3,
#               transform=ccrs.PlateCarree(),
#               zorder=3)

# # ISS track in inset
# axins.plot(iss_lons, iss_lats, color='#fdc18d', linewidth=2, alpha=0.7, linestyle='--', zorder=4)

# # Change of angle dot in inset
# axins.scatter(target_lon, target_lat, color='#c2501b', s=60, marker='x', zorder=5)

# # Remove tick labels
# axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Rectangle on main map to indicate zoom region
rect = Rectangle((x0, y0), zoom_width, zoom_height,
                 linewidth=1, edgecolor='white', facecolor='none',
                 transform=ccrs.PlateCarree(), zorder=6)
ax.add_patch(rect)




plt.savefig("GLD_filtered_ISS.pdf", format='pdf', bbox_inches='tight')

# Show plot
plt.show()

#%% ALL DATA

# Define DTU red and white
dtu_red = '#990000'
white = '#ffffff'
dtu_navy = '#030F4F'

# Spatial filter for plot
lon_min_plot, lon_max_plot = 10, 42
lat_min_plot, lat_max_plot = -35, 0

# Create linear colormap from DTU red to white
dtu_reds = LinearSegmentedColormap.from_list("dtu_reds", [dtu_navy,dtu_red])

# Set up figure and map
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Load and convert background image to grayscale
img = mpimg.imread("C:/Users/user/OneDrive/Desktop/NASA_Dark_Marble.jpg")
gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale

# Show grayscale background
ax.imshow(gray, origin='upper', extent=[-180, 180, -90, 90],
          cmap='gray', transform=ccrs.PlateCarree(), zorder=0)

# Zoom to region of interest
ax.set_extent([lon_min_plot, lon_max_plot, lat_min_plot, lat_max_plot], crs=ccrs.PlateCarree())

# Map features
ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.5, zorder=2)

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Scatter lightning strikes
sc = ax.scatter(df['Longitude'],
                df['Latitude'],
                c=df['seconds_from_start'],
                cmap='Oranges',
                s=35,
                alpha=0.7,
                edgecolor='w',
                linewidth=0.2,
                transform=ccrs.PlateCarree(),
                zorder=3)

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

# Plot ISS track as a line
ax.plot(iss_lons, iss_lats, color='blue', linewidth=2, alpha=0.7, label='ISS Track', zorder=4)

# Find the index for the time 2 min and 10 sec
target_time = ts.utc(2023, 12, 30, 20, 4 + 2 + 10 / 60)  # Time is 2 min 10 sec after the epoch
target_index = np.abs(times - target_time).argmin()  # Find the closest index

# Get corresponding lat and lon for the target time
target_lat = iss_lats[target_index]
target_lon = iss_lons[target_index]

# Plot a dot at that specific point
change_of_angle_dot = ax.scatter(target_lon, target_lat, color='red', s=50, marker='x', label='Change of Angle', zorder=5)

# Colorbar
cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.75,
                    label='Time [s]', fraction=0.2, pad=0.04)
cbar.ax.tick_params(labelsize=10)

# Legend with scatter plot and other labels
ax.legend(loc='lower left', handles=[sc, ax.lines[-1], change_of_angle_dot], 
          labels=['Lightning Strikes', 'ISS Track', 'Change of Angle'])

# Show plot
plt.show()



#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Bin into 1-second intervals (floor to integer seconds)
filtered_df['second_bin'] = filtered_df['seconds'].astype(int)

# Aggregate by second: sum SignalStrengthKA
binned_df = filtered_df.groupby('second_bin')['SignalStrengthKA'].sum().reset_index()

binned_df['SignalStrengthKA_abs'] = binned_df['SignalStrengthKA'].abs()

# Plot
plt.figure(figsize=(14, 4))
plt.bar(
    binned_df['second_bin'],
    binned_df['SignalStrengthKA_abs'],
    width=0.9,
    color='skyblue',
    edgecolor='black',
    alpha=0.8
)

plt.xlabel('Seconds Since Start')
plt.ylabel('Summed Signal Strength (kA)')
plt.title('1-Second Binned Signal Strength of Lightning Events')

plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()


#%%
from geopy.distance import geodesic

# Precompute ISS track as list of (lat, lon) pairs
iss_track = list(zip(iss_lats, iss_lons))

# Function to find the minimum geodesic distance from one point to the ISS track
def min_distance_to_track(lat, lon, track):
    return min([geodesic((lat, lon), (t_lat, t_lon)).km for t_lat, t_lon in track])

# Apply to each lightning strike in filtered_df
filtered_df['distance_to_iss_km'] = [
    min_distance_to_track(lat, lon, iss_track)
    for lat, lon in zip(filtered_df['Latitude'], filtered_df['Longitude'])
]

# Define weighting function — can tweak `epsilon` and exponent for smoother or sharper effects
power = 2  # increase to reduce impact of distant strikes more aggressively
filtered_df['weighted_strength'] = filtered_df['SignalStrengthKA_abs'] / (filtered_df['distance_to_iss_km'])**power *1000

print(filtered_df['weighted_strength'].describe())

#%%

# Bin into 1-second intervals (floor to integer seconds)
filtered_df['second_bin'] = filtered_df['seconds'].astype(int)

# Aggregate by second: sum SignalStrengthKA
binned_df = filtered_df.groupby('second_bin')['weighted_strength'].sum().reset_index()

binned_df['weighted_strength'] = binned_df['weighted_strength'].abs()
binned_df['weighted_strength'] = 1

# Plot
plt.figure(figsize=(14, 4))
plt.bar(
    binned_df['second_bin'],
    binned_df['weighted_strength'],
    width=0.9,
    color='skyblue',
    edgecolor='black',
    alpha=0.8
)

plt.xlabel('Seconds Since Start')
plt.ylabel('Summed Signal Strength (kA) (weighted)')
plt.title('1-Second Binned Signal Strength of Lightning Events')

plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()

#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from skyfield.api import load, EarthSatellite
from matplotlib.colors import LinearSegmentedColormap

# Define DTU colors
dtu_red = '#990000'
white = '#ffffff'
dtu_navy = '#030F4F'

# Create colormap (even if unused now)
dtu_reds = LinearSegmentedColormap.from_list("dtu_reds", [dtu_navy, dtu_red])

# Define zoomed-in extent
zoom_center_lon = 23.8
zoom_center_lat = -25.5
zoom_width = 1.6
zoom_height = 1.6
x0, x1 = zoom_center_lon - zoom_width / 2, zoom_center_lon + zoom_width / 2
y0, y1 = zoom_center_lat - zoom_height / 2, zoom_center_lat + zoom_height / 2

# Load grayscale background
img = mpimg.imread("C:/Users/user/OneDrive/Desktop/NASA_Dark_Marble.jpg")
gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale

# Set up plot
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

# Background and coastlines
ax.imshow(gray, origin='upper', extent=[-180, 180, -90, 90], cmap='gray',
          transform=ccrs.PlateCarree(), zorder=0)
ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.5, zorder=2)

# Gridlines (optional)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Lightning scatter
sc = ax.scatter(filtered_df['Longitude'],
                filtered_df['Latitude'],
                c=filtered_df['seconds'],
                cmap='Oranges',
                s=35,
                alpha=0.9,
                edgecolor='w',
                linewidth=0.3,
                transform=ccrs.PlateCarree(),
                zorder=3)

# Colorbar
cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.915,
                    label='Detection Time [s]', fraction=0.05, pad=0.04)
cbar.ax.tick_params(labelsize=12)

plt.savefig("GLD_zoom_only.pdf", format='pdf', bbox_inches='tight')
plt.show()
