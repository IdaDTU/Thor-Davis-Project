import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from skyfield.api import load, EarthSatellite


# Load and parse data
LYN = np.load("C:/Users/user/OneDrive/Desktop/ThorDavis/data/Nikon_2023_12_30_20_04_46.mov.LYN.npy", allow_pickle=True)

# Construct DataFrame
df = pd.DataFrame({
    'Longitude': [entry['location']['coordinates'][0] for entry in LYN],
    'Latitude': [entry['location']['coordinates'][1] for entry in LYN],
    'SignalStrengthKA': [entry['signalStrengthKA'] for entry in LYN],
    'SignalStrengthKA_abs': [abs(entry['signalStrengthKA']) for entry in LYN],
    'datetime': [entry['time'] for entry in LYN],
    'cloud': [entry['cloud'] for entry in LYN],
    'multiplicity': [entry.get('multiplicity') for entry in LYN]})

# Convert to datetime *before* filtering
df['datetime'] = pd.to_datetime(df['datetime'])
#%%
print(df['datetime'])
#%%

# Time range filtering
start = pd.Timestamp("2023-12-30 20:04:28.000000+00:00")
end = pd.Timestamp("2023-12-30 20:06:56.000000+00:00")
filt_df = df[(df['datetime'] >= start) & (df['datetime'] <= end)].sort_values(by='datetime')

# Calculate time delta from the first timestamp
df['delta'] = df['datetime'] - df['datetime'].iloc[0]

# Calculate seconds and minutes from start
df['seconds_from_start'] = df['delta'].dt.total_seconds()
df['minutes_from_start'] = df['seconds_from_start'] / 60

# Spatial filtering
lon_min, lon_max = 21, 26
lat_min, lat_max = -27, -24.5
filt_df = df[(df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
             (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]

# Filter strikes that happened within the first 20 seconds
first_130_sec_filt_df = filt_df[filt_df['seconds_from_start'] <= 110]

print("Number of lightning strikes within the first 130 seconds after filtering:", len(first_130_sec_filt_df))

#%% Bin into 1-second intervals (to integer seconds)
filt_df['second_bin'] = filt_df['seconds_from_start'].astype(int)
print(df['seconds_from_start'])

#%% Aggregate by second – set value to 1 if *any* signal is present
binned_df = filt_df.groupby('second_bin').size().reset_index(name='count')
binned_df['constant'] = 1  # since presence of any count means we want a 1

# Plot
plt.figure(figsize=(14, 4))
plt.bar(binned_df['second_bin'],
        binned_df['constant'],
        width=0.9,
        color='skyblue',
        edgecolor='black',
        alpha=0.8)

plt.xlabel('Seconds Since Start')
plt.ylabel('Summed Signal Strength (kA) (weighted)')
plt.title('1-Second Binned Signal Strength of Lightning Events')

plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.show()

print('Finished filtering!')

print(binned_df.describe())


#%% plot

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Set global font settings
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',  # Computer Modern
})

# Define DTU colors
dtu_red = '#990000'
white = '#ffffff'
dtu_navy = '#030F4F'

# Spatial filter for plot
lon_min_plot, lon_max_plot = 10, 42
lat_min_plot, lat_max_plot = -35, 0

# Create linear colormap from DTU navy to DTU red
dtu_reds = LinearSegmentedColormap.from_list("dtu_reds", [dtu_navy, dtu_red])

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
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

# Scatter lightning strikes
sc = ax.scatter(df['Longitude'],
                df['Latitude'],
                c=df['seconds_from_start'],
                cmap='Oranges',
                s=30,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                edgecolor='w',
                linewidth=0.1,
                zorder=3,
                label='Detected Lightning')

# Colorbar
cbar = fig.colorbar(sc, ax=ax, orientation='vertical',
                    label='Detection Time [s]', fraction=0.03, pad=0.04)
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Detection Time [s]', fontsize=14)

plt.legend()
plt.savefig("GLD_all_lightning_map.pdf", format='pdf', bbox_inches='tight')

# Show plot
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd

# Set global font settings
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

# Define DTU colors
dtu_red = '#990000'
white = '#ffffff'
dtu_navy = '#030F4F'

# Spatial filter for plot
lon_min_plot, lon_max_plot = 10, 42
lat_min_plot, lat_max_plot = -35, 0

# Define color map for lightning types
type_colors = {
    'MS + CC': '#66a386',  # Orange
    'MS + CG': '#1d468b',  # Blue
    'CC': '#c0281b',       # Red
    'CG': '#6fbcbc',       # Purple
    'Unknown': '#9370db'   # Grey
}


cmap = ListedColormap(list(type_colors.values()))

# Assign type to data
df.loc[(df['multiplicity'] > 0) & (df['cloud'] == 1), 'type'] = 'MS + CC'
df.loc[(df['multiplicity'] > 0) & (df['cloud'] == 0), 'type'] = 'MS + CG'
df.loc[(pd.isna(df['multiplicity'])) & (df['cloud'] == 1), 'type'] = 'CC'
df.loc[(pd.isna(df['multiplicity'])) & (df['cloud'] == 0), 'type'] = 'CG'
df.loc[pd.isna(df['multiplicity']) & pd.isna(df['cloud']), 'type'] = 'Unknown'

# Convert 'type' to categorical codes
df['type_code'] = df['type'].map({k: i for i, k in enumerate(type_colors.keys())})

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
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

# Scatter plot colored by type
sc = ax.scatter(df['Longitude'],
                df['Latitude'],
                c=df['type_code'],
                cmap=cmap,
                s=30,
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                zorder=3)

# Create one scatter point per type for legend
for t in type_colors:
    ax.scatter([], [], 
               c=type_colors[t], 
               label=t, 
               s=30, 
               alpha=0.7)

# Add the legend
ax.legend(loc='lower left', frameon=True, fontsize=14)

plt.savefig("GLD_lightning_map_by_type.pdf", format='pdf', bbox_inches='tight')

plt.tight_layout()
plt.show()
