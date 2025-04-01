import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the data
LYN = np.load("C:/Users/user/OneDrive/Desktop/Nikon_2023_12_30_20_04_46.mov.LYN .npy", allow_pickle=True)


# Extract relevant fields
LON = [entry['location']['coordinates'][0] for entry in LYN]
LAT = [entry['location']['coordinates'][1] for entry in LYN]
CUR = [entry['signalStrengthKA'] for entry in LYN]
TIME = [entry['time'] for entry in LYN]
CLOUD = [entry['cloud'] for entry in LYN]

# Create DataFrame
df = pd.DataFrame({
    'Longitude': LON,
    'Latitude': LAT,
    'SignalStrengthKA': CUR,
    'time': TIME,
    'cloud': CLOUD
})

# Convert time to datetime and compute fractional minutes
df['datetime'] = pd.to_datetime(df['time'])
df['minutes'] = df['datetime'].dt.minute + df['datetime'].dt.second / 60 + df['datetime'].dt.microsecond / 1e6 / 60

# Temporal filter
df = df[(df['minutes'] >= 4.46) & (df['minutes'] <= 11.68)]

# Compute seconds since the start for timeline plotting
start_time = df['datetime'].min()
df['seconds'] = (df['datetime'] - start_time).dt.total_seconds()

# Plot density hexbin map weighted by signal strength
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='whitesmoke', edgecolor='black')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

hb = ax.hexbin(
    df['Longitude'],
    df['Latitude'],
    C=df['SignalStrengthKA'],
    reduce_C_function=np.mean,
    gridsize=50,
    cmap='coolwarm',
    mincnt=1,
    transform=ccrs.PlateCarree()
)

cb = plt.colorbar(hb, ax=ax, orientation='vertical', label='Mean Signal Strength (KA)')
plt.title("Lightning Strike Density (Weighted by Signal Strength)")
plt.show()

# Spatial filter
lon_min, lon_max = 20, 21
lat_min, lat_max = -3, -2

filtered_df = df[
    (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
    (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)
]

# Plot individual strike locations
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='whitesmoke', edgecolor='black')
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ax.scatter(
    filtered_df['Longitude'],
    filtered_df['Latitude'],
    color='red',
    s=10,
    transform=ccrs.PlateCarree(),
    label='Lightning Strikes'
)

plt.title("Lightning Strike Locations in Selected Region")
plt.legend()
plt.show()

# Timeline of strikes in filtered area
plt.figure(figsize=(10, 2))
plt.eventplot(filtered_df['seconds'], orientation='horizontal', colors='red',
              lineoffsets=1, linelengths=0.8)
plt.xlabel("Seconds since start")
plt.yticks([])
plt.title("Lightning Strikes (First 20 Seconds)")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print number of strikes in selected region
print("Number of lightning strikes in selected region:", len(filtered_df))

# Compute distance between ligtning

# Ensure sorted by time
filtered_df = filtered_df.sort_values('seconds').reset_index(drop=True)
print(filtered_df)

# Calculate time difference to next strike
filtered_df['seconds_to_next'] = filtered_df['seconds'].shift(-1) - filtered_df['seconds']

# Last row will have NaN since there's no "next" event
# Timeline of time diffrence for strikes in filtered area
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(filtered_df.index, filtered_df['seconds_to_next'], marker='o', linestyle='-')

ax.set_xlabel("Strike Index")
ax.set_ylabel("Time to Next Strike (ms)")
ax.set_title("Time Difference Between Lightning Strikes")
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()
plt.show()





















