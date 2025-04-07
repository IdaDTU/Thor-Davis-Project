import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.image as mpimg


# Load the data
LYN = np.load("C:/Users/user/OneDrive/Desktop/Nikon_2023_12_30_20_04_46.mov.LYN .npy", allow_pickle=True)

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


filtered_df = df[
    (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
    (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)]

print('Finished filtering!')

#------------------------------------------------------------------------#
# Set up figure and map
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Background image
img = mpimg.imread("C:/Users/user/OneDrive/Desktop/NASA_Dark_Marble.jpg")
ax.imshow(img, origin='upper', extent=[-180, 180, -90, 90],
          transform=ccrs.PlateCarree(), zorder=0)

# Zoom in
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Map features
ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.5, zorder=2)

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Scatter lightning strikes with colorbar for multiplicity
sc = ax.scatter(filtered_df['Longitude'],
                filtered_df['Latitude'],
                c=filtered_df['seconds'],
                cmap='plasma',
                s=25,
                edgecolor='k',
                linewidth=0.2,
                transform=ccrs.PlateCarree(),
                zorder=3)

# Vertical colorbar
cbar = fig.colorbar(sc, ax=ax, orientation='vertical', label='Multiplicity', fraction=0.05, pad=0.04)
cbar.ax.tick_params(labelsize=10)

# Legend
ax.legend(['Lightning Strikes'], loc='lower right')

# Show plot
plt.show()



#%% ------------------------------------------------------------------------#

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

#%% Timeline of time diffrence for strikes in filtered area
fig, ax = plt.subplots(figsize=(10, 3))

# Bar plot instead of line plot
ax.bar(filtered_df.index, filtered_df['seconds_to_next'], width=1.0, color='skyblue')

ax.set_xlabel("Strike Index")
ax.set_ylabel("Time to Next Strike (ms)")
ax.set_title("Time Difference Between Lightning Strikes")
ax.grid(True, linestyle='--', alpha=0.7)

fig.tight_layout()
plt.show()


#%% Plot signal strength over time (seconds since start)
plt.figure(figsize=(10, 4))
plt.plot(filtered_df['seconds'], abs(filtered_df['SignalStrengthKA']), marker='o', linestyle='-')
plt.xlabel("Seconds since start")
plt.ylabel("Signal Strength (KA)")
plt.title("Signal Strength of Lightning Strikes Over Time")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#---------------------EXAMPLES----------------------------#
#%% Filter strikes that happened within the first 10 seconds
first_10_sec_df = filtered_df[filtered_df['seconds'] <= 10]
print(first_10_sec_df['multiplicity'])

#%% Print the result





#%%


IM = plt.imread("C:/Users/user/Downloads/ThorDavis_2023-12-30 20_07_56.657450.png")
plt.imshow(IM)

#%%

fig, ax = plt.subplots(figsize=(10, 3))

# Bar plot instead of line plot
ax.bar(first_10_sec_df.index, first_10_sec_df['seconds_to_next'], width=1.0, color='skyblue')

ax.set_xlabel("Strike Index")
ax.set_ylabel("Time to Next Strike (ms)")
ax.set_title("Time Difference Between Lightning Strikes")
ax.grid(True, linestyle='--', alpha=0.7)

fig.tight_layout()
plt.show()


#%% Plot signal strength over time (seconds since start)
plt.figure(figsize=(10, 4))
plt.plot(filtered_df['seconds'], abs(filtered_df['SignalStrengthKA']), marker='o', linestyle='-')
plt.xlabel("Seconds since start")
plt.ylabel("Signal Strength (KA)")
plt.title("Signal Strength of Lightning Strikes Over Time")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#---------------------EXAMPLES----------------------------#
#%% Filter strikes that happened within the first 20 seconds
first_10_sec_df = filtered_df[filtered_df['seconds'] <= 20]
print(first_10_sec_df['multiplicity'])
print("Number of lightning strikes within the first 10 seconds:", len(first_10_sec_df))
print("Number of lightning strikes within the entire dataframe:", len(filtered_df))

















