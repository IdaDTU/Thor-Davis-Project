# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:35:42 2025
@author: ida
"""

import dv_processing as dvp
import glob
from video_information import count_batches, create_dataframe
from plotting import plot_event_distribution

# Data
data  = glob.glob("C:/Users/user/OneDrive/Desktop/ThorDavis/dvSave-2023_12_30_20_04_46.aedat4",
                  recursive='True')[0]

# Read the file
reader = dvp.io.MonoCameraRecording(data)
print(f"Opened an AEDAT4 file which contains data from [{reader.getCameraName()}] camera")

print(reader.isFrameStreamAvailable)
#%% Calculate total number of batches
total_batches = count_batches(data) 

# Set number of batches to process
n = total_batches

# Create dataframe
df_histogram = create_dataframe(data, 
                                num_bins=100*n,
                                batches = n)
print(f"computed {n} batches with {100*n} bins...")

# Create plot
plot_event_distribution(df_histogram)

#%% Read current frame interval duration value
\

#%% Find the index of the maximum count
max_index = df_histogram['count'].idxmax()

# Get the corresponding timestamp
max_timestamp = df_histogram.loc[max_index, 'timestamps_sec']

print(f"Timestamp with max event count: {max_timestamp} s")
