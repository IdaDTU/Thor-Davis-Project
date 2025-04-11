##

from plotting import plot_event_distribution, plot_event_distribution2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def calculate_total_events(df, df_histogram, xaxis='time', filename=None, show=False):
    """
    Calculate the total events for each merged cluster and store the relevant data.
    
    Args:
    - merged_clusters (list): A list of merged clusters, where each cluster is a list of event indices.
    - start (list or np.ndarray): Start times for each merged cluster.
    - end (list or np.ndarray): End times for each merged cluster.
    - df_histogram (DataFrame): The histogram DataFrame used for plotting event distributions.
    
    Returns:
    - total_events_list (list): List of the total events for each merged cluster.
    """
    
    merged_clusters = df['clusters']
    
    if xaxis == 'time':
        start = df['start']
        end = df['end']
    else:
        start = df['start frame']
        end = df['end frame']
    
    total_events_list = []
    max_event_list = []

    # Iterate over merged clusters
    for i, merged_cluster_indices in enumerate(merged_clusters):

        total_events, max_event = plot_event_distribution2(
            df=df,
            df_histogram=df_histogram,
            start=start.iloc[i].item(),
            end=end.iloc[i].item(),
            xaxis=xaxis,
            filename=filename,
            show=show
            )
        total_events_list.append(total_events)
        max_event_list.append(max_event)

    return total_events_list, max_event_list

def calculate_cluster_sizes(df):
    """
    Calculate the size (maximum distance between any two points) for each merged cluster.
    
    Args:
    - df (pd.DataFrame): DataFrame containing 'clusters', 'x', 'y' columns.
    
    Returns:
    - cluster_sizes (list): Maximum pairwise distance within each cluster.
    """
    merged_clusters = df['clusters']
    x = df['x']
    y = df['y']

    # Flatten coordinates
    x_coords_valid = np.concatenate(x)
    y_coords_valid = np.concatenate(y)
    
    #x_coords_valid = df['x'].values
    #y_coords_valid = df['y'].values

    cluster_sizes = []

    for cluster_indices in merged_clusters:
        # Select only the coordinates for this cluster
        cluster_x = x_coords_valid[cluster_indices]
        cluster_y = y_coords_valid[cluster_indices]

        # Combine into 2D array
        coordinates = np.stack((cluster_x, cluster_y), axis=1)

        if len(coordinates) < 2:
            max_distance = 0  # Or np.nan if preferred
        else:
            distances = cdist(coordinates, coordinates)
            max_distance = np.max(distances)

        cluster_sizes.append(max_distance)

    return cluster_sizes

def calculate_cluster_sizes2(df):
    """
    Calculate the size (max spatial distance) of each cluster in filtered_df,
    where each row contains the x and y values for one merged cluster.
    """
    cluster_sizes = []

    for i in range(len(df)):
        cluster_x = df.iloc[i]['x']
        cluster_y = df.iloc[i]['y']

        # Safety: convert to array in case it's a list
        coordinates = np.stack((np.array(cluster_x), np.array(cluster_y)), axis=1)

        if len(coordinates) < 2:
            max_distance = 0
        else:
            distances = cdist(coordinates, coordinates)
            max_distance = np.max(distances)

        cluster_sizes.append(max_distance)

    return cluster_sizes



def calculate_event_rate(total_events, periods):
    """
    Calculate the event rate as Total Events / Period [s], given total events and periods in ms.
    
    Args:
    - total_events (list or np.ndarray): List or array containing the total events for each cluster.
    - periods (list or np.ndarray): List or array containing the period in milliseconds for each cluster.
    
    Returns:
    - event_rates (list): List of event rates, rounded to 3 decimal places, for each cluster.
    """
    # Calculate the event rate (Total Events / Period [s])
    event_rates = total_events / periods
    
    # Round the event rates to 3 decimal places
    event_rates = np.round(event_rates, 3)
    
    return event_rates

def create_and_save_df(df, size, total_events, event_rate, filename=None, decimals=3):
    # Create a dictionary of the input data
    
    start = df['start']
    end = df['end']
    period = df['period']
    mean_time = df['mean time']
    std_time = df['std_time']
    std_size = df['std_size']
    frame_mean = [np.mean(frames) for frames in df['frames']]
    frame_period = df['frame period']
    frame_start = df['start frame']
    frame_end = df['end frame']
    
    data = {
        'start': start,
        'end': end,
        'period [ms]': period*1000,
        'mean time': mean_time,
        'size [px]': size,
        'total events': total_events,
        'event rate [event/ms]': event_rate/1000,
        'std (time) [ms]': std_time*1000, #so its the same as periods (ms)
        'std (size) [px]': std_size,
        'mean frame': frame_mean,
        'frame period': frame_period,
        'start frame': frame_start,
        'end frame': frame_end
    }

    # Create DataFrame from the data
    df = pd.DataFrame(data)
    df["eps study"] = np.sqrt(df["size [px]"]**2 + ((df["start"] - df["end"]) / 100)**2)
    
    # Round the numeric columns to the specified number of decimal places
    df['start'] = df['start'].round(decimals)
    df['end'] = df['end'].round(decimals)
    df['period [ms]'] = df['period [ms]'].round(decimals)
    df['mean time'] = df['mean time'].round(decimals)
    df['size [px]'] = df['size [px]'].round(decimals)
    df['total events'] = df['total events'].round(decimals)
    df['event rate [event/ms]'] = df['event rate [event/ms]'].round(decimals)
    df['std (time) [ms]'] = df['std (time) [ms]'].round(decimals)
    df['std (size) [px]'] = df['std (size) [px]'].round(decimals)
    df['mean frame'] = df['mean frame'].round()
    df['eps study'] = df['eps study'].round(decimals)

    if filename:
        df.to_csv(filename, index=False)
        print(f"Output dataframe saved as {filename}")
    
    return df

def create_and_save_df2(df, df_histogram, size, total_events, max_event, event_rate, filename=None, decimals=3):
    import pandas as pd
    import numpy as np

    # Convert size, total_events, etc. to Series if they aren’t already
    size = pd.Series(size)
    total_events = pd.Series(total_events)
    max_event = pd.Series(max_event)
    event_rate = pd.Series(event_rate)
    
    # Calculate frame_mean from lists in df['frames']
    frame_mean = [np.mean(frames) for frames in df['frames']]
    
    # Timestamp of highest event count in each cluster
    max_event_times = []
    for times, counts in zip(df['timestamps'], df_histogram['count']):
        max_idx = np.argmax(counts)
        max_event_times.append(times.iloc[max_idx])

    cluster_number = list(range(0, len(df)))
    
    # Assemble data dictionary row-wise
    data = {
        'cluster': cluster_number,
        'start': df['start'],
        'end': df['end'],
        'period [ms]': df['period'] * 1000,
        'mean time': df['mean time'],
        'max event time': max_event_times,
        'size [px]': size,
        'total events': total_events,
        'norm size': total_events*size,
        'max event': max_event,
        'event rate [event/ms]': event_rate / 1000,
        'std (time) [ms]': df['std_time'] * 1000,
        'std (size) [px]': df['std_size'],
        'mean frame': frame_mean,
        'frame period': df['frame period'],
        'start frame': df['start frame'],
        'end frame': df['end frame']
    }
    
    # Create DataFrame
    result_df = pd.DataFrame(data)

    # Row-wise eps study calculation
    result_df["eps study"] = np.sqrt(
        result_df["size [px]"]**2 + ((result_df["start"] - result_df["end"]) / 100)**2 #måske slet de 100
    )

    # Time until next strike
    result_df["time until next strike [ms]"] = (result_df["start"].shift(-1) - result_df["start"])

    # Round values
    for col in ['start', 'end', 'period [ms]', 'mean time', 'max event time', 'size [px]',
                'total events', 'norm size', 'event rate [event/ms]', 'std (time) [ms]',
                'std (size) [px]', 'eps study', 'time until next strike [ms]']:
        result_df[col] = result_df[col].round(decimals)

    result_df['mean frame'] = result_df['mean frame'].round()

    # Save if filename is provided
    if filename:
        result_df.to_csv(filename, index=False)
        print(f"Output dataframe saved as {filename}")

    return result_df




