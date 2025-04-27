## Functions used to create the output df
from plotting import plot_event_distribution
import numpy as np
import pandas as pd

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

        total_events, max_event = plot_event_distribution(
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


from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

def calculate_cluster_sizes(df):
    """
    Efficiently calculate the maximum spatial distance in each cluster.
    """
    cluster_sizes = []

    for i in range(len(df)):
        cluster_x = df.iloc[i]['x']
        cluster_y = df.iloc[i]['y']
        coordinates = np.stack((np.array(cluster_x), np.array(cluster_y)), axis=1)

        if len(coordinates) < 2:
            max_distance = 0
        else:
            try:
                # Use convex hull points only for efficiency
                hull = ConvexHull(coordinates)
                hull_points = coordinates[hull.vertices]
                max_distance = np.max(pdist(hull_points))
            except:
                # fallback if ConvexHull fails (e.g., all points colinear)
                max_distance = np.max(pdist(coordinates))

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

def create_and_save_df(df, df_histogram, size, total_events, max_event, event_rate, filename=None, decimals=3):

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

