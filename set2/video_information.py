import dv_processing as dvp
import numpy as np
import pandas as pd

def create_dataframe(file_path: str,
                     num_bins: int,
                     frame_rate: int,
                     batches: int) -> pd.DataFrame:
    """
    Processes event timestamps from a dvSave (.aedat4) file and returns a histogram DataFrame.

    Parameters:
        file_path (str): Path to the .aedat4 file.
        num_bins (int): Number of bins for the histogram.
        batches (int): Maximum number of batches to process

    Returns:
        pd.DataFrame: A DataFrame containing timestamps (in seconds) and event counts.
    """
    capture = dvp.io.MonoCameraRecording(file_path)
    all_on_timestamps = []

    for i in range(batches):
        events = capture.getNextEventBatch()
        for e in events:
            if e.polarity():  # On event
                all_on_timestamps.append(e.timestamp())

    all_on_timestamps = np.array(all_on_timestamps)
    all_on_timestamps -= np.min(all_on_timestamps)  # Normalize timestamps

    bin_edges = np.linspace(0, np.max(all_on_timestamps), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_counts, _ = np.histogram(all_on_timestamps, bins=bin_edges)

    return pd.DataFrame({
        'timestamps_sec': bin_centers / 1e6,  # Convert to seconds
        'count': hist_counts,
        'frames': frame_rate * bin_centers / 1e6
    })

def collect_on_events(capture, num_batches, df_histogram):
    """
    Collect 'on' polarity events from the given capture object over a specified number of batches,
    filtering events based on df_histogram's relative time range.

    Args:
    - capture: The DV MonoCameraRecording object.
    - num_batches (int): Number of event batches to process.
    - df_histogram (pd.DataFrame): DataFrame with a 'timestamps_sec' column representing relative time (in seconds).

    Returns:
    - on_events_list (list): List of (x, y, timestamp) tuples (timestamps in seconds, relative to first_timestamp).
    - event_array (np.ndarray): NumPy array of the same events.
    - first_timestamp (float): First event timestamp (in seconds, absolute time).
    """

    # Get the filtering range from df_histogram
    time1 = df_histogram['timestamps_sec'].min()
    time2 = df_histogram['timestamps_sec'].max()

    # Initialize empty list for storing filtered ON events
    on_events_list = []
    first_timestamp = None
    batch_count = 0

    for _ in range(num_batches):
        events = capture.getNextEventBatch()
        if len(events) == 0:
            continue

        if first_timestamp is None:
            first_timestamp = events.getLowestTime() / 1_000_000  # seconds

        for e in events:
            if e.polarity():
                event_time = e.timestamp() / 1_000_000  # absolute timestamp in seconds
                #relative_time = event_time - first_timestamp  # make relative to start

                if (time1+first_timestamp) <= event_time <= (time2+first_timestamp):
                    on_events_list.append((e.x(), e.y(), event_time*1000_00)) # event_array time is in 10 microsec

        batch_count += 1

    print(f"Finding ON events between {time1:.6f} and {time2:.6f} seconds")
    print(f"Number of ON events collected: {len(on_events_list)}")

    event_array = np.array(on_events_list)  # shape: (N, 3)

    return event_array, first_timestamp
