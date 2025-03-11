import dv_processing as dvp
import numpy as np
import pandas as pd

def count_batches(file_path: str, max_batches: int = 10000200) -> int:
    """
    Counts the number of event batches in a dvSave (.aedat4) file.
    
    Parameters:
        file_path (str): Path to the .aedat4 file.
        max_batches (int): Maximum number of batches to process (default: 10000200).
    
    Returns:
        int: Total number of event batches in the file.
    """
    # Open the file
    capture = dvp.io.MonoCameraRecording(file_path)
    batch_count = 0
    
    # Iterate through batches until no more batches exist
    for i in range(max_batches):
        events = capture.getNextEventBatch()
        if events is None or len(events) == 0:  # Stop if there are no more events
            break
        batch_count += 1
    
    return batch_count

def create_dataframe(file_path: str, 
                     num_bins: int,
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
        'count': hist_counts
    })