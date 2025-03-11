import matplotlib.pyplot as plt

def plot_event_distribution(df_histogram):
    """
    Plots the event distribution over time from a dataframe.
    
    Parameters:
    df_histogram (pd.DataFrame): DataFrame containing 'timestamps_sec' and 'count' columns
    """
    plt.figure(figsize=(15, 5))
    plt.plot(
        df_histogram['timestamps_sec'],
        df_histogram['count'],
        marker='o',
        linestyle='-',
        color='lightsteelblue'
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Event Count")
    plt.title("Event Distribution Over Time")
    plt.grid(True)  # optional, for better readability
    plt.show()
