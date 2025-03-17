import matplotlib.pyplot as plt

def plot_event_distribution(df_histogram, window_size=100, filename='event_distribution.pdf'): 
    """
    Plots the event distribution over time from a dataframe.
    Adds average and moving average lines. Saves the plot as a PDF.

    Parameters:
    df_histogram (pd.DataFrame): DataFrame containing 'timestamps_sec' and 'count' columns
    window_size (int): Window size for calculating the moving average
    filename (str): The name of the PDF file to save the plot
    """

    # Calculate average and moving average
    avg_count = df_histogram['count'].mean()
    moving_avg = df_histogram['count'].rolling(window=window_size).mean()

    # Plotting
    plt.figure(figsize=(15, 5))

    # Event count
    plt.plot(
        df_histogram['timestamps_sec'],
        df_histogram['count'],
        marker='o',
        linestyle='-',
        label='Event Count',
        color='lightsteelblue')

    # Moving average line
    plt.plot(
        df_histogram['timestamps_sec'],
        moving_avg,
        color='steelblue',
        linestyle='-',
        linewidth=2,
        label='Moving Average')
    
    # Average line
    plt.axhline(y=avg_count, color='darkblue', linestyle='--', linewidth=2, label="Average")

    # Labels and title
    plt.xlabel("Time [s]")
    plt.ylabel("Event Count")

    # Grid and legend
    plt.grid(True)
    plt.legend()

    # Save as PDF
    plt.savefig(filename, format='pdf', bbox_inches='tight')

    # Optional: Show the plot
    plt.show()

    # Close the figure to free memory if in a loop
    plt.close()

def plot_cluster_locations(merged_clusters, x_coords_valid, y_coords_valid, merged_mean_time, title="Clusters with More Than 10 '100 Events' lasting less than 500 ms"):
    """
    Plot the locations of clusters with their mean time.
    
    Args:
    - merged_clusters (list): A list of merged clusters, where each cluster is a list of event indices.
    - x_coords_valid (np.ndarray): The x-coordinates of the valid events.
    - y_coords_valid (np.ndarray): The y-coordinates of the valid events.
    - merged_mean_time (list): A list of mean times for each merged cluster.
    - title (str): The title of the plot. Default is "Clusters with More Than 10 '100 Events' lasting less than 500 ms".
    """
    # Create the plot
    for i, cluster_group in enumerate(merged_clusters):
        # Merge all the event indices for the current cluster group
        all_indices = np.array(cluster_group)

        # Plot the events for this merged cluster
        plt.scatter(x_coords_valid[all_indices], y_coords_valid[all_indices], 
                    label=f'Lightning @ time: {merged_mean_time[i]:.2f} s', s=10)
    
    # Set axis limits based on the valid coordinates
    plt.xlim(0, max(x_coords_valid))
    plt.ylim(0, max(y_coords_valid))

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()

# Example of calling the function
# plot_event_distribution(df_histogram, window_size=100, filename='my_plot.pdf')
