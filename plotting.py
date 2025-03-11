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

# Example of calling the function
# plot_event_distribution(df_histogram, window_size=100, filename='my_plot.pdf')
