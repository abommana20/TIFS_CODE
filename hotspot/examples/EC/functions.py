import matplotlib.pyplot as plt
import re

def plot_values_and_distances(data_dict, save_filename=None):
    """
    Plots the values from a dictionary as bars and their distances from the first value as a line plot on a secondary y-axis.
    Optionally saves the plot to a file.

    Parameters:
        data_dict (dict): A dictionary where keys are labels and values are numeric.
        save_filename (str, optional): Filename to save the plot. If None, the plot will not be saved.
    """
    # Calculate distances from the first key's value
    first_value = list(data_dict.values())[0]
    distance = {key: value - first_value for key, value in data_dict.items()}

    # Keys and values for plotting
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    distances = list(distance.values())

    # Create figure and axes
    fig, ax1 = plt.subplots()

    # Plotting the original values with bars
    bars = ax1.bar(keys, values, color='blue', alpha=0.6)
    # Setting the first axis limits
    ax1.set_ylim([min(values)-2, max(values) + 2])  # add some padding above the max value

    # Adding labels and title to the first axis
    ax1.set_xlabel('Percentgae of Fault Coverage')
    ax1.set_ylabel('Percentage of Conduatance drop', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Bar Chart of Values and Line Plot of Distances')

    # Create a second y-axis for the distances
    ax2 = ax1.twinx()
    line = ax2.plot(keys, distances, 'r-o')
    ax2.set_ylim([min(distances) - 1, max(distances) + 1])
    # Setting the second y-axis label
    ax2.set_ylabel('Additional drop due to EC comapred to FF (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Adding legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Show plot
    plt.show()

    # Save plot if filename is provided
    if save_filename:
        fig.savefig(save_filename, format='png', dpi=300)  # Save as PNG with high resolution

def plot_values(data_dict, save_filename=None):
    """
    Plots the values from a dictionary as bars against their keys.

    Parameters:
        data_dict (dict): A dictionary where keys are labels and values are numeric.
        save_filename (str, optional): Filename to save the plot. If None, the plot will not be saved.
    """
    # Extract keys and values for plotting
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plotting the values as bars
    ax.bar(keys, values, color='blue')
    ax.set_ylim(min(values)-2, max(values)+2)  # Add some padding above the max value
    # Adding labels and title
    ax.set_xlabel('Percentage of Fault Coverage')
    ax.set_ylabel('Maximum temperature')
    ax.set_title('Bar Chart of Values')

    # Add legend
    ax.legend()

    # Show plot
    plt.show()

    # Save plot if filename is provided
    if save_filename:
        fig.savefig(save_filename, format='png', dpi=300)  # Save as PNG with high resolution

def extract_max_value(filename):
    pattern = re.compile(r'layer_1_tile\d+\s+(\d+\.\d+)')
    max_value = float('-inf')

    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                value = float(match.group(1))
                if value > max_value:
                    max_value = value

    return max_value