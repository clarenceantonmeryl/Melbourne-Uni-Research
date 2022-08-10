# Import Libraries and Classes

import matplotlib.pyplot as plt
import numpy as np

from DataGenerator import DataGenerator

# Initialise the data_generator
data_generator = DataGenerator()

MEASUREMENTS = 500

# Initialise the x values
X = [x for x in range(0, MEASUREMENTS)]

# Initialise the squid and seal population lists using the data_generator
squid_population_list, seal_population_list = data_generator.get_population_lists(a_n=50, b_n=0.2)


def calculate_gradient(y2, y1, x2, x1):
    # calculates the gradient using the rise/run formula
    gradient = (y2 - y1) / (x2 - x1)
    return gradient


def config_graph():
    # Sets up the graphs similar to main.py
    plt.figure(figsize=(24, 14))
    plt.xlim(0, MEASUREMENTS)
    plt.xlabel("Time (n units of time)", fontsize=15)
    plt.ylabel("Gradient", fontsize=15)
    plt.plot(X, np.zeros((MEASUREMENTS,)))  # Add an x axis


def get_gradient_lists():
    # Initialise the squid and seal gradients with 0
    squid_gradient_list = [0]
    seal_gradient_list = [0]

    # Calculate squid gradients
    for index in range(1, MEASUREMENTS):
        # Add to the squid gradient list the gradient
        squid_gradient_list.append(
            calculate_gradient(
                y2=squid_population_list[index],  # Current population density (y value)
                y1=squid_population_list[index - 1],  # Previous population density (previous y value)
                x2=index,  # Current index (x value)
                x1=index - 1  # Previous index (previous x value)
            )
        )

    # Calculate seal gradients
    for index in range(1, MEASUREMENTS):
        # Add to the seal gradient list the gradient
        seal_gradient_list.append(
            calculate_gradient(
                y2=seal_population_list[index],  # Current population density (y value)
                y1=seal_population_list[index - 1],  # Previous population density (previous y value)
                x2=index,  # Current index (x value)
                x1=index - 1  # Previous index (previous x value)
            )
        )

    return squid_gradient_list, seal_gradient_list


def plot_gradient_graphs(squid_gradients, seal_gradients):
    # Plot graphs in a similar way to main.py
    config_graph()

    plt.title(f"Gradient of the Graph of Squid Population Density ({MEASUREMENTS} measurements)",
              fontsize=17)
    plt.plot(X, squid_gradients, color="red")

    plt.show()

    config_graph()

    plt.title(f"Gradient of the Graph of Seal Population Density ({MEASUREMENTS} measurements)",
              fontsize=17)
    plt.plot(X, seal_gradients, color="red")

    plt.show()


# Call the functions to first get the gradient lists, and then plot them
squid_gradient_list, seal_gradient_list = get_gradient_lists()
plot_gradient_graphs(squid_gradients=squid_gradient_list, seal_gradients=seal_gradient_list)
