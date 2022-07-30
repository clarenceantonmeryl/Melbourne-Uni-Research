
import matplotlib.pyplot as plt
import numpy as np

from DataGenerator import DataGenerator

data_generator = DataGenerator()

MEASUREMENTS = 500

X = np.linspace(0, MEASUREMENTS, MEASUREMENTS)

squid_population_list, seal_population_list = data_generator.get_population_lists(a_n=50, b_n=0.2)

def calculate_gradient(y2, y1, x2, x1):
    gradient = (y2 - y1) / (x2 - x1)
    return gradient


def config_graph():
    plt.figure(figsize=(24, 14))
    plt.xlim(0, MEASUREMENTS)
    plt.xlabel("Time (n units of time)", fontsize=15)
    plt.ylabel("Population density (animals/km²)", fontsize=15)
    plt.plot(X, np.zeros((MEASUREMENTS,)))


def get_gradient_lists():
    squid_gradient_list = [0]
    seal_gradient_list = [0]

    # Calculate squid gradients
    for index in range(1, MEASUREMENTS):
        squid_gradient_list.append(
            calculate_gradient(
                y2=squid_population_list[index],
                y1=squid_population_list[index - 1],
                x2=index,
                x1=index - 1
            )
        )

    # Calculate seal gradients
    for index in range(1, MEASUREMENTS):
        seal_gradient_list.append(
            calculate_gradient(
                y2=seal_population_list[index],
                y1=seal_population_list[index - 1],
                x2=index,
                x1=index - 1
            )
        )

    return squid_gradient_list, seal_gradient_list


def plot_gradient_graphs(squid_gradients, seal_gradients):

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


squid_gradient_list, seal_gradient_list = get_gradient_lists()
plot_gradient_graphs(squid_gradients=squid_gradient_list, seal_gradients=seal_gradient_list)
