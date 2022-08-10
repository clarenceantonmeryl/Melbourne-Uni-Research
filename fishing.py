# Import Libraries and Classes

import matplotlib.pyplot as plt

from DataGenerator import DataGenerator

# Initialise the data_generator object
data_generator = DataGenerator()

MEASUREMENTS = 500

# Initialise the x value
X = [x for x in range(0, MEASUREMENTS + 1)]


def config_graph():
    # Set up the graph in a similar way to main.py
    plt.figure(figsize=(24, 14))
    plt.xlim(0, MEASUREMENTS)
    plt.xlabel("Time (n units of time)", fontsize=15)
    plt.ylabel("Population density (animals/km²)", fontsize=15)


def plot_data_squid_fishing(k=25000, P=0.9, A=3000):
    # Uses a method from the data_generator object to get the squid and seal population density data
    squid_population_list_fishing, seal_population_list_squid_fishing = data_generator.get_population_lists_squid_fished(
        a_n=50,
        b_n=0.2,
        k=k,
        P=P,
        A=A
    )

    # Get population density data without fishing using the data_generator
    squid_population_list, seal_population_list = data_generator.get_population_lists(a_n=50, b_n=0.2)

    # Graphs are plotted in a similar way to how it is done in main.py
    config_graph()

    ax = plt.gca()
    ax.set_yticks(range(0, 96, 5))  # Set the increments of the graph (y values go up in fives)

    plt.title(f"Population Density of Squid Over Time With and Without Fishing ({MEASUREMENTS} measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_fishing, color="red", alpha=0.6)
    plt.plot(X, squid_population_list, color="blue", alpha=0.6)

    plt.show()

    config_graph()

    plt.title(f"Population Density of Seal Over Time With and Without Squid Fishing ({MEASUREMENTS} measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_squid_fishing, color="red", alpha=0.6)
    plt.plot(X, seal_population_list, color="blue", alpha=0.6)

    plt.show()

    config_graph()

    plt.title(f"Environment With Squid Fishing ({MEASUREMENTS} measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_squid_fishing, color="red", alpha=0.6)
    plt.plot(X, squid_population_list_fishing, color="blue", alpha=0.6)

    plt.show()


def plot_data_seal_fishing(k=198, P=0.9, A=3000):
    # Uses a method from the data_generator object to get the squid and seal population density data
    squid_population_list_seal_fishing, seal_population_list_fishing = data_generator.get_population_lists_seal_fished(
        a_n=50,
        b_n=0.2,
        k=k,
        P=P,
        A=A
    )

    # Get the population density data without fishing
    squid_population_list, seal_population_list = data_generator.get_population_lists(a_n=50, b_n=0.2)

    # Plot the graphs in a similar way to the plot_data_squid_fishing function
    config_graph()

    ax = plt.gca()
    ax.set_yticks(range(0, 101, 5))  # Make the y values go up in fives

    plt.title(f"Population Density of Squid Over Time With and Without Seal Fishing ({MEASUREMENTS} measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_seal_fishing, color="red", alpha=0.6)
    plt.plot(X, squid_population_list, color="blue", alpha=0.6)

    plt.show()

    config_graph()

    plt.title(f"Population Density of Seal Over Time With and Without Fishing ({MEASUREMENTS} measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_fishing, color="red", alpha=0.6)
    plt.plot(X, seal_population_list, color="blue", alpha=0.6)

    plt.show()

    config_graph()

    plt.title(f"Environment With Seal Fishing ({MEASUREMENTS} measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_fishing, color="red", alpha=0.6)
    plt.plot(X, squid_population_list_seal_fishing, color="blue", alpha=0.6)

    plt.show()


def plot_data_both_fishing(k_squid=25000, P_squid=0.9, A_squid=3000, k_seal=198, P_seal=0.9, A_seal=3000):
    # Uses a method from the data_generator object to get the squid and seal population density data
    squid_population_list_both_fishing, seal_population_both_fishing = data_generator.get_population_lists_both_fished(
        a_n=50,
        b_n=0.2,
        k_squid=k_squid,
        P_squid=P_squid,
        A_squid=A_squid,
        k_seal=k_seal,
        P_seal=P_seal,
        A_seal=A_seal
    )

    # Get the population density data without fishing
    squid_population_list, seal_population_list = data_generator.get_population_lists(a_n=50, b_n=0.2)

    # Plot the graphs in a similar way in the plot_data_squid_fishing function
    config_graph()

    ax = plt.gca()
    ax.set_yticks(range(0, 96, 5))

    plt.title(
        f"Population Density of Squid Over Time With and Without Both Species Fished ({MEASUREMENTS} measurements)",
        fontsize=17)
    plt.plot(X, squid_population_list_both_fishing, color="red", alpha=0.6)
    plt.plot(X, squid_population_list, color="blue", alpha=0.6)

    plt.show()

    config_graph()

    plt.title(
        f"Population Density of Seal Over Time With and Without Both Species Fished ({MEASUREMENTS} measurements)",
        fontsize=17)
    plt.plot(X, seal_population_both_fishing, color="red", alpha=0.6)
    plt.plot(X, seal_population_list, color="blue", alpha=0.6)

    plt.show()

    config_graph()

    plt.title(
        f"Environment With Both Species Fished ({MEASUREMENTS} measurements)",
        fontsize=17)
    plt.plot(X, seal_population_both_fishing, color="red", alpha=0.6)
    plt.plot(X, squid_population_list_both_fishing, color="blue", alpha=0.6)

    plt.show()


# Call the functions that plot the graphs
plot_data_squid_fishing()
plot_data_seal_fishing()
plot_data_both_fishing()
