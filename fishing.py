import matplotlib.pyplot as plt
import numpy as np

from DataGenerator import DataGenerator

data_generator = DataGenerator()

MEASUREMENTS = 500


X = np.linspace(0, MEASUREMENTS, MEASUREMENTS)


def config_graph():
    plt.figure(figsize=(24, 14))
    plt.xlim(0, MEASUREMENTS)
    plt.xlabel("Time (n units of time)", fontsize=15)
    plt.ylabel("Population density (animals/km²)", fontsize=15)


def plot_data_squid_fishing(k=25000, P=0.9, A=3000):
    squid_population_list_fishing, seal_population_list_squid_fishing = data_generator.get_population_lists_squid_fished(
        a_n=50,
        b_n=0.2,
        k=k,
        P=P,
        A=A
    )

    squid_population_list, seal_population_list = data_generator.get_population_lists(a_n=50, b_n=0.2)

    config_graph()

    ax = plt.gca()
    ax.set_yticks(range(0, 96, 5))

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
    squid_population_list_seal_fishing, seal_population_list_fishing = data_generator.get_population_lists_seal_fished(
        a_n=50,
        b_n=0.2,
        k=k,
        P=P,
        A=A
    )

    squid_population_list, seal_population_list = data_generator.get_population_lists(a_n=50, b_n=0.2)

    config_graph()

    ax = plt.gca()
    ax.set_yticks(range(0, 101, 5))

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

    squid_population_list, seal_population_list = data_generator.get_population_lists(a_n=50, b_n=0.2)

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


# plot_data_squid_fishing()
# plot_data_seal_fishing()
# plot_data_both_fishing()
