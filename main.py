
import matplotlib.pyplot as plt
import numpy as np
from DataGenerator import DataGenerator

MEASUREMENTS = 500

data_generator = DataGenerator


def squid_population(A_n, B_n, R, A_E, C):
    A_n_plus_one = R * A_n - ((R - 1) / A_E) * (A_n ** 2) - (C * A_n * B_n)
    return A_n_plus_one


def seal_population(A_n, B_n, A_E, r):
    B_n_plus_one = (r / A_E) * A_n * B_n
    return B_n_plus_one


def get_population_lists(a_n, b_n, R=3.0, A_E=100.0, C=0.5, r=2.0):
    squid_population_list = []
    seal_population_list = []

    squid_population_list.append(a_n)
    seal_population_list.append(b_n)

    for _ in range(len(X) - 1):
        squid_population_list.append(
            squid_population(
                A_n=squid_population_list[-1],
                B_n=seal_population_list[-1],
                R=R,
                A_E=A_E,
                C=C
            )
        )

        seal_population_list.append(
            seal_population(
                A_n=squid_population_list[-2],
                B_n=seal_population_list[-1],
                A_E=A_E,
                r=r
            )
        )

    return squid_population_list, seal_population_list


def get_details(squid_population_list, seal_population_list):
    print("Mean squid, seal")
    print(np.mean(squid_population_list))
    print(np.mean(seal_population_list))
    print("Max squid, seal")
    print(np.max(squid_population_list))
    print(np.max(seal_population_list))
    print("Min squid, seal")
    print(np.min(squid_population_list))
    print(np.min(seal_population_list))


def config_graph():
    plt.figure(figsize=(24, 14))
    plt.xlim(0, MEASUREMENTS)
    plt.xlabel("Time (n units of time)", fontsize=15)
    plt.ylabel("Population density (animals/km²)", fontsize=15)


def plot_data():
    squid_population_list, seal_population_list = get_population_lists(a_n=50, b_n=0.2)

    config_graph()
    plt.title(f"Squid Population Density Over Time ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list)

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list, color='red')

    plt.show()

    config_graph()

    plt.title(f"Population Densities of Squid and Seal Over Time ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list)
    plt.plot(X, seal_population_list, color='red')

    plt.show()


# Variables to experiment: R, A_E, C, r

def plot_data_varying_R():
    squid_population_list_normal_R, seal_population_list_normal_R = get_population_lists(a_n=50, b_n=0.2)
    squid_population_list_high_R, seal_population_list_high_R = get_population_lists(a_n=50, b_n=0.2, R=4.23)
    squid_population_list_low_R, seal_population_list_low_R = get_population_lists(a_n=50, b_n=0.2, R=1.5)

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and Low R Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_R, color="blue")
    plt.plot(X, squid_population_list_low_R, color="green")

    plt.show()

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and High R Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_R, color="blue")
    plt.plot(X, squid_population_list_high_R, color="red")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and Low R Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_R, color="blue")
    plt.plot(X, seal_population_list_low_R, color="green")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and High R Values ({MEASUREMENTS}) Measurements",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_R, color="blue")
    plt.plot(X, seal_population_list_high_R, color="red")

    plt.show()


def plot_data_varying_A_E():
    squid_population_list_normal_A_E, seal_population_list_normal_A_E = get_population_lists(a_n=50, b_n=0.2)
    squid_population_list_high_A_E, seal_population_list_high_A_E = get_population_lists(a_n=50, b_n=0.2, A_E=200)
    squid_population_list_low_A_E, seal_population_list_low_A_E = get_population_lists(a_n=50, b_n=0.2, A_E=50)

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and Low A_E Values ({MEASUREMENTS} Measurements",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_A_E, color="blue")
    plt.plot(X, squid_population_list_low_A_E, color="green")

    plt.show()

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and High A_E Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_A_E, color="blue")
    plt.plot(X, squid_population_list_high_A_E, color="red")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and Low A_E Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_A_E, color="blue")
    plt.plot(X, seal_population_list_low_A_E, color="green")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and High A_E Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_A_E, color="blue")
    plt.plot(X, seal_population_list_high_A_E, color="red")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Low and High A_E Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_low_A_E, color="green")
    plt.plot(X, seal_population_list_high_A_E, color="red")

    plt.show()


def plot_data_varying_C():
    squid_population_list_normal_C, seal_population_list_normal_C = get_population_lists(a_n=50, b_n=0.2)
    squid_population_list_high_C, seal_population_list_high_C = get_population_lists(a_n=50, b_n=0.2, C=1)
    squid_population_list_low_C, seal_population_list_low_C = get_population_lists(a_n=50, b_n=0.2, C=0.25)

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and Low C Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_C, color="blue")
    plt.plot(X, squid_population_list_low_C, color="green")

    plt.show()

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and High C Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_C, color="blue")
    plt.plot(X, squid_population_list_high_C, color="red")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and Low C Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_C, color="blue")
    plt.plot(X, seal_population_list_low_C, color="green")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and High C Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_C, color="blue")
    plt.plot(X, seal_population_list_high_C, color="red")

    plt.show()


def plot_data_varying_r():
    squid_population_list_normal_r, seal_population_list_normal_r = get_population_lists(a_n=50, b_n=0.2)
    squid_population_list_high_r, seal_population_list_high_r = get_population_lists(a_n=50, b_n=0.2, r=2.88)
    squid_population_list_low_r, seal_population_list_low_r = get_population_lists(a_n=50, b_n=0.2, r=1)

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and Low r Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_r, color="blue")
    plt.plot(X, squid_population_list_low_r, color="green")

    plt.show()

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and High r Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_r, color="blue")
    plt.plot(X, squid_population_list_high_r, color="red")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and Low r Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_r, color="blue")
    plt.plot(X, seal_population_list_low_r, color="green")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and High r Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_r, color="blue")
    plt.plot(X, seal_population_list_high_r, color="red")

    plt.show()


def plot_data_varying_starting_a():
    squid_population_list_normal_starting_a, seal_population_list_normal_starting_a = get_population_lists(a_n=50,
                                                                                                           b_n=0.2)
    squid_population_list_high_starting_a, seal_population_list_high_starting_a = get_population_lists(a_n=125, b_n=0.2)
    squid_population_list_low_starting_a, seal_population_list_low_starting_a = get_population_lists(a_n=12.5, b_n=0.2)

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and Low A_0 Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_starting_a, color="blue")
    plt.plot(X, squid_population_list_low_starting_a, color="green")

    plt.show()

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and High A_0 Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_starting_a, color="blue")
    plt.plot(X, squid_population_list_high_starting_a, color="red")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and Low A_0 Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_starting_a, color="blue")
    plt.plot(X, seal_population_list_low_starting_a, color="green")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and High A_0 Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_starting_a, color="blue")
    plt.plot(X, seal_population_list_high_starting_a, color="red")

    plt.show()


def plot_data_varying_starting_b():
    squid_population_list_normal_starting_b, seal_population_list_normal_starting_b = get_population_lists(a_n=50,
                                                                                                           b_n=0.2)
    squid_population_list_high_starting_b, seal_population_list_high_starting_b = get_population_lists(a_n=50, b_n=0.8)
    squid_population_list_low_starting_b, seal_population_list_low_starting_b = get_population_lists(a_n=50, b_n=0.05)

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and Low B_0 Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_starting_b, color="blue")
    plt.plot(X, squid_population_list_low_starting_b, color="green")

    plt.show()

    config_graph()

    plt.title(f"Squid Population Density Over Time With Normal and High B_0 Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_normal_starting_b, color="blue")
    plt.plot(X, squid_population_list_high_starting_b, color="red")

    plt.show()

    config_graph()
    plt.title(f"Seal Population Density Over Time With Normal and Low B_0 Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_starting_b, color="blue")
    plt.plot(X, seal_population_list_low_starting_b, color="green")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With Normal and High B_0 Values ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list_normal_starting_b, color="blue")
    plt.plot(X, seal_population_list_high_starting_b, color="red")

    plt.show()


def plot_data_no_squid_start():
    squid_population_list, seal_population_list = get_population_lists(a_n=50, b_n=0.2)
    squid_population_list_no_squid, seal_population_list_no_squid = get_population_lists(a_n=0, b_n=0.2)

    config_graph()

    plt.title(f"Squid Population Density Over Time With and Without Squids Initially ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list, color="blue")
    plt.plot(X, squid_population_list_no_squid, color="green")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With and Without Squids Initially ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list, color="blue")
    plt.plot(X, seal_population_list_no_squid, color="green")

    plt.show()

    config_graph()

    plt.title(f"Environment Without Squids Initially ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_no_squid, color="green")
    plt.plot(X, seal_population_list_no_squid, color="red", alpha=0.5)

    plt.show()


def plot_data_no_seal_start():
    squid_population_list, seal_population_list = get_population_lists(a_n=50, b_n=0.2)
    squid_population_list_no_seal, seal_population_list_no_seal = get_population_lists(a_n=50, b_n=0)

    config_graph()

    plt.title(f"Squid Population Density Over Time With and Without Seals Initially ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list, color="blue")
    plt.plot(X, squid_population_list_no_seal, color="green")

    plt.show()

    config_graph()

    plt.title(f"Seal Population Density Over Time With and Without Seals Initially ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, seal_population_list, color="blue")
    plt.plot(X, seal_population_list_no_seal, color="green")

    plt.show()

    config_graph()

    plt.title(f"Environment Without Seals Initially ({MEASUREMENTS} Measurements)",
              fontsize=17)
    plt.plot(X, squid_population_list_no_seal, color="green")
    plt.plot(X, seal_population_list_no_seal, color="red", alpha=0.5)

    plt.show()


# plot_data()
# plot_data_varying_R()
# plot_data_varying_A_E()
# plot_data_varying_C()
# plot_data_varying_r()
# plot_data_varying_starting_a()
# plot_data_varying_starting_b()
# plot_data_no_squid_start()
# plot_data_no_seal_start()
