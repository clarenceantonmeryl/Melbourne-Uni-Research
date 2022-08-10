# Note: In this file, and other files, population is used sometimes instead of population density
# For example, population_list actually means the population density list

# Import Libraries and Classes

# matplotlib will be used to plot graphs
import matplotlib.pyplot as plt

# numpy will be used to handle arrays
import numpy as np

# DataGenerator is a class from the DataGenerator.py file
from DataGenerator import DataGenerator

# Initialise the amount of measurements per graph
MEASUREMENTS = 500

# Initialise a DataGenerator object
data_generator = DataGenerator()

# Initialise the X list, which will act as a list of x values for graphs
X = [x for x in range(0, MEASUREMENTS + 1)]


def squid_population(A_n, B_n, R, A_E, C):
    # Compute the squid population density given the parameters
    A_n_plus_one = R * A_n - ((R - 1) / A_E) * (A_n ** 2) - (C * A_n * B_n)
    return A_n_plus_one


def seal_population(A_n, B_n, A_E, r):
    # Compute the seal population density given the parameters
    B_n_plus_one = (r / A_E) * A_n * B_n
    return B_n_plus_one


def get_population_lists(a_n, b_n, R=3.0, A_E=100.0, C=0.5, r=2.0):
    # Initialise the population density lists for squids and seals, which will act as the y values for graphs
    squid_population_list = []
    seal_population_list = []

    # Add the initial squid and seal population densities
    squid_population_list.append(a_n)
    seal_population_list.append(b_n)

    for _ in range(len(X) - 1):
        # This loop will run for the amount of x values

        # Compute the squid population density with the given parameters (an index of -1, such as list[-1] gets the last item of a list)
        squid_population_list.append(
            squid_population(
                A_n=squid_population_list[-1],
                B_n=seal_population_list[-1],
                R=R,
                A_E=A_E,
                C=C
            )
        )

        # Compute the seal population density with the given parameters (an index of -2 is used because the new squid population density would have already been added)
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
    # Given two lists, print the means, maximums and minimums for squids and seals
    print("Mean squid, seal")
    print(np.mean(squid_population_list))
    print(np.mean(seal_population_list))
    print("Max squid, seal")
    print(np.max(squid_population_list))
    print(np.max(seal_population_list))
    print("Min squid, seal")
    print(np.min(squid_population_list))
    print(np.min(seal_population_list))


def present(list_presented):
    # Print the list in a readable format in the Python Console
    for index in range(0, len(list_presented)):
        print(f"{index}: {list_presented[index]}")


def config_graph():
    # Sets up the graphs by setting the size, axis titles, font sizes and x limits
    plt.figure(figsize=(24, 14))
    plt.xlim(0, MEASUREMENTS)
    plt.xlabel("Time (n units of time)", fontsize=15)
    plt.ylabel("Population density (animals/km²)", fontsize=15)


def plot_data():

    # Get squid and seal population density data
    squid_population_list, seal_population_list = get_population_lists(a_n=50, b_n=0.2)

    # A sample of how get_details can be used
    get_details(squid_population_list, seal_population_list)

    # A sample of how present can be used
    present(squid_population_list)
    present(seal_population_list)

    # Initialise the graph
    config_graph()

    # Set the title of the graph, along with its font size
    plt.title(f"Squid Population Density Over Time ({MEASUREMENTS} Measurements)",
              fontsize=17)

    # Plotting the actual graph
    plt.plot(X, squid_population_list)

    # Showing the graph (this and the above 3 steps will be used repeatedly to generate a graph in this and other plot functions)
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


def plot_data_varying_R():
    # Get the squid and seal population density data for various R values
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
    # Get the squid and seal population density data for various A_E values
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
    # Get the squid and seal population density data for various C values
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
    # Get the squid and seal population density data for various r values
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
    # Get the squid and seal population density data for various A_0 values
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
    # Get the squid and seal population density data for various B_0 values
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
    # Get the squid and seal population density data for when no squids are present initially and when they are present initially
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
    # Get the squid and seal population density data for when no seals are present initially and when they are present initially
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


# Calling the functions
plot_data()
plot_data_varying_R()
plot_data_varying_A_E()
plot_data_varying_C()
plot_data_varying_r()
plot_data_varying_starting_a()
plot_data_varying_starting_b()
plot_data_no_squid_start()
plot_data_no_seal_start()
