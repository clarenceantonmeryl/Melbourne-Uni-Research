# Import Libraries and Classes

import pandas as pd  # Handles the data in a DataFrame
import numpy as np  # Handles the arrays
import timeit  # Times the code
from DataGenerator import DataGenerator  # For general use in computing the squid and seal population density data
import matplotlib.pyplot as plt  # Plots the graphs
import seaborn as sns  # Also plots graphs
import json  # Stores the data in separate files so that the data doesn't have to be generated each run

# Initialise the number of measurements and x values
MEASUREMENTS = 500
X = [x for x in range(0, MEASUREMENTS + 1)]

# Create an object of the Class DataGenerator
data_generator = DataGenerator()


# Used to calculate the most stable environment
def stable_environment():
    # Initialise the lists that will contain the values of all the parameters
    R_list = []
    r_list = []
    C_list = []
    A_E_list = []
    # This is the target value
    gradient_means = []

    # For loop runs for each combination of values within a range
    for R_raw in range(14, 44):
        R = R_raw / 10

        for r_raw in range(10, 30):
            r = r_raw / 10

            for C_raw in range(2, 11):
                C = C_raw / 10

                for A_E in range(50, 201):

                    # Attempts to compute the squid and seal population density lists
                    try:
                        squid_population_list, seal_population_list = data_generator.get_population_lists(
                            a_n=50,
                            b_n=0.2,
                            R=R,
                            A_E=A_E,
                            C=C,
                            r=r
                        )

                    # However, an Overflow error can occur as the population density becomes too high or low for the computer to handle
                    # Thus, instead of terminating the code, this error is acknowledged and action is taken for it (in this case just to move on)
                    except OverflowError:
                        print(f"{R} {r} {C} {A_E} OVERFLOW")
                        continue
                    else:
                        # When there are no errors, the values of all the parameters will be added to their lists
                        R_list.append(R)
                        r_list.append(r)
                        C_list.append(C)
                        A_E_list.append(A_E)

                        # The gradients are calculated and stored in a list
                        squid_gradient, seal_gradient = data_generator.get_gradient_lists(
                            squid_population_list,
                            seal_population_list
                        )

                        # The medians are calculated
                        squid_median = np.median(squid_gradient)
                        seal_median = np.median(seal_gradient)

                        # The mean of these medians are calculated
                        squid_seal_mean = (squid_median + seal_median) / 2

                        # Since it is not the lowest gradient that is desired, but the gradient closest to 0 that is wanted, the absolute value is taken
                        # This will help with the sorting process later
                        if squid_seal_mean < 0:
                            squid_seal_mean = squid_seal_mean * -1
                        # Adds the result to the gradient_means list
                        gradient_means.append(squid_seal_mean)
                        # Prints what has happened in the console for the sake of records
                        print(f"{R} {r} {C} {A_E} {squid_seal_mean}")

    return R_list, r_list, C_list, A_E_list, gradient_means


def optimal_environment():
    # Initialises all lists
    R_list = []
    r_list = []
    C_list = []
    A_E_list = []
    density_means = []
    seal_medians = []
    squid_medians = []

    # Loop runs for every combination of parameter
    # Note the range is the combined range of both upper and lower ranges discussed in Part 6 of the report
    for R_raw in range(1, 43):
        R = R_raw / 10

        for r_raw in range(1, 30):
            r = r_raw / 10

            for C_raw in range(1, 11):
                C = C_raw / 10

                for A_E in range(50, 201):

                    # Ensures the program doesn't crash due to an Overflow error
                    try:
                        squid_population_list, seal_population_list = data_generator.get_population_lists(
                            a_n=50,
                            b_n=0.2,
                            R=R,
                            A_E=A_E,
                            C=C,
                            r=r
                        )

                    except OverflowError:
                        print(f"{R} {r} {C} {A_E} OVERFLOW")
                        continue
                    else:
                        # Add the combination of parameters to their lists
                        R_list.append(R)
                        r_list.append(r)
                        C_list.append(C)
                        A_E_list.append(A_E)

                        # Calculates the medians
                        squid_median = np.median(squid_population_list)
                        seal_median = np.median(seal_population_list)

                        # Adds these medians to their lists
                        squid_medians.append(squid_median)
                        seal_medians.append(seal_median)

                        # Calculates the mean of the medians and adds it to its list
                        squid_seal_mean = (squid_median + seal_median) / 2
                        density_means.append(squid_seal_mean)
                        # Prints the result in the console
                        print(f"{R} {r} {C} {A_E} {squid_seal_mean}")

    return R_list, r_list, C_list, A_E_list, density_means, squid_medians, seal_medians


def generate_data():
    # Initialises the lists through calling the stable_environment and optimal_environment functions
    R_list, r_list, C_list, A_E_list, gradient_means = stable_environment()
    R_list_optimal, r_list_optimal, C_list_optimal, A_E_list_optimal, density_means, squid_medians, seal_medians = optimal_environment()

    # Saves the data from the stable_environment function into a dictionary
    dictionary_stable = {
        "R": R_list,
        "r": r_list,
        "C": C_list,
        "A_E": A_E_list,
        "Gradient mean": gradient_means
    }

    print(len(R_list_optimal), len(r_list_optimal), len(C_list_optimal), len(A_E_list_optimal), len(density_means),
          len(squid_medians), len(seal_medians))

    # Saves the data from the optimal_environment function into a dictionary
    dictionary_optimal = {
        "R": R_list_optimal,
        "r": r_list_optimal,
        "C": C_list_optimal,
        "A_E": A_E_list_optimal,
        "Density mean": density_means,
        "Squid medians": squid_medians,
        "Seal medians": seal_medians
    }

    # Creates pandas DataFrames with these dictionaries
    df_stable = pd.DataFrame(data=dictionary_stable)

    df_optimal = pd.DataFrame(data=dictionary_optimal)

    # Creates more DataFrames, but these ones are sorted by their desired variable
    df_stable_sorted = df_stable.sort_values(by="Gradient mean")
    print(df_stable_sorted.head())

    df_optimal_overall_sorted = df_optimal.sort_values(by="Density mean", ascending=False)
    print(df_optimal_overall_sorted.head())

    df_optimal_squid_sorted = df_optimal.sort_values(by="Squid medians", ascending=False)
    print(df_optimal_squid_sorted.head())

    df_optimal_seal_sorted = df_optimal.sort_values(by="Seal medians", ascending=False)
    print(df_optimal_seal_sorted.head())

    # Save all the data to a json file to prevent having to load the data again
    df_stable_sorted.to_json('data/stable.json')
    df_optimal_overall_sorted.to_json('data/optimal-overall.json')
    df_optimal_squid_sorted.to_json('data/optimal-squid-sorted.json')
    df_optimal_seal_sorted.to_json('data/optimal-seal-sorted.json')


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


def plot_stable_optimal():
    # Gets the squid and seal population density data through the data_generator and plots the graphs in a similar way to main.py
    squid_population_list_stable, seal_population_list_stable = data_generator.get_population_lists(a_n=50, b_n=0.2,
                                                                                                    R=2.8,
                                                                                                    A_E=182, C=0.7,
                                                                                                    r=1.3)

    squid_population_list_overall_optimal, seal_population_list_overall_optimal = data_generator.get_population_lists(
        a_n=50, b_n=0.2, R=2.9, A_E=200, C=0.3, r=0.9)

    squid_population_list_squid_optimal, seal_population_list_squid_optimal = data_generator.get_population_lists(
        a_n=50,
        b_n=0.2,
        R=1.7,
        A_E=200,
        C=0.1,
        r=0.5)

    squid_population_list_seal_optimal, seal_population_list_seal_optimal = data_generator.get_population_lists(a_n=50,
                                                                                                                b_n=0.2,
                                                                                                                R=4.2,
                                                                                                                A_E=160,
                                                                                                                C=0.1,
                                                                                                                r=2.2)

    get_details(squid_population_list_seal_optimal, seal_population_list_seal_optimal)

    config_graph()
    plt.title(f"Most Stable Environment ({MEASUREMENTS} Measurements)", fontsize=17)
    plt.plot(X, squid_population_list_stable, color="blue")
    plt.plot(X, seal_population_list_stable, color="red")
    plt.show()

    config_graph()
    plt.title(f"Most Optimal Overall Environment ({MEASUREMENTS} Measurements)", fontsize=17)
    plt.plot(X, squid_population_list_overall_optimal, color="blue")
    plt.plot(X, seal_population_list_overall_optimal, color="red")
    plt.show()

    config_graph()
    plt.title(f"Most Optimal Environment for Squids ({MEASUREMENTS} Measurements)", fontsize=17)
    plt.plot(X, squid_population_list_squid_optimal, color="blue")
    plt.plot(X, seal_population_list_squid_optimal, color="red")
    plt.show()

    config_graph()
    plt.title(f"Most Optimal Environment for Seals ({MEASUREMENTS} Measurements)", fontsize=17)
    plt.plot(X, squid_population_list_seal_optimal, color="blue")
    plt.plot(X, seal_population_list_seal_optimal, color="red")
    plt.show()


def load_data():
    # Loads the json files back into pandas DataFrames for use in the code
    with open(file="data/optimal.json", mode="r") as upper_section_optimal_file:  # Opens the file
        print('start')  # Records Progress
        upper_section_data = json.load(upper_section_optimal_file)  # Saves the json data into a variable
        print('done') # Records Progress

    with open(file="data/optimal-1.json", mode="r") as lower_section_optimal_file:
        print('start')
        lower_section_data = json.load(lower_section_optimal_file)
        print('done')

    with open(file="data/optimal-overall.json", mode="r") as optimal_overall_file:
        print('start')
        optimal_overall_data = json.load(optimal_overall_file)
        print('done')

    with open(file="data/optimal-squid-sorted.json", mode="r") as optimal_squid_file:
        print('start')
        optimal_squid_data = json.load(optimal_squid_file)
        print('done')

    with open(file="data/optimal-seal-sorted.json", mode="r") as optimal_seal_file:
        print('start')
        optimal_seal_data = json.load(optimal_seal_file)
        print('done')

    with open(file="data/stable.json", mode="r") as stable_file:
        print('start')
        stable_data = json.load(stable_file)
        print('done')

    # Convert all the variables defined above to DataFrames
    upper_section_optimal_df = pd.DataFrame(data=upper_section_data)
    lower_section_optimal_df = pd.DataFrame(data=lower_section_data)
    overall_optimal_df = pd.DataFrame(data=optimal_overall_data)
    squid_optimal_df = pd.DataFrame(data=optimal_squid_data)
    seal_optimal_df = pd.DataFrame(data=optimal_seal_data)
    stable_df = pd.DataFrame(data=stable_data)

    # Store the frequency of each value as a separate json file
    overall_optimal_df['Density mean'].value_counts().to_json("data/val_count.json")

    squid_optimal_df['Squid medians'].value_counts().to_json("data/val_count_1.json")

    return upper_section_optimal_df, lower_section_optimal_df, overall_optimal_df, squid_optimal_df, seal_optimal_df, stable_df


def plot_lmplot(x, y, df, title, file_name):
    # Create the plot with the regression line
    sns.lmplot(x=x, y=y, data=df, height=10, scatter_kws={'alpha': 0.005}, line_kws={'color': 'darkred'})
    plt.title(f"{title}")  # Set the title
    plt.tight_layout()  # Allow all axis decorations to be visible
    plt.savefig(f"data/graph/{file_name}.png")  # Save the plot onto the computer
    plt.show()  # Show the plot


def generate_title(variable1, variable2):
    return f"The Effect of {variable1} on {variable2}"  # Format the title


def lmplots():
    upper_section_optimal_df, lower_section_optimal_df, overall_optimal_df, squid_optimal_df, seal_optimal_df, stable_df = load_data()

    # Stable
    print("Stable")  # Records Progress
    plot_lmplot(x='R', y='Gradient mean', df=stable_df, title=generate_title("R", "Gradient mean"),
                file_name="stable_R_1")  # Plot the lmplot using the plot_lmplot function defined above
    plot_lmplot(x='r', y='Gradient mean', df=stable_df, title=generate_title("r", "Gradient mean"),
                file_name="stable_r_2")
    plot_lmplot(x='C', y='Gradient mean', df=stable_df, title=generate_title("C", "Gradient mean"),
                file_name="stable_C")
    plot_lmplot(x='A_E', y='Gradient mean', df=stable_df, title=generate_title("A_E", "Gradient mean"),
                file_name="stable_A_E")

    # Upper
    print("Upper")
    plot_lmplot(x='R', y='Density mean', df=upper_section_optimal_df, title=generate_title("R", "Density mean"),
                file_name="upper_R_1")
    plot_lmplot(x='r', y='Density mean', df=upper_section_optimal_df, title=generate_title("r", "Density mean"),
                file_name="upper_r_2")
    plot_lmplot(x='C', y='Density mean', df=upper_section_optimal_df, title=generate_title("C", "Density mean"),
                file_name="upper_C")
    plot_lmplot(x='A_E', y='Density mean', df=upper_section_optimal_df, title=generate_title("A_E", "Density mean"),
                file_name="upper_A_E")

    # Lower
    print("Lower")
    plot_lmplot(x='R', y='Density mean', df=lower_section_optimal_df, title=generate_title("R", "Density mean"),
                file_name="lower_R_1")
    plot_lmplot(x='r', y='Density mean', df=lower_section_optimal_df, title=generate_title("r", "Density mean"),
                file_name="lower_r_2")
    plot_lmplot(x='C', y='Density mean', df=lower_section_optimal_df, title=generate_title("C", "Density mean"),
                file_name="lower_C")
    plot_lmplot(x='A_E', y='Density mean', df=lower_section_optimal_df, title=generate_title("A_E", "Density mean"),
                file_name="lower_A_E")

    # Overall
    print("Overall")
    plot_lmplot(x='R', y='Density mean', df=overall_optimal_df, title=generate_title("R", "Density mean"),
                file_name="overall_R_1")
    plot_lmplot(x='r', y='Density mean', df=overall_optimal_df, title=generate_title("r", "Density mean"),
                file_name="overall_r_2")
    plot_lmplot(x='C', y='Density mean', df=overall_optimal_df, title=generate_title("C", "Density mean"),
                file_name="overall_C")
    plot_lmplot(x='A_E', y='Density mean', df=overall_optimal_df, title=generate_title("A_E", "Density mean"),
                file_name="overall_A_E")

    # Squid
    print("Squid Optimal")
    plot_lmplot(x='R', y='Squid medians', df=squid_optimal_df, title=generate_title("R", "Squid medians"),
                file_name="squid_R_1")
    plot_lmplot(x='r', y='Squid medians', df=squid_optimal_df, title=generate_title("r", "Squid medians"),
                file_name="squid_r_2")
    plot_lmplot(x='C', y='Squid medians', df=squid_optimal_df, title=generate_title("C", "Squid medians"),
                file_name="squid_C")
    plot_lmplot(x='A_E', y='Squid medians', df=squid_optimal_df, title=generate_title("A_E", "Squid medians"),
                file_name="squid_A_E")

    # Seal
    print("Seal Optimal")
    plot_lmplot(x='R', y='Seal medians', df=seal_optimal_df, title=generate_title("R", "Seal medians"),
                file_name="seal_R_1")
    plot_lmplot(x='r', y='Seal medians', df=seal_optimal_df, title=generate_title("r", "Seal medians"),
                file_name="seal_r_2")
    plot_lmplot(x='C', y='Seal medians', df=seal_optimal_df, title=generate_title("C", "Seal medians"),
                file_name="seal_C")
    plot_lmplot(x='A_E', y='Seal medians', df=seal_optimal_df, title=generate_title("A_E", "Seal medians"),
                file_name="seal_A_E")


start_time = timeit.default_timer()  # Initialise the time for running
# Call all the functions
generate_data()
load_data()
plot_stable_optimal()
lmplots()

print(timeit.default_timer() - start_time)  # Prints the total run time
