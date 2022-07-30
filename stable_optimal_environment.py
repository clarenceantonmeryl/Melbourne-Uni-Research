


# Calculate the most stable ecosystem by finding the median gradient of seal and squid

# Create a for loop for each parameter, and at the most inner of the loop: generate the list, find the median  gradient
# of each species, find the mean of this and add this to a dataframe. Dataframe should have an index, entry for each
# parameter value, and the mean of the medians. Median is used to take care of the outliers (spikes and drops)

# R: 0-5
# r: 0-5
# C: 0-25
# AE: 0-500
# A0: 0-100
# B0: 0-100

import pandas as pd
import numpy as np
import timeit
from DataGenerator import DataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json

MEASUREMENTS = 500
X = np.linspace(0, MEASUREMENTS, MEASUREMENTS)

data_generator = DataGenerator()


def stable_environment():
    R_list = []
    r_list = []
    C_list = []
    A_E_list = []
    gradient_means = []
    for R_raw in range(14, 44):
        R = R_raw / 10

        for r_raw in range(10, 30):
            r = r_raw / 10

            for C_raw in range(2, 11):
                C = C_raw / 10

                for A_E in range(50, 201):

                    R_list.append(R)
                    r_list.append(r)
                    C_list.append(C)
                    A_E_list.append(A_E)

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
                        gradient_means.append(10000000)
                        print(f"{R} {r} {C} {A_E} OVERFLOW")

                    else:
                        squid_gradient, seal_gradient = data_generator.get_gradient_lists(
                            squid_population_list,
                            seal_population_list
                        )

                        squid_median = np.median(squid_gradient)
                        seal_median = np.median(seal_gradient)

                        squid_seal_mean = (squid_median + seal_median) / 2
                        gradient_means.append(abs(squid_seal_mean))
                        print(f"{R} {r} {C} {A_E} {squid_seal_mean}")

    return R_list, r_list, C_list, A_E_list, gradient_means


def optimal_environment():
    R_list = []
    r_list = []
    C_list = []
    A_E_list = []
    density_means = []
    seal_medians = []
    squid_medians = []

    for R_raw in range(1, 44):
        R = R_raw / 10

        for r_raw in range(1, 30):
            r = r_raw / 10

            for C_raw in range(1, 11):
                C = C_raw / 10

                for A_E in range(50, 201):

                    R_list.append(R)
                    r_list.append(r)
                    C_list.append(C)
                    A_E_list.append(A_E)

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
                        density_means.append(-1000000)
                        squid_medians.append(-1000000)
                        seal_medians.append(-1000000)
                    else:
                        squid_median = np.median(squid_population_list)
                        seal_median = np.median(seal_population_list)

                        squid_medians.append(squid_median)
                        seal_medians.append(seal_median)

                        squid_seal_mean = (squid_median + seal_median) / 2
                        density_means.append(squid_seal_mean)
                        print(f"{R} {r} {C} {A_E} {squid_seal_mean}")

    return R_list, r_list, C_list, A_E_list, density_means, squid_medians, seal_medians


def generate_data():
    R_list, r_list, C_list, A_E_list, gradient_means = stable_environment()
    R_list_optimal, r_list_optimal, C_list_optimal, A_E_list_optimal, density_means, squid_medians, seal_medians = optimal_environment()

    dictionary_stable = {
        "R": R_list,
        "r": r_list,
        "C": C_list,
        "A_E": A_E_list,
        "Gradient mean": gradient_means
    }

    print(len(R_list_optimal), len(r_list_optimal), len(C_list_optimal), len(A_E_list_optimal), len(density_means),
          len(squid_medians), len(seal_medians))

    dictionary_optimal = {
        "R": R_list_optimal,
        "r": r_list_optimal,
        "C": C_list_optimal,
        "A_E": A_E_list_optimal,
        "Density mean": density_means,
        "Squid medians": squid_medians,
        "Seal medians": seal_medians
    }

    df_stable = pd.DataFrame(data=dictionary_stable)

    df_optimal = pd.DataFrame(data=dictionary_optimal)

    df_stable_sorted = df_stable.sort_values(by="Gradient mean")
    print(df_stable_sorted.head())

    df_optimal_overall_sorted = df_optimal.sort_values(by="Density mean", ascending=False)
    print(df_optimal_overall_sorted.head())

    df_optimal_squid_sorted = df_optimal.sort_values(by="Squid medians", ascending=False)
    print(df_optimal_squid_sorted.head())

    df_optimal_seal_sorted = df_optimal.sort_values(by="Seal medians", ascending=False)
    print(df_optimal_seal_sorted.head())

    df_stable_sorted.to_json('stable.json')
    df_optimal_overall_sorted.to_json('optimal-overall.json')
    df_optimal_squid_sorted.to_json('optimal-squid-sorted.json')
    df_optimal_seal_sorted.to_json('optimal-seal-sorted.json')


def config_graph():
    plt.figure(figsize=(24, 14))
    plt.xlim(0, MEASUREMENTS)
    plt.xlabel("Time (n units of time)", fontsize=15)
    plt.ylabel("Population density (animals/km²)", fontsize=15)


def plot_stable_optimal():
    squid_population_list_stable, seal_population_list_stable = data_generator.get_population_lists(a_n=50, b_n=0.2,
                                                                                                    R=2.7,
                                                                                                    A_E=119, C=0.5,
                                                                                                    r=1.4)

    squid_population_list_overall_optimal, seal_population_list_overall_optimal = data_generator.get_population_lists(
        a_n=50, b_n=0.2, R=1.4, A_E=200, C=0.2, r=1.0)

    squid_population_list_squid_optimal, seal_population_list_squid_optimal = data_generator.get_population_lists(
        a_n=50,
        b_n=0.2,
        R=1.7,
        A_E=200,
        C=0.3,
        r=0.3)

    squid_population_list_seal_optimal, seal_population_list_seal_optimal = data_generator.get_population_lists(a_n=50,
                                                                                                                b_n=0.2,
                                                                                                                R=4.3,
                                                                                                                A_E=57,
                                                                                                                C=0.1,
                                                                                                                r=2.2)

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
    with open(file="/data/optimal-overall.json", mode="r") as optimal_overall_file:
        print('start')
        optimal_overall_data = json.load(optimal_overall_file)
        print('done')

    with open(file="/data/optimal-squid-sorted.json", mode="r") as optimal_squid_file:
        print('start')
        optimal_squid_data = json.load(optimal_squid_file)
        print('done')

    with open(file="/data/optimal-seal-sorted.json", mode="r") as optimal_seal_file:
        print('start')
        optimal_seal_data = json.load(optimal_seal_file)
        print('done')

    with open(file="/data/stable.json", mode="r") as stable_file:
        print('start')
        stable_data = json.load(stable_file)
        print('done')

    overall_optimal_df = pd.DataFrame(data=optimal_overall_data)
    squid_optimal_df = pd.DataFrame(data=optimal_squid_data)
    seal_optimal_df = pd.DataFrame(data=optimal_seal_data)
    stable_df = pd.DataFrame(data=stable_data)

    return overall_optimal_df, squid_optimal_df, seal_optimal_df, stable_df


start_time = timeit.default_timer()

overall_optimal_df, squid_optimal_df, seal_optimal_df, stable_df = load_data()

mask_stable = np.zeros_like(stable_df.corr())
triangle_indices = np.triu_indices_from(mask_stable)
mask_stable[triangle_indices] = True

mask_overall = np.zeros_like(overall_optimal_df.corr())
triangle_indices = np.triu_indices_from(mask_overall)
mask_overall[triangle_indices] = True

mask_squid = np.zeros_like(squid_optimal_df.corr())
triangle_indices = np.triu_indices_from(mask_squid)
mask_squid[triangle_indices] = True

mask_seal = np.zeros_like(seal_optimal_df.corr())
triangle_indices = np.triu_indices_from(mask_seal)
mask_seal[triangle_indices] = True

plt.figure(figsize=(24, 14))
plt.title("Correlations Between Variables for Most Stable Ecosystem", fontsize=17)
sns.heatmap(stable_df.corr(), mask=mask_stable, annot=True, annot_kws={"size": 15})
sns.set_style('white')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure(figsize=(24, 14))
plt.title("Correlations Between Variables for Most Optimal Ecosystem Overall", fontsize=17)
sns.heatmap(overall_optimal_df.corr(), mask=mask_overall, annot=True, annot_kws={"size": 15})
sns.set_style('white')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure(figsize=(24, 14))
plt.title("Correlations Between Variables for Most Optimal Ecosystem for Squids", fontsize=17)
sns.heatmap(squid_optimal_df.corr(), mask=mask_squid, annot=True, annot_kws={"size": 15})
sns.set_style('white')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure(figsize=(24, 14))
plt.title("Correlations Between Variables for Most Optimal Ecosystem for Seals", fontsize=17)
sns.heatmap(seal_optimal_df.corr(), mask=mask_seal, annot=True, annot_kws={"size": 15})
sns.set_style('white')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

sns.pairplot(stable_df, kind='reg', plot_kws={'line_kws': {'color': 'red'}})
plt.show()

sns.pairplot(overall_optimal_df, kind='reg', plot_kws={'line_kws': {'color': 'red'}})
plt.show()

sns.pairplot(squid_optimal_df, kind='reg', plot_kws={'line_kws': {'color': 'red'}})
plt.show()

sns.pairplot(seal_optimal_df, kind='reg', plot_kws={'line_kws': {'color': 'red'}})
plt.show()

print(timeit.default_timer() - start_time)
