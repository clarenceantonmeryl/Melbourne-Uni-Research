# Calculate the most stable environment by finding the median gradient of seal and squid

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
                        R_list.append(R)
                        r_list.append(r)
                        C_list.append(C)
                        A_E_list.append(A_E)

                        squid_gradient, seal_gradient = data_generator.get_gradient_lists(
                            squid_population_list,
                            seal_population_list
                        )

                        squid_median = np.median(squid_gradient)
                        seal_median = np.median(seal_gradient)

                        squid_seal_mean = (squid_median + seal_median) / 2
                        if squid_seal_mean < 0:
                            squid_seal_mean = squid_seal_mean * -1
                        gradient_means.append(squid_seal_mean)
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

    for R_raw in range(1, 43):
        R = R_raw / 10

        for r_raw in range(1, 30):
            r = r_raw / 10

            for C_raw in range(1, 11):
                C = C_raw / 10

                for A_E in range(50, 201):

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
                        R_list.append(R)
                        r_list.append(r)
                        C_list.append(C)
                        A_E_list.append(A_E)

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
    with open(file="data/optimal.json", mode="r") as upper_section_optimal_file:
        print('start')
        upper_section_data = json.load(upper_section_optimal_file)
        print('done')

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

    upper_section_optimal_df = pd.DataFrame(data=upper_section_data)
    lower_section_optimal_df = pd.DataFrame(data=lower_section_data)
    overall_optimal_df = pd.DataFrame(data=optimal_overall_data)
    squid_optimal_df = pd.DataFrame(data=optimal_squid_data)
    seal_optimal_df = pd.DataFrame(data=optimal_seal_data)
    stable_df = pd.DataFrame(data=stable_data)

    return upper_section_optimal_df, lower_section_optimal_df, overall_optimal_df, squid_optimal_df, seal_optimal_df, stable_df


def get_mask(df):
    mask = np.zeros_like(df.corr())
    triangle_indices = np.triu_indices_from(mask)
    mask[triangle_indices] = True

    return mask


def plot_heatmap(df, mask, title):
    plt.figure(figsize=(24, 14))
    plt.title(f"{title}", fontsize=17)
    sns.heatmap(df.corr(), mask=mask, annot=True, annot_kws={"size": 15})
    sns.set_style('white')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def heatmaps():
    upper_section_optimal_df, lower_section_optimal_df, overall_optimal_df, squid_optimal_df, seal_optimal_df, stable_df = load_data()

    upper_section_mask = get_mask(df=upper_section_optimal_df)
    lower_section_mask = get_mask(df=lower_section_optimal_df)
    mask_stable = get_mask(df=stable_df)
    mask_overall = get_mask(df=overall_optimal_df)

    plot_heatmap(df=upper_section_optimal_df, mask=upper_section_mask,
                 title="Correlations Between Variables for the Upper Section Optimal Environment")
    plot_heatmap(df=lower_section_optimal_df, mask=lower_section_mask,
                 title="Correlations Between Variables for the Lower Section Optimal Environment")
    plot_heatmap(df=stable_df, mask=mask_stable, title="Correlations Between Variables for Most Stable Environment")
    plot_heatmap(df=overall_optimal_df, mask=mask_overall,
                 title="Correlations Between Variables for Most Optimal Environment Overall, for Squids and for Seals")


def plot_lmplot(x, y, df, title):
    sns.lmplot(x=x, y=y, data=df, height=10, scatter_kws={'alpha': 0.6}, line_kws={'color': 'darkred'})
    plt.title(f"{title}")
    plt.tight_layout()
    plt.show()


def generate_title(variable1, variable2):
    return f"Graph of {variable1} Against {variable2}"


def lmplots():
    upper_section_optimal_df, lower_section_optimal_df, overall_optimal_df, squid_optimal_df, seal_optimal_df, stable_df = load_data()

    # Stable
    # print("Stable")
    # plot_lmplot(x='R', y='Gradient mean', df=stable_df, title=generate_title("R", "Gradient mean"))
    # plot_lmplot(x='r', y='Gradient mean', df=stable_df, title=generate_title("r", "Gradient mean"))
    # plot_lmplot(x='C', y='Gradient mean', df=stable_df, title=generate_title("C", "Gradient mean"))
    # plot_lmplot(x='A_E', y='Gradient mean', df=stable_df, title=generate_title("A_E", "Gradient mean"))

    # Upper
    # print("Upper")
    # plot_lmplot(x='R', y='Density mean', df=upper_section_optimal_df, title=generate_title("R", "Density mean"))
    # plot_lmplot(x='r', y='Density mean', df=upper_section_optimal_df, title=generate_title("r", "Density mean"))
    # plot_lmplot(x='C', y='Density mean', df=upper_section_optimal_df, title=generate_title("C", "Density mean"))
    # plot_lmplot(x='A_E', y='Density mean', df=upper_section_optimal_df, title=generate_title("A_E", "Density mean"))

    # Lower
    # print("Lower")
    # plot_lmplot(x='R', y='Density mean', df=lower_section_optimal_df, title=generate_title("R", "Density mean"))
    # plot_lmplot(x='r', y='Density mean', df=lower_section_optimal_df, title=generate_title("r", "Density mean"))
    # plot_lmplot(x='C', y='Density mean', df=lower_section_optimal_df, title=generate_title("C", "Density mean"))
    # plot_lmplot(x='A_E', y='Density mean', df=lower_section_optimal_df, title=generate_title("A_E", "Density mean"))

    # Overall
    # print("Overall")
    # plot_lmplot(x='R', y='Density mean', df=overall_optimal_df, title=generate_title("R", "Density mean"))
    # plot_lmplot(x='r', y='Density mean', df=overall_optimal_df, title=generate_title("r", "Density mean"))
    # plot_lmplot(x='C', y='Density mean', df=overall_optimal_df, title=generate_title("C", "Density mean"))
    # plot_lmplot(x='A_E', y='Density mean', df=overall_optimal_df, title=generate_title("A_E", "Density mean"))

    # Squid
    # print("Squid Optimal")
    # plot_lmplot(x='R', y='Squid medians', df=squid_optimal_df, title=generate_title("R", "Squid medians"))
    # plot_lmplot(x='r', y='Squid medians', df=squid_optimal_df, title=generate_title("r", "Squid medians"))
    # plot_lmplot(x='C', y='Squid medians', df=squid_optimal_df, title=generate_title("C", "Squid medians"))
    # plot_lmplot(x='A_E', y='Squid medians', df=squid_optimal_df, title=generate_title("A_E", "Squid medians"))

    # Seal
    # print("Seal Optimal")
    # plot_lmplot(x='R', y='Seal medians', df=seal_optimal_df, title=generate_title("R", "Seal medians"))
    plot_lmplot(x='r', y='Seal medians', df=seal_optimal_df, title=generate_title("r", "Seal medians"))
    # plot_lmplot(x='C', y='Seal medians', df=seal_optimal_df, title=generate_title("C", "Seal medians"))
    # plot_lmplot(x='A_E', y='Seal medians', df=seal_optimal_df, title=generate_title("A_E", "Seal medians"))


start_time = timeit.default_timer()
# generate_data()
# heatmaps()
lmplots()
# plot_stable_optimal()
print(timeit.default_timer() - start_time)
