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
from DataGenerator import DataGenerator


density_means = []

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

                for A_E_raw in range(50, 201):
                    A_E = A_E_raw

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


# def optimal_environment():
#     for R_raw in range(30, 50):
#         R = R_raw / 100
#
#         for r_raw in range(30, 50):
#             r = r_raw / 100
#
#             for C_raw in range(75, 100):
#                 C = C_raw / 100
#
#                 for A_E_raw in range(75, 100):
#                     A_E = A_E_raw / 100
#
#                     for A_0_raw in range(75, 100):
#                         A_0 = A_0_raw / 100
#
#                         for B_0_raw in range(75, 100):
#                             B_0 = B_0_raw / 100
#
#                             R_list.append(R)
#                             r_list.append(r)
#                             C_list.append(C)
#                             A_E_list.append(A_E)
#                             A_0_list.append(50)
#                             B_0_list.append(0.2)
#
#                             try:
#                                 squid_population_list, seal_population_list = data_generator.get_population_lists(
#                                     a_n=50,
#                                     b_n=0.2,
#                                     R=R,
#                                     A_E=A_E,
#                                     C=C,
#                                     r=r
#                                 )
#
#                             except OverflowError:
#                                 density_means.append(-1)
#
#                             else:
#                                 squid_median = np.median(squid_population_list)
#                                 seal_median = np.median(seal_population_list)
#
#                                 squid_seal_mean = (squid_median + seal_median) / 2
#                                 density_means.append(squid_seal_mean)


R_list, r_list, C_list, A_E_list, gradient_means = stable_environment()

dictionary_stable = {
    "R": R_list,
    "r": r_list,
    "C": C_list,
    "A_E": A_E_list,
    "Gradient mean": gradient_means
}

df = pd.DataFrame(data=dictionary_stable)

df.sort_values(by="Gradient mean")
print(df.head())
