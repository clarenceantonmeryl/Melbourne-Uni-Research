# A universal Class to be used in other files to avoid defining functions again. Borrows code from numerous files

# Import numpy for arrays
import numpy as np

# Initialise the amount of measurements
MEASUREMENTS = 500

# Initialise the x values
X = [x for x in range(0, MEASUREMENTS + 1)]


class DataGenerator:

    # Initialise the class, but there is nothing to initialise
    def __init__(self):
        pass

    # Uses code from gradient.py to calculate the gradient given y2, y1, x2, x1.
    # These are static methods as they do not need an instance of the Class to be called
    @staticmethod
    def calculate_gradient(y2, y1, x2, x1):
        gradient = (y2 - y1) / (x2 - x1)
        return gradient

    # Uses code from main.py to calculate the squid population density when given parameters
    @staticmethod
    def squid_population(A_n, B_n, R, A_E, C):
        A_n_plus_one = R * A_n - ((R - 1) / A_E) * (A_n ** 2) - (C * A_n * B_n)
        return A_n_plus_one

    # Uses code from main.py to calculate the seal population density when given parameters
    @staticmethod
    def seal_population(A_n, B_n, A_E, r):
        B_n_plus_one = (r / A_E) * A_n * B_n
        return B_n_plus_one

    # Applies the formula stated in Part 5 of the report to simulate squid fishing
    @staticmethod
    def squid_population_fished(A_n, B_n, R, A_E, C, k, P, A):
        A_n_plus_one = (R * A_n - ((R - 1) / A_E) * (A_n ** 2) - (C * A_n * B_n)) - ((k * P) / A)
        return A_n_plus_one

    # Applies the formula stated in Part 5 of the report to simulate seal fishing
    @staticmethod
    def seal_population_fished(A_n, B_n, A_E, r, k, P, A):
        B_n_plus_one = ((r / A_E) * A_n * B_n) - ((k * P) / A)
        return B_n_plus_one

    # Uses code from main.py to get the details of a single population density list (not two this time).
    # Also combines code from the present function in main.py
    @staticmethod
    def get_details(population_list):
        print("Mean")
        print(np.mean(population_list))
        print("\n Max")
        print(np.max(population_list))
        print("\n Min")
        print(np.min(population_list))
        print("\n")

        for index in range(len(population_list)):
            print(f"{index}: {population_list[index]}")

    # Uses code from main.py to get the squid and seal population density lists when given the parameters.
    # This is not a static method, thus, needs an instance of the Class to run
    def get_population_lists(self, a_n, b_n, R=3.0, A_E=100.0, C=0.5, r=2.0):
        squid_population_list = []
        seal_population_list = []

        squid_population_list.append(a_n)
        seal_population_list.append(b_n)

        for _ in range(len(X) - 1):
            squid_population_list.append(
                self.squid_population(
                    A_n=squid_population_list[-1],
                    B_n=seal_population_list[-1],
                    R=R,
                    A_E=A_E,
                    C=C
                )
            )

            seal_population_list.append(
                self.seal_population(
                    A_n=squid_population_list[-2],
                    B_n=seal_population_list[-1],
                    A_E=A_E,
                    r=r
                )
            )

        return squid_population_list, seal_population_list

    # Uses similar code to the get_population_lists function, but instead takes the k, P, A parameters and applies the squid_population_fished function
    def get_population_lists_squid_fished(self, a_n, b_n, k, P, A, R=3.0, A_E=100.0, C=0.5, r=2.0):
        squid_population_list = []
        seal_population_list = []

        squid_population_list.append(a_n)
        seal_population_list.append(b_n)

        for _ in range(len(X) - 1):
            squid_population_list.append(
                self.squid_population_fished(
                    A_n=squid_population_list[-1],
                    B_n=seal_population_list[-1],
                    R=R,
                    A_E=A_E,
                    C=C,
                    k=k,
                    P=P,
                    A=A
                )
            )

            seal_population_list.append(
                self.seal_population(
                    A_n=squid_population_list[-2],
                    B_n=seal_population_list[-1],
                    A_E=A_E,
                    r=r
                )
            )

        return squid_population_list, seal_population_list

    # Uses similar code to the get_population_lists function, but instead takes the k, P, A parameters and applies the seal_population_fished function
    def get_population_lists_seal_fished(self, a_n, b_n, k, P, A, R=3.0, A_E=100.0, C=0.5, r=2.0):
        squid_population_list = []
        seal_population_list = []

        squid_population_list.append(a_n)
        seal_population_list.append(b_n)

        for _ in range(len(X) - 1):
            squid_population_list.append(
                self.squid_population(
                    A_n=squid_population_list[-1],
                    B_n=seal_population_list[-1],
                    R=R,
                    A_E=A_E,
                    C=C
                )
            )

            seal_population_list.append(
                self.seal_population_fished(
                    A_n=squid_population_list[-2],
                    B_n=seal_population_list[-1],
                    A_E=A_E,
                    r=r,
                    k=k,
                    P=P,
                    A=A
                )
            )

        return squid_population_list, seal_population_list

    # This time, the function takes a k, P and A for each species. Moreover, it applies both the squid_population_fished and seal_population_fished functions
    def get_population_lists_both_fished(
            self,
            a_n,
            b_n,
            k_squid,
            P_squid,
            A_squid,
            k_seal,
            P_seal,
            A_seal,
            R=3.0,
            A_E=100.0,
            C=0.5,
            r=2.0
    ):
        squid_population_list = []
        seal_population_list = []

        squid_population_list.append(a_n)
        seal_population_list.append(b_n)

        for _ in range(len(X) - 1):
            squid_population_list.append(
                self.squid_population_fished(
                    A_n=squid_population_list[-1],
                    B_n=seal_population_list[-1],
                    R=R,
                    A_E=A_E,
                    C=C,
                    k=k_squid,
                    P=P_squid,
                    A=A_squid
                )
            )

            seal_population_list.append(
                self.seal_population_fished(
                    A_n=squid_population_list[-2],
                    B_n=seal_population_list[-1],
                    A_E=A_E,
                    r=r,
                    k=k_seal,
                    P=P_seal,
                    A=A_seal
                )
            )

        return squid_population_list, seal_population_list

    # Uses code from gradient.py to get a list of the gradients given the squid and seal population density lists
    def get_gradient_lists(self, squid_population_list, seal_population_list):
        squid_gradient_list = [0]
        seal_gradient_list = [0]

        # Calculate squid gradients
        for index in range(1, MEASUREMENTS):
            squid_gradient_list.append(
                self.calculate_gradient(
                    y2=squid_population_list[index],
                    y1=squid_population_list[index - 1],
                    x2=index,
                    x1=index - 1
                )
            )

        # Calculate seal gradients
        for index in range(1, MEASUREMENTS):
            seal_gradient_list.append(
                self.calculate_gradient(
                    y2=seal_population_list[index],
                    y1=seal_population_list[index - 1],
                    x2=index,
                    x1=index - 1
                )
            )

        return squid_gradient_list, seal_gradient_list
