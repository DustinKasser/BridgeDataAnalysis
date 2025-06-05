import csv
import matplotlib.pyplot as plt
import numpy as np


def get_min_max_values(value_regression_path: str, smallest_default = 100, largest_default = 0) -> (float, float):
    file = open(value_regression_path, "r")
    reader = csv.reader(file)
    next(reader)


    largest = largest_default
    smallest = smallest_default

    for row in reader:
        value = float(row[1])
        if value < smallest:
            smallest = value
        if value > largest:
            largest = value

    return smallest, largest

def create_box_and_whiskers(max_value: float, value_regression_path):
    file = open(value_regression_path, "r")
    reader = csv.reader(file)
    next(reader)

    max_int = int(max_value)
    array = [[] for _ in range(max_int + 1)]

    for row in reader:
        index = round(float(row[1]))
        array[index].append(float(row[2]))

    x_vals = np.arange(len(array))
    y_prediction_degree_1 = 1.03639864 + 0.31630926 * x_vals
    plt.plot(x_vals, y_prediction_degree_1, color="red")
    y_prediction_degree_2 = 1.95171354e+00 + 2.46361390e-01 * x_vals + 1.30580718e-03 * x_vals**2
    plt.plot(x_vals, y_prediction_degree_2, color="blue")

    plt.boxplot(array, positions=x_vals)
    plt.title("Box and Whisker Plots")
    plt.xlabel("Value")
    plt.ylabel("MeanTricksTaken")
    plt.grid(True)
    plt.show()

