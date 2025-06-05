import csv


def calculate_mean_squared_error_slow(regression_path: str):
    """
    This function calculates the mean_squared error using a regression file that is fitted to give predictive trick values.
    To make this faster, I should probably be using batches of 10_000, but I just want to sketch this code to see how my errors compare, so I'm hacking it out.
    TODO: Come back and build a better version of this
    :param regression_path:
    :return:
    """
    with open(regression_path, "r") as file:
        reader = csv.reader(file)

        weights = 0.0
        total_error = 0.0

        for row in reader:
            #only calculates the error on BundleIDs ending with 0 or 1, which were reserved for testing purposes (no training was done on these)
            if row[0].endswith(("0", "1")):
                prediction = float(row[1])
                actual = float(row[2])
                weight = float(row[3])

                weighted_error = weight * ((prediction - actual) ** 2)
                weights += weight
                total_error += weighted_error

        adjusted_error = total_error/weights
        print(f"Mean Squared Error: {adjusted_error}")