import csv
import math

import numpy
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import digamma
from scipy.stats import gamma


def determine_cutoffs(bundle_path: str, percentage: float) -> int:
    """
    Determines what cutoff for samples should be used to ensure a certain percentage of bundles are included in gamma fit calculations
    :param bundle_path: path to bundle_file.csv
    :param percentage: percentage of bundles to be included in calculations, should be a decimal (0.10 instead of 10%)
    :return: cutoff value
    """

    # initialize the counting map and count of total bundles
    sample_map = {}
    total = 0

    # initialize file reader and skip the header
    bundle_file = open(bundle_path, "r")
    reader = csv.reader(bundle_file)
    next(reader)

    # counts how many bundles have each number of samples
    for row in reader:
        total += 1

        samples = int(row[6])
        if samples in sample_map:
            sample_map[samples] += 1
        else:
            sample_map[samples] = 1

    #obtains sorted keys in descending order
    sorted_keys = sorted(sample_map.keys(), reverse=True)

    cutoff = sorted_keys[0]
    cutoff_index = 0
    included_count = 0
    while included_count < percentage * total:
        included_count += sample_map[cutoff]
        cutoff_index += 1
        if cutoff_index < len(sorted_keys):
            cutoff = sorted_keys[cutoff_index]
        else:
            cutoff = 0

    return cutoff



def fit_gamma_prior(bundle_path: str, gamma_path: str, cutoff: int):
    """
    Fits a gamma distribution to the variances from the bundle_path.csv file
    :return: [alpha, scale] parameters for the gamma function
    """

    #initialize starting values
    s1 = 0.0
    s_log = 0.0
    w = 0.0

    #initialize file reader and skip header
    file = open(bundle_path, "r")
    reader = csv.reader(file)
    next(reader)

    # Row format for bundle_file.csv
    # 0: HandID
    # 1: BundleID
    # 2: TrumpSuit
    # 3: Direction
    # 4: MeanTricksTaken
    # 5: SampleVariance
    # 6: Samples

    #reads each row in bundle_file.csv and if it has more samples than the cutoff, uses it in a weighted calculation of the gamma parameters.
    for row in reader:
        if int(row[6]) > cutoff:
            weight = int(row[6]) - 1
            variance = float(row[5])

            s1 += weight * variance
            if variance == 0:
                s_log += weight * -6
            else:
                s_log += weight * math.log(variance)
            w += weight

    #closes the bundle_file.csv to be reused
    file.close()

    #defines temporary parameters
    mu = s1/w
    mu_log = s_log/w
    print(f"Mu: {mu}, mu_log: {mu_log}")

    #defines target function
    def target_function(alpha):
        return digamma(alpha) - numpy.log(alpha) + numpy.log(mu) - mu_log

    #solves for alpha and scale
    sol = root_scalar(target_function, bracket=[1e-5, 1e5], method="bisect")
    alpha = sol.root
    scale = mu/alpha

    #Stores alpha and scale for re-use
    file_out = open(gamma_path, "w", newline="")
    writer = csv.writer(file_out)
    writer.writerow(["ParameterName", "ParameterValue"])
    writer.writerow(["Alpha", alpha])
    writer.writerow(["Scale", scale])
    file_out.close()

    #returns scale parameters
    return [alpha, scale]





class PosteriorCalculator:
    """
    This class is used to calculate the posterior variance of a bundle of hand results
    """

    #the fitted prior's parameters
    prior_alpha: float
    prior_scale: float
    #the arrays listing which y-values will give non-trivial weights in a given distribution_n_sample file
    #indexed by sample size 2-15
    valid_distribution_arrays: dict
    #the distribution files with the trivial columns removed for memory purposes
    # {samples: [{y_value: distribution}]}
    sample_arrays_stripped: dict
    #the key to pull x-values from sample_arrays_stripped
    X_KEY = "X_KEY"
    #pre-computed posteriors for when the sample size is in the range 2-15
    #{samples: {y_value: posterior}}
    pre_computed_expected_values: dict


    def load_valid_distribution_arrays(self):
        """
        loads the distribution_valid files into a dictionary {samples: valid_array}
        """
        #builds the initial dict
        self.valid_distribution_arrays = {}

        #iterates over all possible sample sizes
        for samples in range(2, 16):

            #instantiates the array and reader
            sample_array = []
            file_valid = open(f"empirical_bayesian/distributions_{samples}_valid.csv", "r")
            reader = csv.reader(file_valid)
            next(reader)

            #adds each row's value to the sample array
            for row in reader:
                sample_array.append(float(row[0]))

            #adds the array to the dict and closes the file
            self.valid_distribution_arrays[samples] = sample_array
            file_valid.close()

    def calculate_index(self, y_value: float) -> int:
        """
        Calculates the index corresponding to a sample variance to be used in an empirical_bayesian/distribution file
        This is a utility function for load_sample_arrays_stripped
        :param y_value: a sample variance of the form xx.xx5
        :return: the index as an integer
        """
        y_value -= 0.005  # removes the intermediate adjustment
        y_value = y_value * 100.0  # scales up to an integer
        return round(y_value) + 1  # accounts for the first column of the csv acting as a key for the row

    def load_sample_arrays_stripped(self):
        """
        Loads distribution_n_sample files into a 2D array. Columns that are all 0 are stripped out to save memory
        This function assumes that the valid_distribution_arrays have been loaded
        """

        #creates the initial array
        self.sample_arrays_stripped = {}

        #iterates over each possible sample size
        for samples in range(2,16):

            #pairs the valid list with their corresponding indexes
            valid_list = self.valid_distribution_arrays[samples]
            valid_indexes = []
            for valid_y in valid_list:
                valid_indexes.append(self.calculate_index(valid_y))

            #instantiates the reader
            distribution_file = open(f"empirical_bayesian/distributions_{samples}_samples.csv")
            reader = csv.reader(distribution_file)
            next(reader)

            #builds the array for this sample and adds it to the dict
            sample_array = []
            self.sample_arrays_stripped[samples] = sample_array

            #iterates over all rows in the file
            for row in reader:

                #builds the dict for that row
                row_dict = {self.X_KEY: float(row[0])}
                for i in range(len(valid_list)):
                    row_dict[valid_list[i]] = int(row[valid_indexes[i]])
                sample_array.append(row_dict)


    def pre_compute_bayes_expected_value(self, samples: int, y: float) -> float:
        """
        Uses the sample_arrays_stripped to estimate the posterior expected value based on the prior gamma distribution
        :param samples: number of samples in observation, should be in the range [2, 15]
        :param y: the valid_distribution observation being updated on, acts as a key
        :return: posterior expected variance
        """

        distribution = self.sample_arrays_stripped[samples]

        # instantiates probability variables for max and min estimates
        total_p_of_y = 0
        sum_expected_x = 0

        # compiles expected values
        for row in distribution:
            #Computes x and pdf(x)
            x = row[self.X_KEY]
            gamma_value_x = gamma.pdf(x, a=self.prior_alpha, scale=self.prior_scale)

            #updates running sums
            weight = row[y]
            total_p_of_y += weight * gamma_value_x
            sum_expected_x += weight * gamma_value_x * x

        # normalizes expected value and returns it
        expected = sum_expected_x/total_p_of_y
        return expected

    def pre_compute_expected_values(self):
        """
        pre-computes posterior variances for valid y_values
        expects sample_arrays_stripped to have been built
        """

        #bulids the dictionary
        self.pre_computed_expected_values = {}

        #iterates over sample sizes
        for samples in range(2, 16):

            #builds the sample dictionary and adds it to the master dictionary
            sample_dict = {}
            self.pre_computed_expected_values[samples] = sample_dict

            #computes the expected values
            for y in self.valid_distribution_arrays[samples]:
                sample_dict[y] = self.pre_compute_bayes_expected_value(samples, y)

            print(f"Pre-Computed for samples={samples}")


    def __init__(self, prior_alpha: float, prior_scale: float):
        self.prior_alpha = prior_alpha
        self.prior_scale = prior_scale
        self.load_valid_distribution_arrays()
        print("loaded validity arrays")
        self.load_sample_arrays_stripped()
        print("loaded stripped distributions")
        self.pre_compute_expected_values()
        print("pre-computed expected values")
        print("Posterior Calculator initialized!")



    def calculate_weight_below(self, min_y_value: float, max_y_value: float, y: float) -> float:
        """
        Calculates the weight to be placed upon the min_y_value
        :param min_y_value: a y_value from a distribution_x_valid file that is less than y
        :param max_y_value: a y_value from a distribution_x_valid file that is greater than y
        :param y: a sample variance
        :return: a float representing the weight to be placed upon min_y_value in a linear interpolation
        """
        difference = max_y_value - min_y_value
        distance_to_max = max_y_value - y
        index_below_weight = distance_to_max / difference
        return index_below_weight

    def obtain_prior_interpolation(self, samples: int, sample_variance: float) -> list:
        """
        Finds the values with non-trivial weights from monte-carlo simulations so that a posterior can be accurately calculated
        :param samples: number of samples
        :param sample_variance: observed variance
        :return: [min_y_value: float, min_y_weight: float, max_y_value: float, max_y_weight: float]
        """

        # initializes variables
        valid_array = self.valid_distribution_arrays[samples]
        min_y_value = 0.005
        max_y_value = 0.005

        # calculates min and max y_values
        for y in valid_array:
            if y <= sample_variance:
                min_y_value = y
                max_y_value = y
            else:
                max_y_value = y
                break

        # handles the case where the sample variance is out of bounds
        if min_y_value == max_y_value:
            return [min_y_value, 1.0, min_y_value, 0.0]
        # handles the case where the sample variance is within bounds
        else:
            min_y_weight = self.calculate_weight_below(min_y_value, max_y_value, sample_variance)
            max_y_weight = 1.0 - min_y_weight
            return [min_y_value, min_y_weight, max_y_value, max_y_weight]


    def calculate_bayes_expected_value(self, samples: int, sample_variance: float) -> float:
        """
        Uses the pre_computed expected values to interpolate the posterior expected variance
        :param samples: number of samples in observation
        :param sample_variance: variance in observation
        :return: posterior expected variance
        """

        # handles the trivial case in which nothing about variance is known
        if samples == 1:
            return self.prior_alpha * self.prior_scale

        # obtains prior interpolation
        prior_interpolation = self.obtain_prior_interpolation(samples, sample_variance)

        #calculate posterior interpolation and return it
        posterior_interpolation = (self.pre_computed_expected_values[samples][prior_interpolation[0]] * prior_interpolation[1] +
                                   self.pre_computed_expected_values[samples][prior_interpolation[2]] * prior_interpolation[3])
        return posterior_interpolation


    # Technically you could use this instead of calculate_bayes_expected_value, but that would be a bit inefficient.
    def refit_gamma_as_posterior(self, prior_alpha: float, prior_scale: float, samples: int, sample_variance: float):
        """
            Uses the sample_arrays_stripped to estimate the posterior gamma distribution based on a prior gamma distribution
            :param prior_alpha: alpha parameter for the prior gamma distribution
            :param prior_scale: scale parameter for the prior gamma distribution
            :param samples: number of samples in observation
            :param sample_variance: variance in observation
            :return: [posterior_alpha, posterior_scale]
            """

        # handles the trivial case in which nothing about variance is known
        if samples == 1:
            return [prior_alpha, prior_scale]

        # obtains prior interpolation
        prior_interpolation = self.obtain_prior_interpolation(samples, sample_variance)
        keys = [prior_interpolation[0], prior_interpolation[2]]
        weights = [prior_interpolation[1], prior_interpolation[3]]

        distribution = self.sample_arrays_stripped[samples]

        # instantiates the variables
        s1 = 0.0
        s_log = 0.0
        w = 0.0

        # reads each row in the distribution array and uses it in a weighted calculation of the gamma parameters.
        for row in distribution:
            x = row[self.X_KEY]
            prior_gamma_value_x = gamma.pdf(x, a=prior_alpha, scale=prior_scale)

            weight = row[keys[0]] * weights[0] + row[keys[1]] * weights[1]

            s1 += weight * x * prior_gamma_value_x
            s_log += weight * math.log(x) * prior_gamma_value_x
            w += weight * prior_gamma_value_x

        # defines temporary parameters
        mu = s1 / w
        mu_log = s_log / w
        print(f"Mu: {mu}, mu_log: {mu_log}")

        # defines target function
        def target_function(alpha):
            return digamma(alpha) - numpy.log(alpha) + numpy.log(mu) - mu_log

        #checks endpoints before solving
        left_endpoint = 1e-5
        right_endpoint = 1e5
        while target_function(left_endpoint) > 0:
            left_endpoint = left_endpoint/10
        while target_function(right_endpoint) < 0:
            right_endpoint = right_endpoint * 10

        # solves for alpha and scale
        sol = root_scalar(target_function, bracket=[left_endpoint, right_endpoint], method="bisect")
        posterior_alpha = sol.root
        posterior_scale = mu / posterior_alpha

        return [posterior_alpha, posterior_scale]

    def small_re_bundle(self, sample_size: int, observed_variance: float,
                        bundle_id: int, raw_reader: csv.reader, raw_row: list) -> (float, list):
        """
        Calculates the posterior variance for bundles of size <= 16
        :param sample_size: size of the bundle
        :param observed_variance: variance of the bundle
        :param bundle_id: bundle_id of the bundle
        :param raw_reader: the input stream for raw_results.csv
        :param raw_row: the most recent row read from raw_results.csv
        :return: posterior_variance, next_raw_row (might be None)
        """

        # calculates the variance
        variance = self.calculate_bayes_expected_value(sample_size, observed_variance)

        # clears all raw results in the bundle for the reader so that it is ready for the next bundle
        try:
            while int(raw_row[0]) == bundle_id:
                raw_row = next(raw_reader)
        except StopIteration:
            raw_row = None

        # returns values
        return variance, raw_row

    def accept_parameter(self, number: float):
        #if scale or alpha are 0, this would give us 0 expectation, which we think of as being impossible
        if number == 0:
            return False
        #If something is NaN or inf, there will be issues, and we should reject it
        elif math.isfinite(number):
            return False
        #If something is negative, that's really bad
        elif number < 0:
            return False
        else:
            return True

    def large_re_bundle(self, bundle_id: int, raw_reader: csv.reader,
                        raw_row: list) -> (float, list):
        """
        Calculates the posterior variance for bundles of size > 16 via repeated Bayesian updates
        :param bundle_id: bundle_id of the bundle
        :param raw_reader: the input stream for raw_results.csv
        :param raw_row: the most recent row read from raw_results.csv
        :return: posterior_variance, next_raw_row (might be None)
        """

        alpha_prior = self.prior_alpha
        scale_prior = self.prior_scale
        expectation = alpha_prior * scale_prior

        # iterates while there are still raw values left in the bundle
        while raw_row is not None and int(raw_row[0]) == bundle_id:

            # begins a batch of 15
            sample_count = 0
            value_array = []

            # creates the batch, ending the loop if the raw_reader becomes empty
            try:
                while sample_count < 15 and int(raw_row[0]) == bundle_id:
                    value_array.append(float(raw_row[1]))
                    sample_count += 1
                    raw_row = next(raw_reader)
            except StopIteration:
                raw_row = None

            # updates gamma parameters
            value_array = np.array(value_array)
            sample_variance = value_array.var()
            posterior_parameters = self.refit_gamma_as_posterior(alpha_prior, scale_prior, sample_count, sample_variance)
            alpha_prior = posterior_parameters[0]
            scale_prior = posterior_parameters[1]

            #Sometimes there can be a NaN error; this usually happens when the scale factor becomes so small that float arithmetic issues kick in.
            #To bypass this, the expectation is stored before the NaN error occurs, and is returned instead in this case.
            #While it is a bit less efficient, the code here will continue to run as though there were no NaN error. Hypothetically one could abort in
            #such a case, but the queue from raw_files still would need to be cleared, so for now I am leaving this code as is.
            #TODO: add an escape function to optimize this code.
            if self.accept_parameter(alpha_prior) and self.accept_parameter(scale_prior):
                temp_expectation = alpha_prior * scale_prior
                if self.accept_parameter(temp_expectation):
                    expectation = temp_expectation

        # having batched through all samples, the expected value of the gamma is the expected posterior for the bundle
        return expectation, raw_row

    def calculate_bundle_posterior_variance(self, bundle_row: list,
                                            raw_reader: csv.reader, raw_row: list) -> (float, list):
        """
        Sorts bundles and extracts values for small_re_bundle and large_re_bundle
        :param bundle_row: the row from bundle_file.csv
        :param raw_reader: input stream from raw_results.csv
        :param raw_row: most recent row from raw_results.csv
        :return: posterior_variance, next_raw_row (might be None)
        """
        if int(bundle_row[6]) <= 15:
            return self.small_re_bundle(int(bundle_row[6]), float(bundle_row[5]),
                                   int(bundle_row[1]), raw_reader, raw_row)
        else:
            return self.large_re_bundle(int(bundle_row[1]), raw_reader, raw_row)

    # ------------------------------------------------------------------------------------
    # An error occurred, causing some bundles to get NaN variance. The following is to try and address those issues.

    def skip_defined(self, writer: csv.writer, raw_reader: csv.reader, bundle_row) -> list:
        """
        This skips a bundle where the variance is defined (Not NaN)
        :param writer: writer to the output file
        :param raw_reader: reader for raw_results.csv
        :param bundle_row: row from re_bundle_file.csv
        :return: the first row from raw_reader that is of a different bundle_id (Might be None)
        """

        #stores the bundle_id and writes the bundle_row
        bundle_id = bundle_row[1]
        writer.writerow(bundle_row)

        #iterates over the rows with the same bundle_id to remove them from the queue
        next_raw = None
        for row in raw_reader:
            next_raw = row
            if next_raw[0] != bundle_id:
                break
        return next_raw

    def correct_nan_row(self, bundle_row: list, raw_reader: csv.reader, raw_row: list, writer: csv.writer) -> list:
        """
        This corrects a row with NaN values
        :param bundle_row: the row from re_bundle_file being read
        :param raw_reader: the reader of raw_results.csv
        :param raw_row: the most recent row read from raw_reader
        :param writer: the writer for the output file
        :return: the first row from raw_reader that is of a different bundle_id (Might be None)
        """
        variance, raw_row = self.calculate_bundle_posterior_variance(bundle_row, raw_reader, raw_row)
        bundle_row[5] = str(variance)
        writer.writerow(bundle_row)
        return raw_row


    def handle_row(self, bundle_row: list, raw_reader: csv.reader, raw_row: list, writer: csv.writer) -> list:
        """
        This handles a bundle_row and writes it to the output file for the process of correcting nan variances
        :param bundle_row: the row from re_bundle_file being read
        :param raw_reader: the reader for raw_results.csv
        :param raw_row: the most recent row read from raw_reader
        :param writer: the writer for the output file
        :return: the first row from raw_reader that is of a different bundle_id (Might be None)
        """
        # corrects NaN errors
        print(bundle_row[0])
        if bundle_row[5] == "nan":
            return self.correct_nan_row(bundle_row, raw_reader, raw_row, writer)
        # skips valid rows
        else:
            return self.skip_defined(writer, raw_reader, bundle_row)

    def nan_correct(self, re_bundle_path: str, re_bundle_resolved_path: str, raw_path: str):
        re_bundle_file = open(re_bundle_path, "r")
        reader_re_bundle = csv.reader(re_bundle_file)
        header = next(reader_re_bundle)

        re_bundle_resolved_file = open(re_bundle_resolved_path, "w", newline="")
        writer = csv.writer(re_bundle_resolved_file)
        writer.writerow(header)


        raw_file = open(raw_path, "r")
        raw_reader = csv.reader(raw_file)
        next(raw_reader)

        raw_row = next(raw_reader)

        for bundle_row in reader_re_bundle:
            raw_row = self.handle_row(bundle_row, raw_reader, raw_row, writer)


# A refresher of the structure of the bundle_file.csv
# 0: HandID
# 1: BundleID
# 2: TrumpSuit
# 3: Direction
# 4: MeanTricksTaken
# 5: SampleVariance
# 6: Samples

def re_bundle_with_updated_variances(alpha_prior: float, scale_prior: float, bundle_path: str, raw_path: str, re_bundle_path: str):
    """
    Moves data from a bundle_file.csv to re_bundle_file.csv replacing variances with bayesian updated variances
    :param alpha_prior: alpha parameter for the prior gamma function
    :param scale_prior: scale parameter for the prior gamma function
    :param bundle_path: path to read the bundle
    :param raw_path: path to the raw_results file
    :param re_bundle_path: path to write the re_bundle
    """

    #sets up readers and writers
    file_bundle_in = open(bundle_path, "r")
    bundle_reader = csv.reader(file_bundle_in)
    header = next(bundle_reader)

    file_raw_in = open(raw_path, "r")
    raw_reader = csv.reader(file_raw_in)
    next(raw_reader)

    file_out = open(re_bundle_path, "w", newline="")
    writer = csv.writer(file_out)
    writer.writerow(header)

    #reads the first line of the raw file
    raw_row = next(raw_reader)

    #instantiates the calculator
    calculator = PosteriorCalculator(alpha_prior, scale_prior)

    #for each bundle, calculates the posterior variance and writes the updated bundle to a re_bundle file
    for bundle_row in bundle_reader:
        print(bundle_row[0])
        variance, raw_row = calculator.calculate_bundle_posterior_variance(bundle_row, raw_reader, raw_row)
        bundle_row[5] = str(variance)
        writer.writerow(bundle_row)


# calc = PosteriorCalculator(1.6967717265505544,0.4789094374663921)
# calc.nan_correct("AllMajorsRaw/re_bundle_file.csv",
#                  "AllMajorsRaw/re_bundle_file_corrected.csv",
#                  "AllMajorsRaw/raw_results.csv")


# re_bundle_with_updated_variances(
#     1.355165258610728,
#     0.6896207533708566,
#     "clean/AllMajorsRaw/bundle_file.csv",
#     "clean/AllMajorsRaw/raw_results.csv",
#     "clean/AllMajorsRaw/re_bundle_file.csv")

#I need to investigate the NaN values... something is amiss in denmark!