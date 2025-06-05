from matplotlib import pyplot as plt

import variance_calculator
import selector
import hand_valuation
import data_visualizer
from clean.bundler import Bundler
import mean_squared_error
import regression

def build_all(bundle_selector):
    bundler = Bundler(bundle_selector)
    bundler.create_initial_bundles()
    print("Created initial bundles!")

    bundle_path = f"{bundle_selector.return_name()}/bundle_file.csv"
    raw_path = f"{bundle_selector.return_name()}/raw_results.csv"
    gamma_path = f"{bundle_selector.return_name()}/gamma.csv"

    cutoffs = variance_calculator.determine_cutoffs(bundle_path, 0.10)
    print("Calculated cutoffs!")
    priors = variance_calculator.fit_gamma_prior(bundle_path, gamma_path, cutoffs)
    print("Calculated priors!")
    prior_alpha = priors[0]
    prior_scale = priors[1]

    re_bundle_path = f"{bundle_selector.return_name()}/re_bundle_file.csv"
    variance_calculator.re_bundle_with_updated_variances(prior_alpha, prior_scale, bundle_path, raw_path, re_bundle_path)

# hand_valuation.create_feature_file("clean/aggregated_bridge_data/hands.csv",
#                                    "clean/AllMajorsRaw/re_bundle_file_corrected.csv",
#                                    "clean/AllMajorsRaw/hand_features.csv",
#                                    hand_valuation.get_standard_feature_counter())

# hand_valuation.create_valuation_regression_file("clean/AllMajorsRaw/re_bundle_file_corrected.csv",
#                                                 "clean/AllMajorsRaw/hand_features.csv",
#                                                 "clean/AllMajorsRaw/value_regression.csv",
#                                                 hand_valuation.get_standard_valuation_system())

# hand_valuation.create_valuation_regression_file("clean/AllMajorsRaw/re_bundle_file_corrected.csv",
#                                                 "clean/AllMajorsRaw/hand_features.csv",
#                                                 "clean/AllMajorsRaw/advanced_trick_regression.csv",
#                                                 hand_valuation.get_advanced_valuation_system())

#data_visualizer.create_box_and_whiskers(45.0, "clean/AllMajorsRaw/value_regression.csv")

#regression.polynomial_regression_on_hand_valuation("clean/AllMajorsRaw/value_regression.csv", 2, 10_000)

# regression.weight_regression("clean/AllMajorsRaw/hand_features.csv",
#                              10_000,
#                              "Aces,Kings,Queens,Jacks,Tens,Length5,Length6,Length7,Length8,Length9,Length10,Length11,Length12,Doubletons,Singletons,Voids".split(","))

print("Calculating error for standard HCP ...")
mean_squared_error.calculate_mean_squared_error_slow("clean/AllMajorsRaw/standard_trick_regression.csv")
print("Calculating error for advanced hand valuation ...")
mean_squared_error.calculate_mean_squared_error_slow("clean/AllMajorsRaw/advanced_trick_regression.csv")