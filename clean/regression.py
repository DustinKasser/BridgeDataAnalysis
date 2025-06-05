import numpy as np
import pandas as pd


#TODO: Generalize this to run a regression on multiple degrees at once (if speed is slow)
#TODO: For now, I am using the linear regression I think. The quadratic seemed roughly the same, so this isn't an insane comparison.
def polynomial_regression_on_hand_valuation(value_regression_path: str, target_degree: int, chunk_size: int):

    #create empty matrices
    weighted_value_only_matrix = np.zeros((target_degree + 1, target_degree + 1)) # this represents the values in B_j * x^j * B_k * x^k for each j
    weighted_value_trick_vector = np.zeros(target_degree + 1) # this represents B_j * x^j * y for each j

    for i, chunk in enumerate(pd.read_csv(value_regression_path, chunksize=chunk_size)):

        print(i)

        #bundleIDs ending with 0 or 1 are reserved for testing purposes
        is_test = chunk["BundleID"].astype(str).str.endswith(("0", "1"))
        train = chunk[~is_test]

        values =train["Value"]
        tricks = train["MeanTricksTaken"]
        weights = train["Weight"]

        values_poly = np.vander(values, N=target_degree + 1, increasing=True)
        weight_matrix = np.diag(weights)

        weighted_value_only_matrix += values_poly.T @ weight_matrix @ values_poly
        weighted_value_trick_vector += values_poly.T @ (weights * tricks)

    beta = np.linalg.solve(weighted_value_only_matrix, weighted_value_trick_vector)
    print(beta)

def weight_regression(hand_feature_path: str, chunk_size: int, header_targets: list[str]):

    weighted_feature_only_matrix = np.zeros((len(header_targets) + 1, len(header_targets) + 1))
    weighted_feature_trick_vector = np.zeros(len(header_targets) + 1)

    for i, chunk in enumerate(pd.read_csv(hand_feature_path, chunksize=chunk_size)):

        print(i)

        # bundleIDs ending with 0 or 1 are reserved for testing purposes
        is_test = chunk["BundleID"].astype(str).str.endswith(("0", "1"))
        train = chunk[~is_test]

        #get features and add an intercept
        features = train[header_targets].astype(int).to_numpy()
        ones = np.ones((features.shape[0], 1))
        features_with_intercept = np.hstack([ones, features])

        #set up the other two vectors
        tricks = train["MeanTricksTaken"].astype(float).to_numpy()
        weights = train["Weight"].astype(float).to_numpy()
        weight_matrix = np.diag(weights)

        weighted_feature_only_matrix += features_with_intercept.T @ weight_matrix @ features_with_intercept
        weighted_feature_trick_vector += features_with_intercept.T @ (weights * tricks)

    diagonal_feature_only = np.diag(weighted_feature_only_matrix)
    used_mask = diagonal_feature_only > 0  # True for intercept and features that occur

    # 2) reduce the matrices to only those rows/cols
    reduced_matrix = weighted_feature_only_matrix[used_mask][:, used_mask]
    reduced_vector = weighted_feature_trick_vector[used_mask]

    # 3) solve the smaller system
    beta_reduced = np.linalg.solve(reduced_matrix, reduced_vector)

    # 4) reduce your labels
    label_string = "Intercept,Aces,Kings,Queens,Jacks,Tens,Length5,Length6,Length7,Length8,Length9,Length10,Length11,Length12,Doubletons,Singletons,Voids"
    labels = label_string.split(",")
    labels_reduced = [label for label, keep in zip(labels, used_mask) if keep]

    # 5) print them
    print(f"{'Label':<12} {'Value':>10}")
    print("-" * 24)
    for lab, val in zip(labels_reduced, beta_reduced):
        print(f"{lab:<12} {val:10.4f}")



#Aces,Kings,Queens,Jacks,Tens,Length5,Length6,Length7,Length8,Length9,Length10,Length11,Length12,Doubletons,Singletons,Voids