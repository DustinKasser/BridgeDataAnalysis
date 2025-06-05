import csv
from abc import ABC, abstractmethod
from math import trunc
from tkinter.font import names

import numpy as np


def determine_is_long(hand: list[str], partner_hand: list[str], trump_index) -> list[bool]:
    """
    Determines whether hand has a longer trump suit that partner_hand
    :param hand: [spades, hearts, diamond, clubs]
    :param partner_hand: partner's hand [spades, hearts, diamond, clubs]
    :param trump_index: False if the trump suit is shorter or the contract is NT
    :return: [am_i_long, is_partner_long]
    """

    #handsles the NT case
    if trump_index == -1:
        return [False, False]

    #determines the longer suits, choosing False for both if they are the same length
    am_i_long = len(hand[trump_index]) > len(partner_hand[trump_index])
    is_partner_long = len(hand[trump_index]) < len(partner_hand[trump_index])
    return [am_i_long, is_partner_long]

class ValueFeature(ABC):
    """
    An abstract class that identifies a feature (and the number of that feature) in a given hand
    """

    @abstractmethod
    def get_feature_name(self) -> str:
        """
        returns the name of the feature
        :return:
        """
        pass

    @abstractmethod
    def calculate_feature_count(self, hand: list[str], trump_index: int, is_long: int) -> int:
        """
        returns the count of the feature in a given hand
        :param hand: [spades, hearts, diamonds, clubs]
        :param trump_index: The index of the trump suit in the hand list, will be -1 if in NT
        :param is_long: Whether this hand holds a longer suit than partner (if lengths are equal, this should be False)
        :return: the count of the feature in this hand
        """
        pass


class CardFeature(ValueFeature):
    """
    Counts the number of a particular card in the hand
    """

    card: str
    name: str

    def __init__(self, name, card):
        """
        :param name: the name of the feature
        :param card: the card to be counted
        """
        self.card = card
        self.name = name

    def get_feature_name(self) -> str:
        return self.name

    def calculate_feature_count(self, hand: list[str], trump_index: int, is_long: int) -> int:
        """
        Returns how many of the card defined at initialization is in the hand
        :param hand: [spades, hearts, diamonds, clubs]
        :param trump_index: UNUSED
        :param is_long: UNUSED
        :return:
        """
        count = 0
        for suit in hand:
            if self.card in suit:
                count += 1
        return count


class LengthFeature(ValueFeature):
    """
    Counts the number of suits of a specific length.
    This will only count them if the contract is NT or if is_long=True
    """

    length: int
    name: str

    def __init__(self, name, length):
        """
        :param name: the name of the feature
        :param length: the length of suit to be counted
        """
        self.name = name
        self.length = length

    def get_feature_name(self) -> str:
        return self.name

    def calculate_feature_count(self, hand: list[str], trump_index: int, is_long: int) -> int:
        """
        :param hand: [spades, hearts, diamonds, clubs]
        :param trump_index: The index of the trump suit in the hand list, will be -1 if in NT
        :param is_long: Whether this hand holds a longer suit than partner (if lengths are equal, this should be False)
        :return: count of suits of the specified length
        """
        #checks whether to use length distribution points
        if trump_index == -1 or is_long:
            #counts the number of long suits
            count = 0
            for suit in hand:
                if len(suit) == self.length:
                    count += 1
            return count
        #otherwise returns false
        else:
            return 0


class ShortFeature(ValueFeature):
    """
    Counts the number of suits of a specific length
    This will only count them if in a suit contract and is_long=False
    """

    length: int
    name: str

    def __init__(self, name, length):
        """
        :param name: the name of the feature
        :param length: the length of suit to be counted
        """
        self.name = name
        self.length = length

    def get_feature_name(self) -> str:
        return self.name

    def calculate_feature_count(self, hand: list[str], trump_index: int, is_long: int) -> int:
        """
        :param hand: [spades, hearts, diamonds, clubs]
        :param trump_index: The index of the trump suit in the hand list, will be -1 if in NT
        :param is_long: Whether this hand holds a longer suit than partner (if lengths are equal, this should be False)
        :return: count of suits of the specified length
        """

        #checks whether to use shortness distribution points
        if trump_index == -1 or is_long:
            return 0
        else:
            count = 0

            #counts shortness in each non-trump suit
            for i in range(4):
                if i != trump_index and len(hand[i]) == self.length:
                    count += 1

            return count

class FeatureCounter:
    """
    Counts features and returns the counts as a list of strings headed by HandID and BundleID
    """
    non_feature_columns = ["HandID", "BundleID", "MeanTricksTaken", "Weight"]
    number_non_features = len(non_feature_columns)
    features: list[ValueFeature]
    number_features: int

    def __init__(self, features: list[ValueFeature]):
        """
        :param features: A list of classes extending ValueFeature
        """
        self.features = features
        self.number_features = len(self.features)

    def get_header(self) -> list[str]:
        """
        :return: Header for CSV with names of features as column names starting with HandID and BundleID
        """
        arr = [""] * (self.number_non_features + self.number_features)
        for i in range(0, self.number_non_features):
            arr[i] = self.non_feature_columns[i]
        for i in range(self.number_non_features, self.number_non_features + self.number_features):
            arr[i] = self.features[i-self.number_non_features].get_feature_name()
        return arr

    def get_row(self, hand: list[str], partner_hand: list[str], trump_index: int, hand_id: str, bundle_id: str, mean_tricks_taken: str, weight: float) -> list[str]:

        #determines is_long for each hand
        long_arr = determine_is_long(hand, partner_hand, trump_index)


        feature_count_arr = [""] * (self.number_non_features + self.number_features)
        feature_count_arr[0] = hand_id
        feature_count_arr[1] = bundle_id
        feature_count_arr[2] = mean_tricks_taken
        feature_count_arr[3] = str(weight)
        # iterates over each feature
        for i in range(self.number_features):
            #adds the count of features between the partnership
            feature_count = self.features[i].calculate_feature_count(hand, trump_index, long_arr[0]) + self.features[i].calculate_feature_count(partner_hand, trump_index, long_arr[1])
            #input into array with offset for HandID and BundleID
            feature_count_arr[i + self.number_non_features] = str(feature_count)
        return feature_count_arr

def get_standard_feature_counter() -> FeatureCounter:
    """
    :return: A FeatureCounter for the features used in a standard hand valuation
    """

    feature_arr = []

    #builds the standard HCP features plus tens
    feature_arr.append(CardFeature("Aces", "A"))
    feature_arr.append(CardFeature("Kings", "K"))
    feature_arr.append(CardFeature("Queens", "Q"))
    feature_arr.append(CardFeature("Jacks", "J"))
    feature_arr.append(CardFeature("Tens", "T"))

    #builds the standard length counts
    for length in range(5, 13):
        feature_arr.append(LengthFeature(f"Length{length}", length))

    #builds the standard shortness counts
    feature_arr.append(ShortFeature("Doubletons", 2))
    feature_arr.append(ShortFeature("Singletons", 1))
    feature_arr.append(ShortFeature("Voids", 0))

    return FeatureCounter(feature_arr)

def get_hands_north_south(hand_row: list[str]) -> (list[str], list[str]):
    """
    Separates the north and south hands and returns them
    :param hand_row: a row read from hands.csv
    :return: hand_north, hand_south
    """
    hand_north = hand_row[1:5]
    hand_south = hand_row[9:13]
    return hand_north, hand_south

def get_hands_east_west(hand_row: list[str]) -> (list[str], list[str]):
    """
    Separates the east and west hands and returns them
    :param hand_row: a row read from hands.csv
    :return: hand_east, hand_west
    """
    hand_east = hand_row[5:9]
    hand_west = hand_row[13:17]
    return hand_east, hand_west

def get_partnership_hands(hand_row: list[str], direction: str) -> (list[str], list[str]):
    """
    Separates the hands for the declaring partnership and returns them
    :param hand_row: a row from hands.csv
    :param direction: declarer's direction
    :return: hand_north/hand_east, hand_south/hand_west, depending on whether declarer is NS or EW
    """
    if direction == "N" or direction == "S":
        return get_hands_north_south(hand_row)
    elif direction == "E" or direction == "W":
        return get_hands_east_west(hand_row)
    else:
        raise ValueError("Declarer direction not recognized")

def get_trump_index(trump_suit: str) -> int:
    """
    Converts from a string trump_suit to an integer trump_index
    :param trump_suit:
    :return: trump_index
    """
    if trump_suit == "nt" or trump_suit == "pass":
        return -1
    elif trump_suit == "s":
        return 0
    elif trump_suit == "h":
        return 1
    elif trump_suit == "d":
        return 2
    elif trump_suit == "c":
        return 3
    else:
        raise ValueError("Trump suit not recognized")

def get_weight(bundle_row: list[str]) -> float:
    """
    Returns the weight to be used in a regression for this bundle.
    :param bundle_row: the row read from the bundle csv file (typically re_bundle_file.csv)
    :return: sample_size/variation, which is the weight to be applied
    """

    #Yes, I know this could be a single call, but I thought it might be nice for the reader to know a little more detail about how weights are calculated
    return int(bundle_row[6])/float(bundle_row[5])

def create_feature_file(hand_path: str, bundle_path: str, output_path: str, feature_counter: FeatureCounter):
    """
    Creates features.csv for a specific FeatureCounter
    :param hand_path: path to hands.csv
    :param bundle_path: path to the csv file for the bundles (typically re_bundle_file.csv)
    :param output_path: path to write output to
    :param feature_counter: the FeatureCounter to be used
    """

    file_hand = open(hand_path, "r")
    hand_reader = csv.reader(file_hand)
    next(hand_reader)

    file_bundle = open(bundle_path, "r")
    bundle_reader = csv.reader(file_bundle)
    next(bundle_reader)

    file_output = open(output_path, "w", newline="")
    writer = csv.writer(file_output)
    writer.writerow(feature_counter.get_header())

    bundle_row = next(bundle_reader)

    #iterates over each hand row
    for hand_row in hand_reader:
        print(hand_row[0])

        #iterates over each bundle_row with the same hand_id (there may be none, in which case nothing will be executed for that hand
        while bundle_row[0] == hand_row[0]:
            #extracts value
            trump_suit = bundle_row[2]
            direction = bundle_row[3]

            hand_1, hand_2 = get_partnership_hands(hand_row, direction)
            trump_index = get_trump_index(trump_suit)

            #obtains row from feature_counter and writes it
            row = feature_counter.get_row(hand_1, hand_2, trump_index, bundle_row[0], bundle_row[1], bundle_row[4], get_weight(bundle_row))
            writer.writerow(row)

            try:
                bundle_row = next(bundle_reader)
            except StopIteration:
                #ensures that for loop will not be entered again
                bundle_row = ["END_OF_BUNDLES"]

    file_output.close()
    file_bundle.close()
    file_hand.close()

class ValuationSystem:

    feature_value_map: dict[str: float]
    intercept = 0.0

    def __init__(self):
        self.feature_value_map = {}

    def add_feature_value_pair(self, feature: str, value: float):
        self.feature_value_map[feature] = value

    def get_value_array(self, header: list[str]) -> list[float]:
        array = [0.0] * len(header)

        for i in range(len(header)):
            if header[i] in self.feature_value_map:
                array[i] = self.feature_value_map[header[i]]

        return array

def get_standard_valuation_system():

    val = ValuationSystem()

    val.add_feature_value_pair("Aces", 4)
    val.add_feature_value_pair("Kings", 3)
    val.add_feature_value_pair("Queens", 2)
    val.add_feature_value_pair("Jacks", 1)

    # builds the standard length counts
    for length in range(5, 13):
        val.add_feature_value_pair(f"Length{length}", length - 4)

    # builds the standard shortness counts
    val.add_feature_value_pair("Doubletons", 1)
    val.add_feature_value_pair("Singletons", 3)
    val.add_feature_value_pair("Voids", 5)

    return val

def get_standard_valuation_system_converted_to_tricks():

    val = ValuationSystem()
    #1.03639864 + 0.31630926 * x_vals
    val.intercept = 1.03639864

    val.add_feature_value_pair("Aces", 4 * 0.31630926)
    val.add_feature_value_pair("Kings", 3 * 0.31630926)
    val.add_feature_value_pair("Queens", 2 * 0.31630926)
    val.add_feature_value_pair("Jacks", 1 * 0.31630926)

    for length in range(5, 13):
        val.add_feature_value_pair(f"Length{length}", (length - 4) * 0.31630926)

    val.add_feature_value_pair("Doubletons", 1 * 0.31630926)
    val.add_feature_value_pair("Singletons", 3 * 0.31630926)
    val.add_feature_value_pair("Voids", 5 * 0.31630926)

    return val

def get_advanced_valuation_system():

    val = ValuationSystem()

    val.intercept = 0.8386

    val.add_feature_value_pair("Aces", 1.3034)
    val.add_feature_value_pair("Kings", 0.9178)
    val.add_feature_value_pair("Queens", 0.5548)
    val.add_feature_value_pair("Jacks", 0.2694)
    val.add_feature_value_pair("Tens", 0.1166)

    val.add_feature_value_pair("Length5", 0.2545)
    val.add_feature_value_pair("Length6", 0.9230)
    val.add_feature_value_pair("Length7", 0.9230)
    val.add_feature_value_pair("Length8", 2.2830)
    val.add_feature_value_pair("Length9", 2.8085)
    val.add_feature_value_pair("Length10", 3.6011)
    val.add_feature_value_pair("Length11", 5.1163)

    val.add_feature_value_pair("Doubletons", 0.6076)
    val.add_feature_value_pair("Singletons", 1.1008)
    val.add_feature_value_pair("Voids", 1.6114)

    return val

def get_value(hand_features_row: list[str], value_array, intercept):
    features_array = np.array(hand_features_row).astype(float)
    return np.dot(features_array, value_array) + intercept


#TODO: Include an option to cap valuations at 13, it's silly to predict more tricks than that, so we can tolerate a piecewise here.
def create_valuation_regression_file(bundle_path: str, hand_features_path: str, output_path: str, valuation_system: ValuationSystem):
    """
    Creates a csv file with columns [BundleID, Value, MeanTricksTaken, Weight]
    :param bundle_path: path to bundle csv file
    :param hand_features_path: path to hand_features csv file
    :param output_path: path to write to
    :param valuation_system: a ValuationSystem to be used
    """

    #set up readers, writers and value_array
    file_bundle = open(bundle_path, "r") #TODO: Now that all of the features are rolled into the hand_features file, there's no reason to open a file_bundle here
    bundle_reader = csv.reader(file_bundle)
    next(bundle_reader)

    file_hand_features = open(hand_features_path, "r")
    hand_features_reader = csv.reader(file_hand_features)
    header_hand_features = next(hand_features_reader)

    file_output = open(output_path, "w", newline="")
    writer = csv.writer(file_output)
    writer.writerow(["BundleID", "Value", "MeanTricksTaken", "Weight"])

    value_array = valuation_system.get_value_array(header_hand_features)
    value_array = np.array(value_array)

    intercept = valuation_system.intercept

    #we iterate this way instead of a for loop because bundle_row and features_row should have exactly the same number of rows, so it is easier to iterate over them simultaneously
    try:
        while True:

            bundle_row = next(bundle_reader)
            features_row = next(hand_features_reader)

            print(bundle_row[0])

            bundle_id = bundle_row[1]
            value = get_value(features_row, value_array, intercept)
            mean_tricks_taken = bundle_row[4]
            weight = get_weight(bundle_row)

            writer.writerow([bundle_id, value, mean_tricks_taken, weight])

    except StopIteration:
        pass