import csv
import os

import numpy as np

from clean.selector import AllMajorsRaw, Selector
from nt_offset_model import write_bundle


class Bundler:

    def __init__(self, selector: Selector):
        """
        Initializes all variables and opens all files (all files should be closed before instantiating a Bundler)
        :param selector: the selector to determine which contracts are bundled
        """
        # -- Instantiate all variables

        self.selector = selector
        self.open = True

        # create input files and writers
        self.result_file = open("aggregated_bridge_data/board_results_fixed.csv", "r")
        self.result_reader = csv.reader(self.result_file)
        next(self.result_reader)
        self.hand_file = open("aggregated_bridge_data/hands.csv", "r")
        self.hand_reader = csv.reader(self.hand_file)
        next(self.hand_reader)

        # create a folder to store the outputs in
        os.makedirs(f"{selector.return_name()}", exist_ok=True)

        # create output files and writers
        self.bundle_file = open(f"{selector.return_name()}/bundle_file.csv", "w", newline="")
        self.bundle_writer = csv.writer(self.bundle_file)
        self.raw_results = open(f"{selector.return_name()}/raw_results.csv", "w", newline="")
        self.raw_writer = csv.writer(self.raw_results)

        self.bundle_count = 0

    def close_all(self):
        """
        Closes all files
        """
        self.result_file.close()
        self.hand_file.close()
        self.bundle_file.close()
        self.raw_results.close()
        self.open = False


    def create_initial_bundles(self):
        """
        This will create a directory with the selector's name, as well as two csv files
        bundle_file.csv has columns [HandID, BundleID, TrumpSuit, Direction, MeanTricksTaken, SampleVariance, Samples]
        raw_results.csv has columns [BundleID, TricksTaken]
        :return:
        """

        # write headers
        self.bundle_writer.writerow(
            ["HandID", "BundleID", "TrumpSuit", "Direction", "MeanTricksTaken", "SampleVariance", "Samples"])
        self.raw_writer.writerow(["BundleID", "TricksTaken"])

        # -- Read all rows and perform bundling operations

        #wraps the loop, as we need to iterate over hand_reader and result_reader at different rates
        try:
            next_result = next(self.result_reader)

            while True:
                next_hand = next(self.hand_reader)

                #creates an array to be fed into the selector
                hand_id_rows = []

                #adds all board_results with a corresponding hand_id to the array
                try:
                    while next_result[0] == next_hand[0]:
                        hand_id_rows.append(next_result)
                        next_result = next(self.result_reader)
                except StopIteration:
                    pass

                #bundles and writes the bundles
                bundles = self.selector.select_bundles(hand_id_rows, next_hand)
                self.write_bundles(bundles)

        except StopIteration:
            self.close_all()


    # For reference, this is the structure of board_results_fixed.csv
    # 0: HandID
    # 1: ContractLevel
    # 2: TrumpSuit
    # 3: Doubled
    # 4: Direction
    # 5: TricksTaken
    def write_bundles(self, bundles: list[list]):
        """
        Accepts the bundles and writes them to csv files
        :param bundles: the bundles from selector.select_bundles
        """

        for bundle in bundles:

            #builds a list of all trick values and writes them to raw_results.csv
            tricks = [0] * len(bundle)
            for i in range(len(bundle)):
                tricks[i] = int(float(bundle[i][5]))
                self.raw_writer.writerow([self.bundle_count, tricks[i]])

            #compiles information for bundle_file.csv
            trick_array = np.array(tricks)
            mean = trick_array.mean()
            variance = trick_array.var()
            bundle_row = [bundle[0][0], self.bundle_count, bundle[0][2], bundle[0][4], mean, variance, len(bundle)]
            self.bundle_writer.writerow(bundle_row)

            #increments the bundle_count so that bundle_id can act as a unique key
            self.bundle_count += 1

def test():
    select = AllMajorsRaw()
    bundler = Bundler(select)
    bundler.create_initial_bundles()

