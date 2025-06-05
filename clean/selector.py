from abc import ABC, abstractmethod

class Selector(ABC):
    """
    An abstract class to group similar contracts and determine which contracts should be included for the analysis
    """
    @abstractmethod
    def select_bundles(self, hand_id_rows: list, hand_record: list) -> list:
        """
        this method should select the rows via some method and then bundle them together into clean groups.
        It may also adjust the "tricks taken" to make things more comparable
        :param hand_id_rows: all contracts with the same hand_id
        :param hand_record: the hand record corresponding to the hand_id
        :return: an array of bundled contracts (a list of lists). Bundles all be non-empty lists of rows in the format of board_results_fixed.csv
        """
        pass

    @abstractmethod
    def return_name(self) -> str:
        """
        This method returns the name of the selector to be used as the folder name for generated files
        :return: name of the folder and selectors
        """
        pass



class AllMajorsRaw(Selector):
    """
    This class selects all No Trump contracts and Major Suit contracts with an 8+ card fit. Tricks taken are NOT adjusted.
    """

    def select_bundles(self, hand_id_rows: list, hand_record: list) -> list:
        """
        This method bundles together contracts by suit, excluding
            1) all minor suit contracts and
            2) major suit contracts without an 8 card fit
        :param hand_id_rows: all contracts with the same hand_id
        :param hand_record: the hand record corresponding to the hand_id
        :return: an array of bundled contracts (a list of lists). Bundles are all non-empty lists of rows in the format of board_results_fixed.csv
        """

        # bundles are to be added to and exported eventually if non-empty
        # allow triggers are for whether there is at least an 8 card fit
        nt_ns_bundle = []
        s_ns_bundle = []
        h_ns_bundle = []

        s_ns_allow = len(hand_record[0]) + len(hand_record[8]) >= 8
        h_ns_allow = len(hand_record[1]) + len(hand_record[9]) >= 8

        nt_ew_bundle = []
        s_ew_bundle = []
        h_ew_bundle = []

        s_ew_allow = len(hand_record[4]) + len(hand_record[12]) >= 8
        h_ew_allow = len(hand_record[5]) + len(hand_record[13]) >= 8


        # iterate through all rows and bundles appropriately, excluding forbidden contracts
        for row in hand_id_rows:
            if row[4] == "N" or row[4] == "S": #checks the direction
                if row[2] == "nt": #checks the contract and bundles appropriately
                    nt_ns_bundle.append(row)
                if row[2] == "s" and s_ns_allow:
                    s_ns_bundle.append(row)
                if row[2] == "h" and h_ns_allow:
                    h_ns_bundle.append(row)
            else: #same for EW
                if row[2] == "nt":
                    nt_ew_bundle.append(row)
                if row[2] == "s" and s_ew_allow:
                    s_ew_bundle.append(row)
                if row[2] == "h" and h_ew_allow:
                    h_ew_bundle.append(row)


        #add all non-empty bundles to the list to return
        to_return = []
        if len(nt_ns_bundle) > 0:
            to_return.append(nt_ns_bundle)
        if len(s_ns_bundle) > 0:
            to_return.append(s_ns_bundle)
        if len(h_ns_bundle) > 0:
            to_return.append(h_ns_bundle)
        if len(nt_ew_bundle) > 0:
            to_return.append(nt_ew_bundle)
        if len(s_ew_bundle) > 0:
            to_return.append(s_ew_bundle)
        if len(h_ew_bundle) > 0:
            to_return.append(h_ew_bundle)


        return to_return


    def return_name(self):
        return "AllMajorsRaw"