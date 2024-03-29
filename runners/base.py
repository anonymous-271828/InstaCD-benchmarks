import argparse
import pandas as pd

from abc import ABC, abstractmethod

from instacd_benchmarks.utils.ecd_processor import DataFormat, ECDProcessor

class BaseRunner(ABC):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='InstaCD Benchmarks')
        self.add_common_arguments()
        self.add_specific_arguments()
        self.args = self.parse_arguments()
        self.setup_format()
        self.setup_processor()

    def add_common_arguments(self):
        self.parser.add_argument('--sheet_name', type=str, required=True, help='Name of the sheet in the dataset information file')
        self.parser.add_argument('--dataset_repo', type=str, required=False, default='submodules/example-causal-datasets', help='Path to the dataset repository')
        self.parser.add_argument('--dataset_information_sheet', type=str, default='submodules/example-causal-datasets/ECD_data_info.xlsx', required=False, help='Path to the dataset information file. Tells the runner which experiments to run.')
        self.parser.add_argument('--timeout', type=int, required=False, default=10800, help='Timeout for the algorithm')
        self.parser.add_argument('--result_file', type=str, required=False, default='results.csv', help='Complete filepath to save the results') 
        self.parser.add_argument('--network_result_file', type=str, required=False, default='network_results.dat', help='Complete filepath to save the network results')
        self.parser.add_argument('--format', type=str, required=False, default='continuous', help='The format of the data')

    def setup_processor(self):
        self.ecd_processor = ECDProcessor(self.args.dataset_repo)
        self.ecd_info = pd.read_excel(self.args.dataset_information_sheet, sheet_name=None)

    def setup_format(self):
        self.format = DataFormat()
        if self.args.format == 'continuous':
            self.format.continuous = True
        elif self.args.format == 'discrete':
            self.format.discrete = True
        elif self.args.format == 'mixed':
            self.format.mixed = True
        elif self.args.format == 'mixed_numeric':
            self.format.mixed = True
            self.format.numeric = True
        elif self.args.format == 'entropy':
            self.format.entropy = True
            self.format.discrete = True
        else:
            raise ValueError('Invalid format')        

    @abstractmethod
    def add_specific_arguments(self):
        # This method will be implemented by each specific runner
        pass

    def parse_arguments(self):
        # This method parses the arguments and returns them
        return self.parser.parse_args()

    @abstractmethod
    def run(self):
        # This method will be implemented by each specific runner
        pass


