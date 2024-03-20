# Processor Class for Example-Causal-Datasets Repository: https://github.com/cmu-phil/example-causal-datasets

from collections import defaultdict
import os
import numpy as np
import pandas as pd


class DataFormat:
    def __init__(
            self,
            continuous: bool = False,
            discrete: bool = False,
            mixed: bool = False,
            maximum: int = None,
            numeric: bool = False,
            json: bool = None,
            cov: bool = False,
    ):
        """
        Constructor for DataFormat
        
        :param continuous: Whether the data is continuous
        :param discrete: Whether the data is discrete
        :param mixed: Whether the data is mixed
        :param maximum: Maximum value of the data
        :param numeric: Whether the data is numerical
        :param json: JSON format of the data
        :param cov: Whether the data is in covariance format
        """
        self.continuous = continuous
        self.discrete = discrete
        self.mixed = mixed
        self.maximum = maximum
        self.numeric = numeric
        self.json = json
        self.cov = cov

    def __str__(self):
        s = ''
        if self.continuous:
            s += 'continuous '
        if self.discrete:
            s += 'discrete '
        if self.mixed:
            s += 'mixed '
        if self.numeric:
            s += 'numeric '
        if self.json:
            s += 'json '
        if self.maximum:
            s += f'maximum {self.maximum} '   
        if self.cov:
            s += 'cov ' 
        return s[:-1]
    
    def __hash__(self) -> int:
        return hash(str(self))

class Data:
    def __init__(self, path, format: DataFormat):
        """
        Constructor for Data
        
        :param path: Path to the data
        :param format: Format of the data
        """
        self.path = path
        self.format = format

    def load_data(self):
        """
        Load data from the path
        """
        self.df = pd.read_csv(self.path, delimiter='\t', header=0)


class Dataset:
    def __init__(self, name:str, path_to_folder:str):
        """
        Constructor for Dataset
        
        :param name: Name of the dataset
        :param path_to_folder: Path to the folder containing the dataset
        """
        self.name = name
        self.path_to_folder = path_to_folder
        self.extract_data_formats()

    def __str__(self):
        return f'{self.name} at {self.path_to_folder}'
    
    @staticmethod
    def extract_format_frome_name(name:str):
        """
        Extract format from the name
        
        :param name: Name of the format
        :return: DataFormat object
        """
        format = {}
        split = name.split('.')
        if 'continuous' in split:
            format['continuous'] = True
        if 'discrete' in split:
            format['discrete'] = True
        if 'mixed' in split:
            format['mixed'] = True
        if 'numeric' in split:
            format['numeric'] = True
        if 'maximum' in split:
            format['maximum'] = int(split[split.index('maximum') + 1])
        if 'json' in split:
            format['json'] = True
        if 'cov' in split:
            format['cov'] = True
        return DataFormat(**format)
    
    def extract_data_formats(self):
        self.formats = {}
        self.data = []
        data_path = os.path.join(self.path_to_folder, 'data')
        if os.path.exists(data_path):
            for f in os.listdir(data_path):
                # Seperate Extension
                f_name, _ = os.path.splitext(f)
                # Separate description
                split = f_name.split('.')
                # Extract format
                format = Dataset.extract_format_frome_name(f_name)
                data = Data(os.path.join(data_path, f), format)
                self.formats[format] = data
                self.data.append(data)

    def get_data_by_format(self, format: DataFormat):
        """
        Get data for a given format
        
        :param format: Format of the data
        :return: Data object
        """
        for f in self.formats:
            if f.continuous == format.continuous and f.discrete == format.discrete and f.mixed == format.mixed and f.numeric == format.numeric and f.json == format.json and f.maximum == format.maximum and f.cov == format.cov:
                return Data(os.path.join(self.path_to_folder, 'data', f'{self.name}.txt'), f)
        return None
    
    def available_formats(self):
        """
        Get available formats for the dataset
        """
        return [str(key) for key in self.formats.keys()]

                    
class ECDProcessor:
    def __init__(self, path_to_ecd_repo:str):
        """
        Constructor for ECDProcessor
        
        :param path_to_ecd_repo: Path to the root of the example-causal-datasets repository
        """
        self.path_to_ecd_repo = path_to_ecd_repo
        self.load_available_datasets()

    def get_all_datasets(self):
        """
        Get all datasets
        """
        all_datasets = []
        for _, datasets in self.available_datasets.items():
            all_datasets.extend(datasets)
        return all_datasets

    def get_all_datasets_by_format(self, format: DataFormat):
        """
        Get all datasets for a given format
        
        :param format: Format of the data
        :return: List of datasets for the given format
        """
        datasets = defaultdict(list)
        for dataset in self.get_all_datasets():
            data = dataset.get_data_by_format(format)
            if data:
                datasets[dataset.name].append(data)
        return dict(datasets)

    def load_available_datasets(self):
        """
        Load available datasets from the example-causal-datasets repository
        """
        self.available_datasets = {}
        for root, dirs, _ in os.walk(self.path_to_ecd_repo):
            for dir in dirs:
                if dir[0] == '.':
                    continue
                self.available_datasets[dir] = []
                for root2, dirs2, _ in os.walk(os.path.join(root, dir)):
                    for dir2 in dirs2:
                        if dir2 == 'feedbacks':
                            self.available_datasets[dir].append(NetworkFeedback(dir2, os.path.abspath(os.path.join(root2, dir2))))
                            continue
                        self.available_datasets[dir].append(Dataset(dir2, os.path.abspath(os.path.join(root2, dir2))))
                    break
            break

    def _get_dataset_by_name(self, name:str, category:str):
        """
        Get a dataset by name
        
        :param name: Name of the dataset
        :param category: Category of the dataset
        :return: Dataset object
        """
        for dataset in self.available_datasets[category]:
            if dataset.name == name:
                return dataset
        return None

    def get_available_datasets(self, name:str, simulated=None, real=None):
        """
        Get available datasets for a given name
        
        :param name: Name of the dataset
        :param simulated: Simulated datasets
        :param real: Real datasets
        :return: List of datasets for the given name
        """
        if simulated:
            return self._get_dataset_by_name(name, 'simulated')
        if real:
            return self._get_dataset_by_name(name, 'real')
        for category in self.available_datasets:
            dataset = self._get_dataset_by_name(name, category)
            if dataset:
                return dataset
        return None
    

## Hardcoded Classes For Network Feedback

class NetworkFeedback:
    def __init__(self, name:str, path_to_network_feedback_repo:str):
        """
        Constructor for NetworkFeedback
        
        :param path_to_network_feedback_repo: Path to the root of the network-feedback repository
        """
        self.name = name
        self.path_to_network_feedback_repo = path_to_network_feedback_repo
        self.load_networks_and_simulations()

    def get_data_by_format(self, format: DataFormat):
        """
        Get data for a given format
        
        :param format: Format of the data
        :return: Data object
        """
        datas = []
        for network, simulations in self.available_simulations.items():
            for simulation, data in simulations.items():
                if data.format.continuous == format.continuous and data.format.discrete == format.discrete and data.format.mixed == format.mixed and data.format.numeric == format.numeric and data.format.json == format.json and data.format.maximum == format.maximum and data.format.cov == format.cov:
                    datas.append(data)
        return datas

    def load_networks_and_simulations(self):
        """
        Load networks and simulations from the network-feedback repository
        """
        self.available_networks = {}
        self.available_simulations = {}
        data_path = os.path.join(self.path_to_network_feedback_repo, 'data')
        for network_name in os.listdir(data_path):
            self.available_networks[network_name] = os.path.abspath(os.path.join(data_path, network_name))
            self.available_simulations[network_name] = {}
            for simulation in os.listdir(os.path.join(data_path, network_name)):
                f_name, _ = os.path.splitext(simulation)
                split = f_name.split('.')
                sim_name = split[0]
                format = Dataset.extract_format_frome_name(f_name)
                self.available_simulations[network_name][sim_name] = Data(os.path.join(data_path, network_name, simulation), format)
