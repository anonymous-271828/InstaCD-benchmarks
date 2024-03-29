# Processor Class for Example-Causal-Datasets Repository: https://github.com/cmu-phil/example-causal-datasets

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import re
import numpy as np
import pandas as pd

from .entropy_discretization import entropy_discretize, convert_to_continuous_bins

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
            entropy: bool = False,
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
        self.entropy = entropy

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
        if self.entropy:
            s += 'entropy ' 
        return s[:-1]
    
    def __hash__(self) -> int:
        return hash(str(self))

class Data:
    def __init__(self, path, format: DataFormat=None):
        """
        Constructor for Data
        
        :param path: Path to the data
        :param format: Format of the data
        """
        self.path = path
        self.format = format

    def load_data(self, remove_missing=True):
        """
        Load data from the path
        """
        self.df = pd.read_csv(self.path, delimiter='\t', header=0)
        # Drop missing values, i.e., rows with '*'
        if remove_missing:
            self.df = self.df.replace('*', np.nan)
            self.df = self.df.dropna()

    def _check_if_continuous(self, column, unique_values_cutoff=10):
        """
        Check if the column is continuous
        """
        if self.df[column].dtype == 'float64' or self.df[column].dtype == 'Float64' or self.df[column].dtype == 'int64' or self.df[column].dtype == 'Int64':
            if len(self.df[column].unique()) > unique_values_cutoff:
                return True
        return False
    
    @staticmethod
    def discretize_column(df, column, target, max_depth):
        """
        Helper function to discretize a single column.
        """
        bins = entropy_discretize(df[column], df[target], max_depth=max_depth)
        converted_bins = convert_to_continuous_bins(bins)
        digitized_values = pd.Series(np.digitize(df[column], converted_bins), index=df.index)
        return column, digitized_values, converted_bins

    def discretize_continuous_features(self, target='auto', max_depth=2, n_cpus=None):
        """
        Discretize continuous features using entropy discretization on a target variable.
        """
        if target == 'auto':
            for column in self.df.columns:
                if '*' in column:
                    target = column
                    break
        if target == 'auto':
            raise ValueError("No target variable found and target was set to auto.")
        self.bins = {}
        # Setup multiprocessing pool
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            # Submit discretization tasks
            futures = [executor.submit(Data.discretize_column, self.df, column, target, max_depth)
                   for column in self.df.columns if self._check_if_continuous(column)]
        
            # Process completed tasks
            for future in as_completed(futures):
                column, digitized_values, converted_bins = future.result()
                self.df[column] = digitized_values
                self.bins[column] = converted_bins
        
        # for column in self.df.columns:
        #     if column == target:
        #         continue
        #     if self._check_if_continuous(column):
        #         bins = entropy_discretize(self.df[column], target_data, max_depth=max_depth)
        #         converted_bins = convert_to_continuous_bins(bins)
        #         self.df[column] = pd.Series(np.digitize(self.df[column], converted_bins), index=self.df.index)
        #         self.bins[column] = converted_bins
        # # Now discretize target variable if needed
        # if self._check_if_continuous(target):
        #     bins = entropy_discretize(target_data, self.df[target], max_depth=max_depth)
        #     converted_bins = convert_to_continuous_bins(bins)
        #     self.df[target] = pd.Series(np.digitize(target_data, converted_bins), index=self.df.index)        
        #     self.bins[target] = converted_bins

    def make_features_numerical(self):
        """
        Make all non-numerical features numerical. 
        """
        self.category_map = {}
        for column in self.df.columns:
            if self.df[column].dtype == 'object' or self.df[column].dtype == 'str':
                self.df[column] = pd.Categorical(self.df[column])
                self.category_map[column] = list(self.df[column].cat.categories)
                self.df[column] = self.df[column].cat.codes

    def export_info(self):
        """
        Export the data info
        """
        print(self.df.info())
            
    def unlabeled_edges_to_labeled_edges(self, edges):
        """
        Convert unlabeled edges to labeled edges
        
        :param edges: Edges to convert
        """
        labeled_edges = []
        for edge in edges:
            labeled_edges.append((self.df.columns[edge[0]], self.df.columns[edge[1]]))
        return labeled_edges
    
    def export(self, path, filename):
        """
        Export the data
        """
        self.df.to_csv(os.path.join(path, filename), sep='\t', index=False)

class KnowledgeProcessor:
    def __init__(self, folder_path:str):
        self.folder_path = folder_path
        self.parse_knowledge()
    
    def parse_knowledge(self):
        self.knowledge = {}
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                knowledge_name, _ = os.path.splitext(filename)
                filename = os.path.join(self.folder_path, filename)
                self.knowledge[knowledge_name] = Knowledge(filename)

class Knowledge:
    def __init__(self, filename:str):
        self.filename = filename
        self.temporal_edges = {}
        self.forbidden_edges = []
        self.required_edges = []
        self.graph_nodes = None
        self.graph_edges = None
        self.parse_knowledge_file()

    def parse_knowledge_file(self):
        if 'graph' in self.filename:
            return self.parse_graph()
        with open(self.filename, 'r') as f:
            content = f.read()
        sections = content.split("/knowledge")[-1].strip().split("\n\n")    
        # Merge first two sections together as these are the temporal edges
        other_sections = self.parse_temporal(sections)        

        for section in other_sections:
            title, *lines = section.strip().split("\n")
            if title == "forbiddirect":
                self.forbidden_edges = [tuple(line.split()) for line in lines]
            elif title == "requiredirect":
                self.required_edges = [tuple(line.split()) for line in lines]
        
        # Add temporal edges to forbidden edges
        self.add_forbidden_temporal_edges()

    def parse_graph(self):
        self.graph_nodes = []
        self.graph_edges = []
        with open(self.filename, 'r') as file:
            lines = file.readlines()
        
        # Parsing the nodes
        nodes_line = lines[1]  # Assumes the second line contains the nodes
        self.graph_nodes = [node.strip() for node in re.split(',|;', nodes_line)]

        # Parsing the edges
        edges_lines = lines[4:]  # Assumes the fourth line onwards contains the edges
        for line in edges_lines:
            # First remove the number
            line = line.split(' ')[1:]
            if len(line) == 0:
                continue
            arrow_idx = line.index('-->')
            source = line[arrow_idx - 1]
            target = line[arrow_idx + 1]
            self.graph_edges.append((source.strip(), target.strip()))

        # Add required and forbidden edges
        self.required_edges = self.graph_edges
        self.forbidden_edges = []
        for node in self.graph_nodes:
            for node2 in self.graph_nodes:
                if node != node2 and (node, node2) not in self.graph_edges:
                    self.forbidden_edges.append((node, node2))

    def add_forbidden_temporal_edges(self):
        tiers = list(self.temporal_edges.keys())
        for tier, nodes in self.temporal_edges.items():
            # If star in tier than all edges between nodes in tier are forbidden as well.
            if '*' in tier:
                for node in nodes:
                    for node2 in nodes:
                        if node != node2:
                            self.forbidden_edges.append((node, node2))
                tier = tier.replace('*', '')
            tier = int(tier)
            for tier2 in tiers:
                _tier2 = int(tier2.replace('*', ''))
                if _tier2 >= tier:
                    continue
                for node in nodes:
                    for node2 in self.temporal_edges[tier2]:
                        self.forbidden_edges.append((node, node2))

    def parse_temporal(self, sections):
        if sections[0] == "addtemporal":
            lines = sections[1].strip().split("\n")
        else:
            return sections
        self.temporal_edges = {}
        for line in lines:
            parts = line.split()
            tier = parts[0]
            nodes = parts[1:]
            self.temporal_edges[tier] = nodes
        return sections[2:]

    def validate_graph(self, proposed_edges):
        wrong_edges = {'forbidden': [], 'missing': []}
        # Check for forbidden edges
        for edge in proposed_edges:
            if edge in self.forbidden_edges:
                wrong_edges['forbidden'].append(edge)
        
        # Check for required edges not in the proposed graph
        for edge in self.required_edges:
            if edge not in proposed_edges:
                wrong_edges['missing'].append(edge)

        return wrong_edges


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
        self.load_knowledge()

    def load_knowledge(self):
        """
        Load knowledge from the dataset
        """
        knowledge_path = os.path.join(self.path_to_folder, 'ground.truth')
        if os.path.exists(knowledge_path):
            self.knowledge = KnowledgeProcessor(knowledge_path)
        else:
            self.knowledge = None

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
        if 'entropy' in split:
            format['entropy'] = True
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
        for f, data in self.formats.items():
            if f.continuous == format.continuous and f.discrete == format.discrete and f.mixed == format.mixed and f.numeric == format.numeric and f.json == format.json and f.cov == format.cov and f.entropy == format.entropy:
                return data
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

    def get_all_dataset_names(self):
        """
        Get all dataset names
        """
        return list(self.available_datasets_by_name.keys())

    def get_all_datasets(self):
        """
        Get all datasets
        """
        all_datasets = []
        for _, dataset in self.available_datasets_by_name.items():
            all_datasets.append(dataset)
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
        self.available_datasets_by_category = {}
        self.available_datasets_by_name = {}
        for root, dirs, _ in os.walk(self.path_to_ecd_repo):
            for dir in dirs:
                if dir[0] == '.':
                    continue
                self.available_datasets_by_category[dir] = []
                for root2, dirs2, _ in os.walk(os.path.join(root, dir)):
                    for dir2 in dirs2:
                        if dir2 == 'feedbacks':
                            ds = NetworkFeedback(dir2, os.path.abspath(os.path.join(root2, dir2))) 
                            self.available_datasets_by_category[dir].append(ds)
                            self.available_datasets_by_name[ds.name] = ds
                            continue
                        ds = Dataset(dir2, os.path.abspath(os.path.join(root2, dir2)))
                        self.available_datasets_by_category[dir].append(ds)
                        self.available_datasets_by_name[ds.name] = ds
                    break
            break

    def get_available_dataset(self, name:str):
        """
        Get available datasets for a given name
        
        :param name: Name of the dataset
        :return: List of datasets for the given name
        """
        try:
            return self.available_datasets_by_name[name]
        except KeyError:
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
        self.knowledge = {}
        data_path = os.path.join(self.path_to_network_feedback_repo, 'data')
        knowledge_path = os.path.join(self.path_to_network_feedback_repo, 'ground.truth')
        for network_name in os.listdir(data_path):
            self.available_networks[network_name] = os.path.abspath(os.path.join(data_path, network_name))
            self.knowledge[network_name] = KnowledgeProcessor(os.path.join(knowledge_path, network_name))
            self.available_simulations[network_name] = {}
            for simulation in os.listdir(os.path.join(data_path, network_name)):
                f_name, _ = os.path.splitext(simulation)
                split = f_name.split('.')
                sim_name = split[0]
                format = Dataset.extract_format_frome_name(f_name)
                self.available_simulations[network_name][sim_name] = Data(os.path.join(data_path, network_name, simulation), format)
