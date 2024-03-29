import csv
import sys, os
import time
import numpy as np
import networkx as nx

import lingam
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor

# Add the InstaCD-benchmarks to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instacd_benchmarks.utils.timeout import run_with_timeout
from runners.base import BaseRunner


class RESITRunner(BaseRunner):
    def __init__(self):
        super().__init__()

    def add_specific_arguments(self):
        self.parser.add_argument('--regressor', type=str, default='linear', required=False, help='Regressor to use for RESIT, choice of [linear, gp]')

    def run_linear_resit(self, data):
        resit = lingam.RESIT(regressor=LinearRegression())
        output = run_with_timeout(resit.fit, (data.df,), self.args.timeout)
        return output
    
    def run_gp_resit(self, data):
        resit = lingam.RESIT(regressor=GaussianProcessRegressor())
        output = run_with_timeout(resit.fit, (data.df,), self.args.timeout)
        return output

    def run(self):
        # Initialize results
        metrics = [['dataset-name', 'sheet-name', 'format', 'num-forbidden', 'num-missing', 'num-predicted', 'runtime', 'knowledge-file']]
        network_results = {}

        # Run RESIT for all continuous datasets, can only handle continuous
        for row in self.ecd_info[self.args.sheet_name].iterrows():
            dataset_name = row[1]['dataset-name']
            print(dataset_name)
            # Get dataset obj
            dataset = self.ecd_processor.get_available_dataset(dataset_name)
            data = dataset.get_data_by_format(self.format)
            data.load_data()
            start_time = time.time()
            if self.args.regressor == 'linear':
                output = self.run_linear_resit(data)
            elif self.args.regressor == 'gp':
                output = self.run_gp_resit(data)
            else:
                raise ValueError('Invalid regressor type')
            runtime = np.round(time.time() - start_time, 2)
            if output is None:
                print('Dataset timed out, quitting the rest of the datasets too')
                metrics.append(
                    [dataset_name, self.args.sheet_name, str(data.format), '*', '*', '*', 'time-out', '*']
                )
                with open(self.args.result_file, 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(metrics)
                break
            # If we didn't time out, then we can continue
            network = nx.DiGraph(output.adjacency_matrix_)
            edges = data.unlabeled_edges_to_labeled_edges(list(network.edges))
            network_results[(dataset_name, str(data.format))] = edges
            for k_name, k in dataset.knowledge.knowledge.items():
                res = k.validate_graph(edges)
                metrics.append(
                    [dataset_name, self.args.sheet_name, str(data.format), len(res['forbidden']), len(res['missing']), len(edges), runtime, k_name]
                )
                with open(self.args.result_file, 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(metrics)

if __name__ == '__main__':
    runner = RESITRunner()
    runner.run()