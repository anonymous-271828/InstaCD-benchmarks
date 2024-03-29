import csv
import sys, os
import time
import numpy as np

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci

# Add the InstaCD-benchmarks to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instacd_benchmarks.utils.timeout import run_with_timeout
from runners.base import BaseRunner


class PCRunner(BaseRunner):
    def __init__(self):
        super().__init__()

    def add_specific_arguments(self):
        self.parser.add_argument('--kernelZ', type=str, default='Polynomial', required=False, help="KernelX/Y/Z (condition_set): ['GaussianKernel', 'LinearKernel', 'PolynomialKernel']. (For 'PolynomialKernel', the default degree is 2. Currently, users can change it by setting the 'degree' of 'class PolynomialKernel()'.")
        self.parser.add_argument('--alpha', type=float, default=0.01, required=False, help='Alpha')
        self.parser.add_argument('--ci_test_discrete', type=str, default='gsq', required=False, help='CI test for discrete variables. Choices are gsq or chisq.')

    def run_continuous_pc(self, dataset):
        data = dataset.get_data_by_format(self.format)
        data.load_data()
        data_np = data.df.to_numpy().astype(np.float64)
        return data, run_with_timeout(pc, (data_np, self.args.alpha, kci), self.args.timeout, kwargs={'kernelZ': self.args.kernelZ})
     
    def run_discrete_pc(self, dataset):
        data = dataset.get_data_by_format(self.format)
        data.load_data()
        if not self.format.entropy:
            data.discretize_continuous_features()
            data.make_features_numerical()
        data_np = data.df.to_numpy()
        return data, run_with_timeout(pc, (data_np, self.args.alpha, self.args.ci_test_discrete), self.args.timeout)

    def run(self):
        # Initialize results
        metrics = [['dataset-name', 'sheet-name', 'format', 'num-forbidden', 'num-missing', 'num-predicted', 'runtime', 'knowledge-file']]
        network_results = {}

        # Run Globe for all continuous datasets, can only handle continuous
        for row in self.ecd_info[self.args.sheet_name].iterrows():
            dataset_name = row[1]['dataset-name']
            print(dataset_name)
            # Get dataset obj
            dataset = self.ecd_processor.get_available_dataset(dataset_name)
            start_time = time.time()
            if self.format.continuous:
                data, network = self.run_continuous_pc(dataset)
            else:
                data, network = self.run_discrete_pc(dataset)
            runtime = np.round(time.time() - start_time, 2)
            if network is None:
                print('Dataset timed out, quitting the rest of the datasets too')
                metrics.append(
                    [dataset_name, self.args.sheet_name, str(self.format), '*', '*', '*', 'time-out', '*']
                )
                with open(self.args.result_file, 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(metrics)
                break
            # If we didn't time out, then we can continue
            network.to_nx_graph()
            edges = data.unlabeled_edges_to_labeled_edges(list(network.nx_graph.edges))
            network_results[(dataset_name, str(self.format))] = edges
            for k_name, k in dataset.knowledge.knowledge.items():
                res = k.validate_graph(edges)
                metrics.append(
                    [dataset_name, self.args.sheet_name, str(self.format), len(res['forbidden']), len(res['missing']), len(edges), runtime, k_name]
                )
                with open(self.args.result_file, 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(metrics)

if __name__ == '__main__':
    runner = PCRunner()
    runner.run()