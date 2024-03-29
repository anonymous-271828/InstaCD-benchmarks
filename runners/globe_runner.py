import csv
import sys, os
import time
import numpy as np

# Add the InstaCD-benchmarks to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instacd_benchmarks.utils.timeout import run_with_timeout
from instacd_benchmarks.globe.globeWrapper import GlobeWrapper
from runners.base import BaseRunner

# from ..utils.timeout import run_with_timeout
# from .base import BaseRunner
# from ..globe.globeWrapper import GlobeWrapper

class GlobeRunner(BaseRunner):
    def __init__(self):
        super().__init__()

    def add_specific_arguments(self):
        self.parser.add_argument('--max-interaction', type=int, default=2, required=False, help='Max Interaction')
    
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
            data = dataset.get_data_by_format(self.format)
            data.load_data()
            data_np = data.df.to_numpy()
            globe = GlobeWrapper(self.args.max_interaction, False, False)
            globe.vars = data_np
            start_time = time.time()
            network = run_with_timeout(globe.run, (), self.args.timeout)
            runtime = np.round(time.time() - start_time, 2)
            if network is None:
                print('Dataset timed out, quitting the rest of the datasets too')
                metrics.append(
                    [dataset_name, self.args.sheet_name, str(data.format), '*', '*', '*', 'time-out', '*']
                )
                with open(self.args.result_file, 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(metrics)
                break
            # If we didn't time out, then we can continue
            network = GlobeWrapper.network_to_edge_list(network)
            edges = data.unlabeled_edges_to_labeled_edges(network)
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
    runner = GlobeRunner()
    runner.run()