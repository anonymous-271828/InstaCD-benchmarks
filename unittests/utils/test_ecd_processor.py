import pandas as pd
from instacd_benchmarks.utils.ecd_processor import DataFormat, ECDProcessor
import unittest
import os

class TestEcdProcessor(unittest.TestCase):
    
    def test_ecd_processor(self):
        ecd_processor = ECDProcessor('submodules/example-causal-datasets')
        self.assertEqual(len(ecd_processor.get_all_datasets()), 24)

    def test_knowledge(self):
        ecd_processor = ECDProcessor('submodules/example-causal-datasets')
        abalone = ecd_processor.get_available_dataset('abalone')
        self.assertListEqual(abalone.knowledge.knowledge['abalone.knowledge'].forbidden_edges, [('Diam', 'Sex'), ('Length', 'Sex'), ('Height', 'Sex'), ('Diam', 'Rings'), ('Length', 'Rings'), ('Height', 'Rings'), ('Shucked', 'Rings'), ('Shucked', 'Sex'), ('Shell', 'Rings'), ('Shell', 'Sex'), ('Viscera', 'Rings'), ('Viscera', 'Sex'), ('Whole*', 'Rings'), ('Whole*', 'Sex'), ('Whole*', 'Shucked'), ('Whole*', 'Shell'), ('Whole*', 'Viscera')])
        self.assertListEqual(abalone.knowledge.knowledge['abalone.knowledge'].required_edges, [])
        self.assertDictEqual(abalone.knowledge.knowledge['abalone.knowledge'].temporal_edges, {'1': ['Rings', 'Sex'], '2': ['Shucked', 'Shell', 'Viscera'], '3': ['Whole*']})
        self.assertDictEqual(abalone.knowledge.knowledge['abalone.knowledge'].validate_graph([('Diam', 'Sex'), ('Length', 'Sex'), ('Height', 'Sex'), ('Diam', 'Rings'), ('Rings', 'Whole*'), ('Rings', 'Viscera'), ('Viscera', 'Whole*'), ('Sex', 'Shucked'), ('Whole*', 'Sex')]), {'forbidden': [('Diam', 'Sex'), ('Length', 'Sex'), ('Height', 'Sex'), ('Diam', 'Rings'), ('Whole*', 'Sex')], 'missing': []})

    def test_knowledge_feedbacks(self):
        ecd_processor = ECDProcessor('submodules/example-causal-datasets')
        feedbacks = ecd_processor.get_available_dataset('feedbacks')
        self.assertListEqual(feedbacks.knowledge['Network5_amp'].knowledge['Network5_amp.ground.truth.graph'].graph_edges, [('X1', 'X3'), ('X2', 'X4'), ('X3', 'X4'), ('X3', 'X5'), ('X4', 'X3')])
        self.assertListEqual(feedbacks.knowledge['Network5_amp'].knowledge['Network5_amp.ground.truth.graph'].forbidden_edges, [('X1', 'X2'), ('X1', 'X4'), ('X1', 'X5'), ('X2', 'X1'), ('X2', 'X3'), ('X2', 'X5'), ('X3', 'X1'), ('X3', 'X2'), ('X4', 'X1'), ('X4', 'X2'), ('X4', 'X5'), ('X5', 'X1'), ('X5', 'X2'), ('X5', 'X3'), ('X5', 'X4')])

    def test_all_knowledge_is_not_empty(self):
        ecd_processor = ECDProcessor('submodules/example-causal-datasets')
        for dataset in ecd_processor.get_all_datasets():
            if type(dataset.knowledge) is dict:
                for _, kp in dataset.knowledge.items():
                    for _, k in kp.knowledge.items():
                        self.assertNotEqual(len(k.temporal_edges) + len(k.forbidden_edges) + len(k.required_edges), 0)   
            else:
                for _, k in dataset.knowledge.knowledge.items():
                    self.assertNotEqual(len(k.temporal_edges) + len(k.forbidden_edges) + len(k.required_edges), 0)   

    def test_discretization_mixed_dataset(self):
        ecd_processor = ECDProcessor('submodules/example-causal-datasets')
        autompg = ecd_processor.get_available_dataset('auto-mpg')
        target = 'mpg*'
        data = autompg.get_data_by_format(DataFormat(mixed=True))
        data.load_data()
        data.discretize_continuous_features(target, max_depth=2)
        unique_values = data.df.apply(pd.Series.unique)
        self.assertEqual(len(unique_values['mpg*']), 4)
        self.assertEqual(len(unique_values['cylinders']), 5)
        self.assertEqual(len(unique_values['displacement']), 4)
        self.assertEqual(len(unique_values['horsepower']), 4)
        self.assertEqual(len(unique_values['weight']), 4)
        self.assertEqual(len(unique_values['acceleration']), 4)
        self.assertEqual(len(unique_values['modelyear']), 4)
        self.assertEqual(len(unique_values['origin']), 3)

    def test_discretization_all_mixed_dataset(self):
        OMIT_DATASETS = ['dry-bean', 'covertype', 'apple-watch-fitbit', 'htru2'] # Just to speed up the test
        ecd_processor = ECDProcessor('submodules/example-causal-datasets')
        for dataset_name, datalist in ecd_processor.get_all_datasets_by_format(DataFormat(mixed=True)).items():
            if dataset_name in OMIT_DATASETS:
                continue
            for data in datalist:
                data.load_data()
                data.discretize_continuous_features(target='auto', max_depth=2)
                data.make_features_numerical()