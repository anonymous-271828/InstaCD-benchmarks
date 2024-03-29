import numpy as np
import pandas as pd
from scipy.stats import entropy

def calculate_entropy(target):
    """Calculate the entropy of the given target variable for multiple classes."""
    probabilities = target.value_counts(normalize=True)
    return entropy(probabilities, base=2)

def information_gain(data, target, split):
    """Calculate the information gain of a potential split for multiple classes."""
    total_entropy = calculate_entropy(target)
    
    # Data and target for left split
    left_target = target[data <= split]
    left_entropy = calculate_entropy(left_target)
    
    # Data and target for right split
    right_target = target[data > split]
    right_entropy = calculate_entropy(right_target)
    
    # Weighted sum of the entropy for the splits
    weighted_entropy = (len(left_target) / len(target)) * left_entropy + (len(right_target) / len(target)) * right_entropy
    
    return total_entropy - weighted_entropy

def find_best_split(data, target):
    """Find the best split point for the data, returning the split point and the information gain, for multiple classes."""
    best_gain = 0
    best_split = None
    unique_values = np.unique(data)
    
    for i in range(1, len(unique_values)):
        split = (unique_values[i - 1] + unique_values[i]) / 2
        gain = information_gain(data, target, split)
        if gain > best_gain:
            best_gain = gain
            best_split = split
            
    return best_split, best_gain

def convert_to_continuous_bins(bins):
    """Convert the list of bins to continuous ranges. Takes the midpoint between bins."""
    continuous_bins = []
    for i in range(len(bins) - 1):
        midpoint = (bins[i][1] + bins[i + 1][0]) / 2
        continuous_bins.append(midpoint)
    return continuous_bins

def entropy_discretize(data, target, max_depth=1, current_depth=0):
    """Recursively discretize the data based on entropy, for multiple classes, stopping at the given max depth."""
    if current_depth == max_depth:
        return [(data.min(), data.max())]  # Return the range of the data as the bin
    
    split, gain = find_best_split(data, target)
    if split is None:
        return [(data.min(), data.max())]  # If no split improves information gain, return the range as the bin
    
    left_data = data[data <= split]
    right_data = data[data > split]
    
    left_bins = entropy_discretize(left_data, target[left_data.index], max_depth, current_depth + 1)
    right_bins = entropy_discretize(right_data, target[right_data.index], max_depth, current_depth + 1)

    return left_bins + right_bins