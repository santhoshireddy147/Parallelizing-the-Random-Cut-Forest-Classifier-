from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import random
import time

from sklearn.ensemble import RandomForestClassifier

import psutil



df = pd.read_csv("/home/sk2349/project/nyc_taxi.csv")
    
df['timestamp'] = pd.to_datetime(df['timestamp'])
     #is anomaly? : True => 1, False => 0
df['anomaly'] = 0

          
df.index = df['timestamp']
df.drop(['timestamp'], axis=1, inplace=True)

class RandomCutForestMPI:
    def __init__(self, num_trees=100, num_samples_per_tree=256, num_dimensions=None, threshold=None):
        self.num_trees = num_trees
        self.num_samples_per_tree = num_samples_per_tree
        self.num_dimensions = num_dimensions
        self.threshold = threshold
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.trees = []
        #print(self.rank)
        #print(self.size)

    def fit_predict(self, X):
        if self.num_dimensions is None:
            self.num_dimensions = X.shape[1] if len(X.shape) > 1 else 1

        

        # Scatter the data to different processes
        local_X = comm.scatter(np.array_split(X, self.size), root=0)

        # Build the forest
        for _ in range(self.num_trees):
            tree = self._build_tree(local_X)
            self.trees.append(tree)

        # Compute anomaly scores
        anomaly_scores = self._compute_anomaly_scores(X)
        
        return anomaly_scores

    def _build_tree(self, X):
        num_samples = X.shape[0]

        # Randomly select samples for the tree
        sample_indices = np.random.choice(num_samples, size=self.num_samples_per_tree, replace=False)
        tree_X = X[sample_indices]

        # Randomly select features for the tree
        feature_indices = np.random.choice(self.num_dimensions, size=int(np.sqrt(self.num_dimensions)), replace=False)
        tree_X = tree_X[:, feature_indices]

        # Create a tree structure (e.g., using a simple threshold-based approach)
        tree = {'feature_indices': feature_indices}

        if self.threshold is None:
            # Compute threshold as the mean of a random feature
            random_feature = np.random.choice(tree_X.shape[1])
            tree['threshold'] = np.mean(tree_X[:, random_feature])
        else:
            tree['threshold'] = self.threshold

        return tree

    def _compute_anomaly_scores(self, X):
        anomaly_scores = np.zeros(X.shape[0])

        for tree in self.trees:
            feature_indices = tree['feature_indices']
            thresholds = tree['threshold']

            # Apply the tree to each sample
            tree_scores = np.mean(X[:, feature_indices] < thresholds, axis=1)

            # Update the anomaly scores
            anomaly_scores += tree_scores

        return anomaly_scores / self.num_trees

def monitor_resources():
    cpu_percentages = []
    memory_usages = []

    # Monitor CPU and memory usage over time
    for _ in range(100):
        cpu_percentages.append(psutil.cpu_percent())
        memory_usages.append(psutil.virtual_memory().used)
        time.sleep(0.1)  # Adjust sleep interval as needed

    # Calculate average CPU usage and peak memory usage
    avg_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)
    peak_memory_usage = max(memory_usages) / (1024 * 1024)  # Convert to MB

    return avg_cpu_usage, peak_memory_usage
# Example usage
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Rank {rank} of {size} processes")

    # Assuming `df` is the DataFrame containing the dataset with the feature(s) for anomaly detection
    # Extract the feature for anomaly detection (assuming 'value' column)
    features = np.array(df['value']).reshape(-1, 1)
    # Process 0's work
    if rank == 0:
          start_avg_cpu, start_peak_memory = monitor_resources()

    #start_memory = psutil.virtual_memory().used
    start_time = time.time()
    
    # Inside your MPI script
    
    
    # Instantiate and fit RandomCutForestMPI
    rcf_mpi = RandomCutForestMPI(num_trees=100, num_samples_per_tree=256, num_dimensions=features.shape[1], threshold=None)
    anomaly_scores = rcf_mpi.fit_predict(features)
    
    
    
    
    # Your MPI code here for process 0
    end_time = time.time()

    if rank == 0:
        #end_memory = psutil.virtual_memory().used
        #memory_usage = end_memory - start_memory
        #print(f"Memory Usage: {memory_usage / (1024 * 1024)} MB")
        end_avg_cpu, end_peak_memory = monitor_resources()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")
        
        print(f"Avg CPU Usage: {end_avg_cpu}%")
        print(f"Peak Memory Usage: {end_peak_memory:.2f} MB")
        # Update DataFrame with anomaly scores
        df['anomaly'] = anomaly_scores
        df['anomaly'] = df['anomaly'].astype(int)
        df.to_csv('/home/sk2349/project/rcf_out.csv')

        # Display DataFrame with anomaly scores
        print(df)
    print(f"Process {rank} completed")
