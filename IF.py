from mpi4py import MPI
import numpy as np
import pandas as pd
import time
import psutil
class Node:
    def __init__(self, left, right, split_feature, split_value):
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.split_value = split_value

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit

    def fit(self, X):
        self.root = self._make_tree(X, 0)

    def _make_tree(self, X, current_height):
        if current_height >= self.height_limit or len(X) <= 1:
            return Node(None, None, None, None)
        else:
            split_feature = np.random.choice(X.shape[1])
            split_value = np.random.uniform(np.min(X[:, split_feature]), np.max(X[:, split_feature]))
            left_mask = X[:, split_feature] < split_value
            right_mask = ~left_mask
            return Node(self._make_tree(X[left_mask], current_height + 1),
                        self._make_tree(X[right_mask], current_height + 1),
                        split_feature,
                        split_value)

    def path_length(self, x):
        return self._path_length_helper(x, self.root, 0)

    def _path_length_helper(self, x, node, current_path_length):
        if node.left is None and node.right is None:
            return current_path_length
        elif x[node.split_feature] < node.split_value:
            return self._path_length_helper(x, node.left, current_path_length + 1)
        else:
            return self._path_length_helper(x, node.right, current_path_length + 1)
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
    
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load data
    if rank == 0:
        df = pd.read_csv("/home/sk2349/project/nyc_taxi.csv")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['anomaly'] = 0
        df.set_index('timestamp', inplace=True)
        data = df[['value']].values  # Extract 'value' feature for anomaly detection
    else:
        data = None

    # Broadcast data
    
    
    data = comm.bcast(data, root=0)
    data = np.array_split(data, size)[rank]
    
    if rank == 0:
          start_avg_cpu, start_peak_memory = monitor_resources()
    start_time = time.time()

    # Fit isolation tree
    it = IsolationTree(height_limit=200)
    it.fit(data)

    # Compute path lengths
    path_lengths = np.array([it.path_length(x) for x in data])

    # Compute anomaly scores
    anomaly_scores = path_lengths / it.height_limit

    # Gather anomaly scores
    anomaly_scores = comm.gather(anomaly_scores, root=0)
    
    end_time = time.time()

    # Print anomaly scores
    if rank == 0:  
        end_avg_cpu, end_peak_memory = monitor_resources()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")
        print(f"Avg CPU Usage: {end_avg_cpu}%")
        print(f"Peak Memory Usage: {end_peak_memory:.2f} MB")
        anomaly_scores = np.concatenate(anomaly_scores)
        ascore=pd.DataFrame(anomaly_scores)
        #print(len(anomaly_scores))
        df['anomaly'] = anomaly_scores
        
        df.to_csv('/home/sk2349/project/it.csv')
        
        #print(len(ascore))

if __name__ == '__main__':
    main()
