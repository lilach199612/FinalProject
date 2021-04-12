import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt
import sys
sys.path.append('./kmeans_pytorch')
from kmeans_pytorch.kmeans import lloyd

def data_preparation(n_cluster, data):
    clusters_index, centers = lloyd(data, n_cluster, device=0, tol=1e-1)
    return clusters_index



