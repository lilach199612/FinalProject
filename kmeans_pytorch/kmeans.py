import torch
import numpy as np
from kmeans_pytorch.pairwise import pairwise_distance

def forgy(X, n_clusters):
	_len = len(X)
	indices = np.random.choice(_len, n_clusters)
	initial_state = X[indices]
	return initial_state


def lloyd(X, n_clusters, device=0, tol=1e-4,iterations=20):
        #X = torch.from_numpy(X).float().cuda(device)
    with torch.no_grad():
        initial_state = forgy(X, n_clusters)
        #print('initial_state:')
        #print(initial_state)
        iteration = 0
        while True:
                dis = pairwise_distance(X, initial_state)
                #print('dis:')
                #print(dis)
                choice_cluster = torch.argmin(dis, dim=1)
                #print('choice_cluster:')
                #print(choice_cluster)
                initial_state_pre = initial_state.clone()
                for index in range(n_clusters):
                    selected = torch.nonzero(choice_cluster==index).squeeze()
                    selected = torch.index_select(X, 0, selected)
                    initial_state[index] = selected.mean(dim=0)
                center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
                iteration += 1
                if center_shift ** 2 < tol or iteration > iterations:
                    break
        return choice_cluster, initial_state
        #return choice_cluster.cpu().numpy(), initial_state.cpu().numpy()
