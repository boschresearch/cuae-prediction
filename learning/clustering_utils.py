""" Conditional Unscented Autoencoder.
Copyright (c) 2024 Robert Bosch GmbH
@author: Faris Janjos
@author: Marcel Hallgarten
@author: Anthony Knittel
@author: Maxim Dolgov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import psutil
import ray
import torch
from sklearn.cluster import KMeans
from torch_scatter import scatter_std

from lib.utils import class_from_path

ray.init(num_cpus=psutil.cpu_count(logical=False), log_to_driver=False)


def clustering_function(
    unclustered_data: torch.Tensor,
    path_to_clustering_func: str,
    num_clusters: int,
    clustering_func_kwargs: dict = None,
) -> torch.Tensor:
    """
    Wrapper for calling a clustering function on unclustered data and obtaining clustered data.
    Supported clustering methods are: K-Means.

    Args:
        unclustered_data: shape [batch_size, num_samples, *], * is any number of feature dimensions
        path_to_clustering_func:
        num_clusters: Number of desired clusters
        clustering_func_kwargs: Additional arguments for the clustering function
    Returns:
        cluster_centroids: shape [batch_size, num_clusters, *]
        cluster_std: shape [batch_size, num_clusters, *]
        cluster_weights: shape [batch_size, num_clusters]
    """

    if clustering_func_kwargs is None:
        clustering_func_kwargs = {}

    supported_clustering_functions = [
        "learning.clustering_utils.kmeans_clustering",
    ]
    assert (
        path_to_clustering_func in supported_clustering_functions
    ), f"Clustering path: {path_to_clustering_func} is not among supported: {supported_clustering_functions}"

    clustering_function = class_from_path(path_to_clustering_func)

    # compute clusters and weights
    cluster_centroids, cluster_std, cluster_weights = clustering_function(
        unclustered_data, num_clusters, **clustering_func_kwargs
    )

    return cluster_centroids, cluster_std, cluster_weights


@ray.remote
def cluster(k: int, data: np.ndarray, random_state: int = None):
    """
    Code within this function is adapted from the following paper:
        Title: Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals
        Authors: Deo, Nachiket and Wolff, Eric and Beijbom, Oscar
        Website: https://github.com/nachiket92/PGP
        License: https://github.com/nachiket92/PGP/blob/main/LICENSE
            MIT License
            Copyright (c) 2021 nachiket92

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
    """

    def cluster(n_clusters: int, x: np.ndarray, random_state=None):
        """
        Cluster using Scikit learn
        """
        clustering_op = KMeans(
            n_clusters=n_clusters,
            n_init=1,
            max_iter=100,
            init="random",
            random_state=random_state,
        ).fit(x)
        return clustering_op.labels_, clustering_op.cluster_centers_

    cluster_lbls, cluster_ctrs = cluster(k, data, random_state)
    cluster_cnts = np.unique(cluster_lbls, return_counts=True)[1]
    out = {"lbls": cluster_lbls, "counts": cluster_cnts}
    return out


def kmeans_clustering(traj: torch.Tensor, k: int, random_state: int = 0):
    """
    Code within this function is adapted from the following paper:
        Title: Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals
        Authors: Deo, Nachiket and Wolff, Eric and Beijbom, Oscar
        Website: https://github.com/nachiket92/PGP
        License: https://github.com/nachiket92/PGP/blob/main/LICENSE
            MIT License
            Copyright (c) 2021 nachiket92

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.

    clusters sampled trajectories to output K modes.
    :param k: number of clusters
    :param traj: set of sampled trajectories, shape [batch_size, num_samples, traj_len, 2]
    :return: traj_clustered:  set of clustered trajectories, shape [batch_size, k, traj_len, 2]
             scores: scores for clustered trajectories (basically 1/rank), shape [batch_size, k]
    """

    # Initialize output tensors
    batch_size = traj.shape[0]
    num_samples = traj.shape[1]
    traj_len = traj.shape[2]
    device = traj.device

    # Down-sample traj along time dimension for faster clustering
    data = traj[:, :, 0::3, :]
    data = data.reshape(batch_size, num_samples, -1).detach().cpu().numpy()

    # Cluster and rank
    cluster_ops = ray.get([cluster.remote(k, data_slice) for data_slice in data])
    # if less than k clusters were found, add empty dummy clusters
    for cluster_op in cluster_ops:
        num_missing_clusters = k - cluster_op["counts"].shape[0]
        cluster_op["counts"] = np.concatenate(
            [cluster_op["counts"], np.zeros(num_missing_clusters)]
        )
    cluster_lbls = np.array([cluster_op["lbls"] for cluster_op in cluster_ops])
    cluster_counts = np.array([cluster_op["counts"] for cluster_op in cluster_ops])

    # Compute mean (clustered) traj and scores
    lbls = (
        torch.as_tensor(cluster_lbls, device=device)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, 1, traj_len, 2)
        .long()
    )

    traj_summed = torch.zeros(
        batch_size, k, traj_len, 2, dtype=traj.dtype, device=device
    ).scatter_add(dim=1, index=lbls, src=traj)
    traj_std = torch.zeros(batch_size, k, traj_len, 2, dtype=traj.dtype, device=device)
    scatter_std(src=traj, index=lbls, dim=1, out=traj_std)

    cnt_tensor = (
        torch.as_tensor(cluster_counts, device=device)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, 1, traj_len, 2)
    )
    traj_clustered = traj_summed / cnt_tensor
    # set center of empty dummy clusters to inf
    # to prevent them from being considered in displacement metrics
    traj_clustered[cnt_tensor == 0] = torch.inf
    # return probabilities proportional to counts
    probs = torch.as_tensor(cluster_counts, device=device)
    probs = probs / torch.sum(probs, dim=1)[0]

    return traj_clustered, traj_std, probs
