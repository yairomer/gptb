import numpy as np


def mat_power(mat, p):
    s, v = np.linalg.eig(mat)
    return (v * s ** p) @ v.T


def generate_gaussian_data(data_size, cov_mat, mu, random_seed=0):
    if not isinstance(random_seed, np.random.RandomState):
        random_seed = np.random.RandomState(random_seed)

    t_mat = mat_power(cov_mat, 0.5)

    data_seed = random_seed.randn(data_size, cov_mat.shape[0])
    data = data_seed @ t_mat.T

    data += mu[None, :]

    return data
