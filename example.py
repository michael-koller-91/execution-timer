import numpy as np
import execution_timer


def data_generator():
    for n_dim in [10, 100]:
        print(f"n_dim = {n_dim}")
        yield np.random.randn(n_dim, n_dim),


def sum(a):
    k = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            k += a[i, j]
    return k


if __name__ == "__main__":
    timer = execution_timer.Timer()
    timer.add_function(sum)
    timer.add_function(np.sum, "np.sum")
    timer.timeit(data_generator, compare_results=True)
    timer.evaluate()
