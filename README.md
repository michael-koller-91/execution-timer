# Execution Timer

A wrapper around the `timeit` module to compare function execution times.

## Example

The example ([example.py](example.py)) demonstrates how `execution_timer.py` can be used.
It compares the execution time of `np.sum` and of the function `sum` given by
```python
def sum(a):
    k = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            k += a[i, j]
    return k
```

Running `python example.py` prints for example:
```
n_dim = 10
i_data = 0
 16.3 µs ± 219.79 ns per loop (mean ± std. dev. of 3 runs, 10000 loops each) (sum)
  3.1 µs ±  10.93 ns per loop (mean ± std. dev. of 3 runs, 100000 loops each) (np.sum)
n_dim = 100
i_data = 1
  1.4 ms ±   4.07 µs per loop (mean ± std. dev. of 3 runs, 100 loops each) (sum)
  7.2 µs ±  41.38 ns per loop (mean ± std. dev. of 3 runs, 100000 loops each) (np.sum)
+----+----------+-------------+--------------+------------+
|    |   i_data |   min(secs) |   mean(secs) | function   |
|----+----------+-------------+--------------+------------|
|  0 |        0 | 1.60449e-05 |  1.6296e-05  | sum        |
|  1 |        0 | 3.09632e-06 |  3.11145e-06 | np.sum     |
|  2 |        1 | 0.00137464  |  0.00137769  | sum        |
|  3 |        1 | 7.19585e-06 |  7.24643e-06 | np.sum     |
+----+----------+-------------+--------------+------------+
Elapsed time timeit(): 5.512404441833496 s.
How often the functions achieved the minimum min(secs):
sum: 0
np.sum: 2
How often the functions achieved the minimum mean(secs):
sum: 0
np.sum: 2
```
