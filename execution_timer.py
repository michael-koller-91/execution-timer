import time
import timeit
import numpy as np
import pandas as pd
from tabulate import tabulate


def exec_time_auto(func, func_name="", repeat=7, maxtime=0.2, *args, **kwargs):
    """
    This is basically an equivalent of the %timeit magic. The function func is
    called repeatedly with the args and kwargs provided and the execution time
    is measured. The function is called at least 'repeat' times.
    """
    if not func_name:
        func_name = func.__name__

    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)

        return wrapped

    wrapped = wrapper(func, *args, **kwargs)

    # How many times can the function be called within 'maxtime' seconds?
    for i in range(10):
        number = 10**i
        x = timeit.timeit(wrapped, number=number)
        if x >= maxtime:
            break

    # the actual timing
    secs = np.array(timeit.repeat(wrapped, repeat=repeat, number=number)) / number

    # print results
    nsecs = secs * 1e9
    s_mean = np.mean(nsecs)
    s_std = np.std(nsecs)
    secs_mean, unit_mean = nsecs_to_unit(s_mean)
    secs_std, unit_std = nsecs_to_unit(s_std)
    print(
        "{:5.1f} {} ± {:6.2f} {} per loop (mean ± std. dev. of {} runs, {} loops each) ({})".format(
            secs_mean, unit_mean, secs_std, unit_std, repeat, number, func_name
        )
    )
    return secs


def nsecs_to_unit(nsecs):
    """
    Convert a given number nsecs of nanoseconds to a representation with a
    corresponding unit. For example, if nsecs = 12345678 nanoseconds, then
    this function returns (12.345678, 'ms').
    """
    if nsecs < 1000:
        unit = "ns"
    else:
        nsecs /= 1000
        if nsecs < 1000:
            unit = "µs"
        else:
            nsecs /= 1000
            if nsecs < 1000:
                unit = "ms"
            else:
                nsecs /= 1000
                unit = "s"
    return nsecs, unit


class Timer:
    def __init__(self) -> None:
        self.df = None
        self.functions = list()
        self.function_names = list()
        self.timeit_results = None

    def add_function(self, function, name=""):
        if name:
            self.function_names.append(name)
        else:
            self.function_names.append(function.__name__)
        self.functions.append(function)

    def timeit(
        self,
        data_generator,
        compare_results=False,
        compare_function=np.allclose,
        repeat=3,
        maxtime=0.1,
    ):
        tic = time.time()

        self.timeit_results = {
            "i_data": list(),
            "min(secs)": list(),
            "mean(secs)": list(),
            "function": list(),
        }
        self.return_values = list()

        for c, dataset in enumerate(data_generator()):
            print(f"i_data = {c}")

            if compare_results:
                for i, function in enumerate(self.functions):
                    self.return_values.append(function(*dataset))
                    if len(self.return_values) > 1:
                        if not compare_function(
                            self.return_values[0], self.return_values[i]
                        ):
                            raise ValueError(
                                f"{self.function_names[0]} and {self.function_names[i]} did not yield comparable results."
                            )

            for i, function in enumerate(self.functions):
                secs = exec_time_auto(
                    function, self.function_names[i], repeat, maxtime, *dataset
                )
                self.timeit_results["i_data"].append(c)
                self.timeit_results["min(secs)"].append(np.min(secs))
                self.timeit_results["mean(secs)"].append(np.mean(secs))
                self.timeit_results["function"].append(self.function_names[i])

        self.df = pd.DataFrame(self.timeit_results)
        t = tabulate(self.df, headers="keys", tablefmt="psql")
        print(t)

        toc = time.time()
        nsecs, unit = nsecs_to_unit((toc - tic) * 1e9)
        print(f"Elapsed time {timeit.__name__}(): {nsecs} {unit}.")

    def evaluate(self):
        l_n_run = np.unique(self.df["i_data"].to_numpy())

        d_min = dict()
        d_mean = dict()
        for func_name in self.function_names:
            d_min[func_name] = 0
            d_mean[func_name] = 0

        for n_run in l_n_run:
            df0 = self.df[self.df["i_data"] == n_run]
            arg_min_min = np.argmin(df0["min(secs)"])
            arg_min_mean = np.argmin(df0["mean(secs)"])

            d_min[df0["function"].to_list()[arg_min_min]] += 1
            d_mean[df0["function"].to_list()[arg_min_mean]] += 1

        print("How often the functions achieved the minimum min(secs):")
        for k, v in d_min.items():
            print(f"{k}: {v}")

        print("How often the functions achieved the minimum mean(secs):")
        for k, v in d_mean.items():
            print(f"{k}: {v}")
