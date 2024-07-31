import json
import re
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import torch


def profiler(f):
    """
    Decorator to profile the running time and memory usage of a function.

    Args:
        f (function): The function to profile.

    Returns:
        tuple: A tuple containing:
            - Any: The return values of the profiled function.
            - dict: A dictionary containing profiling information with keys like 'time' and 'memory'.

    Usage:
        @profiler
        def my_function():
            pass

        return_values, profile = my_function()
    """

    def wrapper(*args, **kwargs):

        is_cuda = torch.cuda.is_available()

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            begin_memory_allocated = torch.cuda.memory_allocated()
            begin_memory_reserved = torch.cuda.memory_reserved()

        tracemalloc.start()

        begin_time = time.perf_counter()

        results = f(*args, **kwargs)

        end_time = time.perf_counter()

        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("traceback")
        total_size = sum(stat.size for stat in stats) / (1024**3)
        total_count = sum(stat.count for stat in stats)
        tracemalloc.stop()

        if is_cuda:
            end_memory_allocated = torch.cuda.memory_allocated()
            end_memory_reserved = torch.cuda.memory_reserved()
            max_memory_allocated = torch.cuda.max_memory_allocated()
            max_memory_reserved = torch.cuda.max_memory_reserved()
            memory_usage_allocated = end_memory_allocated - begin_memory_allocated
            memory_usage_reserved = end_memory_reserved - begin_memory_reserved

        else:
            end_memory_allocated = 0
            end_memory_reserved = 0
            max_memory_allocated = 0
            max_memory_reserved = 0
            memory_usage_allocated = 0
            memory_usage_reserved = 0

        running_time = round(end_time - begin_time, 3)

        profile = {
            "running_time": f"{running_time} seconds",
            "GPU_end_memory_allocated": f"{end_memory_allocated / (1024 ** 3):.2f} GB",
            "GPU_end_memory_reserved": f"{end_memory_reserved / (1024 ** 3):.2f} GB",
            "GPU_max_memory_allocated": f"{max_memory_allocated / (1024 ** 3):.2f} GB",
            "GPU_max_memory_reserved": f"{max_memory_reserved / (1024 ** 3):.2f} GB",
            "GPU_memory_usage_allocated": f"{memory_usage_allocated / (1024 ** 3):.2f} GB",
            "GPU_memory_usage_reserved": f"{memory_usage_reserved / (1024 ** 3):.2f} GB",
            "RAM_peak_usage": f"{total_size:.2f} GB",
            "RAM_peak_count": str(total_count),
        }

        return results, profile

    return wrapper


def experiment(experiment_name="", num_experiments=10, save_profile=True):
    """
    Decorator to run an experiment multiple times and save the results and profiles.

    Args:
        experiment_name (str): The name of the experiment.
        num_experiments (int): The number of times to run the experiment.
        save_profile (bool): Whether to save the profiles.

    Returns:
        Any: The return values of the experiment.

    Usage:
        @experiment(experiment_name="my_experiment", num_experiments=10, save_profile=True)
        def my_experiment():
            pass

        return_values = my_experiment()

    """
    assert num_experiments > 0, "num_experiments must be greater than 0"

    def decorator(f):
        def wrapper(*args, **kwargs):

            @profiler
            def run_experiment():
                return f(*args, **kwargs)

            return_values = []
            profiles = []

            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            begin_time = time.perf_counter()

            for i in range(num_experiments):
                print(f"Running experiment {i + 1}/{num_experiments}")

                return_value, profile = run_experiment()
                return_values.append(return_value)
                profiles.append(profile)

            if save_profile:
                end_time = time.perf_counter()

                output_dir = Path(rf"output/experiments/profiles")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{experiment_name}_{now}.json"

                results = {
                    "experiment_time": round(end_time - begin_time, 3),
                    "num_experiments": num_experiments,
                    "first_experiment": profiles[0],
                    "average_profile": {
                        key: sum(
                            float(re.findall(r"\d+\.?\d*", profile[key])[0])
                            for profile in profiles
                        )
                        / num_experiments
                        for key in profiles[0]
                    },
                    "profiles": profiles,
                    "return_values": return_values,
                }

                with open(output_file, "w") as fp:
                    json.dump(results, fp)

            return return_values

        return wrapper

    return decorator
