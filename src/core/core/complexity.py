"""Compute and record complexity for a class."""

import time
from functools import wraps

from core.util import create_logger

LOG = create_logger(__name__)


class ComplexityMonitor:
    def __init__(self):
        self.monitor_log = {}

    def _monitor_function(self, func):
        """Decorator to monitor function complexity."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # Log execution time for the function
            if func.__name__ not in self.monitor_log:
                self.monitor_log[func.__name__] = []
            self.monitor_log[func.__name__].append(elapsed_time)

            return result
        return wrapper

    def report(self):
        """Logs a report of monitored function complexities."""
        report_msg = ["\nComplexity Monitoring Report:"]
        for func_name, times in self.monitor_log.items():
            total_time = sum(times)
            avg_time = total_time / len(times)
            report_msg.append(
                f"{func_name}: called {len(times)} times | total execution time: {total_time:.6f}s | "
                f"average execution time: {avg_time:.6f}"
            )
        LOG.info('\n'.join(report_msg))


class BaseMonitoredClass(ComplexityMonitor):
    """Base class to automatically monitor inherited functions."""
    def __init__(self):
        super().__init__()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and not attr_name.startswith("__"):
                monitored_attr = self._monitor_function(attr)
                setattr(self, attr_name, monitored_attr)
