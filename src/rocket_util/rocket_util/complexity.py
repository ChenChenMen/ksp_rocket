"""Compute and record complexity for a class."""

import time
from functools import wraps

from rocket_util.logconfig import create_logger

LOG = create_logger(__name__)


class ComplexityMonitor:
    def __init__(self):
        self.monitor_log = {}

    def register_monitored_object(self, obj, ignored_methods=None):
        """Register an object to monitor its methods."""
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name)
            # Skip non-callable attributes and private methods
            if not callable(attr) or attr_name.startswith("__"):
                continue
            # Skip ignored methods
            if ignored_methods and attr_name in ignored_methods:
                continue
            # Register the method for monitoring
            monitored_attr = self._monitor_function(attr)
            setattr(obj, attr_name, monitored_attr)

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

    def report_complexity(self):
        """Logs a report of monitored function complexities."""
        report_msg = ["\nComplexity Monitoring Report:"]
        # Iterate through the monitored functions and their execution times
        for func_name, times in self.monitor_log.items():
            total_time = sum(times)
            avg_time = total_time / len(times)
            report_msg.append(
                f"{func_name}: called {len(times)} times | total execution time: {total_time:.6f}s | "
                f"average execution time: {avg_time:.6f}"
            )
        LOG.info('\n'.join(report_msg))


class BaseMonitoredClass:
    """Base class to automatically monitor inherited functions."""

    def __init__(self):
        """Initialize the complexity monitor."""
        self.complexity_monitor = ComplexityMonitor()
        self.complexity_monitor.register_monitored_object(self, ignored_methods=['report_complexity'])

    def report_complexity(self):
        """Report the complexity of the monitored functions."""
        self.complexity_monitor.report_complexity()
