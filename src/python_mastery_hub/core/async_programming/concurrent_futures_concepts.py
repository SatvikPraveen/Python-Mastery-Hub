"""
Concurrent Futures Learning Module.

Comprehensive coverage of concurrent.futures, ThreadPoolExecutor, ProcessPoolExecutor,
and advanced parallel processing patterns.
"""

from typing import Dict, Any, List, Callable
import json


class ConcurrentFuturesConcepts:
    """Learning module for concurrent.futures concepts."""
    
    def __init__(self):
        self.name = "Concurrent Futures"
        self.description = "Master concurrent.futures for advanced parallel programming"
        self.difficulty = "advanced"
        self.topics = [
            "ThreadPoolExecutor basics",
            "ProcessPoolExecutor basics",
            "Future objects",
            "submit() vs map()",
            "Exception handling in futures",
            "Timeout handling",
            "Executor context managers"
        ]
    
    def demonstrate(self) -> Dict[str, Any]:
        """Provide comprehensive demonstration of concurrent.futures."""
        return {
            "explanation": "concurrent.futures provides a high-level interface for asynchronously executing callables using threads and processes",
            "examples": {
                "threadpool_executor": {
                    "explanation": "ThreadPoolExecutor for I/O-bound tasks",
                    "code": '''from concurrent.futures import ThreadPoolExecutor
import time

def fetch_data(item_id):
    """Simulate I/O operation."""
    time.sleep(1)
    return f"Data for {item_id}"

# Using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit tasks
    futures = [executor.submit(fetch_data, i) for i in range(5)]
    
    # Get results as they complete
    for future in futures:
        print(future.result())  # Blocks until result is available
''',
                    "output": "Data for 0\\nData for 1\\nData for 2\\nData for 3\\nData for 4"
                },
                "processpool_executor": {
                    "explanation": "ProcessPoolExecutor for CPU-bound tasks",
                    "code": '''from concurrent.futures import ProcessPoolExecutor
import time

def cpu_intensive_task(n):
    """Simulate CPU-intensive work."""
    result = sum(i * i for i in range(n))
    return result

# Using ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(cpu_intensive_task, 1000000) for _ in range(4)]
    
    for i, future in enumerate(futures):
        result = future.result()
        print(f"Task {i}: {result}")
''',
                    "output": "Task 0: 333333833333500\\nTask 1: 333333833333500\\nTask 2: 333333833333500\\nTask 3: 333333833333500"
                },
                "submit_vs_map": {
                    "explanation": "Difference between submit() and map() methods",
                    "code": '''from concurrent.futures import ThreadPoolExecutor
import time

def process_item(item):
    time.sleep(0.1)
    return item * 2

items = [1, 2, 3, 4, 5]

# Using submit() - more control
print("Using submit():")
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_item, item) for item in items]
    for i, future in enumerate(futures):
        print(f"Item {i}: {future.result()}")

# Using map() - simpler for straightforward tasks
print("\\nUsing map():")
with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(process_item, items)
    for i, result in enumerate(results):
        print(f"Item {i}: {result}")
''',
                    "output": "Using submit():\\nItem 0: 2\\nItem 1: 4\\nItem 2: 6\\nItem 3: 8\\nItem 4: 10\\n\\nUsing map():\\nItem 0: 2\\nItem 1: 4\\nItem 2: 6\\nItem 3: 8\\nItem 4: 10"
                },
                "exception_handling": {
                    "explanation": "Exception handling with futures",
                    "code": '''from concurrent.futures import ThreadPoolExecutor
import time

def risky_operation(x):
    if x == 2:
        raise ValueError(f"Error processing {x}")
    return x * 2

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(risky_operation, i) for i in range(5)]
    
    for i, future in enumerate(futures):
        try:
            result = future.result(timeout=1)
            print(f"Task {i}: {result}")
        except ValueError as e:
            print(f"Task {i}: Exception - {e}")
        except TimeoutError:
            print(f"Task {i}: Timeout")
''',
                    "output": "Task 0: 0\\nTask 1: 2\\nTask 2: Exception - Error processing 2\\nTask 3: 6\\nTask 4: 8"
                },
                "as_completed": {
                    "explanation": "Processing futures as they complete",
                    "code": '''from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

def fetch_url(url, delay):
    time.sleep(delay)
    return f"Response from {url}"

urls = ["http://api1.com", "http://api2.com", "http://api3.com"]

with ThreadPoolExecutor(max_workers=2) as executor:
    # Submit all tasks
    futures = {
        executor.submit(fetch_url, url, random.uniform(0.5, 2)): url 
        for url in urls
    }
    
    # Process results as they complete (not in submission order)
    for future in as_completed(futures):
        url = futures[future]
        result = future.result()
        print(result)
''',
                    "output": "Response from http://api1.com\\nResponse from http://api2.com\\nResponse from http://api3.com"
                },
                "timeout_handling": {
                    "explanation": "Timeout handling with result() method",
                    "code": '''from concurrent.futures import ThreadPoolExecutor
import time

def slow_task(duration):
    time.sleep(duration)
    return f"Completed after {duration}s"

with ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(slow_task, 0.5)
    future2 = executor.submit(slow_task, 3)
    
    # This succeeds (0.5s < 1s timeout)
    try:
        result = future1.result(timeout=1)
        print(result)
    except TimeoutError:
        print("Task 1 timed out")
    
    # This times out (3s > 1s timeout)
    try:
        result = future2.result(timeout=1)
        print(result)
    except TimeoutError:
        print("Task 2 timed out")
''',
                    "output": "Completed after 0.5s\\nTask 2 timed out"
                },
                "batch_processing": {
                    "explanation": "Batch processing with chunking",
                    "code": '''from concurrent.futures import ThreadPoolExecutor
import time

def process_batch(batch):
    """Process a batch of items."""
    time.sleep(0.1)
    return sum(batch)

def chunk_list(lst, n):
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

data = list(range(1, 11))  # [1, 2, 3, ..., 10]
batch_size = 3

with ThreadPoolExecutor(max_workers=2) as executor:
    batches = list(chunk_list(data, batch_size))
    futures = [executor.submit(process_batch, batch) for batch in batches]
    
    total = sum(future.result() for future in futures)
    print(f"Total sum: {total}")
    print(f"Batch results: {[f.result() for f in futures]}")
''',
                    "output": "Total sum: 55\\nBatch results: [6, 15, 24, 10]"
                }
            },
            "best_practices": [
                "Use ThreadPoolExecutor for I/O-bound tasks (network, files)",
                "Use ProcessPoolExecutor for CPU-bound tasks (calculations)",
                "Always use context managers (with statement) for executor cleanup",
                "Set appropriate max_workers based on system capabilities",
                "Handle exceptions in futures to prevent silent failures",
                "Use as_completed() for results as they arrive",
                "Set reasonable timeouts to prevent hanging",
                "Consider the GIL when using ThreadPoolExecutor for CPU work"
            ]
        }


__all__ = ["ConcurrentFuturesConcepts"]
