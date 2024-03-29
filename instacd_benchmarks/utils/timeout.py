# from multiprocessing import Process, Queue
# import time


# def run_with_timeout(func, args, timeout, kwargs=None):
#     if kwargs is None:
#         kwargs = {}
#     # Define a wrapper function to put the result in the queue
#     def wrapper_func(queue, *args, **kwargs):
#         queue.put(func(*args, **kwargs))
    
#     # Create a queue to share data between processes
#     queue = Queue()
    
#     # Create and start a process to run the function
#     process = Process(target=wrapper_func, args=(queue,) + args, kwargs=kwargs)
#     process.start()
    
#     # Wait for the function to complete or timeout
#     process.join(timeout)
    
#     if process.is_alive():
#         print("Function timed out")
#         process.terminate()
#         process.join()
#         return None  # Or any default value to indicate timeout
#     else:
#         print("Function completed within timeout")
#         # Get the result from the queue
#         return queue.get()

from multiprocessing import Pool
from multiprocessing import TimeoutError
from functools import partial

def run_with_timeout(func, args, timeout, kwargs=None):
    if kwargs is None:
        kwargs = {}
    
    # Partial function application to include args and kwargs
    partial_func = partial(func, *args, **kwargs)

    # Create a pool with a single worker process
    with Pool(1) as pool:
        # Use apply_async to execute the function asynchronously
        result_async = pool.apply_async(partial_func)
        
        try:
            # Attempt to get the result with the specified timeout
            result = result_async.get(timeout=timeout)
            print("Function completed within timeout")
            return result
        except TimeoutError:
            print("Function timed out")
            return None  # Or any default value to indicate timeout