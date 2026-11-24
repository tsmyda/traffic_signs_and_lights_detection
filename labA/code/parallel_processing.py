import multiprocessing
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import cv2
from threadpoolctl import threadpool_limits


THREAD_LIMIT = None


def initialize_worker(initializer, arguments):
    global THREAD_LIMIT
    cv2.setNumThreads(1)
    THREAD_LIMIT = threadpool_limits(limits=1)
    if initializer is not None:
        initializer(*arguments)


def parallel_map(function, tasks, workers, initializer=None, initargs=()):
    if workers < 1:
        raise ValueError("Worker count must be positive")
    if workers == 1:
        initialize_worker(initializer, initargs)
        for task in tasks:
            yield function(task)
        return
    iterator = iter(tasks)
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker, initargs=(initializer, initargs))
    pending = deque()
    try:
        for _ in range(2 * workers):
            task = next(iterator, None)
            if task is None:
                break
            pending.append(executor.submit(function, task))
        while pending:
            result = pending.popleft().result()
            task = next(iterator, None)
            if task is not None:
                pending.append(executor.submit(function, task))
            yield result
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
