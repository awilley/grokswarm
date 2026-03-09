"""
task_queue.py -- Async Task Queue with Worker Pool

A functional async task queue with DELIBERATE BUGS:
  1. Race condition: self.completed_count is incremented without a lock
  2. Race condition: self.results dict is mutated from multiple workers without a lock
  3. Bug: tasks can be added after shutdown() is called

These bugs cause intermittent failures under concurrent workloads.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    coro_fn: Callable[..., Coroutine]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: int = 0  # Higher = higher priority
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority > other.priority  # Higher priority first


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str = ""
    duration: float = 0.0


class AsyncTaskQueue:
    """Async task queue with configurable worker pool.

    Supports priority-based scheduling, timeouts, and callbacks.
    """

    def __init__(self, num_workers: int = 3, task_timeout: float = 30.0):
        self.num_workers = num_workers
        self.task_timeout = task_timeout
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._tasks: dict[str, Task] = {}
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._shutdown = False  # BUG 3: this flag exists but is never checked in add_task

        # BUG 1 & 2: These shared variables are accessed from multiple workers
        # without any synchronization primitive
        self.completed_count = 0
        self.failed_count = 0
        self.results: dict[str, TaskResult] = {}

        # Callbacks
        self._on_complete: Callable[[TaskResult], None] | None = None
        self._on_error: Callable[[TaskResult], None] | None = None

    def on_complete(self, callback: Callable[[TaskResult], None]):
        """Register a callback for completed tasks."""
        self._on_complete = callback

    def on_error(self, callback: Callable[[TaskResult], None]):
        """Register a callback for failed tasks."""
        self._on_error = callback

    async def add_task(self, task_id: str,
                       coro_fn: Callable[..., Coroutine],
                       *args,
                       priority: int = 0,
                       **kwargs) -> Task:
        """Add a task to the queue.

        BUG 3: Does not check self._shutdown -- tasks can be added after
        shutdown() is called, leading to tasks that never complete.
        """
        task = Task(
            id=task_id,
            coro_fn=coro_fn,
            args=args,
            kwargs=kwargs,
            priority=priority,
        )
        self._tasks[task_id] = task
        await self._queue.put(task)
        return task

    async def _worker(self, worker_id: int):
        """Worker loop that processes tasks from the queue."""
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

            try:
                result = await asyncio.wait_for(
                    task.coro_fn(*task.args, **task.kwargs),
                    timeout=self.task_timeout,
                )
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = time.time()

                task_result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    duration=task.completed_at - task.started_at,
                )

                # BUG 1: No lock around shared counter increment.
                # Under concurrent access, increments can be lost:
                #   worker A reads completed_count = 5
                #   worker B reads completed_count = 5
                #   worker A writes completed_count = 6
                #   worker B writes completed_count = 6  (should be 7!)
                self.completed_count += 1

                # BUG 2: No lock around shared dict mutation.
                # Multiple workers writing simultaneously can cause
                # lost updates or (in CPython) usually works but is
                # technically unsafe and wrong.
                self.results[task.id] = task_result

                if self._on_complete:
                    self._on_complete(task_result)

            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = f"Task timed out after {self.task_timeout}s"
                task.completed_at = time.time()

                task_result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=task.error,
                    duration=task.completed_at - task.started_at,
                )
                self.failed_count += 1
                self.results[task.id] = task_result

                if self._on_error:
                    self._on_error(task_result)

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()

                task_result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    duration=task.completed_at - task.started_at,
                )
                self.failed_count += 1
                self.results[task.id] = task_result

                if self._on_error:
                    self._on_error(task_result)

            finally:
                self._queue.task_done()

    async def start(self):
        """Start the worker pool."""
        self._running = True
        self._shutdown = False
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.num_workers)
        ]

    async def shutdown(self, wait: bool = True):
        """Stop accepting tasks and optionally wait for completion."""
        self._shutdown = True

        if wait:
            # Wait for queue to drain
            await self._queue.join()

        self._running = False

        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> TaskResult | None:
        """Get result for a completed task."""
        return self.results.get(task_id)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def total_tasks(self) -> int:
        return len(self._tasks)

    def get_summary(self) -> dict:
        """Get queue statistics."""
        return {
            "total": self.total_tasks,
            "pending": self.pending_count,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "results_recorded": len(self.results),
            "workers": self.num_workers,
            "running": self._running,
        }
