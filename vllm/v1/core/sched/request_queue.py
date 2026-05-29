# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    MLFQ = "mlfq"


MLFQ_NUM_LEVELS = 3
MLFQ_QUANTA = (1, 2, 4)
# Skip-Join MLFQ 中 qi 的工程化阈值：按“下一轮预计 token 数”选择初始队列。
# 例如 1 token 的 decode 步进入 Q1，2 token 左右的小块进入 Q2，更大的块进入 Q3。
# 举例：decode 请求下一步通常只算 1 个 token，所以 skip-join 到 Q1；
# 长 prefill/chunked prefill 如果下一轮预计要算 4 个 token，则直接进入 Q3，
# 避免大块 prefill 把短 decode 步堵在最高优先级队列里。
MLFQ_SKIP_JOIN_THRESHOLDS = MLFQ_QUANTA
# Skip-Join MLFQ 中的 alpha：低优先级请求累计等待超过该时间后提升到 Q1。
MLFQ_STARVATION_SECONDS = 1.0


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class MLFQRequestQueue(RequestQueue):
    """Skip-Join multi-level feedback queue.

    新请求不固定进入 Q1，而是根据下一轮预计计算量 skip-join 到合适队列；
    请求消耗完当前层 quantum 后降级；等待过久的请求会提升回 Q1。
    Lower level numbers are served first; each level is FCFS.
    """

    def __init__(self) -> None:
        self._queues: list[deque[Request]] = [
            deque() for _ in range(MLFQ_NUM_LEVELS)
        ]

    @staticmethod
    def _level(request: Request) -> int:
        level = getattr(request, "mlfq_level", 0)
        return min(max(level, 0), MLFQ_NUM_LEVELS - 1)

    def add_request(self, request: Request) -> None:
        request.record_mlfq_enqueue()
        self._queues[self._level(request)].append(request)

    def pop_request(self) -> Request:
        for queue in self._queues:
            if queue:
                request = queue.popleft()
                request.record_mlfq_dequeue()
                return request
        raise IndexError("pop from empty MLFQ")

    def peek_request(self) -> Request:
        for queue in self._queues:
            if queue:
                return queue[0]
        raise IndexError("peek from empty MLFQ")

    def prepend_request(self, request: Request) -> None:
        request.record_mlfq_enqueue()
        self._queues[self._level(request)].appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in reversed(list(requests)):
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        for queue in self._queues:
            try:
                queue.remove(request)
                request.record_mlfq_dequeue()
                return
            except ValueError:
                pass
        raise ValueError("request not in MLFQ")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        for queue in self._queues:
            filtered_requests = []
            for req in queue:
                if req in requests_to_remove:
                    req.record_mlfq_dequeue()
                else:
                    filtered_requests.append(req)
            queue.clear()
            queue.extend(filtered_requests)

    def __bool__(self) -> bool:
        return any(self._queues)

    def __len__(self) -> int:
        return sum(len(queue) for queue in self._queues)

    def __iter__(self) -> Iterator[Request]:
        for queue in self._queues:
            yield from queue


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.MLFQ:
        return MLFQRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
