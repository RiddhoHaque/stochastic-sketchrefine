from queue import PriorityQueue

class Heap:

    def __init__(self, is_max_heap: bool)\
        -> None:
        self.__pqueue = PriorityQueue()
        self.__is_max = is_max_heap
        self.__sum = 0

    def push(self, value: float):
        self.__sum += value
        if self.__is_max:
            value *= -1
        self.__pqueue.put(value)
    
    def pop(self) -> float:
        value = self.__pqueue.get()
        if self.__is_max:
            value *= -1
        self.__sum -= value
        return value
    
    def size(self) -> int:
        return self.__pqueue.qsize()
    
    def sum(self) -> float:
        return self.__sum