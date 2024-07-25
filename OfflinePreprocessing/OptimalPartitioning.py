from OfflinePreprocessing.MonotonicDeque import MonotonicDeque


class OptimalPartitioning:

    @staticmethod
    def get_next_divisions(distances: list[float, int],
                           diameter: float) -> list[int]:
        tuple_count = len(distances)
        next_division = [_+1 for _ in range(tuple_count)]
        index = tuple_count - 1
        max_index = tuple_count
        deque = MonotonicDeque()
        deque.add_node(tuple_count, 0)
        while index >= 0:
            while distances[max_index-1][0] > distances[index][0]\
                + diameter:
                max_index -= 1
            tail = deque.get_tail(max_index)
            next_division[index] = tail.get_index()
            deque.add_node(index, tail.get_value()+1)
            index -= 1
        return next_division