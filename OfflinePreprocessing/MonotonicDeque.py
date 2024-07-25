class Node:

    def __init__(self, index: int,
                 value: int) -> None:
        self.__index = index
        self.__value = value
        self.__next = None
        self.__prev = None
    
    def get_index(self) -> int:
        return self.__index
    
    def get_value(self) -> int:
        return self.__value
    
    def get_next(self):
        return self.__next
    
    def get_prev(self):
        return self.__prev
    
    def set_next(self, next):
        self.__next = next
    
    def set_prev(self, prev):
        self.__prev = prev

    

class MonotonicDeque:
    
    def __init__(self):
        self.__head = None
        self.__tail = None
    
    def add_node(self, index: int,
                 value: int):
        while self.__head is not None and \
            value <= self.__head.get_value():
            head = self.__head
            self.__head = self.__head.get_next()
            if self.__head is not None:
                self.__head.set_prev(None)
            del head

        node = Node(index, value)
            
        if self.__head is None:
            self.__head = node
            self.__tail = node
            return
        
        node.set_next(self.__head)
        self.__head.set_prev(node)
        self.__head = node

    def get_tail(self, max_index: int) -> Node:
        while self.__tail is not None and \
            self.__tail.get_index() > max_index:
            tail = self.__tail
            self.__tail = tail.get_prev()
            if self.__tail is not None:
                self.__tail.set_next(None)
            del tail

        if self.__tail is None:
            self.__head = None
        return self.__tail
