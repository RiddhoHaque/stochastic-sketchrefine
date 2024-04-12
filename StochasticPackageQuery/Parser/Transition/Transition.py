class Transition:

    def __init__(self, trigger: chr, next_state) -> None:
        self.__trigger = trigger
        self.__next_state = next_state
    
    def get_trigger(self) -> chr:
        return self.__trigger
    
    def fires(self, char: chr) -> bool:
        if char == self.__trigger:
            return True
        return False

    def get_next_state(self):
        return self.__next_state