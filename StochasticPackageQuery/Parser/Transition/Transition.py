class Transition:

    def __init__(self, trigger: chr, next_state,
                 anything_but_transition = False,
                 accept_anything_transition = False) -> None:
        self.__trigger = trigger
        self.__next_state = next_state
        self.__anything_but_transition = \
            anything_but_transition
        self.__accept_anything_transition = \
            accept_anything_transition
    
    def get_trigger(self) -> chr:
        return self.__trigger
    
    def fires(self, char: chr) -> bool:
        if self.__accept_anything_transition:
            return True
        if char == self.__trigger:
            if self.__anything_but_transition:
                return False
            return True
        if self.__anything_but_transition:
            return True
        return False

    def get_next_state(self):
        return self.__next_state