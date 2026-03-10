from abc import ABC, abstractmethod

class Explainer(ABC):
    """This is an abstract superclass of all explainers"""
    def __init__():
        pass
    
    @abstractmethod
    def explain(self):
        """
        Must be implemented by subclasses
        
        The core explainer logic in here.
        """
        pass