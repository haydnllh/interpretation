from abc import ABC, abstractmethod
from ..models.wrapper import wrap_model

class Explainer(ABC):
    """This is an abstract superclass of all explainers"""
    
    def __init__(self, input_model):
        """Initialise with wrapped model"""
        
        self.model = wrap_model(input_model)
    
    @abstractmethod
    def explain(self):
        """
        Must be implemented by subclasses
        
        The core explainer logic in here.
        """
        
        pass