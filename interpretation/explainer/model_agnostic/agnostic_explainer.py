from abc import abstractmethod
from ..explainer import Explainer
from ...models.wrapper import wrap_model


class AgnosticExplainer(Explainer):
    """This is an abstract superclass of all model-agnostic explainers"""
    
    def __init__(self, input_model):
        super().__init__(input_model)
        self.model = wrap_model(input_model)
    
    @abstractmethod
    def explain(self):
        """
        Must be implemented by subclasses
        
        The core explainer logic in here.
        """
        
        pass