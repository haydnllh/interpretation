from abc import abstractmethod
from ..explainer import Explainer

class SpecificExplainer(Explainer):
    """This is an abstract superclass of all model-specific explainers"""
    
    def __init__(self, input_model):
        super().__init__(input_model)