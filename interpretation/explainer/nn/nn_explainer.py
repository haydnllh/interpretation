from ..explainer import Explainer
from...models import NNModel

class NNExplainer(Explainer):
    """This is an abstract superclass of all neural network explainers"""
    def __init__(self, input_model):
        super().__init__(input_model)
        self.model = NNModel(input_model)