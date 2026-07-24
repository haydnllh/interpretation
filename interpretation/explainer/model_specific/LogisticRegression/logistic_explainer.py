from ..specific_explainer import SpecificExplainer

class LogisticExplainer(SpecificExplainer):
    def __init__(self, input_model):
        super().__init__(input_model)