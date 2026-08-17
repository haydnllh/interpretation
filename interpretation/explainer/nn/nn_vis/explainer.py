from ..nn_explainer import NNExplainer

class NNVis(NNExplainer):
    """Visualise learned features of neural networks via optimisation.""" 
    def __init__(self, input_model):
        super().__init__(input_model)
    
    def visualise(
        self,
        layer_identifier: int | str
    ):
        pass