from ..nn_explainer import NNExplainer


class Adversarial(NNExplainer):
    """
    ML models may completely change their decision over small variations that are not observed by humans.
    E.g. a stop sign with a little dirt on it
    This shows how easily machine learning models are fooled
    """
    def __init__(self, input_model):
        super().__init__(input_model)
    
    def generate_adversarial(self, X, y, target_label):
        pass