from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from interpretation.explainer.agnostic.counterfactual import CounterfactualExplainer

X, y = make_regression(
    n_samples=200,
    n_features=4,
    noise=0.1,
    random_state=42
)

model = LinearRegression()
model.fit(X, y)

explainer = CounterfactualExplainer(
    input_model=model,
    input_data=X
)

desired_y = 1.0
result = explainer.explain(
    X[0],
    desired_y=desired_y,
    method="wachter"
)

print(
f"""
    The counterfactual result that produces the value {desired_y} is: 
    Original: {X[0]}, Prediction: {model.predict(X[:1])}
    Counterfactual: {result}, Prediction: {model.predict(result[None, :])}
"""
)