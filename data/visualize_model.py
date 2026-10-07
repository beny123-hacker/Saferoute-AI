import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. Load the trained regressor and feature names
model = joblib.load("risk_model_regressor.pkl")
features = joblib.load("model_features.pkl")

# 2. Extract the first decision tree from the forest
tree_to_plot = model.estimators_[0]

# 3. Plot the top levels of the decision tree
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(
    tree_to_plot,
    feature_names=features,
    max_depth=3,  # Show top 3 levels for clean readability
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("SafeRoute AI: Random Forest Decision Rules (Tree #1)", fontsize=14)
plt.tight_layout()
plt.savefig("random_forest_tree.png")
print("Saved decision tree image as 'random_forest_tree.png'. Open it in VS Code!")