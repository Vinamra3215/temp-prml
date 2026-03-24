"""SHAP feature importance analysis."""
import numpy as np
import matplotlib.pyplot as plt


def shap_feature_importance(model, X_train, X_test, feature_names=None, save_path=None):
    """Compute and plot SHAP values for model interpretability."""
    try:
        import shap
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(X_test[:50])
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names, show=False)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        print("Install shap: pip install shap")
