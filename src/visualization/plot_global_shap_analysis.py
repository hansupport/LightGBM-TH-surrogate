import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TARGET, X_SAMPLE_SCAL_PATH, SHAP_SAMPLE_SCAL_PATH, GLOBAL_SHAP_FIG_DIR, scenario_feature_names

x_sample_scal = np.load(X_SAMPLE_SCAL_PATH)
shap_sample_scal = np.load(SHAP_SAMPLE_SCAL_PATH)

C = x_sample_scal.shape[2]
F = len(scenario_feature_names)

for target_idx in range(C):
    shap_values_scen = shap_sample_scal[:, :, target_idx, :F].reshape(-1, F)
    x_test_scen = x_sample_scal[:, :, target_idx, :F].reshape(-1, F)

    plt.figure(figsize=(8, 10))
    shap.summary_plot(
        shap_values_scen,
        x_test_scen,
        feature_names=scenario_feature_names,
        show=False
    )
    plt.tight_layout()
    save_path = os.path.join(GLOBAL_SHAP_FIG_DIR, f"shap_summary_{TARGET[target_idx]}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"saved: {save_path}")