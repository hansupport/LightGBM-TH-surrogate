import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TARGET, SAMPLE_IDS, TRUE_SAMPLE_SCAL_PATH, PRED_SAMPLE_SCAL_PATH, TRUE_PRED_FIG_DIR

true_sample_scal = np.load(TRUE_SAMPLE_SCAL_PATH)
pred_sample_scal = np.load(PRED_SAMPLE_SCAL_PATH)

C = true_sample_scal.shape[2]

def plot_sample_oof(samples):
    for target_idx in range(C):
        target_name = TARGET[target_idx]

        true_sample = true_sample_scal[:, :, target_idx]
        pred_sample = pred_sample_scal[:, :, target_idx]

        for sample_id in samples:
            plt.figure(figsize=(9, 5))
            plt.title(f"{target_name} sample{sample_id}")
            plt.grid(True, color="0.55", linewidth=0.8)

            plt.plot(true_sample[sample_id], label="TRUE", color="black", linewidth=3.2,
                     marker="o", markersize=6, markevery=25)
            plt.plot(pred_sample[sample_id], label="LGBM", color="tab:red", linewidth=1.2,
                     marker="s", markersize=6, markevery=25)

            plt.legend()

            save_path = os.path.join(TRUE_PRED_FIG_DIR, f"true_vs_pred_sample{sample_id}_{target_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"saved: {save_path}")

plot_sample_oof(SAMPLE_IDS)