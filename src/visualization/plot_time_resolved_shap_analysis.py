import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TARGET,
    TIME_DIM,
    SAMPLE_IDS,
    TRUE_SAMPLE_SCAL_PATH,
    PRED_SAMPLE_SCAL_PATH,
    SHAP_SAMPLE_SCAL_PATH,
    TIME_RESOLVED_SHAP_FIG_DIR,
    scenario_feature_names,
)

true_sample_scal = np.load(TRUE_SAMPLE_SCAL_PATH)   # (N, T, C)
pred_sample_scal = np.load(PRED_SAMPLE_SCAL_PATH)   # (N, T, C)
shap_sample_scal = np.load(SHAP_SAMPLE_SCAL_PATH)   # (N, T, C, P)

N = true_sample_scal.shape[0]
T = true_sample_scal.shape[1]
C = true_sample_scal.shape[2]
P = shap_sample_scal.shape[3]
F = len(scenario_feature_names)

def plot_pred_and_shap_exact(sample_ids):
    base_fontsize = 16
    font_scale = 1.42
    dpi = 300
    fig_height = 13
    fig_width = 11
    top_k = 5
    title_pad = 16
    marker_scale = 1.5
    markevery_top = 25
    markevery_bottom = 25
    grid_color = "0.55"
    grid_alpha = 0.9
    grid_lw = 0.7
    zero_lw = 2.2
    zero_alpha = 0.9
    save_dir = TIME_RESOLVED_SHAP_FIG_DIR
    fname_prefix = "pred_shap"

    ylabel_map = {
        "PPS": "Pressure",
        "ZW_VESSEL": "Water Level",
        "TCRHOT": "Temperature",
        "ZWPZ": "Water Level",
    }

    fs = base_fontsize * font_scale
    plt.rcParams.update({
        "font.size": fs,
        "axes.titlesize": (base_fontsize + 2) * font_scale,
        "axes.labelsize": base_fontsize * font_scale,
        "xtick.labelsize": (base_fontsize - 1) * font_scale,
        "ytick.labelsize": (base_fontsize - 1) * font_scale,
        "legend.fontsize": (base_fontsize - 2) * font_scale,
        "legend.title_fontsize": (base_fontsize - 1) * font_scale,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
    })

    os.makedirs(save_dir, exist_ok=True)

    bottom_colors = ["C1", "C4", "C5", "C8", "C9"]
    bottom_styles = [
        {"linestyle": "-",  "marker": "^"},
        {"linestyle": "-",  "marker": None},
        {"linestyle": "-.", "marker": None},
        {"linestyle": "--", "marker": None},
        {"linestyle": ":",  "marker": None},
    ]

    true_ms = 6.5 * marker_scale
    pred_ms = 6.0 * marker_scale
    bottom_ms = 6.0 * marker_scale

    var_order = ["PPS", "ZW_VESSEL", "TCRHOT", "ZWPZ"]
    target_name_to_idx = {v: k for k, v in TARGET.items()}
    target_indices = [target_name_to_idx[v] for v in var_order]

    rmse_all = np.sqrt(np.mean((pred_sample_scal - true_sample_scal) ** 2, axis=1)) # (N, C)

    for sample_id in sample_ids:
        print(f"\n================= SAMPLE {sample_id} (all targets) =================")

        for target_idx in target_indices:
            target_name = TARGET[target_idx]
            y_label = ylabel_map.get(target_name, target_name)

            true_seq = true_sample_scal[:, :, target_idx] # (N, T)
            pred_seq = pred_sample_scal[:, :, target_idx] # (N, T)
            shap_seq_all = shap_sample_scal[:, :, target_idx, :] # (N, T, P)
            shap_seq_scenario = shap_seq_all[:, :, :F] # (N, T, F)
            shap_seq_time = shap_seq_all[:, :, F:] # (N, T, TIME_DIM)

            rmse_target = rmse_all[:, target_idx] # (N,)
            order = np.argsort(rmse_target)
            rank = np.empty(len(rmse_target), dtype=int)
            rank[order] = np.arange(len(rmse_target)) + 1
            rmse = float(rmse_target[sample_id])
            top_pct = float(rank[sample_id]) / float(len(rmse_target)) * 100.0
            print(f"target={target_name} sample={sample_id} RMSE={rmse:.4f} (top {top_pct:.2f}%)")

            shap_sample = shap_seq_scenario[sample_id] # (T, F)
            scenario_sum = shap_seq_scenario.sum(axis=2) # (N, T)
            time_sum = shap_seq_time.sum(axis=2) # (N, T)

            base_time = pred_seq[sample_id] - scenario_sum[sample_id] # (T,)

            mean_abs_local = np.mean(np.abs(shap_sample), axis=0) # (F,)
            top_indices = np.argsort(-mean_abs_local)[:top_k]

            fig, axes = plt.subplots(
                2, 1,
                figsize=(fig_width, fig_height),
                dpi=dpi,
                sharex=True,
                gridspec_kw={"height_ratios": [1.2, 2.0]}
            )

            t = np.arange(true_seq.shape[1])

            ax0 = axes[0]
            ax0.plot(
                t, true_seq[sample_id],
                label="TRUE",
                color="black",
                linewidth=5.2,
                marker="o",
                markersize=true_ms,
                markevery=markevery_top,
            )
            ax0.plot(
                t, pred_seq[sample_id],
                label="PRED",
                color="C3",
                linewidth=2.6,
                marker="s",
                markersize=pred_ms,
                markevery=markevery_top,
            )
            ax0.plot(
                t, base_time,
                label="BASE + TIME",
                color="C2",
                linestyle="--",
                linewidth=2.6,
            )
            ax0.set_ylabel(y_label)
            ax0.set_title(f"{target_name} sample {sample_id}", pad=title_pad)
            ax0.grid(True, linewidth=grid_lw, alpha=grid_alpha, color=grid_color)
            ax0.legend(
                loc="upper right",
                frameon=True,
                framealpha=0.9,
                borderpad=0.6,
                labelspacing=0.4,
                handlelength=2.2,
            )

            ax1 = axes[1]
            for j, feat_idx in enumerate(top_indices):
                feat_name = scenario_feature_names[feat_idx]
                color = bottom_colors[j]
                style = bottom_styles[j]

                ax1.plot(
                    t, shap_sample[:, feat_idx],
                    label=feat_name,
                    color=color,
                    linestyle=style["linestyle"],
                    linewidth=2.4,
                    marker=style["marker"],
                    markersize=bottom_ms if style["marker"] else None,
                    markevery=markevery_bottom if style["marker"] else None,
                )

            ax1.axhline(0.0, linestyle="--", linewidth=zero_lw, color="black", alpha=zero_alpha)
            ax1.set_xlabel("Time index")
            ax1.set_ylabel("SHAP value")
            ax1.grid(True, linewidth=grid_lw, alpha=grid_alpha, color=grid_color)
            ax1.legend(
                title="Feature",
                frameon=True,
                framealpha=0.9,
                borderpad=0.6,
                labelspacing=0.35,
                handlelength=2.2,
            )

            fig.tight_layout()

            save_path = os.path.join(save_dir, f"time_resolved_shap_sample{sample_id}_{TARGET[target_idx]}.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.close(fig)

plot_pred_and_shap_exact(SAMPLE_IDS)