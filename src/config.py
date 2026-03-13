import os

# Setting
TIME_DIM = 3
FOLD = 5
SEED = 42

TARGET = {
    0: "PPS",
    1: "ZW_VESSEL",
    2: "TCRHOT",
    3: "ZWPZ"
}

parameter = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "min_data_in_leaf": 200,
    "bagging_fraction": 0.7,
    "num_leaves": 127,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

SAMPLE_IDS = list(range(20))

scenario_feature_names = [
    "01: AC power 1 on time",
    "02: AC power 1 off time",
    "03: AC power 2 on time",
    "04: AC power 2 off time",
    "05: AC power 3 on time",
    "06: AC power 3 off time",
    "07: AC power 4 on time",
    "08: AC power 4 off time",
    "09: HPSI on time",
    "10: HPSI off time",
    "11: TDAFW on time",
    "12: TDAFW off time",
    "13: MDAFW on time",
    "14: MDAFW off time",
    "15: PLPP on time",
    "16: PLPP off time",
    "17: Flag PSV Stuck",
    "18: SDS open time",
    "19: CSS on time",
    "20: CSS off time",
    "21: Recirculation off time",
    "22: Seal LOCA time",
    "23: Seal LOCA area",
    "24: ET-LOOP",
    "25: SBO-R",
    "26: SBO-S",
    "27: TSLOCA",
]

# Path
DATA_DIR = "../data"
INPUT_PATH = os.path.join(DATA_DIR, "inputs.npy")
OUTPUT_PATH = os.path.join(DATA_DIR, "outputs.npy")

OUT_DIR = "../out"
MODEL_DIR = os.path.join(OUT_DIR, "models", "lgbm")
SCALER_DIR = os.path.join(OUT_DIR, "scalers")
PRED_SCALED_DIR = os.path.join(OUT_DIR, "predictions", "scaled")
PRED_ORIG_DIR = os.path.join(OUT_DIR, "predictions", "original")
SHAP_DIR = os.path.join(OUT_DIR, "shap")
EVAL_DIR = os.path.join(OUT_DIR, "evaluations")

TRUE_SAMPLE_SCAL_PATH = os.path.join(PRED_SCALED_DIR, "true_sample_scal.npy")
PRED_SAMPLE_SCAL_PATH = os.path.join(PRED_SCALED_DIR, "pred_sample_scal.npy")
TRUE_SAMPLE_ORIG_PATH = os.path.join(PRED_ORIG_DIR, "true_sample_orig.npy")
PRED_SAMPLE_ORIG_PATH = os.path.join(PRED_ORIG_DIR, "pred_sample_orig.npy")

X_SAMPLE_SCAL_PATH = os.path.join(SHAP_DIR, "x_sample_scal.npy")
SHAP_SAMPLE_SCAL_PATH = os.path.join(SHAP_DIR, "shap_sample_scal.npy")

FOLD_METRICS_PATH = os.path.join(EVAL_DIR, "fold_metrics.csv")
SUMMARY_METRICS_PATH = os.path.join(EVAL_DIR, "summary_metrics.csv")

FIGURE_DIR = "../figures"
TRUE_PRED_FIG_DIR = os.path.join(FIGURE_DIR, "true_vs_pred")
GLOBAL_SHAP_FIG_DIR = os.path.join(FIGURE_DIR, "global_shap")
TIME_RESOLVED_SHAP_FIG_DIR = os.path.join(FIGURE_DIR, "time_resolved_shap")

for path in [
    MODEL_DIR,
    SCALER_DIR,
    PRED_SCALED_DIR,
    PRED_ORIG_DIR,
    SHAP_DIR,
    EVAL_DIR,
    FIGURE_DIR,
    TRUE_PRED_FIG_DIR,
    GLOBAL_SHAP_FIG_DIR,
    TIME_RESOLVED_SHAP_FIG_DIR,
]:
    os.makedirs(path, exist_ok=True)