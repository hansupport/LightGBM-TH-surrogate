import numpy as np
import os
import csv
from sklearn.model_selection import KFold
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import TIME_DIM, FOLD, SEED, TARGET, INPUT_PATH, OUTPUT_PATH, MODEL_DIR, SCALER_DIR, PRED_SCALED_DIR, PRED_ORIG_DIR, FOLD_METRICS_PATH, SUMMARY_METRICS_PATH
from utils import add_time_feature

X = np.load(INPUT_PATH)[:,:-1] # (N, F)
Y = np.load(OUTPUT_PATH) # (N, T, C)

N = X.shape[0]
F = X.shape[1]
P = F + TIME_DIM
T = Y.shape[1]
C = Y.shape[2]

# print(X.shape)
# print(Y.shape)

kf = KFold(n_splits=FOLD, shuffle=True, random_state=SEED)

true_sample_scal = np.empty((N, T, C))
pred_sample_scal = np.empty((N, T, C))
true_sample_orig = np.empty((N, T, C))
pred_sample_orig = np.empty((N, T, C))

hold_out_eval_rmse = np.empty((N, C))
hold_out_eval_mae = np.empty((N, C))
hold_out_eval_nmae = np.empty((N, C))

fold_metrics = []
summary_metrics = []

for target_idx in range(C):
    print("==================================================")
    print(f"target = {TARGET[target_idx]}")
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(N))):
        x_target_scaler = joblib.load(os.path.join(SCALER_DIR, f"target_{TARGET[target_idx]}_fold_{fold}_x_scaler.joblib"))
        y_target_scaler = joblib.load(os.path.join(SCALER_DIR, f"target_{TARGET[target_idx]}_fold_{fold}_y_scaler.joblib"))

        # train test split
        x_train_init, x_test_init = X[train_idx, :], X[test_idx, :]
        y_train_init, y_test_init = Y[train_idx, :, target_idx], Y[test_idx, :, target_idx]

        N_train, N_test = x_train_init.shape[0], x_test_init.shape[0]

        # scaling
        x_train_scal = x_target_scaler.transform(x_train_init)
        y_train_scal = y_target_scaler.transform(y_train_init.reshape(-1, 1))
        x_test_scal = x_target_scaler.transform(x_test_init)
        y_test_scal = y_target_scaler.transform(y_test_init.reshape(-1, 1))

        # time feature addtion
        # (N, F) -> (N*T, F+TIME_DIM) = (N*T, P)
        x_train_time_scal = add_time_feature(x_train_scal, T, P, TIME_DIM)
        x_test_time_scal = add_time_feature(x_test_scal, T, P, TIME_DIM)

        x_train = x_train_time_scal # (N_train * T, P)
        x_test = x_test_time_scal # (N_test * T, P)
        y_train = y_train_scal.reshape(-1) # (N_train * T, )
        y_test = y_test_scal.reshape(-1) # (N_test * T, )
        
        lgbm = joblib.load(os.path.join(MODEL_DIR, f"target_{TARGET[target_idx]}_fold_{fold}_lgbm_model.joblib"))

        true_sample_scal[test_idx, :, target_idx] = y_test.reshape(N_test, -1) # (N_test, T)
        pred_sample_scal[test_idx, :, target_idx] = lgbm.predict(x_test).reshape(N_test, -1) # (N_test, T)

        true_sample_orig[test_idx, :, target_idx] = y_target_scaler.inverse_transform(
            true_sample_scal[test_idx, :, target_idx].reshape(-1, 1)
        ).reshape(N_test, -1) # (N_test, T)

        pred_sample_orig[test_idx, :, target_idx] = y_target_scaler.inverse_transform(
            pred_sample_scal[test_idx, :, target_idx].reshape(-1, 1)
        ).reshape(N_test, -1) # (N_test, T)
        
        eps = 1e-8
        for i in test_idx:
            hold_out_eval_rmse[i, target_idx] = np.sqrt(mean_squared_error(true_sample_scal[i, :, target_idx], pred_sample_scal[i, :, target_idx])) # (N_test, T)
            hold_out_eval_mae[i, target_idx] = mean_absolute_error(true_sample_scal[i, :, target_idx], pred_sample_scal[i, :, target_idx]) # (N_test, T)
            hold_out_eval_nmae[i, target_idx] = (
                mean_absolute_error(true_sample_orig[i, :, target_idx], pred_sample_orig[i, :, target_idx]) / (np.mean(true_sample_orig[i, :, target_idx]) + eps)
            ) * 100 # (N_test, T)

        fold_rmse_mean = np.mean(hold_out_eval_rmse[test_idx, target_idx])
        fold_rmse_median = np.median(hold_out_eval_rmse[test_idx, target_idx])
        fold_rmse_std = np.std(hold_out_eval_rmse[test_idx, target_idx])

        fold_mae_mean = np.mean(hold_out_eval_mae[test_idx, target_idx])
        fold_mae_median = np.median(hold_out_eval_mae[test_idx, target_idx])
        fold_mae_std = np.std(hold_out_eval_mae[test_idx, target_idx])

        fold_nmae_mean = np.mean(hold_out_eval_nmae[test_idx, target_idx])
        fold_nmae_median = np.median(hold_out_eval_nmae[test_idx, target_idx])
        fold_nmae_std = np.std(hold_out_eval_nmae[test_idx, target_idx])

        print(f"Fold{fold+1} {TARGET[target_idx]} RMSE mean", fold_rmse_mean)
        print(f"Fold{fold+1} {TARGET[target_idx]} RMSE median", fold_rmse_median)
        print(f"Fold{fold+1} {TARGET[target_idx]} RMSE std", fold_rmse_std)
        print(f"Fold{fold+1} {TARGET[target_idx]} MAE mean", fold_mae_mean)
        print(f"Fold{fold+1} {TARGET[target_idx]} MAE median", fold_mae_median)
        print(f"Fold{fold+1} {TARGET[target_idx]} MAE std", fold_mae_std)
        print(f"Fold{fold+1} {TARGET[target_idx]} NMAE mean", fold_nmae_mean)
        print(f"Fold{fold+1} {TARGET[target_idx]} NMAE median", fold_nmae_median)
        print(f"Fold{fold+1} {TARGET[target_idx]} NMAE std", fold_nmae_std)

        fold_metrics.append({
            "target": TARGET[target_idx],
            "fold": fold + 1,
            "rmse_mean": fold_rmse_mean,
            "rmse_median": fold_rmse_median,
            "rmse_std": fold_rmse_std,
            "mae_mean": fold_mae_mean,
            "mae_median": fold_mae_median,
            "mae_std": fold_mae_std,
            "nmae_mean": fold_nmae_mean,
            "nmae_median": fold_nmae_median,
            "nmae_std": fold_nmae_std,
        })

    print("==================================================")
    print(f"summary for 5fold")

    summary_rmse_mean = np.mean(hold_out_eval_rmse[:, target_idx])
    summary_rmse_median = np.median(hold_out_eval_rmse[:, target_idx])
    summary_rmse_std = np.std(hold_out_eval_rmse[:, target_idx])

    summary_mae_mean = np.mean(hold_out_eval_mae[:, target_idx])
    summary_mae_median = np.median(hold_out_eval_mae[:, target_idx])
    summary_mae_std = np.std(hold_out_eval_mae[:, target_idx])

    summary_nmae_mean = np.mean(hold_out_eval_nmae[:, target_idx])
    summary_nmae_median = np.median(hold_out_eval_nmae[:, target_idx])
    summary_nmae_std = np.std(hold_out_eval_nmae[:, target_idx])

    print(f"{TARGET[target_idx]} RMSE mean", summary_rmse_mean)
    print(f"{TARGET[target_idx]} RMSE median", summary_rmse_median)
    print(f"{TARGET[target_idx]} RMSE std", summary_rmse_std)

    print(f"{TARGET[target_idx]} MAE mean", summary_mae_mean)
    print(f"{TARGET[target_idx]} MAE median", summary_mae_median)
    print(f"{TARGET[target_idx]} MAE std", summary_mae_std)

    print(f"{TARGET[target_idx]} NMAE mean", summary_nmae_mean)
    print(f"{TARGET[target_idx]} NMAE median", summary_nmae_median)
    print(f"{TARGET[target_idx]} NMAE std", summary_nmae_std)
    print("==================================================")

    summary_metrics.append({
        "target": TARGET[target_idx],
        "rmse_mean": summary_rmse_mean,
        "rmse_median": summary_rmse_median,
        "rmse_std": summary_rmse_std,
        "mae_mean": summary_mae_mean,
        "mae_median": summary_mae_median,
        "mae_std": summary_mae_std,
        "nmae_mean": summary_nmae_mean,
        "nmae_median": summary_nmae_median,
        "nmae_std": summary_nmae_std,
    })

np.save(os.path.join(PRED_SCALED_DIR, "true_sample_scal"), true_sample_scal)
np.save(os.path.join(PRED_SCALED_DIR, "pred_sample_scal"), pred_sample_scal)
np.save(os.path.join(PRED_ORIG_DIR, "true_sample_orig"), true_sample_orig)
np.save(os.path.join(PRED_ORIG_DIR, "pred_sample_orig"), pred_sample_orig)

with open(FOLD_METRICS_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "target",
            "fold",
            "rmse_mean",
            "rmse_median",
            "rmse_std",
            "mae_mean",
            "mae_median",
            "mae_std",
            "nmae_mean",
            "nmae_median",
            "nmae_std",
        ]
    )
    writer.writeheader()
    writer.writerows(fold_metrics)

with open(SUMMARY_METRICS_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "target",
            "rmse_mean",
            "rmse_median",
            "rmse_std",
            "mae_mean",
            "mae_median",
            "mae_std",
            "nmae_mean",
            "nmae_median",
            "nmae_std",
        ]
    )
    writer.writeheader()
    writer.writerows(summary_metrics)