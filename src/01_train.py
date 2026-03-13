import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
from lightgbm import LGBMRegressor
import joblib

from config import TIME_DIM, FOLD, SEED, TARGET, parameter, INPUT_PATH, OUTPUT_PATH, MODEL_DIR, SCALER_DIR
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

for target_idx in range(C):
    for (fold, (train_idx, test_idx)) in tqdm(enumerate(kf.split(range(N))), total=FOLD):

        # train test split
        x_train_init, x_test_init = X[train_idx, :], X[test_idx, :]
        y_train_init, y_test_init = Y[train_idx, :, target_idx], Y[test_idx, :, target_idx]

        N_train, N_test = x_train_init.shape[0], x_test_init.shape[0]

        # scaling
        x_target_scaler, y_target_scaler = StandardScaler(), StandardScaler()
        x_train_scal = x_target_scaler.fit_transform(x_train_init)
        y_train_scal = y_target_scaler.fit_transform(y_train_init.reshape(-1, 1))
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

        # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        
        # LGBM
        lgbm = LGBMRegressor(
            n_estimators=parameter["n_estimators"],
            learning_rate=parameter["learning_rate"],
            min_data_in_leaf=parameter["min_data_in_leaf"],
            bagging_fraction=parameter["bagging_fraction"],
            num_leaves=parameter["num_leaves"],
            reg_alpha=parameter["reg_alpha"],
            reg_lambda=parameter["reg_lambda"],
            random_state=SEED,
            verbose=-1
        )

        # train
        lgbm.fit(x_train, y_train)

        joblib.dump(lgbm, os.path.join(MODEL_DIR, f"target_{TARGET[target_idx]}_fold_{fold}_lgbm_model.joblib"))
        joblib.dump(x_target_scaler, os.path.join(SCALER_DIR, f"target_{TARGET[target_idx]}_fold_{fold}_x_scaler.joblib"))
        joblib.dump(y_target_scaler, os.path.join(SCALER_DIR, f"target_{TARGET[target_idx]}_fold_{fold}_y_scaler.joblib"))