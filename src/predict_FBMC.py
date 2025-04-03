import os
import torch
import pandas as pd
from tqdm import tqdm
from data_loader import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
from model import *
import config
import data_loader
import train_FBMC

MODEL_NAME = 'Net'
BORDER_TYPE = 'MAXBEX'
LOSS = 'SmoothL1Loss'
TRAINING_SET = 'BL_FBMC_FULL'

PCA_COMP = train_FBMC.PCA_COMP
SEQ_LEN = train_FBMC.SEQ_LEN


if MODEL_NAME == 'LSTM':
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../model_params/{BORDER_TYPE}/{MODEL_NAME}/SEQ_LEN={SEQ_LEN}', f"{MODEL_NAME}_{TRAINING_SET}_{LOSS}_{SEQ_LEN}.pth")
    pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions_csv', f"pred_{MODEL_NAME}_{TRAINING_SET}_{LOSS}_{SEQ_LEN}.csv")
else:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../model_params/{BORDER_TYPE}/{MODEL_NAME}', f"{MODEL_NAME}_{TRAINING_SET}_{LOSS}.pth")
    pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/predictions_csv', f"pred_{MODEL_NAME}_{TRAINING_SET}_{LOSS}.csv")



full_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prep_data', f"{TRAINING_SET}.csv"), index_col=0)


# Dataset X & Y has to merged (only use intersecting timestamps), they are separated again here..
if BORDER_TYPE == "MAXBEX":
    first_target_idx = full_df.columns.get_loc("AUS_CZE")
elif BORDER_TYPE == "NTC":
    first_target_idx = full_df.columns.get_loc("AT_to_IT_NORD")
else:
    raise ValueError("Wrong BORDER_TYPE!")

X = full_df.iloc[:, :first_target_idx]
Y = full_df.iloc[:, first_target_idx:]

split_index = int(len(full_df) * train_FBMC.TRAIN_SPLIT)
X_test = X.iloc[split_index:]
Y_test = Y.iloc[split_index:]

pca = PCA(n_components=PCA_COMP)

Y_scaler = StandardScaler()

X_test = pca.fit_transform(X_test)
Y_test_scaled = Y_scaler.fit_transform(Y_test)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=train_FBMC.BATCH_SIZE, shuffle=False)

test_indices = range(split_index, len(full_df))
test_timestamps = full_df.index[test_indices]
test_timestamps = pd.to_datetime(test_timestamps)

sample_X = next(iter(test_loader))[0]
input_dim = sample_X.shape[-1] 

output_dim = Y_test.shape[1]

model = get_model(MODEL_NAME, input_dim, output_dim)
print(f"\nLoading model: {MODEL_NAME}\n")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

predictions = []
timestamps = []


with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="Making Predictions", leave=True)
    
    for X_batch, _ in progress_bar:

        if MODEL_NAME == "LSTM":
            X_batch = X_batch.view(X_batch.shape[0], 1, X_batch.shape[1])

        y_pred = model(X_batch)
        predictions.extend(y_pred.cpu().numpy().tolist())
        progress_bar.set_postfix(current_pred=y_pred[0, 0].item()) 

pred_numpy = np.array(predictions)
pred_real = Y_scaler.inverse_transform(pred_numpy)

border_names = Y_test.columns.to_list()
border_names = [f"{name}_pred" for name in border_names]

pred_df = pd.DataFrame(pred_real, columns=border_names)
pred_df.insert(0, "timestamp", pd.to_datetime(test_timestamps))

pred_df.to_csv(pred_path, index=False)
print("Predictions saved")
