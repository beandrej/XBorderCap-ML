import os
import sys; sys.dont_write_bytecode = True
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.train_utils import (
    getTimestamps,
    loadDataset,
    splitXY,
    prepareData,
    prepareTestLoader,
    prepareTestPaths,
    setSeed
)
from model import getModel




def main(dataset, model_name):

    """
    ************************************************************************************************************************************************************************************
                                                                            PRE-PROCESSING
    ************************************************************************************************************************************************************************************
    """

    setSeed()
    
    border_type = dataset.split('_')[1]
    timestamps = getTimestamps(dataset)
    df = loadDataset(dataset)
    _, Y = splitXY(df, border_type)
    all_preds = pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
    metrics = []

    """
    ************************************************************************************************************************************************************************************
                                                                            TEST PREDICTION
    ************************************************************************************************************************************************************************************
    """

    for border in Y.columns:

        model_path, pred_path, metrics_path = prepareTestPaths(dataset, model_name, border)

        X_test, Y_test = prepareData(dataset, border)

        test_loader, input_dim = prepareTestLoader(X_test, model_name)

        model = getModel(model_name, input_dim=input_dim, output_dim=1).to(config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
        print(f"Loaded model from: {model_path}")
        model.eval()

        Y_test = Y[[border]]
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                if model_name == "LSTM":
                    X_batch, _, lengths = batch
                    X_batch = X_batch.to(config.DEVICE)
                    lengths = lengths.to(config.DEVICE)
                    y_pred = model(X_batch, lengths)
                else:
                    (X_batch,) = batch
                    X_batch = X_batch.to(config.DEVICE)
                    y_pred = model(X_batch)

                predictions.append(y_pred.cpu().numpy())

        predictions_np = np.vstack(predictions)

        predictions_real = predictions_np
        ground_truth = Y_test.values[:len(predictions_real)].flatten()

        valid_timestamps = timestamps[:len(predictions_real)]
        all_preds = all_preds.iloc[:len(valid_timestamps)]
        all_preds[f"{border}_pred"] = predictions_real.flatten()

        r2 = r2_score(ground_truth, predictions_real.flatten())
        mae = mean_absolute_error(ground_truth, predictions_real.flatten())
        metrics.append({"border": border, "r2": r2, "mae": mae})

        print(f"{border}: R² = {r2:.4f}, MAE = {mae:.2f}")

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    all_preds.to_csv(pred_path, index=False)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    print(f"\nPredictions saved to: {pred_path}")
    print(f"Test metrics saved to: {metrics_path}")
