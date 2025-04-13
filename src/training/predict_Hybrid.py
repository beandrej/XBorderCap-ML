import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.metrics._regression")
import torch
import pandas as pd
import numpy as np
import sys; sys.dont_write_bytecode = True
import os
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import Hybrid
from utils.train_utils import (
    prepareDataHybrid,
    loadHybridModelConfig,
    loadHybridClassMap,
    prepareTestPaths,
    getTimestamps,
    getLoader,
    splitXY,
    loadDataset,
    setSeed
    )

def main(dataset, model_name):

    setSeed()
    border_type = dataset.split('_')[1]
    timestamps = getTimestamps(dataset)
    df = loadDataset(dataset)
    _, Y = splitXY(df, border_type)
    pred_df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
    metrics = []

    for target_col in Y.columns:

        X_test, Y_test = prepareDataHybrid(dataset, target_col)
        test_loader = getLoader(X_test)

        print(f"\nPredicting border: {target_col}")

        model_path, pred_path, metrics_path, classmapping_path, model_config_path = prepareTestPaths(dataset, model_name, target_col)
        config_data = loadHybridModelConfig(model_config_path)
        if config_data is None:
            print(f"Skipping {target_col} — config not found.")
            continue

        input_dim = config_data["input_dim"]
        cls_cols = config_data["cls_cols"]
        reg_cols = config_data["reg_cols"]
        cls_dims = config_data["cls_dims"]
        hidden_dim = config_data["hidden_dim"]
        reg_count = len(reg_cols)

        model = Hybrid(input_dim=input_dim, cls_dims=cls_dims, reg_count=reg_count)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        cls_preds, reg_preds = [], []

        with torch.no_grad():
            for batch in test_loader:
                X_batch = batch[0]
                cls_outs, reg_outs = model(X_batch)

                if target_col in cls_cols:
                    class_map = loadHybridClassMap(target_col, classmapping_path)
                    col_idx = cls_cols.index(target_col)
                    preds_idx = cls_outs[col_idx].argmax(dim=1).cpu().numpy()
                    preds = [float(class_map.get(str(idx), str(idx))) for idx in preds_idx]
                    cls_preds.extend(preds)

                if target_col in reg_cols:
                    col_idx = reg_cols.index(target_col)
                    preds = reg_outs[col_idx].cpu().numpy()
                    reg_preds.extend(preds)

        # Save predictions to DataFrame
        pred_df = pred_df.iloc[:len(cls_preds) if cls_preds else len(reg_preds)]
        if cls_preds:
            pred_df[f"{target_col}_pred"] = cls_preds
        elif reg_preds:
            pred_df[f"{target_col}_pred"] = np.array(reg_preds).flatten()

        # === Metrics ===
        ground_truth = Y_test[target_col].values[:len(pred_df)]
        predictions = (
            np.array(cls_preds)
            if target_col in cls_cols
            else np.array(reg_preds).flatten()
        )

        mae = mean_absolute_error(ground_truth, predictions)

        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]

        if target_col in reg_cols:
            predictions = np.array(predictions, dtype=np.float64)
            ground_truth = np.array(ground_truth, dtype=np.float64)

            r2 = r2_score(ground_truth, predictions)
            mae = mean_absolute_error(ground_truth, predictions)
            acc = 0.0
        else:
            r2 = 0.0
            acc = np.mean(np.array(predictions) == np.array(ground_truth))


        metrics.append({
            "border": target_col,
            "r2": r2,
            "mae": mae,
            "accuracy": acc
        })

        print(f"{target_col}: R² = {r2:.4f} | MAE = {mae:.2f} | Accuracy = {acc:.2f}")

    # --- SAVE ---
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pred_df.to_csv(pred_path, index=False)
    print(f"\nAll predictions saved to {pred_path}")

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    print(f"Test metrics saved to {metrics_path}")
