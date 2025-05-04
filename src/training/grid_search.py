if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    from sklearn.model_selection import ParameterGrid
    from train_Hybrid import main as train_main
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    import config
    from utils.train_utils import preparePaths

    param_grid = {
        'SHARED_HIDDEN': [256],
        'CLS_HIDDEN': [128],
        'DROPOUT': [0.4],
        'BATCH_SIZE': [256],
        'LEARNING_RATE': [3e-3],
        'DILATION': [[1, 3, 8, 16], [2, 8, 12, 24], [1, 4, 10, 16]]
    }

    dataset = "BL_FBMC_NORM"
    model_name = "TCNHybrid"
    border = "GER_FRA"

    results = []

    all_configs = list(ParameterGrid(param_grid))
    # valid_configs = [cfg for cfg in all_configs if cfg['KERNEL_SIZE'] <= cfg['SEQ_LEN']]
    # skipped_configs = [cfg for cfg in all_configs if cfg['KERNEL_SIZE'] > cfg['SEQ_LEN']]

    # if skipped_configs:
    #     print(f"\n⚠ Skipping {len(skipped_configs)} configs where KERNEL_SIZE > SEQ_LEN")


    for i, params in enumerate(all_configs):
        print(f"\n--- Config {i+1}/{len(all_configs)}: {params}")
        for k, v in params.items():
            setattr(config, k, v)

        try:
            train_main(dataset, model_name, border)
        except Exception as e:
            print(f"[ERROR] Skipping config due to exception: {e}")
            continue

        # Collect metrics
        from train_Hybrid import preparePaths
        _, _, metrics_path, _, _ = preparePaths(dataset, model_name, border)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        if os.path.exists(metrics_path):
            try:
                df = pd.read_csv(metrics_path)
                if 'val_mae' in df.columns:
                    best_val = df['val_mae'].min()
                else:
                    best_val = float('inf')

                result = params.copy()
                result['val_mae'] = best_val
                results.append(result)

            except Exception as e:
                print(f"[WARNING] Failed to read or write metrics: {e}")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='val_mae', ascending=True)

        output_dir = os.path.join(config.PROJECT_ROOT, "grid_results")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{model_name}_{dataset}_{border}_gridsearch.csv")
        results_df.to_csv(output_path, index=False)

        print(f"\n✅ Grid search complete. Results saved to: {output_path}")
        print("\n🏆 Best Configuration:")
        print(results_df.iloc[0])
    else:
        print("\n⚠ No results collected. Check for training errors or empty logs.")
