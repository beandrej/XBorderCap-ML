import config
import os
import torch
import warnings
warnings.filterwarnings("ignore", message=".*Cannot set number of intraop threads.*")
import sys; sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run():
    print(f"Pipeline start: TRAIN={config.DO_TRAIN}, PREDICT={config.DO_PREDICT}, PLOT={config.DO_PLOT}")
    assert config.DO_TRAIN and not config.DO_PREDICT or not config.DO_TRAIN and config.DO_PREDICT, "Either training or prediction should be enabled, not both."
    torch.set_num_threads(8)  # or a number that fits your CPU (like 4, 8, etc.)
    torch.set_num_interop_threads(2)  # Optional: controls inter-op parallelism

    if config.DO_TRAIN:
        for model_name in config.MODELS:
            for dataset in config.DATASETS:
                
                if model_name in ("Hybrid", "TCN"):
                    from training.train_Hybrid import main as train_main
                else:
                    from training.train_NN import main as train_main
                    
                for target_col in config.ALL_BORDERS:
                    print(f"\nTraining Model: {model_name} on {dataset} | Target: {target_col}")
                    train_main(dataset, model_name, target_col)

    if config.DO_PREDICT:
        for model_name in config.MODELS:
            for dataset in config.DATASETS:
                if model_name in ("Hybrid", "TCN"):
                    from training.predict_Hybrid import main as predict_main
                else:
                    from training.predict_NN import main as predict_main
                print(f"\nPredicting with Model: {model_name} on {dataset}\n")
                predict_main(dataset, model_name)

if __name__ == '__main__':
    run()
