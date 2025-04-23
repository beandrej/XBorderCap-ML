import config
import os
import sys; sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(f"Pipeline start: TRAIN={config.DO_TRAIN}, PREDICT={config.DO_PREDICT}, PLOT={config.DO_PLOT}")

assert config.DO_TRAIN and not config.DO_PREDICT or not config.DO_TRAIN and config.DO_PREDICT, "Either training or prediction should be enabled, not both."

if config.DO_TRAIN:
    for model_name in config.MODELS:
        for dataset in config.DATASETS:

            if model_name == "Hybrid":
                from training.train_Hybrid import main as train_main
                from utils.train_utils import buildTrainValTestSet
                _, Y_all, _, _, _, _ = buildTrainValTestSet(dataset, border=None)
                for target_col in Y_all.columns:
                    print(f"\nTraining Model: {model_name} on {dataset} | Target: {target_col}")
                    train_main(dataset, model_name, target_col)
            else:
                from training.train_NN import main as train_main
                from utils.train_utils import buildTrainValTestSet
                _, Y_all, _, _, _, _ = buildTrainValTestSet(dataset, border=None)
                for target_col in Y_all.columns:
                    print(f"\nTraining Model: {model_name} on {dataset} | Target: {target_col}")
                    train_main(dataset, model_name, target_col)

if config.DO_PREDICT:
    for model_name in config.MODELS:
        for dataset in config.DATASETS:
            
            if model_name == "Hybrid":
                from training.predict_Hybrid import main as predict_main
                print(f"\nPredicting with Model: {model_name} on {dataset}\n")
                predict_main(dataset, model_name)
            else:
                from training.predict_NN import main as predict_main
                print(f"\nPredicting with Model: {model_name} on {dataset}\n")
                predict_main(dataset, model_name)

# if config.DO_PLOT:
#     for model_name in config.MODELS:
#         for dataset in config.DATASETS:
#             from utils.plot_utils import *

