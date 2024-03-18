# explicit
version = "simple"               # "simple" or "full"
file_name = "static_features"   # "behavior_features" or "static_features"
model_short_name = "bert"       # "bert" or "longformer"

# local_model = "checkpoints/bert_2024-03-18-10-28-35/checkpoint_epoch_6.pth.tar"
local_model = None              # None if you want to use the model from the Hugging Face model hub
test_only = False               # Set to True if you want to skip training and only test the model

models = {
    "bert": "bert-base-uncased",
    "longformer": "allenai/longformer-base-4096"
}

hyperparams = {
    "model_name": models[model_short_name],
    "max_length": 512 if model_short_name == "bert" else 4096,
    "batch_size": 5,
    "lr": 5e-4,
    "epochs": 5,
    "training_set_size": 0.8,
    "checkpoint_frequency": 10,
}

# implicit
json_path = f'data/full_{file_name}.json' if version == "full" else f'data/simple_{file_name}.json'

