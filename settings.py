# explicit
version = "small"
file_name = "static_features"
model = "bert"

models = {
    "bert": "bert-base-uncased",
    "longformer": "allenai/longformer-base-4096"
}

hyperparams = {
    "model_name": models[model],
    "num_labels": 3,
    "max_length": 512 if model == "bert" else 4096,
    "batch_size": 10,
    "lr": 5e-4,
    "epochs": 10,
    "training_set_size": 0.8
}

# implicit
json_path = f'data/{file_name}.json' if version != "small" else f'data/simple_{file_name}.json'

