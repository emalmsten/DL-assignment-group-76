
# explicit
sets = {
    "file_name": "static_features",  # "behavior_features" or "static_features"
    "model_short_name": "bert",  # "bert" or "longformer"

    "local_model": None,  # None if you want to use the model from the Hugging Face model hub
    "test_only": False,  # Set to True if you want to skip training and only test the model

    "truncation_needed": False,  # Set to False iff truncation already done
    "truncation_size": 512,
}