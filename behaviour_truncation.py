import json
from token_analyzer.gpt_tokenizer import GptTokenizer

t = GptTokenizer()

name = 'behavior_features_small'
input_file= f'data/{name}.json'

def remove_jsons(data):
    data = data[:len(data) // 50]
    new_file = "data/behavior_features_small.json"
    with open(new_file, 'w') as file:
        json.dump(data, file, indent=4)


def print_token_info(data):
    data_strings = [json.dumps(feature, separators=(',', ':')) for feature in data]
    tokens = [t.num_tokens_from_string(feature) for feature in data_strings]
    min_tokens = min(tokens)
    max_tokens = max(tokens)
    avg_tokens = round(sum(tokens) / len(tokens), 3)
    tokens_above_512 = [token for token in tokens if token > 512]
    print(f"Min tokens: {min_tokens}, Max tokens: {max_tokens}, Avg tokens: {avg_tokens}", f"Tokens above 512: {len(tokens_above_512)}")
    return tokens_above_512

def remove_dup_adresses(behaviors):
    for behavior in behaviors:
        for key, value in behavior.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                # Only process lists of strings containing paths
                behavior[key] = path_to_json(value)

def path_to_json(paths):
    """
    Convert a list of file paths into a nested JSON-like structure.
    """
    root = {}
    if paths is None:
        return root

    for path in paths:
        if path is None or "\\" not in path:
            continue
        parts = path.strip("\\").split("\\")
        current_level = root
        for part in parts[:-1]:  # Go up to the second-to-last part for directories
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        # The last part is always a file, add to the current directory list
        if "f" not in current_level:
            current_level["f"] = []
        current_level["f"].append(parts[-1])
    return root

# Load the original JSON data
with open(input_file, 'r') as file:
    data = json.load(file)

    behaviors = [entry['Behavior'] for entry in data]

    print_token_info(behaviors)

    remove_dup_adresses(behaviors)
    print_token_info(behaviors)
    tokens = [t.num_tokens_from_string(json.dumps(behavior, separators=(',', ':'))) for behavior in behaviors]
    max_entry = tokens.index(max(tokens))

    print(json.dumps(behaviors[max_entry], indent=2))





