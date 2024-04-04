import json
from collections import defaultdict

from token_analyzer.gpt_tokenizer import GptTokenizer
from math import floor, log10


def binning(features):
    # Calculate bin counts
    bin_counts = recursive_bin_count(features, start=True)

    # Threshold for filtering
    l_threshold = 300
    u_threshold = len(features)

    # Filter the data
    filtered_data = {}
    for key, sub_dict in bin_counts.items():
        # Use dictionary comprehension to filter out entries below the threshold
        filtered_sub_dict = {sub_key: value for sub_key, value in sub_dict.items() if l_threshold <= value <= u_threshold}
        if filtered_sub_dict:
            filtered_data[key] = filtered_sub_dict

    bin_counts = filtered_data
    # The filtered_data dictionary contains only the items with values above 300
    sorted_list_with_values = sorted(bin_counts.items(), key=lambda x: list(x[1].values())[0], reverse=True)

    # Transform the sorted list into the desired format
    formatted_list = [{k: list(v.values())[0]} for k, v in sorted_list_with_values]
    return formatted_list

    # Convert the result to a regular dict for better readability


# Function to recursively count bin values
def recursive_bin_count(data, counts=None, start=False):
    if counts is None:
        counts = defaultdict(lambda: defaultdict(int))

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "Opcode-Occurrence":
                continue

            if isinstance(value, dict):
                recursive_bin_count(value, counts)
            elif isinstance(value, list):
                value = str(value)
                counts[key][value] += 1
            else:
                counts[key][value] += 1
    elif isinstance(data, list) and start:
        for item in data:
            recursive_bin_count(item, counts)

    return counts


def round_numbers(value):
    if isinstance(value, dict):  # If the value is a dictionary
        return {key: round_numbers(val) for key, val in value.items()}
    elif isinstance(value, list):  # If the value is a list
        return [round_numbers(item) for item in value]
    elif isinstance(value, float) or isinstance(value, int):  # If the value is a number
        rounded_value = round(value, 3 - int(floor(log10(abs(value)))) - 1) if value != 0 else 0
        # If the rounded value is an integer, convert it to int to remove trailing .0
        return rounded_value
    else:
        return value

def remove_undesirable_keys(data, undesirables):
    if isinstance(data, dict):
        # Create a copy to iterate over to avoid RuntimeError for changing size during iteration
        for key in list(data.keys()):
            if key in undesirables:
                del data[key]
            else:
                # Recurse into nested dictionaries or lists
                data[key] = remove_undesirable_keys(data[key], undesirables)
    elif isinstance(data, list):
        # Recurse into each item in the list, which could be dictionaries
        data = [remove_undesirable_keys(item, undesirables) for item in data]
    return data


def shorten_key(key):
    # Split the key into words by both spaces and underscores, then take the first three letters of each
    if key == "e_res2":
        return key
    words = key.replace('_', ' ').replace('-', ' ').split()  # Split on spaces after replacing underscores with spaces
    return "_".join(word[:3] for word in words)

def shorten_keys(obj):
    if isinstance(obj, dict):
        # Recursively apply transformation for dictionaries
        return {shorten_key(k): shorten_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Apply transformation for each element in a list
        return [shorten_keys(element) for element in obj]
    else:
        # Return the item as is if it's neither a dict nor a list
        return obj

def print_token_info(data):
    data_strings = [json.dumps(feature, separators=(',', ':')) for feature in data]
    tokens = [t.num_tokens_from_string(feature) for feature in data_strings]
    min_tokens = min(tokens)
    max_tokens = max(tokens)
    avg_tokens = round(sum(tokens) / len(tokens), 3)
    tokens_above_512 = [token for token in tokens if token > 512]
    print(f"Min tokens: {min_tokens}, Max tokens: {max_tokens}, Avg tokens: {avg_tokens}", f"Tokens above 512: {len(tokens_above_512)}")
    return tokens_above_512

t = GptTokenizer()
json_file = 'data/simple_static_features.json'
with open(json_file, 'r') as file:
    data_list = json.load(file)
    # take a subset of the data
    # for all entries in the json list, take the features
    features = [json.loads(entry['features_json']) for entry in data_list]
    print_token_info(features)

    features = [round_numbers(feature) for feature in features]
    features = remove_undesirable_keys(features, ["Opcodes"])
    features = shorten_keys(features)

    tokens_above_512 = print_token_info(features)

    occurences = binning(features)
    occ_keys = [list(occ.keys())[0] for occ in occurences]
    chunk = 5
    # remove 10 entires from the list at the time until the amount of tokens above 512 is 0
    while len(tokens_above_512) > 0 and len(occ_keys) > 0:
        undersirables = occ_keys[:chunk]
        occ_keys = occ_keys[chunk:]

        features = remove_undesirable_keys(features, undersirables)

        tokens = [t.num_tokens_from_string(json.dumps(feature, separators=(',', ':'))) for feature in features]
        tokens_above_512 = [token for token in tokens if token > 512]


    print_token_info(features)



