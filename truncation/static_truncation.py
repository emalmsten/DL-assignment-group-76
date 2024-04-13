import json
from collections import defaultdict
from math import floor, log10
import random

from settings import sets

from truncation.gpt_tokenizer import GptTokenizer


trunc_size = sets["truncation_size"]
t = GptTokenizer()

# This is step 4 of the truncation pipeline
def remove_most_common_KV_pairs(features, l_amount=0.7):

    # Function to recursively count how many times the same value occurs for a key
    def recursive_bin_count(data, counts=None, start=False):
        # Initialize the counts dictionary if it's not provided
        if counts is None:
            counts = defaultdict(lambda: defaultdict(int))

        # Recursively count the occurrences of values for each key
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    recursive_bin_count(value, counts)
                elif isinstance(value, list):
                    value = str(value)
                    counts[key][value] += 1
                else:
                    counts[key][value] += 1
        # Recurse into each item in the list, which could be dictionaries
        elif isinstance(data, list) and start:
            for item in data:
                recursive_bin_count(item, counts)

        return counts

    # Function to remove key-value pairs from a dictionary
    def remove_KV_pairs(data, undesirables):
        if isinstance(data, dict):
            # Create a copy to iterate over to avoid RuntimeError for changing size during iteration
            for key in list(data.keys()):
                if key in undesirables:
                    del data[key]
                else:
                    # Recurse into nested dictionaries or lists
                    data[key] = remove_KV_pairs(data[key], undesirables)
        elif isinstance(data, list):
            # Recurse into each item in the list, which could be dictionaries
            data = [remove_KV_pairs(item, undesirables) for item in data]
        return data

    # Calculate bin counts
    bin_counts = recursive_bin_count(features, start=True)

    # Threshold for filtering
    l_threshold = len(features) * l_amount
    u_threshold = len(features)

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
    occurences = [list(occ.keys())[0] for occ in formatted_list]

    return remove_KV_pairs(features, occurences + ["Opc"])





# Step 1 of the truncation pipeline, rounding the numbers in the dictionary a specified number of decimal places
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




# Step 2 of the truncation pipeline, shortening the keys in the dictionary
def shorten_keys(obj, trunc_size=512):
    def shorten_key(key, length=3):
        # Split the key into words by both spaces and underscores, then take the first three letters of each
        if key == "e_res2":
            return key
        words = key.replace('_', ' ').replace('-', ' ').split()  # Split on spaces after replacing underscores with spaces
        return "_".join(word[:length] for word in words)

    length = 3 if trunc_size == 512 else 5
    if isinstance(obj, dict):
        # Recursively apply transformation for dictionaries
        return {shorten_key(k): shorten_keys(v, length) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Apply transformation for each element in a list
        return [shorten_keys(element, length) for element in obj]
    elif isinstance(obj, str):
        # Return the string as is if it's a string
        length = 8 if trunc_size == 512 else 12
        return shorten_key(obj, length=length)
    else:
        # Return the item as is if it's neither a dict nor a list
        return obj


# Step 3 of the truncation pipeline, truncating strings in the dictionary
def trunc_strings(obj, trunc_size=512):
    length = 15 if trunc_size == 512 else 30
    if isinstance(obj, dict):
        # Recursively apply transformation for dictionaries
        return {k: trunc_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Apply transformation for each element in a list
        return [trunc_strings(element) for element in obj]
    elif isinstance(obj, str):
        # Return the string as is if it's a string
        return obj[:length]
    else:
        # Return the item as is if it's neither a dict nor a list
        return obj


# Step 6 of the truncation pipeline, pruning entries from dictionaries or lists nested within the provided data structure
def iterative_static_truncation(features):
    """
    Recursively prunes entries from dictionaries or lists nested within the provided data structure, except for the top level.
    If a dictionary or list has more than `n` entries, each entry is removed with a probability of `t`%.
    After pruning at a level, it does not go deeper into that level.

    :param data: The dictionary or list to prune.
    :param n: The threshold number of entries above which pruning should occur.
    :param t: The probability (as a percentage) of removing each entry.
    """
    def prune_entries(data, n, t):
        def prune(data):
            if isinstance(data, dict):
                if len(data) > n:
                    # Randomly select keys to remove
                    keys_to_remove = [key for key in data if random.randint(1, 100) <= t]
                    for key in keys_to_remove:
                        del data[key]
                    return  # Stop going deeper once pruning is done at this level
                else:
                    for value in data.values():
                        prune(value)
            elif isinstance(data, list):
                if len(data) > n:
                    # Randomly select indices to remove
                    indices_to_remove = [i for i in range(len(data)) if random.randint(1, 100) <= t]
                    # Remove the items in reverse order to avoid changing the indices of the items to remove
                    for i in sorted(indices_to_remove, reverse=True):
                        del data[i]
                    return  # Stop going deeper once pruning is done at this level
                else:
                    for item in data:
                        prune(item)

        # Start pruning from the second level to avoid pruning the top level
        for key in data:
            prune(data[key])

    # Prune entries from the features
    for i, feature in enumerate(features):
        if i % 10000 == 0:
            print(f"Processing entry {i}/{len(features)}")
        while True:
            # Calculate the number of tokens in the feature
            tokens = t.num_tokens_from_string(json.dumps(feature, separators=(',', ':')).replace('"', ''))
            if tokens <= trunc_size:
                break

            # Calculate the percentage of tokens that need to be pruned
            percentage_missing = 100 * (tokens - trunc_size) / tokens

            # Prune 5% if the percentage missing is less than 10%, otherwise half of the percentage missing
            prune_percentage = max(percentage_missing / 2, 5)
            prune_entries(feature, 3, prune_percentage)

            if tokens <= trunc_size:
                break


def print_token_info(data):
    data_strings = [json.dumps(feature, separators=(',', ':')) for feature in data]
    data_strings = [feature.replace('"', '') for feature in data_strings]
    tokens = [t.num_tokens_from_string(feature) for feature in data_strings]

    min_tokens = min(tokens)
    max_tokens = max(tokens)
    avg_tokens = round(sum(tokens) / len(tokens), 3)
    tokens_above_trunc = [token for token in tokens if token > trunc_size]

    print(f"Min tokens: {min_tokens}, Max tokens: {max_tokens}, Avg tokens: {avg_tokens}", f"Tokens above {trunc_size}: {len(tokens_above_trunc)}")
    # print max token entry
    max_token_entry = data[tokens.index(max_tokens)]
    print(json.dumps(max_token_entry, separators=(',', ':')))
    return tokens_above_trunc

def truncate_static():
    name = "static_features"
    input_file = f'data/{name}.json'
    output_file = f'data/truncated_{name}_to_{trunc_size}_small.json'

    with open(input_file, 'r') as file:
        data = json.load(file)
        print(f"Loaded {len(data)} entries")

        # for all entries in the json list, take the features
        features = [json.loads(v["features_json"]) for k, v in data.items()]

        print("\nLoaded features")
        print_token_info(features)

        features = [round_numbers(feature) for feature in features]
        print("\nRounded numbers")
        print_token_info(features)

        features = shorten_keys(features, trunc_size)
        print("\nShortened keys")
        print_token_info(features)

        features = trunc_strings(features, trunc_size)
        print("\nTruncated strings")
        print_token_info(features)

        l_thres = 0.7 if trunc_size == 512 else 0.9
        features = remove_most_common_KV_pairs(features, l_thres)
        print("\nRemoved undesirable keys")
        print_token_info(features)

        iterative_static_truncation(features)
        print("\nIterative truncation")
        print_token_info(features)

        features_strings = [json.dumps(feature, separators=(',', ':')).replace('"', '') for feature in features]
        for i, (k,v) in enumerate(data.items()):
            v["features_json"] = features_strings[i]

        with open(output_file, 'w') as file:
            json.dump(data, file, indent=4)




