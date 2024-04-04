import copy
import json
from token_analyzer.gpt_tokenizer import GptTokenizer
import re


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
    # replace all " in the strings
    data_strings = [feature.replace('"', '') for feature in data_strings]

    tokens = [t.num_tokens_from_string(feature) for feature in data_strings]
    min_tokens = min(tokens)
    max_tokens = max(tokens)
    avg_tokens = round(sum(tokens) / len(tokens), 3)
    tokens_above_512 = [token for token in tokens if token > 512]
    print(f"Min tokens: {min_tokens}, Max tokens: {max_tokens}, Avg tokens: {avg_tokens}", f"Tokens above 512: {len(tokens_above_512)}")

    # find and print the entry with the most tokens
    max_entry = tokens.index(max(tokens))
    print(f"Max entry: {data_strings[max_entry]}")

    return tokens_above_512

def remove_dup_adresses(behaviors):
    for behavior in behaviors:
        for key, value in behavior.items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                # Only process lists of strings containing paths
                behavior[key] = path_to_json(value)


def truncate_sequence(sequence):
    """Truncate a sequence based on the specified rules."""
    # For mixed sequences of letters and numbers or pure numbers
    if re.search(r'[a-zA-Z].*\d|\d.*[a-zA-Z]', sequence):
        return sequence[-1:]
    elif sequence.isdigit():
        return sequence[-3:]
    # For sequences of characters
    else:
        return sequence[:7]


def truncate_mixed_string(s):
    """Apply truncation rules to mixed sequences within a string."""
    # Define delimiters and compile a regex pattern for splitting
    delimiters = ' |-|_|\.'
    pattern = re.compile(f'({delimiters})')
    parts = pattern.split(s)
    new_parts = []

    for part in parts:
        # Check if part is not a delimiter
        if not re.match(pattern, part):
            # Truncate the part based on its type
            trunc_seq = truncate_sequence(part)
            if trunc_seq is not None:
                new_parts.append(trunc_seq)
        else:
            new_parts.append(part)  # Keep delimiters unchanged

    return ''.join(new_parts)


def adjust_truncate_json(obj):
    """Adjust the function to correctly apply truncation rules."""
    if isinstance(obj, dict):
        return {truncate_mixed_string(k): adjust_truncate_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [adjust_truncate_json(element) for element in obj]
    elif isinstance(obj, str):
        return truncate_mixed_string(obj)
    elif isinstance(obj, (int, float)):
        # Convert numbers to string and apply truncation
        return truncate_sequence(str(obj))
    else:
        return obj

def categorize_by_extension(data):
    # Recursive function to process each item in the dictionary
    def process(item):
        if isinstance(item, dict):
            # If the item is a dictionary, recursively process each value
            for key, value in item.items():
                item[key] = process(value)
        elif isinstance(item, list):
            # If the item is a list, check each element
            new_list = []
            for elem in item:
                # Process each element based on its type
                if isinstance(elem, (dict, list)):
                    # Recursively process dictionaries and lists
                    new_list.append(process(elem))
                elif isinstance(elem, str):
                    # Handle the string elements here
                    # Assuming we are in the special "f" list, categorize by file extension
                    categorized = {}
                    for filename in item:
                        # Splitting by '.' to separate name and extension, and handle files without extension
                        parts = filename.rsplit('.', 1)
                        if len(parts) == 2:
                            name, extension = parts
                        else:
                            # Handling filenames without an extension
                            name = parts[0]
                            extension = 'misc'
                        if extension not in categorized:
                            categorized[extension] = []
                        categorized[extension].append(name)
                    # Return the categorized dictionary to replace the original list
                    return categorized
                # If the element is not a dict, list, or string, just add it back to the list
                else:
                    new_list.append(elem)
            # If the list contained non-string elements, return the processed list
            if new_list:
                return new_list
        return item

    return process(data)



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

def find_and_delete_longest_list(json_data, l_size = 1):
    """
    Recursively finds the longest nesting containing a list of size 1 and deletes that list.
    """
    def find_longest_list(data, path=[]):
        # Initialize variables to track the longest list and its path.
        longest_list_path = []
        longest_list_depth = -1

        # Recursively search for lists of size 1.
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = path + [key]
                list_path, list_depth = find_longest_list(value, current_path)
                if list_depth > longest_list_depth:
                    longest_list_path, longest_list_depth = list_path, list_depth
        elif isinstance(data, list) and len(data) == l_size:
            return path, len(path)  # Return the current path and depth if the list size is 1.
        return longest_list_path, longest_list_depth

    def delete_by_path(data, path):
        # Navigate to the item just before the target, then delete the target.
        for step in path[:-1]:
            data = data[step]
        if path:
            del data[path[-1]]

        while path and isinstance(data, dict) and not data:
            path.pop()
            data = json_data
            for step in path[:-1]:
                data = data[step]
            if path:
                del data[path[-1]]

    # Find the longest list and its path.
    longest_list_path, _ = find_longest_list(json_data)

    # Delete the longest list if it exists.
    if longest_list_path:
        delete_by_path(json_data, longest_list_path)

    return json_data

def remove_empty_lists(data):
    """
    Removes any key-value pair where the value is an empty list from a list of dictionaries.

    :param data: List of dictionaries to be processed.
    :return: Processed list of dictionaries with no empty list values.
    """
    # Process each dictionary in the list
    for d in data:
        # Use a list of keys to remove to avoid RuntimeError for changing dictionary size during iteration
        keys_to_remove = [key for key, value in d.items() if isinstance(value, list) and not value]
        for key in keys_to_remove:
            del d[key]
    return data

# Load the original JSON data
with open(input_file, 'r') as file:
    data = json.load(file)

    behaviors = [entry['Behavior'] for entry in data]
    behaviors_deepcopy = copy.deepcopy(behaviors)

    print_token_info(behaviors)

    remove_empty_lists(behaviors)

    remove_dup_adresses(behaviors)
    print_token_info(behaviors)

    behaviors = adjust_truncate_json(behaviors)
    print_token_info(behaviors)

    behaviors = categorize_by_extension(behaviors)
    print_token_info(behaviors)

    tokens = [t.num_tokens_from_string(json.dumps(behavior, separators=(',', ':'))) for behavior in behaviors]

    behaviors_under = []
    max_l_size =0

    for i, behavior in enumerate(behaviors):
        tokens = t.num_tokens_from_string(json.dumps(behavior, separators=(',', ':')))
        l_size = 1

        while tokens > 512:
            if i % 100 == 0:
                print(f"Entry {i}")
            tokens_old = tokens
            find_and_delete_longest_list(behavior, l_size)
            tokens = t.num_tokens_from_string(json.dumps(behavior, separators=(',', ':')))
            if tokens == tokens_old:
                l_size += 1
                if l_size > max_l_size:
                    max_l_size = l_size
    print(f"Max l_size: {max_l_size}")










    # print(json.dumps(behaviors[max_entry], indent=2))
    # print("______________________________________________________")
    # print(json.dumps(behaviors_deepcopy[max_entry], indent=2))





