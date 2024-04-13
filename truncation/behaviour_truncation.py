import json
import re

from settings import sets

from truncation.gpt_tokenizer import GptTokenizer

trunc_size = sets["truncation_size"]
t = GptTokenizer()

def print_token_info(data):
    data_strings = [json.dumps(feature, separators=(',', ':')) for feature in data]
    data_strings = [feature.replace('"', '') for feature in data_strings]

    tokens = [t.num_tokens_from_string(feature) for feature in data_strings]
    min_tokens = min(tokens)
    max_tokens = max(tokens)
    avg_tokens = round(sum(tokens) / len(tokens), 3)
    tokens_above_context_size = [token for token in tokens if token > trunc_size]
    print(f"Min tokens: {min_tokens}, Max tokens: {max_tokens}, Avg tokens: {avg_tokens}", f"Tokens above {trunc_size}: {len(tokens_above_context_size)}")

    # find and print the entry with the most tokens
    max_entry = tokens.index(max(tokens))
    print(f"Max entry: {data_strings[max_entry]}")

    return tokens_above_context_size

# Step 5 of the truncation pipeline
def concat_all_duplicates(obj):

    def concat_duplicates(data_list):
        """Concatenate duplicate entries in a list."""
        groups = group_entries(data_list)
        return format_groups(groups)

    def simplify_difference(entry1, entry2):
        """Simplify the method to find a base pattern or commonality between two entries."""
        # Find common prefix
        common_prefix = ''.join(c1 for c1, c2 in zip(entry1, entry2) if c1 == c2)

        # Remove trailing non-alphanumeric characters for cleaner output
        common_prefix = common_prefix.rstrip('_.-')

        return common_prefix

    def format_groups(groups):
        formatted_list = []
        for group in groups:
            if len(group) == 1:
                # Directly add the entry if the group has only one item
                formatted_list.append(group[0])
            else:
                # For groups larger than 1, find a simplified common base or pattern
                base_pattern = simplify_difference(group[0], group[1])
                formatted_entry = f'{len(group)}x {base_pattern} '
                formatted_list.append(formatted_entry)
        return formatted_list

    def similarity_score(str1, str2):
        """Calculate similarity score based on characters in the same position."""
        length = min(len(str1), len(str2))
        match_count = sum(1 for i in range(length) if str1[i] == str2[i])
        return match_count / length if length > 0 else 0

    def group_entries(entries, threshold=0.4):
        """Group entries based on similarity score."""
        groups = []
        for entry in entries:
            placed = False
            for group in groups:
                if any(similarity_score(entry, member) >= threshold for member in group):
                    group.append(entry)
                    placed = True
                    break
            if not placed:
                groups.append([entry])
        return groups

    if isinstance(obj, dict):
        return {k: concat_all_duplicates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        if all(isinstance(element, str) for element in obj):
            return concat_duplicates(obj)
        return [concat_all_duplicates(element) for element in obj]
    else:
        return obj


# Step 3 of the truncation pipeline
def remove_long_sequences(obj):
    """Adjust the function to correctly apply truncation rules."""

    def truncate_sequence(sequence):
        """Truncate a sequence based on the specified rules."""
        if trunc_size == 512:
            mix = 1
            dig = 3
            char = 10
        else:
            mix = 4
            dig = 5
            char = 10

        # For mixed sequences of letters and numbers or pure numbers
        if re.search(r'[a-zA-Z].*\d|\d.*[a-zA-Z]', sequence):
            return sequence[-mix:]
        elif sequence.isdigit():
            return sequence[-dig:]
        # For sequences of characters
        else:
            return sequence[:char]

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

    if isinstance(obj, dict):
        return {truncate_mixed_string(k): remove_long_sequences(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_long_sequences(element) for element in obj]
    elif isinstance(obj, str):
        return truncate_mixed_string(obj)
    elif isinstance(obj, (int, float)):
        # Convert numbers to string and apply truncation
        return truncate_sequence(str(obj))
    else:
        return obj


# Step 4 of the truncation pipeline
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


# Step 2 of the truncation pipeline
def concatanate_addresses(behaviors):

    # Function to concatenate similar addresses
    def transform_entries(json_list):
        transformed_entries = []

        # Process each entry in the list
        for entry_key, paths in json_list.items():
            entry_dict = {}
            none_files = []
            # Process each adress in the list
            for path in paths:

                # remove all nones from the list
                if path is None:
                    continue

                # Splitting on both '/' and '\\' to handle different types of paths
                parts = [p for p in path.replace("\\", "/").split("/") if p and p != "f"]
                # Limiting the number of parts to 20 to avoid excessive nesting
                parts = parts[0:20]


                if len(parts) == 1 and ('/' not in path and '\\' not in path):
                    # If there's no path separator, add to none_files
                    none_files.append(parts[0])
                elif len(parts) == 0:
                    # If the path is empty, skip it
                    continue

                else:
                    # Otherwise, construct the nested dictionary structure
                    current_level = entry_dict
                    # Traverse the parts list to create the nested dictionary structure
                    for part in parts[:-1]:
                        # Check if the part is already in the current level
                        if part not in current_level:
                            # If not, add it as a new dictionary
                            current_level[part] = {}
                        # Move to the next level
                        current_level = current_level[part]
                    # Add the file to the 'f' list under the current level
                    if 'f' not in current_level:
                        current_level['f'] = []

                    # Add the file to the 'f' list under the current level
                    current_level['f'].append(parts[-1])

            # Adding the none_files list under a special 'other' key
            if none_files:
                entry_dict['oth'] = none_files

            transformed_entries.append({entry_key: entry_dict})

        return transformed_entries

    new_behaviors = []
    # Process each behavior in the list
    for i, behavior in enumerate(behaviors):
        new_behaviors.append(transform_entries(behavior))

    return new_behaviors


def find_and_delete_longest_list(json_data, l_size = 1):
    """
    Recursively finds the longest nesting containing a list of size 1 and deletes that list.
    """
    def find_longest_list(data, path=[]):
        # Initialize variables to track the longest list and its path.
        longest_list_path = []
        longest_list_depth = -1

        # Recursively search for lists of size l_size
        if isinstance(data, dict):
            for key, value in data.items():
                # Recursively search for lists in the value
                current_path = path + [key]
                # Find the longest list in the value
                list_path, list_depth = find_longest_list(value, current_path)
                if list_depth > longest_list_depth:
                    longest_list_path, longest_list_depth = list_path, list_depth
        # Check if the current data is a list of size l_size
        elif isinstance(data, list) and len(data) == l_size and isinstance(data[0], str):
            return path, len(path)  # Return the current path and depth if the list size is 1.
        return longest_list_path, longest_list_depth

    # Function to delete the target item by its path
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

# Step 1 of the truncation pipeline
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



def iterative_behaviour_truncation(behaviors):
    for i, behavior in enumerate(behaviors):
        l_size = 1
        max_l_size = 1
        tokens_old = -1

        if (i + 1) % 100 == 0:
            print(f"Truncated {i + 1} of {len(behaviors)}")

        while True:

            tokens = t.num_tokens_from_string(json.dumps(behavior, separators=(',', ':')).replace('"', ''))

            if tokens <= trunc_size:
                break

            if tokens == tokens_old:
                l_size += 1

                if l_size > max_l_size:
                    max_l_size = l_size

            tokens_old = tokens

            behavior = find_and_delete_longest_list(behavior, l_size)

def truncate_behaviour():
    name = "behavior_features"
    input_file = f'data/{name}.json'
    output_file = f'data/truncated_{name}_to_{trunc_size}.json'

    # Load the original JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)

        behaviors = [entry['Behavior'] for entry in data]

        print(f"Loaded {len(behaviors)} entries")
        print_token_info(behaviors)

        remove_empty_lists(behaviors)
        print("removed empty lists")
        print_token_info(behaviors)

        behaviors = concatanate_addresses(behaviors)
        print("Concatenated adresses")
        print_token_info(behaviors)

        new_behaviors = []

        # convert the list of dictionaries back to a dictionary of lists
        for behavior in behaviors:
            converted_dict = {}
            for entry in behavior:
                for key, value in entry.items():
                    converted_dict[key] = value
            new_behaviors.append(converted_dict)
        behaviors = new_behaviors


        new_behaviors = []
        for i, behavior in enumerate(behaviors):
            new_behaviors.append(remove_long_sequences(behavior))
        print("adjusted truncate json")
        behaviors = new_behaviors
        print_token_info(behaviors)

        behaviors = categorize_by_extension(behaviors)
        print("categorized by extension")
        print_token_info(behaviors)

        behaviors = concat_all_duplicates(behaviors)
        print("concat all duplicates")
        print_token_info(behaviors)

        iterative_behaviour_truncation(behaviors)
        print("iterative behaviour truncation")
        print_token_info(behaviors)

        behavior_strings = [json.dumps(behavior, separators=(',', ':')).replace('"', '') for behavior in behaviors]

        # in the data file, replace the old behavior with the new behavior
        for i, entry in enumerate(data):
            entry['Behavior'] = behavior_strings[i]

        with open(output_file, 'w') as file:
            json.dump(data, file, indent=4)


