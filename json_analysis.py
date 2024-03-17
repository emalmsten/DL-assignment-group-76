import json
import csv

def analyze_table(file_name):
    import csv
    from collections import Counter

    full_file_name = f'json_info/table_for_{file_name}.csv'

    # Read the previously created CSV file
    with open(full_file_name, 'r') as file:
        reader = csv.DictReader(file)
        family_counts = Counter()

        # Count the occurrence of each family name
        for row in reader:
            family_counts[row['family_name']] += 1

    # Print the amount of unique families
    print(f"Number of unique families: {len(family_counts)}")

    # Sort the family counts by count in descending order
    sorted_family_counts = sorted(family_counts.items(), key=lambda item: item[1], reverse=True)

    # Write the sorted counts to a new CSV file
    with open(f'json_info/family_counts_for_{file_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['family_name', 'count', 'rank'])

        # Write each sorted family and its count to the CSV
        for i, (family, count) in enumerate(sorted_family_counts):
            writer.writerow([family, count, i])



def make_table(file_name):
    # Load the JSON data from the file
    full_file_name = f'data/{file_name}.json'
    print(f"loading {full_file_name}")

    with open(full_file_name, 'r') as file:
        data = json.load(file)

    print("data loaded")

    # Open a new CSV file to write the data into
    with open(f'json_info/table_for_{file_name}.csv', 'w', newline='') as file:

        writer = csv.writer(file)
        # Write the header row

        # Loop over the JSON data and write each row to the CSV
        writer.writerow(['name', 'tokens', 'family_name'])

        # Inside the loop, calculate the length of embeddings and write it to the CSV along with other attributes
        for name, attributes in data.items():
            writer.writerow([name, attributes['tokens'], attributes['family_name']])

    print("done")




#name = "behavior-features"
name = "static_features"
analyze_table(name)

# write_table(name)
