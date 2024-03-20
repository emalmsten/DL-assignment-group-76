import json

name = 'behavior_features'
input_file= f'data/{name}.json'


# If you want the full dataset
full = False

output_file = f'data/full_{name}.json' if full else f'data/simple_{name}.json'

# Define the desired families you want to keep in your new CSV
desired_families = ['BlisterLoader', 'Necurs', 'Gamaredon', 'Limerat', 'RaccoonStealer']

# Load the original JSON data
with open(input_file, 'r') as file:
    data = json.load(file)

simplified_data = []
for details in data:
    if full or details.get('Family Name') in desired_families:
        simplified_data.append({
            "name": str(details.get('SHA')),
            "features_json": str(details.get('Behavior')),
            "family_name": str(details.get('Family Name'))
        })


# Write the filtered data to a new JSON file
with open(output_file, 'w', encoding='utf-8-sig') as file:
    json.dump(simplified_data, file, indent=4)

print('Filtered JSON has been written to filtered.json')
