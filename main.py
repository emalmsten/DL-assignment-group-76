import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from settings import hyperparams as params, version, file_name, json_path
import csv
import json


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def get_data(categories):
    with open(json_path, 'r') as file:
        data = json.load(file)

    inverted_categories = {v: k for k, v in categories.items()}

    # for all data make a list of tuples
    data_list = []
    for details in data:
        index = inverted_categories[details.get('family_name')]
        features = details.get('features_json')
        data_list.append((features, index))

    # Split the data into a training and test set give the training set size
    split = int(len(data_list) * params["training_set_size"])
    training_set = data_list[:split]
    test_set = data_list[split:]

    return training_set, test_set


def process_data(data, tokenizer):
    data_texts, data_labels = zip(*data)
    data_encodings = tokenizer(list(data_texts), truncation=True, padding=True, max_length=params["max_length"])
    return SentimentDataset(data_encodings, list(data_labels))


def get_categories(small):
    # Replace 'your_file.csv' with the path to your CSV file
    csv_file_path = f'json_info/family_counts_for_{file_name}.csv'
    if small:
        desired_families = ['BlisterLoader', 'Necurs', 'Gamaredon', 'Limerat', 'RaccoonStealer']

    # Initialize an empty dictionary
    categories = {}

    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        # Create a CSV reader
        csv_reader = csv.DictReader(file)

        cnt = 0

        # Enumerate over the reader to keep track of the row index
        for row in csv_reader:
            # Assuming 'family_name' is the column name
            index = int(row['rank'])
            if small:
                if row['family_name'] not in desired_families:
                    continue
                index = cnt
                cnt += 1

            family_name = row['family_name']

            # Add the family name to the dictionary with the index as its key
            categories[index] = family_name

    return categories


def get_accuracy(preds, labels):
    preds_tensor = torch.tensor(preds)
    labels_tensor = torch.tensor(labels)
    correct = (preds_tensor == labels_tensor).float().sum()
    accuracy = correct / len(labels)
    return accuracy.item()  # Converts tensor to Python float


def calc_remain_time(epochs, epoch, batch_size, batch, start_time):
    elapsed_time = time.time() - start_time
    remaining_batches = ((epochs - epoch) * batch_size) + (batch_size - batch)
    remaining_batches_percent = remaining_batches / batch

    seconds = elapsed_time / remaining_batches_percent

    hours = seconds // 3600
    minutes = seconds // 60

    return f'{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds',


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", params["model_name"])
    print("Version", version)
    print("File", file_name)
    print(params)

    small_version = version == "small"
    categories = get_categories(small_version)
    num_labels = len(categories)

    # Initialize tokenizer and model
    model_name = params['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Freeze all layers except the classifier
    for param in model.base_model.parameters():
        param.requires_grad = False

    print("Model and tokenizer initialized")

    training_set, test_set = get_data(categories)
    training_set = process_data(training_set, tokenizer)
    test_set = process_data(test_set, tokenizer)

    print("Data processed")

    # Preparing the data
    train(model, training_set)
    test(model, test_set)


def train(model, training_set):
    print("Training...")
    # Training loop
    model.train()

    train_loader = DataLoader(training_set, batch_size=params["batch_size"], shuffle=True)
    print("Data loaded")
    # Training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    losses = []
    accuracies = []
    all_preds = []
    all_labels = []

    training_start_time = time.time()

    for epoch in range(params["epochs"]):
        for i, batch in enumerate(train_loader):
            remaining_time = calc_remain_time(params["epochs"], epoch, params["batch_size"], i + 1, training_start_time)

            print(
                f'\rEpoch {epoch + 1}/{params["epochs"]}, Batch {i + 1}/{len(train_loader)}, Remaining time {remaining_time}',
                end="")

            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            all_preds.extend(torch.argmax(outputs.logits, dim=1).tolist())
            all_labels.extend(labels.tolist())

            losses.append(loss.item())
            accuracies.append(get_accuracy(all_preds, all_labels))

        print(
            f'Epoch {epoch + 1}/{params["epochs"]}, average loss: {sum(losses) / len(losses)}, average accuracy: {sum(accuracies) / len(accuracies)}')


def test(model, test_set):
    # Put the model in evaluation mode
    model.eval()

    # DataLoader for test set
    test_loader = DataLoader(test_set, batch_size=params["batch_size"], shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask)

            # Convert logits to probabilities to get the predicted class (highest probability)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Calculate the accuracy
    accuracy = get_accuracy(all_preds, all_labels)
    print(f'Test Accuracy: {accuracy}')


run()