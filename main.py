import csv
import json
import random
import time
import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from format_json_data import desired_families
from settings import hyperparams as params, version, file_name, json_path, model_short_name, local_model, test_only

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = f'checkpoints/{model_short_name}_{time.strftime("%Y-%m-%d-%H-%M-%S")}'


# Dataaset Class
class MalwareDataset(Dataset):
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


# Gets the data from the fromatted json file and returns a training and test set
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

    # shuffle the data
    random.shuffle(data_list)

    # Split the data into a training and test set give the training set size
    split = int(len(data_list) * params["training_set_size"])

    training_set = data_list[:split]
    test_set = data_list[split:]

    return training_set, test_set


# Processes the data into a form the transfomrer can understand and returns a MalwareDataset
def process_data(data, tokenizer):
    data_texts, data_labels = zip(*data)
    data_encodings = tokenizer(list(data_texts), truncation=True, padding=True, max_length=params["max_length"])
    return MalwareDataset(data_encodings, list(data_labels))

# gets all the labels and their corresponding index
# [(0, 'BlisterLoader'), (1, 'Necurs'), (2, 'Gamaredon'), (3, 'Limerat'), (4, 'RaccoonStealer')] for example
def get_categories(simple):
    # Replace 'your_file.csv' with the path to your CSV file
    csv_file_path = f'json_info/family_counts_for_{file_name}.csv'

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
            if simple:
                if row['family_name'] not in desired_families:
                    continue
                index = cnt
                cnt += 1

            family_name = row['family_name']

            # Add the family name to the dictionary with the index as its key
            categories[index] = family_name

    return categories


# gets the accuracy of the model
def get_accuracy(preds, labels):
    preds_tensor = torch.tensor(preds)
    labels_tensor = torch.tensor(labels)
    correct = (preds_tensor == labels_tensor).float().sum()
    accuracy = correct / len(labels)
    return accuracy.item()  # Converts tensor to Python float


# saves the model and optimizer to a checkpoint, can later be used
def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    # Ensure directory exists
    filename = os.path.join(checkpoint_dir, filename)

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)


# loads a checkpoint
def load_checkpoint(checkpoint_path, model, optimizer, device):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load the saved model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move optimizer state to device
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return model, optimizer

def run():
    print("Device:", device)
    print("Model:", params["model_name"])
    print("Version", version)
    print("File", file_name)
    print(params)

    simple_version = version == "simple"
    categories = get_categories(simple_version)
    num_labels = len(categories)

    # Initialize tokenizer and model
    model_name = params['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    # If a local model should be ran intead of the model from the Hugging Face model hub
    if local_model is not None:
        print(f"Loading local model {local_model}" )
        model, optimizer = load_checkpoint(local_model, model, optimizer, device)

    # make sure it runs on cuda
    model.to(device)

    # Freeze all layers except the classifier
    for param in model.base_model.parameters():
        param.requires_grad = False

    print("Model and tokenizer initialized")

    training_set, test_set = get_data(categories)
    training_set = process_data(training_set, tokenizer)
    test_set = process_data(test_set, tokenizer)

    print("Data processed")

    # Preparing the data
    if not test_only:
        os.makedirs(checkpoint_dir, exist_ok=True)
        train(model, optimizer, training_set)
    test(model, test_set)


def train(model, optimizer, training_set):
    print("Training...")
    # Training loop
    model.train()

    train_loader = DataLoader(training_set, batch_size=params["batch_size"], shuffle=True)
    print("Data loaded")
    # Training settings


    for epoch in range(params["epochs"]):
        losses = []
        accuracies = []
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{params["epochs"]}'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()

            all_preds.extend(torch.argmax(outputs.logits, dim=1).tolist())
            all_labels.extend(labels.tolist())

            losses.append(loss.item())
            accuracies.append(get_accuracy(all_preds, all_labels))

            optimizer.step()

        if (epoch + 1) % params['checkpoint_frequency'] == 0:
            checkpoint_filename = f"checkpoint_epoch_{epoch + 1}.pth.tar"
            save_checkpoint(model, optimizer, filename=checkpoint_filename)
            print(f"Checkpoint saved: {checkpoint_filename}")
        print()
        print(
            f'Epoch {epoch + 1}/{params["epochs"]}, average loss: {sum(losses) / len(losses)}, average accuracy: {sum(accuracies) / len(accuracies)}')

    print("Training finished")
    save_checkpoint(model, optimizer, filename="final_checkpoint.pth.tar")


def test(model, test_set):
    # Put the model in evaluation mode
    model.eval()
    print("Testing...")

    # DataLoader for test set
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False)

    all_preds = []
    all_labels = []
    losses = []
    accuracies = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Convert logits to probabilities to get the predicted class (highest probability)
            all_preds.extend(torch.argmax(outputs.logits, dim=1).tolist())
            all_labels.extend(labels.tolist())
            losses.append(outputs.loss.item())
            accuracies.append(get_accuracy(all_preds, all_labels))

    accuracy = get_accuracy(all_preds, all_labels)
    print(f'Test Accuracy: {accuracy}')


run()
