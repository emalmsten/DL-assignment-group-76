from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

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

# Training set: short Harry Potter movie reviews with labels
training_set = [
    ("The pancakes were undercooked and tasteless", 0),
    ("Pancakes were dry and flavorless", 0),
    ("Dry pancakes and too hard", 0),
    ("Pancakes were burnt and unpleasant", 0),
    ("Pancakes bland, lacking syrup", 0),
    ("Disappointing pancakes, not enjoyable", 0),
    ("Pancakes soggy and undercooked", 0),
    ("Pancakes too salty and greasy", 0),
    ("Pancakes not worth the price", 0),
    ("Pancakes left a bad taste", 0),
    ("Would not recommend these pancakes", 0),
    ("Pancakes too sweet, artificial taste", 0),
    ("Unsatisfactory pancakes, bad texture", 0),
    ("Pancakes lacked freshness and flavor", 0),
    ("Poor choice, pancakes very bad", 0),
    ("Pancakes delicious and fluffy", 1),
    ("Pancakes perfectly sweet and savory", 1),
    ("Pancakes mouthwatering and satisfying", 1),
    ("Best pancakes ever, absolutely delicious", 1),
    ("Pancakes fresh, light, flavorful", 1),
    ("Good pancakes with maple syrup", 1),
    ("Highly recommend these delicious pancakes", 1),
    ("Delightful pancakes, a breakfast treat", 1),
    ("Pancakes incredibly soft and tasty", 1),
    ("Perfectly cooked, pancakes amazing", 1),
    ("Pancakes amazing texture, rich taste", 1),
    ("Pancakes full of berries, delicious", 1),
    ("Never had pancakes this good", 1),
    ("Pancakes enjoyable and memorable", 1),
    ("Pancakes exceeded expectations, fantastic", 1)
]


# Test set: sentences without labels for sentiment analysis
test_set = [
    "These pancakes were absolutely delicious",
    "Pancakes too bland, bad",
    "Amazing pancakes, texture was perfect",
    "Sadly, pancakes were too burnt",
    "Loved the pancakes, every bite was delightful",
    "Pancakes far too greasy, not enjoyable",
    "Pancakes made a perfect breakfast",
    "Disliked these pancakes, too artificial",
    "Pancakes wonderfully fluffy, so light",
    "Pancakes left much to be desired"
]


# Initialize tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Freeze all layers except the classifier
for param in model.base_model.parameters():
    param.requires_grad = False

# Preparing the data
train_texts, train_labels = zip(*training_set)
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
train_dataset = SentimentDataset(train_encodings, list(train_labels))

test_encodings = tokenizer(test_set, truncation=True, padding=True, max_length=512)
test_dataset = SentimentDataset(test_encodings)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Training settings
optimizer = AdamW(model.parameters(), lr=5e-3)
epochs = 50  # Set your epochs

# Training loop
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Prediction on test set
model.eval()
test_loader = DataLoader(test_dataset, batch_size=10)  # Batch size can be equal to test set size if it's small
predictions = []

for batch in test_loader:
    with torch.no_grad():
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())

# Mapping predictions to sentiment categories
sentiment_categories = {0: "Negative", 1: "Positive"}
predicted_sentiments = [sentiment_categories[pred] for pred in predictions]

for sentence, sentiment in zip(test_set, predicted_sentiments):
    print(f"Sentence: {sentence}, Predicted sentiment: {sentiment}")
