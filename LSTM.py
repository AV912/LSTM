import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EMBED_DIM = 100
HIDDEN_DIM = 256
NUM_CLASSES = 2
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# Load IMDB dataset
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# Tokenization
tokenizer = get_tokenizer('basic_english')

# Build vocabulary
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Text pipeline
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1 if x == 'pos' else 0

# Collate function for DataLoader
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    padded_text_list = pad_sequence(text_list, batch_first=True)
    return padded_text_list, label_list, lengths

# Create DataLoader
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch)

# Define the LSTM model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out

# Initialize the model
model = SentimentLSTM(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training function
def train(dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels, _ = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, _ = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_acc += (outputs.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_acc / total_count

# Training loop
for epoch in range(NUM_EPOCHS):
    train_loss = train(train_dataloader)
    train_acc = evaluate(train_dataloader)
    test_acc = evaluate(test_dataloader)
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# Final evaluation
final_test_acc = evaluate(test_dataloader)
print(f'Final Test Accuracy: {final_test_acc:.4f}')

# Example prediction
def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        text_tensor = torch.tensor(text_pipeline(text)).unsqueeze(0).to(device)
        output = model(text_tensor)
        return "Positive" if output.argmax(1).item() == 1 else "Negative"

sample_text = "This movie was fantastic! I really enjoyed every moment of it."
print(f"Sample text: '{sample_text}'")
print(f"Predicted sentiment: {predict_sentiment(sample_text)}")