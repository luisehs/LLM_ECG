import os
import wfdb
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Path to PTB-XL data
data_path = 'L:/physionet.org/files/ptb-xl/1.0.3/'
sampling_rate = 100  # The sampling rate is 100 Hz in PTB-XL

# ---- Load Metadata ----
metadata = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'))

# ---- ECG Preprocessing ----
def preprocess_ecg(record_path, sampling_rate=100):
    
    #Preprocess ECG signals: FFT, basic filtering, and segmentation.
    
    # Load the WFDB signal file
    record = wfdb.rdrecord(record_path)

    # Extract the ECG signals
    ecg_signal = record.p_signal[:, 0]  # Using only the first lead for simplicity

    # Fourier Transform for frequency domain analysis
    freq_domain = fft(ecg_signal)

    return ecg_signal, freq_domain

# ---- Create a Label Mapping for 'report' column ----
# Convert string labels in the 'report' column to integers
unique_labels = metadata['report'].unique()
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# ---- Prepare the Dataset ----
signals = []
labels = []

for index, row in metadata.iterrows():
    # Path to each ECG file in WFDB format
    record_path = os.path.join(data_path, row['filename_hr'])
    
    # Preprocess the ECG signal
    ecg_signal, _ = preprocess_ecg(record_path)
    ecg_signal = np.pad(ecg_signal, (0, 5000 - len(ecg_signal)), 'constant')[:5000]  # Resize to 5000

    signals.append(ecg_signal)
    
    # Map the string label in 'report' column to an integer using the label_mapping
    labels.append(label_mapping[row['report']])

# Convert signals and labels to tensors
signals_tensor = torch.tensor(np.array(signals), dtype=torch.float32)  # Ensure it's a NumPy array first
labels_tensor = torch.tensor(labels, dtype=torch.long)  # Now labels are integers

# ---- Create a Dataset Class ----
class ECGDataset(Dataset):
    def __len__(self):
        return len(signals)

    def __getitem__(self, idx):
        # Ensure it returns both the ECG signal and the label
        signal = signals_tensor[idx]
        label = labels_tensor[idx]
        return signal, label

# ---- Define the Transformer Model for ECG ----
class ECGTransformerEncoder(nn.Module):
    def __init__(self, d_model=5000, nhead=4, num_layers=4):  # Input size adjusted to 5000
        super(ECGTransformerEncoder, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),  # Ensure batch_first is True
            num_layers=num_layers
        )
        # Add a linear layer to reduce the size to 768
        self.linear_reduce = nn.Linear(d_model, 768)

    def forward(self, ecg_embedding):
        if len(ecg_embedding.shape) == 2:  # If input has only two dimensions
            ecg_embedding = ecg_embedding.unsqueeze(1)  # Add sequence dimension (batch_size, 1, embedding_dim)
        x = self.transformer_encoder(ecg_embedding)
        x = self.linear_reduce(x)  # Reduce to 768
        return x.squeeze(1)  # Remove the sequence dimension for compatibility with BERT

# ---- Model with Bio_ClinicalBERT ----
class ECGDiagnosisModel(nn.Module):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        super(ECGDiagnosisModel, self).__init__()
        # Load pretrained Bio_ClinicalBERT model
        self.bert = AutoModel.from_pretrained(model_name)
        # Transformer encoder for ECG signals
        self.ecg_transformer = ECGTransformerEncoder()
        # Classification layer
        self.fc = nn.Linear(768, len(label_mapping))  # Number of possible diagnoses

    def forward(self, ecg_signal, attention_mask=None):
        # Pass ECG signal through the transformer
        ecg_embedding = self.ecg_transformer(ecg_signal)
        
        # Check shape of ecg_embedding (now should have two dimensions: batch_size, embedding_dim)
        batch_size, embedding_dim = ecg_embedding.shape

        # Use Bio_ClinicalBERT to transform ECG embeddings into text representations
        bert_output = self.bert(inputs_embeds=ecg_embedding.view(batch_size, 1, -1), attention_mask=attention_mask)
        
        # Final linear layer for classification
        logits = self.fc(bert_output.last_hidden_state[:, 0, :])  # [CLS] token output
        return logits

# ---- Function to Calculate Accuracy ----
def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)  # Get the index of the max log-probability
    correct = (predicted == labels).sum().item()  # Count correct predictions
    return correct

# ---- Model Training with Accuracy and Loss Tracking ----
def train_model(model, dataloader, optimizer, epochs=10):
    criterion = nn.CrossEntropyLoss()

    # Lists to store loss and accuracy per epoch
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for ecg_signals, labels in dataloader:
            optimizer.zero_grad()

            # Forward pass
            logits = model(ecg_signals)
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

            # Calculate accuracy
            correct_predictions += calculate_accuracy(logits, labels)
            total_predictions += labels.size(0)  # Total number of samples

        # Calculate and store the average loss and accuracy for this epoch
        avg_loss = running_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions * 100  # Accuracy in percentage
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return loss_history, accuracy_history

# ---- Function to Test the Model on Test Data ----
def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ecg_signals, labels in test_loader:
            # Forward pass
            logits = model(ecg_signals)
            _, predicted = torch.max(logits, 1)

            # Collect predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy using sklearn's accuracy_score
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Create the dataset and dataloader
dataset = ECGDataset()

# Split the dataset into training and test sets (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create model and optimizer
model = ECGDiagnosisModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model for 10 epochs
loss_history, accuracy_history = train_model(model, train_loader, optimizer, epochs=10)

# Test the model on the test set
test_model(model, test_loader)
