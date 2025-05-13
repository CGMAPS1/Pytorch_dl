import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ===== 1. Enhanced Data Generation =====
def generate_data(num_samples, seq_length):
    """Generates more complex sequence patterns"""
    X = np.random.randint(0, 2, size=(num_samples, seq_length))
    y = np.zeros(num_samples)
    for i in range(num_samples):
        # More complex pattern: 1 followed by 0 after 2 steps
        for j in range(seq_length - 2):
            if X[i,j] == 1 and X[i,j+2] == 0:
                y[i] = 1
                break
    return X, y

# 10x more data for better learning
num_samples = 10000  
seq_length = 10
X, y = generate_data(num_samples, seq_length)

# Convert to PyTorch tensors
X = torch.FloatTensor(X).unsqueeze(-1)  # Shape: (10000, 10, 1)
y = torch.LongTensor(y)

# Train-test split
train_size = int(0.8 * num_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ===== 2. Improved Model Architecture =====
class EnhancedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,  # Increased capacity
            num_layers=2,    # Stacked LSTM
            dropout=0.2,     # Regularization
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Take last timestep
        return self.fc(x)

model = EnhancedLSTM()

# ===== 3. Optimized Training Setup =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Smaller LR
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# ===== 4. Enhanced Training Loop =====
def train_model(model, X_train, y_train, epochs=15):
    model.train()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

train_model(model, X_train, y_train)

# ===== 5. Comprehensive Evaluation =====
def evaluate(model, X_test, y_test):
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Show prediction confidence distribution
    test_probs = torch.softmax(model(X_test), dim=1)
    print(f"Average confidence on correct predictions: {test_probs.max(dim=1).values.mean():.4f}")

evaluate(model, X_test, y_test)

# ===== 6. Prediction Example =====
def predict_sequence(model, sequence):
    model.eval()
    seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        output = model(seq_tensor)
        prob = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(prob).item()
        confidence = prob[0, predicted_class].item()
    
    print(f"\nSequence: {sequence}")
    print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.4f})")
    print(f"Class probabilities: {prob.squeeze().numpy().round(4)}")

# Test cases
test_seq1 = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  # Should be class 1 (1 followed by 0)
test_seq2 = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  # Should be class 0
predict_sequence(model, test_seq1)
predict_sequence(model, test_seq2)