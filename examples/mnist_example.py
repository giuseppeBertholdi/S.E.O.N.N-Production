import torch
import sys
import os

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from seonn_torch.seonn_model import SEONN_Model
from seonn_torch.utils.data_loader import get_mnist_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Parameters ---
    input_size = 28 * 28  # MNIST image size
    output_size = 10  # 10 digits
    initial_neurons = 800 # Adjusted to accommodate input/output neurons
    initial_connectivity = 0.01 # Adjusted for smaller network
    learning_rate = 1e-3
    epochs = 1 # Keep at 1 for fastest possible run
    batch_size = 128

    # --- Data Loaders ---
    train_loader = get_mnist_loader(batch_size=batch_size, train=True)
    test_loader = get_mnist_loader(batch_size=batch_size, train=False)

    # --- Model ---
    model = SEONN_Model(
        input_size=input_size,
        output_size=output_size,
        initial_neurons=initial_neurons,
        initial_connectivity=initial_connectivity,
        learning_rate=learning_rate
    ).to(device)

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            loss = model.train_step(data, target)
            total_loss += loss
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')
        print(f'Epoch {epoch} Average Loss: {total_loss / len(train_loader):.4f}')

    # --- Evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    main()
