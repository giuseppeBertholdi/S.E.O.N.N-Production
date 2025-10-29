import torch
import sys
import os

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from seonn_torch.seonn_model import SEONN_Model
from seonn_torch.utils.data_loader import get_mnist_loader

def main():
    print("=== Exemplo Básico da SEONN ===")
    print("Treinamento com 10 epochs no dataset MNIST\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Parâmetros Básicos ---
    input_size = 28 * 28  # MNIST: 784 pixels
    output_size = 10      # 10 dígitos (0-9)
    initial_neurons = 1000 # Rede com neurônios suficientes para input/output
    initial_connectivity = 0.02  # 2% de conectividade inicial
    learning_rate = 1e-3
    epochs = 10           # 10 epochs como solicitado
    batch_size = 64

    print(f"Parâmetros da rede:")
    print(f"  - Neurônios iniciais: {initial_neurons}")
    print(f"  - Conectividade inicial: {initial_connectivity*100}%")
    print(f"  - Taxa de aprendizado: {learning_rate}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}\n")

    # --- Carregamento dos Dados ---
    print("Carregando dataset MNIST...")
    train_loader = get_mnist_loader(batch_size=batch_size, train=True)
    test_loader = get_mnist_loader(batch_size=batch_size, train=False)
    print(f"Dataset carregado com sucesso!\n")

    # --- Criação do Modelo ---
    print("Criando modelo SEONN...")
    model = SEONN_Model(
        input_size=input_size,
        output_size=output_size,
        initial_neurons=initial_neurons,
        initial_connectivity=initial_connectivity,
        learning_rate=learning_rate
    ).to(device)
    
    print(f"Modelo criado com {model.num_neurons} neurônios")
    print(f"Conexões iniciais: {model.weights.numel()}\n")

    # --- Loop de Treinamento ---
    print("Iniciando treinamento...")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            loss = model.train_step(data, target)
            total_loss += loss
            num_batches += 1
            
            # Mostrar progresso a cada 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1:2d}/{epochs} | Batch {batch_idx:4d} | Loss: {loss:.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1:2d}/{epochs} | Loss médio: {avg_loss:.4f}')
        print("-" * 50)

    # --- Avaliação ---
    print("\nAvaliando modelo...")
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

    accuracy = 100 * correct / total
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}%")
    print(f"Total de amostras testadas: {total}")
    print(f"Previsões corretas: {correct}")

    # --- Demonstração da Evolução ---
    print(f"\nDemonstrando evolução da rede...")
    print(f"Conexões antes da evolução: {model.weights.numel()}")
    
    # Evoluir a rede (pruning + crescimento)
    pruned_indices, new_indices = model.evolve(pruning_threshold=0.001)
    
    print(f"Conexões após evolução: {model.weights.numel()}")
    print(f"Conexões removidas: {pruned_indices.shape[1] if pruned_indices.numel() > 0 else 0}")
    print(f"Conexões adicionadas: {new_indices.shape[1] if new_indices.numel() > 0 else 0}")
    
    print("\n=== Exemplo Concluído ===")
    print("A SEONN demonstrou:")
    print("✓ Treinamento com backpropagation")
    print("✓ Evolução estrutural (pruning + crescimento)")
    print("✓ Adaptação dinâmica da topologia")

if __name__ == "__main__":
    main()
