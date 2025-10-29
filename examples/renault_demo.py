import torch
import sys
import os
import time

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from seonn_torch.seonn_model import SEONN_Model
from seonn_torch.utils.road_data_loader import get_road_condition_loader
from seonn_torch.utils.visualization import visualize_graph

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Parameters for the demo ---
    input_size = 25  # 5x5 pixel grid
    output_size = 4  # straight, left_turn, right_turn, obstacle
    initial_neurons = 500 # A reasonable number for visualization
    initial_connectivity = 0.05
    learning_rate = 1e-3
    epochs_sunny = 5 # Epochs for initial training
    epochs_rainy = 5 # Epochs for adaptation and continued learning
    batch_size = 16

    print("\n--- SEONN: O Cérebro Adaptativo para Veículos Renault do Futuro ---")
    print("Simulando a capacidade da SEONN de se adaptar a novas condições de estrada.\n")

    # --- Phase 1: Initial Training (Sunny Day) ---
    print("Fase 1: Treinamento Inicial (Dia Ensolarado)")
    print("A SEONN está aprendendo a dirigir em condições normais de estrada.\n")
    
    model = SEONN_Model(
        input_size=input_size,
        output_size=output_size,
        initial_neurons=initial_neurons,
        initial_connectivity=initial_connectivity,
        learning_rate=learning_rate
    ).to(device)

    sunny_day_loader = get_road_condition_loader(batch_size=batch_size, num_samples_per_type=200, noise_level=0.0)

    for epoch in range(epochs_sunny):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(sunny_day_loader):
            data, target = data.to(device), target.to(device)
            loss = model.train_step(data, target)
            total_loss += loss
        print(f'  Epoch {epoch+1}/{epochs_sunny} Average Loss: {total_loss / len(sunny_day_loader):.4f}')

    # Evaluate after sunny day training
    sunny_day_accuracy = evaluate_model(model, sunny_day_loader, device)
    print(f"Acurácia após Dia Ensolarado: {sunny_day_accuracy:.2f}%")

    # --- Visualize Initial Network ---
    print("\nVisualizando a rede após o treinamento inicial (Dia Ensolarado). Feche a janela para continuar.")
    visualize_graph(
        model.graph.indices,
        model.weights,
        model.num_neurons,
        "SEONN: Rede após Dia Ensolarado"
    )

    # --- Phase 2: Structural Adaptation (Sudden Rain) ---
    print("\nFase 2: Adaptação Estrutural (Chuva Repentina)")
    print("Condições de estrada mudam para chuva. A SEONN precisa se adaptar! Feche a janela para continuar.\n")
    
    # Simulate a sudden change in conditions that triggers evolution
    # Here, we explicitly call evolve. In a real scenario, this could be triggered by performance drop.
    pruned_indices, new_indices = model.evolve(pruning_threshold=0.005) # Stronger pruning for demonstration

    # --- Visualize Adapted Network ---
    print("Visualizando a rede após a adaptação estrutural (Chuva). Feche a janela para continuar.")
    visualize_graph(
        model.graph.indices,
        model.weights,
        model.num_neurons,
        "SEONN: Rede Adaptada para Chuva",
        pruned_indices=pruned_indices,
        new_indices=new_indices
    )

    # --- Phase 3: Continuous Learning (Driving in Rain) ---
    print("\nFase 3: Aprendizado Contínuo (Dirigindo na Chuva)")
    print("A SEONN continua a aprender e refinar seu comportamento nas novas condições.\n")

    rainy_day_loader = get_road_condition_loader(batch_size=batch_size, num_samples_per_type=200, noise_level=0.5) # Add noise for rain

    for epoch in range(epochs_rainy):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(rainy_day_loader):
            data, target = data.to(device), target.to(device)
            loss = model.train_step(data, target)
            total_loss += loss
        print(f'  Epoch {epoch+1}/{epochs_rainy} Average Loss: {total_loss / len(rainy_day_loader):.4f}')

    # Evaluate after rainy day training
    rainy_day_accuracy = evaluate_model(model, rainy_day_loader, device)
    print(f"Acurácia após Chuva: {rainy_day_accuracy:.2f}%")

    print("\nDemonstração Concluída! A SEONN se adaptou e continuou aprendendo.")
    print("Verifique os arquivos 'sunny_day_network.png' e 'rainy_day_network.png' para ver a evolução da estrutura da rede.")

if __name__ == "__main__":
    run_demo()
