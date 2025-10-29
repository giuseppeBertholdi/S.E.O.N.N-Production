import torch
import sys
import os
import matplotlib.pyplot as plt
import time

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from seonn_torch.seonn_model import SEONN_Model
from seonn_torch.utils.data_loader import get_mnist_loader
from seonn_torch.utils.visualization import visualize_graph

def plot_training_progress(losses, accuracies, epochs):
    """Plota o progresso do treinamento em tempo real"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot de loss
    ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o')
    ax1.set_title('Loss durante o Treinamento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(losses) * 1.1)
    
    # Plot de acurácia
    ax2.plot(epochs, accuracies, 'g-', linewidth=2, marker='s')
    ax2.set_title('Acurácia durante o Treinamento', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Acurácia (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def quick_evaluate(model, test_loader, device):
    """Avaliação rápida do modelo"""
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
    return 100 * correct / total

def main():
    print("🚀 DEMO RÁPIDO E VISUAL DA SEONN 🚀")
    print("=" * 50)
    print("Treinamento super rápido com visualizações!")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Dispositivo: {device}")

    # --- Parâmetros Otimizados para Velocidade ---
    input_size = 28 * 28  # MNIST
    output_size = 10
    initial_neurons = 800  # Rede menor para velocidade
    initial_connectivity = 0.01  # Menos conexões = mais rápido
    learning_rate = 5e-3  # Learning rate maior para convergência mais rápida
    epochs = 3  # Apenas 3 epochs para demonstração rápida
    batch_size = 128  # Batch maior para velocidade

    print(f"\n⚙️  Configuração Otimizada:")
    print(f"   • Neurônios: {initial_neurons}")
    print(f"   • Conectividade: {initial_connectivity*100}%")
    print(f"   • Learning Rate: {learning_rate}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch Size: {batch_size}")

    # --- Carregamento Rápido dos Dados ---
    print(f"\n📊 Carregando MNIST...")
    train_loader = get_mnist_loader(batch_size=batch_size, train=True)
    test_loader = get_mnist_loader(batch_size=batch_size, train=False)
    print(f"✅ Dataset carregado!")

    # --- Criação do Modelo ---
    print(f"\n🧠 Criando SEONN...")
    model = SEONN_Model(
        input_size=input_size,
        output_size=output_size,
        initial_neurons=initial_neurons,
        initial_connectivity=initial_connectivity,
        learning_rate=learning_rate
    ).to(device)
    
    print(f"✅ Modelo criado com {model.num_neurons} neurônios")
    print(f"🔗 Conexões iniciais: {model.weights.numel()}")

    # --- Visualização da Rede Inicial ---
    print(f"\n🎨 Visualizando rede inicial...")
    visualize_graph(
        model.graph.indices,
        model.weights,
        model.num_neurons,
        "SEONN: Rede Inicial (Antes do Treinamento)"
    )

    # --- Treinamento Rápido com Visualização ---
    print(f"\n🏃‍♂️ Iniciando treinamento rápido...")
    losses = []
    accuracies = []
    epoch_list = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            loss = model.train_step(data, target)
            total_loss += loss
            num_batches += 1
            
            # Mostrar progresso a cada 50 batches
            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx:3d} | Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        # Avaliação rápida
        accuracy = quick_evaluate(model, test_loader, device)
        accuracies.append(accuracy)
        epoch_list.append(epoch + 1)
        
        print(f"   📊 Loss médio: {avg_loss:.4f}")
        print(f"   🎯 Acurácia: {accuracy:.2f}%")
        
        # Plot em tempo real
        plot_training_progress(losses, accuracies, epoch_list)
    
    training_time = time.time() - start_time
    print(f"\n⏱️  Tempo de treinamento: {training_time:.2f} segundos")

    # --- Evolução da Rede ---
    print(f"\n🔄 Demonstrando evolução estrutural...")
    print(f"   Conexões antes: {model.weights.numel()}")
    
    pruned_indices, new_indices = model.evolve(pruning_threshold=0.002)
    
    print(f"   Conexões após evolução: {model.weights.numel()}")
    print(f"   ✂️  Conexões removidas: {pruned_indices.shape[1] if pruned_indices.numel() > 0 else 0}")
    print(f"   ➕ Conexões adicionadas: {new_indices.shape[1] if new_indices.numel() > 0 else 0}")

    # --- Visualização da Rede Evoluída ---
    print(f"\n🎨 Visualizando rede após evolução...")
    visualize_graph(
        model.graph.indices,
        model.weights,
        model.num_neurons,
        "SEONN: Rede Após Evolução Estrutural",
        pruned_indices=pruned_indices,
        new_indices=new_indices
    )

    # --- Resultado Final ---
    final_accuracy = quick_evaluate(model, test_loader, device)
    print(f"\n🏆 RESULTADO FINAL:")
    print(f"   🎯 Acurácia final: {final_accuracy:.2f}%")
    print(f"   ⏱️  Tempo total: {training_time:.2f}s")
    print(f"   🔗 Conexões finais: {model.weights.numel()}")
    
    print(f"\n✨ DEMO CONCLUÍDO!")
    print(f"   A SEONN demonstrou:")
    print(f"   ✓ Treinamento super rápido")
    print(f"   ✓ Evolução estrutural dinâmica")
    print(f"   ✓ Visualizações em tempo real")
    print(f"   ✓ Adaptação automática da topologia")

if __name__ == "__main__":
    main()
