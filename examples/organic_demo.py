import torch
import sys
import os
import time
import matplotlib.pyplot as plt

# Adjust path to import from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from seonn_torch.seonn_model import SEONN_Model
from seonn_torch.utils.data_loader import get_mnist_loader
# from seonn_torch.utils.visualization import visualize_graph  # Desabilitado para performance

def plot_biological_metrics(model, epoch, losses, accuracies, biological_data):
    """Plot biological metrics during training"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss and accuracy
    ax1.plot(losses, 'b-', linewidth=2, marker='o')
    ax1.set_title('Loss durante o Treinamento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(accuracies, 'g-', linewidth=2, marker='s')
    ax2.set_title('Acurácia durante o Treinamento', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Acurácia (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Biological metrics
    ax3.plot(biological_data['neuron_health'], 'r-', linewidth=2, marker='^')
    ax3.set_title('Saúde dos Neurônios', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Saúde Média')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    ax4.plot(biological_data['activity_variance'], 'm-', linewidth=2, marker='d')
    ax4.set_title('Variação de Atividade', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Variância da Atividade')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def evaluate_model(model, test_loader, device):
    """Evaluate model performance"""
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
    print("🧬 DEMO ORGÂNICO DA SEONN 🧬")
    print("=" * 60)
    print("Demonstrando mecanismos biológicos e performance otimizada!")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Dispositivo: {device}")

    # --- Parâmetros Biológicos Otimizados ---
    input_size = 28 * 28  # MNIST
    output_size = 10
    initial_neurons = 1000
    initial_connectivity = 0.015  # Ligeiramente mais denso para melhor performance
    learning_rate = 2e-3  # Learning rate otimizado
    plasticity_rate = 0.02  # Plasticidade sináptica
    homeostasis_strength = 0.15  # Homeostase neural
    competition_strength = 0.08  # Competição entre neurônios
    
    epochs = 5
    batch_size = 128

    print(f"\n🧬 Configuração Biológica:")
    print(f"   • Neurônios: {initial_neurons}")
    print(f"   • Conectividade: {initial_connectivity*100}%")
    print(f"   • Learning Rate: {learning_rate}")
    print(f"   • Plasticidade: {plasticity_rate}")
    print(f"   • Homeostase: {homeostasis_strength}")
    print(f"   • Competição: {competition_strength}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch Size: {batch_size}")

    # --- Carregamento dos Dados ---
    print(f"\n📊 Carregando MNIST...")
    train_loader = get_mnist_loader(batch_size=batch_size, train=True)
    test_loader = get_mnist_loader(batch_size=batch_size, train=False)
    print(f"✅ Dataset carregado!")

    # --- Criação do Modelo Orgânico ---
    print(f"\n🧠 Criando SEONN Orgânica...")
    model = SEONN_Model(
        input_size=input_size,
        output_size=output_size,
        initial_neurons=initial_neurons,
        initial_connectivity=initial_connectivity,
        learning_rate=learning_rate,
        plasticity_rate=plasticity_rate,
        homeostasis_strength=homeostasis_strength,
        competition_strength=competition_strength
    ).to(device)
    
    print(f"✅ Modelo orgânico criado com {model.num_neurons} neurônios")
    print(f"🔗 Conexões iniciais: {model.weights.numel()}")
    
    # Estatísticas iniciais do grafo
    graph_stats = model.graph.get_connection_statistics()
    print(f"📈 Estatísticas iniciais: {graph_stats}")

    # --- Visualização da Rede Inicial ---
    print(f"\n🎨 Rede inicial criada (visualização desabilitada para performance)")

    # --- Treinamento com Métricas Biológicas ---
    print(f"\n🏃‍♂️ Iniciando treinamento orgânico...")
    losses = []
    accuracies = []
    biological_data = {
        'neuron_health': [],
        'activity_variance': [],
        'connection_stats': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        print("-" * 40)
        
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
        
        # Avaliação
        accuracy = evaluate_model(model, test_loader, device)
        accuracies.append(accuracy)
        
        # Coletar métricas biológicas
        neuron_health_avg = model.neuron_health.mean().item()
        activity_variance = model.activity_history.var().item()
        connection_stats = model.graph.get_connection_statistics()
        
        biological_data['neuron_health'].append(neuron_health_avg)
        biological_data['activity_variance'].append(activity_variance)
        biological_data['connection_stats'].append(connection_stats)
        
        print(f"   📊 Loss médio: {avg_loss:.4f}")
        print(f"   🎯 Acurácia: {accuracy:.2f}%")
        print(f"   🧬 Saúde neural: {neuron_health_avg:.3f}")
        print(f"   📈 Variação atividade: {activity_variance:.4f}")
        print(f"   🔗 Conexões ativas: {connection_stats['active_connections']}")
        
        # Plot métricas biológicas (desabilitado para performance)
        # plot_biological_metrics(model, epoch, losses, accuracies, biological_data)
    
    training_time = time.time() - start_time
    print(f"\n⏱️  Tempo de treinamento: {training_time:.2f} segundos")

    # --- Evolução Orgânica ---
    print(f"\n🔄 Demonstrando evolução orgânica...")
    print(f"   Conexões antes: {model.weights.numel()}")
    
    pruned_indices, new_indices = model.evolve(pruning_threshold=0.002)
    
    print(f"   Conexões após evolução: {model.weights.numel()}")
    print(f"   ✂️  Conexões removidas: {pruned_indices.shape[1] if pruned_indices.numel() > 0 else 0}")
    print(f"   ➕ Conexões adicionadas: {new_indices.shape[1] if new_indices.numel() > 0 else 0}")

    # --- Visualização da Rede Evoluída ---
    print(f"\n🎨 Rede evoluída (visualização desabilitada para performance)")

    # --- Resultado Final ---
    final_accuracy = evaluate_model(model, test_loader, device)
    final_stats = model.graph.get_connection_statistics()
    
    print(f"\n🏆 RESULTADO FINAL:")
    print(f"   🎯 Acurácia final: {final_accuracy:.2f}%")
    print(f"   ⏱️  Tempo total: {training_time:.2f}s")
    print(f"   🔗 Conexões finais: {final_stats['num_connections']}")
    print(f"   🧬 Saúde neural final: {model.neuron_health.mean().item():.3f}")
    print(f"   📈 Variação atividade final: {model.activity_history.var().item():.4f}")
    
    print(f"\n✨ DEMO ORGÂNICO CONCLUÍDO!")
    print(f"   A SEONN Orgânica demonstrou:")
    print(f"   ✓ Mecanismos biológicos (LTP/LTD, homeostase, competição)")
    print(f"   ✓ Evolução estrutural guiada por atividade")
    print(f"   ✓ Performance otimizada com cache e operações vetorizadas")
    print(f"   ✓ Adaptação dinâmica da topologia")
    print(f"   ✓ Rastreamento de saúde neural e idade das conexões")

if __name__ == "__main__":
    main()
