"""
Exemplo de Uso da SEONN
========================

Demonstra como usar a Self-Evolving Organic Neural Network
para tarefas de classificação e aprendizado contínuo.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import SEONN_Model, TaskContext


def create_synthetic_dataset(num_samples=1000, input_dim=784, num_classes=10):
    """
    Cria um dataset sintético para demonstração.
    """
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Adiciona algum padrão
    for i, label in enumerate(y):
        X[i, :50] += label * 0.1
    
    return X, y


def train_example():
    """
    Exemplo de treinamento básico da SEONN.
    """
    print("=" * 60)
    print("SEONN - Self-Evolving Organic Neural Network")
    print("Exemplo de Treinamento")
    print("=" * 60)
    
    # Configuração
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsando dispositivo: {device}")
    
    # Hiperparâmetros
    num_neurons = 50
    input_dim = 784
    hidden_dim = 128
    output_dim = 10
    batch_size = 32
    num_epochs = 5
    
    # Cria dataset sintético
    print("\n1. Criando dataset sintético...")
    X_train, y_train = create_synthetic_dataset(num_samples=1000)
    X_val, y_val = create_synthetic_dataset(num_samples=200)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Inicializa SEONN
    print("\n2. Inicializando SEONN...")
    model = SEONN_Model(
        num_neurons=num_neurons,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        device=device
    )
    
    # Topologia inicial
    model.initialize(topology='sparse')
    model.to(device)
    
    # Otimizador e critério (otimizados)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Treinamento
    print("\n3. Iniciando treinamento...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Contexto da tarefa
            task_context = TaskContext(
                task_id=f"epoch_{epoch}",
                task_type="classification",
                complexity=0.5,
                required_specialization="general"
            )
            
            # Treinamento
            result = model.train_step(
                x=x_batch,
                y=y_batch,
                task_context=task_context,
                optimizer=optimizer,
                criterion=criterion
            )
            
            train_loss += result['loss']
            train_acc += result['accuracy']
            train_batches += 1
            
            # Log periódico
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss={result['loss']:.4f}, "
                      f"Acc={result['accuracy']*100:.2f}%, "
                      f"Active Neurons={result['num_active_neurons']}")
        
        # Avaliação
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                output = model(x_batch)
                loss = criterion(output, y_batch)
                accuracy = (output.argmax(dim=1) == y_batch).float().mean()
                
                val_loss += loss.item()
                val_acc += accuracy.item()
                val_batches += 1
        
        # Média
        avg_train_loss = train_loss / train_batches
        avg_train_acc = train_acc / train_batches
        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_acc / val_batches
        
        # Print
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc*100:.2f}%")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc*100:.2f}%")
        
        # Estatísticas da SEONN
        stats = model.get_statistics()
        print(f"\n  SEONN Stats:")
        print(f"    Neurônios ativos: {stats['num_active_neurons']}")
        print(f"    Fitness médio: {stats['avg_neuron_fitness']:.3f}")
        print(f"    Conexões no grafo: {stats['graph']['num_edges']}")
        print(f"    Estado crítico: {stats['manager'].get('is_critical_state', False)}")
        
        # Evolução arquitetural
        model.evolve_architecture(evolution_rate=0.1)
    
    print("\n4. Treinamento concluído!")
    
    # Estatísticas finais
    print("\n5. Estatísticas Finais:")
    final_stats = model.get_statistics()
    for key, value in final_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Salvar modelo
    print("\n6. Salvando modelo...")
    model_path = "seonn_model.pth"
    model.save_model(model_path)
    print(f"Modelo salvo em: {model_path}")


def lifelong_learning_example():
    """
    Exemplo de aprendizado contínuo (lifelong learning).
    """
    print("\n" + "=" * 60)
    print("Exemplo: Aprendizado Contínuo (Lifelong Learning)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cria múltiplas tarefas sequenciais
    tasks = [
        ("Task A - Digits 0-2", 0, 3),
        ("Task B - Digits 3-5", 3, 6),
        ("Task C - Digits 6-9", 6, 10),
    ]
    
    # Inicializa modelo
    model = SEONN_Model(
        num_neurons=100,
        input_dim=784,
        hidden_dim=128,
        output_dim=3,
        device=device
    )
    model.initialize(topology='sparse')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTreinando em múltiplas tarefas sequenciais...")
    for task_name, start_class, end_class in tasks:
        print(f"\n{task_name}")
        print("-" * 60)
        
        # Cria nose data para essa tarefa
        X, y = create_synthetic_dataset(num_samples=500, num_classes=end_class)
        X = X.to(device)
        y = y.to(device)
        
        # Treina
        for step in range(10):
            model.train()
            
            # Sample batch
            indices = torch.randint(0, len(X), (32,))
            x_batch = X[indices]
            y_batch = y[indices]
            
            # Contexto
            task_context = TaskContext(
                task_id=task_name,
                task_type="lifelong_learning",
                complexity=0.6,
                required_specialization="general"
            )
            
            # Treinamento
            result = model.train_step(
                x=x_batch,
                y=y_batch,
                task_context=task_context,
                optimizer=optimizer
            )
            
            if step % 3 == 0:
                print(f"  Step {step}: Loss={result['loss']:.4f}, "
                      f"Acc={result['accuracy']*100:.2f}%")
        
        # Estatísticas
        stats = model.get_statistics()
        print(f"\n  Fitness médio: {stats['avg_neuron_fitness']:.3f}")
        print(f"  Conexões ativas: {stats['graph']['num_edges']}")
    
    print("\n✓ Lifelong learning concluído!")


if __name__ == "__main__":
    # Roda exemplo principal
    train_example()
    
    # Roda exemplo de lifelong learning
    # lifelong_learning_example()
    
    print("\n" + "=" * 60)
    print("Todos os exemplos foram executados com sucesso!")
    print("=" * 60)

