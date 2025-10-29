"""
Utilidades de Visualização
==========================
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_network_statistics(stats: Dict):
    """
    Plota estatísticas da rede SEONN.
    
    Args:
        stats: Dicionário com estatísticas
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Evolução de conexões
    axes[0, 0].plot(stats.get('num_edges', []))
    axes[0, 0].set_title('Evolução do Número de Conexões')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Número de Arestas')
    axes[0, 0].grid(True)
    
    # Gráfico 2: Fitness médio
    axes[0, 1].plot(stats.get('avg_fitness', []))
    axes[0, 1].set_title('Fitness Médio dos Neurônios')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Fitness')
    axes[0, 1].grid(True)
    
    # Gráfico 3: Neurônios ativos
    axes[1, 0].plot(stats.get('active_neurons', []))
    axes[1, 0].set_title('Neurônios Ativos por Tarefa')
    axes[1, 0].set_xlabel('Task')
    axes[1, 0].set_ylabel('Número de Neurônios')
    axes[1, 0].grid(True)
    
    # Gráfico 4: Desempenho
    axes[1, 1].plot(stats.get('accuracy', []))
    axes[1, 1].set_title('Acurácia ao Longo do Tempo')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('seonn_statistics.png')
    print("Gráfico salvo em: seonn_statistics.png")

