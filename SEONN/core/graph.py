"""
Grafo Neural Dinâmico
=====================

A estrutura física e lógica da SEONN é representada como um grafo dinâmico,
onde:
- Vértices correspondem aos neurônios
- Arestas representam conexões sinápticas adaptativas
- Evolui com o tempo baseado em:
  - Fluxo de dados
  - Erros e reforços
  - Contexto

Inspirado em Dynamic Graph Convolutional Networks e GNNs adaptativos.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import deque, defaultdict

from .dna_neural import NeuralDNA
from .neuron import AutonomicNeuron
from .plasticity import SynapticConnection


class DynamicNeuralGraph(nn.Module):
    """
    Grafo Neural Dinâmico representando a topologia da SEONN.
    
    O grafo evolui dinamicamente baseado em:
    - Estímulos externos
    - Métricas de desempenho
    - Necessidades de reorganização
    """
    
    def __init__(self, 
                 num_neurons: int,
                 input_dim: int = 784,  # Padrão para MNIST
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # Estrutura de grafo
        # Representação esparsa de arestas (source, target, weight)
        self.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        self.edge_weights = torch.empty(0, device=device)
        
        # Pool de neurônios
        self.neurons: Dict[int, AutonomicNeuron] = {}
        
        # Cache para eficiência
        self._cached_adjacency = None
        self._cache_valid = False
        
        # Estatísticas do grafo
        self.connection_age = torch.zeros(0, dtype=torch.long, device=device)
        self.activation_history = defaultdict(list)
        
    def initialize_neurons(self, initial_topology: str = 'random'):
        """
        Inicializa os neurônios do grafo.
        
        Args:
            initial_topology: 'random', 'fully_connected', 'sparse'
        """
        # Cria pool de neurônios
        for i in range(self.num_neurons):
            neuron_id = f"neuron_{i}"
            dna = NeuralDNA(
                neuron_id=neuron_id,
                generation=0,
                birth_timestamp=0.0,
                specialization="general"
            )
            
            neuron = AutonomicNeuron(
                neuron_id=neuron_id,
                input_dim=self.input_dim,
                output_dim=self.hidden_dim,
                dna=dna
            ).to(self.device)
            
            self.neurons[i] = neuron
        
        # Inicializa topologia
        if initial_topology == 'random':
            self._init_random_topology()
        elif initial_topology == 'fully_connected':
            self._init_fully_connected_topology()
        elif initial_topology == 'sparse':
            self._init_sparse_topology()
    
    def _init_random_topology(self, connection_prob: float = 0.3):
        """Inicializa topologia aleatória"""
        edges = []
        weights = []
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and np.random.rand() < connection_prob:
                    edges.append([i, j])
                    weights.append(np.random.randn() * 0.01)
        
        if edges:
            self.edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
            self.edge_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.connection_age = torch.zeros(len(weights), dtype=torch.long, device=self.device)
    
    def _init_fully_connected_topology(self):
        """Inicializa topologia totalmente conectada"""
        edges = []
        weights = []
        
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j:
                    edges.append([i, j])
                    weights.append(np.random.randn() * 0.01)
        
        self.edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        self.edge_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.connection_age = torch.zeros(len(weights), dtype=torch.long, device=self.device)
    
    def _init_sparse_topology(self, fan_out: int = 5):
        """Inicializa topologia esparsa com fan-out limitado"""
        edges = []
        weights = []
        
        for i in range(self.num_neurons):
            # Conecta a neurônios aleatórios
            targets = np.random.choice(
                [j for j in range(self.num_neurons) if j != i],
                size=min(fan_out, self.num_neurons - 1),
                replace=False
            )
            
            for j in targets:
                edges.append([i, j])
                weights.append(np.random.randn() * 0.01)
        
        if edges:
            self.edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
            self.edge_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.connection_age = torch.zeros(len(weights), dtype=torch.long, device=self.device)
    
    def forward(self, 
               x: torch.Tensor, 
               active_neurons: Optional[Set[int]] = None) -> torch.Tensor:
        """
        Propaga informação através do grafo.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            active_neurons: Conjunto de neurônios ativos
        
        Returns:
            Output tensor (batch_size, output_dim)
        """
        if active_neurons is None:
            active_neurons = set(range(self.num_neurons))
        
        batch_size = x.size(0)
        
        # Propaga através das camadas
        hidden_states = {}
        
        # Camada de entrada -> neurônios ocultos
        for neuron_id, neuron in self.neurons.items():
            if neuron_id in active_neurons:
                hidden_states[neuron_id] = neuron(x)
        
        # Propaga entre neurônios através do grafo
        if len(self.edge_index) > 0:
            # Agrega mensagens
            for i in range(self.edge_index.size(1)):
                src = self.edge_index[0, i].item()
                tgt = self.edge_index[1, i].item()
                weight = self.edge_weights[i]
                
                if src in active_neurons and tgt in active_neurons and src in hidden_states:
                    if tgt not in hidden_states:
                        hidden_states[tgt] = torch.zeros(batch_size, self.hidden_dim, device=self.device)
                    
                    # Soma com peso
                    hidden_states[tgt] = hidden_states[tgt] + hidden_states[src] * weight
        
        # Agrega todos os estados para saída
        if hidden_states:
            aggregated = torch.stack(list(hidden_states.values()))
            output = aggregated.mean(dim=0)  # Agregação por média
        else:
            output = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        
        return output
    
    def add_edge(self, source: int, target: int, weight: float = 0.01):
        """Adiciona uma aresta ao grafo"""
        if source == target:
            return
        
        # Verifica se já existe
        for i in range(self.edge_index.size(1)):
            if (self.edge_index[0, i].item() == source and 
                self.edge_index[1, i].item() == target):
                return  # Já existe
        
        # Adiciona nova aresta
        new_edge = torch.tensor([[source], [target]], dtype=torch.long, device=self.device)
        self.edge_index = torch.cat([self.edge_index, new_edge], dim=1)
        
        new_weight = torch.tensor([weight], dtype=torch.float32, device=self.device)
        self.edge_weights = torch.cat([self.edge_weights, new_weight])
        
        new_age = torch.tensor([0], dtype=torch.long, device=self.device)
        self.connection_age = torch.cat([self.connection_age, new_age])
        
        self._invalidate_cache()
    
    def remove_edge(self, source: int, target: int):
        """Remove uma aresta do grafo"""
        mask = torch.ones(self.edge_index.size(1), dtype=torch.bool, device=self.device)
        
        for i in range(self.edge_index.size(1)):
            if (self.edge_index[0, i].item() == source and 
                self.edge_index[1, i].item() == target):
                mask[i] = False
        
        self.edge_index = self.edge_index[:, mask]
        self.edge_weights = self.edge_weights[mask]
        self.connection_age = self.connection_age[mask]
        
        self._invalidate_cache()
    
    def prune_weak_connections(self, threshold: float = 0.01):
        """Remove conexões fracas"""
        mask = self.edge_weights.abs() > threshold
        
        self.edge_index = self.edge_index[:, mask]
        self.edge_weights = self.edge_weights[mask]
        self.connection_age = self.connection_age[mask]
        
        self._invalidate_cache()
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do grafo"""
        return {
            'num_neurons': self.num_neurons,
            'num_edges': self.edge_index.size(1),
            'avg_degree': self.edge_index.size(1) / max(self.num_neurons, 1),
            'avg_weight': self.edge_weights.abs().mean().item() if len(self.edge_weights) > 0 else 0.0,
            'max_weight': self.edge_weights.abs().max().item() if len(self.edge_weights) > 0 else 0.0,
            'min_weight': self.edge_weights.abs().min().item() if len(self.edge_weights) > 0 else 0.0
        }
    
    def _invalidate_cache(self):
        """Invalida cache"""
        self._cache_valid = False
        self._cached_adjacency = None
    
    def save_state(self) -> Dict:
        """Salva estado do grafo"""
        return {
            'edge_index': self.edge_index.cpu(),
            'edge_weights': self.edge_weights.cpu(),
            'connection_age': self.connection_age.cpu(),
            'neurons': {k: v.state_dict() for k, v in self.neurons.items()}
        }
    
    def load_state(self, state: Dict):
        """Carrega estado do grafo"""
        self.edge_index = state['edge_index'].to(self.device)
        self.edge_weights = state['edge_weights'].to(self.device)
        self.connection_age = state['connection_age'].to(self.device)

