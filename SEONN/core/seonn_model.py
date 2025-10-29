"""
Self-Evolving Organic Neural Network (SEONN) - Modelo Principal
================================================================

Integra todos os componentes:
- Neurônios autô tapi leve and1s inteligentes
- DNA Neural
- Plasticidade sináptica
- Núcleo gerenciador
- Grafo neural dinâmico

Sistema completo de IA evolutiva e adaptativa.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np

from .dna_neural import NeuralDNA
from .neuron import AutonomicNeuron
from .plasticity import PlasticityMechanism
from .manager import NucleusManager, TaskContext
from .graph import DynamicNeuralGraph


class SEONN_Model(nn.Module):
    """
    Self-Evolving Organic Neural Network - Modelo Principal
    
    Arquitetura completa integrando todos os componentes evolutivos.
    """
    
    def __init__(self,
                 num_neurons: int = 100,
                 input_dim: int = 784,
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # Componentes principais
        # 1. Grafo Neural Dinâmico
        self.neural_graph = DynamicNeuralGraph(
            num_neurons=num_neurons,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            device=device
        )
        
        # 2. Mecanismo de Plasticidade
        self.plasticity = PlasticityMechanism()
        
        # 3. Núcleo Gerenciador
        self.nucleus = NucleusManager(
            total_neurons=num_neurons,
            max_active_neurons=int(num_neurons * 0.7)
        )
        
        # Camadas de saída final (melhoradas)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Módulo de evolução
        self.evolution_coordinator = nn.Linear(num_neurons, num_neurons)
        
        # Contadores e estatísticas
        self.step_count = 0
        self.training_history = []
        
    def initialize(self, topology='sparse'):
        """Inicializa a rede"""
        self.neural_graph.initialize_neurons(initial_topology=topology)
        self.to(self.device)
    
    def forward(self, 
               x: torch.Tensor,
               task_context: Optional[TaskContext] = None) -> torch.Tensor:
        """
        Forward pass através da SEONN.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            task_context: Contexto da tarefa
        
        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # 1. Observa estado da rede
        neuron_activities = torch.zeros(self.num_neurons, device=self.device)
        observed_state = self.nucleus.observe_network_state(neuron_activities)
        
        # 2. Seleciona sub-rede ativa baseado em contexto
        if task_context is None:
            task_context = TaskContext(
                task_id="default",
                task_type="general",
                complexity=0.5,
                required_specialization="general"
            )
        
        neuron_fitness = torch.tensor(
            [n.get_dna_fitness() for n in self.neural_graph.neurons.values()],
            device=self.device
        )
        
        active_subnet = self.nucleus.select_active_subnet(task_context, neuron_fitness)
        
        # 3. Propaga através do grafo
        hidden_out = self.neural_graph(x, active_neurons=active_subnet)
        
        # 4. Camadas finais
        # 4. Camadas finais (melhoradas)
        out = torch.relu(self.fc1(hidden_out))
        if out.size(0) > 1:  # BatchNorm requer batch > 1
            out = self.bn1(out)
        out = self.dropout1(out)
        
        out = torch.relu(self.fc2(out))
        if out.size(0) > 1:
            out = self.bn2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        # 5. Atualiza estatísticas
        self.step_count += 1
        neuron_activities = hidden_out.mean(dim=0)
        
        return out
    
    def train_step(self, 
                  x: torch.Tensor, 
                  y: torch.Tensor,
                  task_context: Optional[TaskContext] = None,
                  optimizer: optim.Optimizer = None,
                  criterion = None) -> Dict:
        """
        Realiza um passo de treinamento.
        """
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # Forward
        output = self.forward(x, task_context)
        
        # Cálculo de loss
        loss = criterion(output, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Atualização evolutiva
        with torch.no_grad():
            self._evolve_step(task_context, output, y)
        
        # Métricas
        accuracy = (output.argmax(dim=1) == y).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'num_active_neurons': len(self.nucleus.active_subset)
        }
    
    def _evolve_step(self, 
                    task_context: TaskContext,
                    predictions: torch.Tensor,
                    targets: torch.Tensor):
        """
        Passo evolutivo: adapta plasticidade e especialização.
        """
        # Calcula sucesso
        correct = (predictions.argmax(dim=1) == targets).float().mean()
        reward_signal = correct - 0.5  # Normaliza entre -0.5 e 0.5
        
        # Atualiza plasticidade - CONEXÕES ENTRE NEURÔNIOS
        neuron_list = list(self.neural_graph.neurons.keys())
        
        for i, neuron_id in enumerate(neuron_list):
            if neuron_id in self.nucleus.active_subset:
                neuron = self.neural_graph.neurons[neuron_id]
                
                # Adiciona conexões entre neurônios ativos
                for j, other_neuron_id in enumerate(neuron_list):
                    if j != i and other_neuron_id in self.nucleus.active_subset:
                        activation_level = correct.item()
                        success = correct > 0.5
                        
                        self.plasticity.update_connection(
                            source_id=neuron.neuron_id,
                            target_id=self.neural_graph.neurons[other_neuron_id].neuron_id,
                            activation_level=activation_level,
                            success=success
                        )
                        
                        # Atualiza pesos do grafo baseado na plasticidade
                        strength = self.plasticity.get_connection_strength(
                            neuron.neuron_id, 
                            self.neural_graph.neurons[other_neuron_id].neuron_id
                        )
                        
                        # Atualiza peso no grafo se conexão existir
                        if strength > 0:
                            key = (neuron_id, other_neuron_id)
                            for k in range(self.neural_graph.edge_index.size(1)):
                                if (self.neural_graph.edge_index[0, k].item() == neuron_id and 
                                    self.neural_graph.edge_index[1, k].item() == other_neuron_id):
                                    self.neural_graph.edge_weights[k] = strength * 0.01
                                    break
                
                # Atualiza especialização
                performance = correct.item()
                neuron.update_specialization(task_context.task_type, performance)
                
                # Evolui DNA do neurônio
                if self.step_count % 50 == 0:
                    neuron.evolve_dna(mutation_rate=0.05)
        
        # Poda conexões fracas periodicamente
        if self.step_count % 100 == 0:
            self.plasticity.prune_connections(min_strength=0.05)
            self.neural_graph.prune_weak_connections(threshold=0.001)
    
    def evolve_architecture(self, evolution_rate: float = 0.1):
        """
        Evolui a arquitetura: adiciona/remove neurônios e conexões.
        """
        stats = self.neural_graph.get_statistics()
        
        # Análise de desempenho
        avg_fitness = np.mean([n.get_dna_fitness() for n in self.neural_graph.neurons.values()])
        
        # Evolução baseada em desempenho
        if avg_fitness > 0.8 and stats['num_edges'] < self.num_neurons * 5:
            # Performance boa: expande
            self._expand_network()
        elif avg_fitness < 0.3:
            # Performance ruim: contrai
            self._contract_network()
    
    def _expand_network(self):
        """Expande a rede adicionando conexões"""
        # Adiciona algumas conexões novas aleatórias
        num_new_connections = min(10, self.num_neurons // 10)
        
        for _ in range(num_new_connections):
            src = np.random.randint(0, self.num_neurons)
            tgt = np.random.randint(0, self.num_neurons)
            if src != tgt:
                self.neural_graph.add_edge(src, tgt, weight=0.01)
    
    def _contract_network(self):
        """Contrai a rede removendo conexões fracas"""
        # Remove conexões mais fracas
        self.neural_graph.prune_weak_connections(threshold=0.03)
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas completas da SEONN"""
        graph_stats = self.neural_graph.get_statistics()
        plasticity_stats = self.plasticity.get_topology_statistics()
        manager_stats = self.nucleus.get_network_statistics()
        
        neuron_fitnesses = [n.get_dna_fitness() for n in self.neural_graph.neurons.values()]
        
        return {
            'step_count': self.step_count,
            'graph': graph_stats,
            'plasticity': plasticity_stats,
            'manager': manager_stats,
            'avg_neuron_fitness': np.mean(neuron_fitnesses),
            'std_neuron_fitness': np.std(neuron_fitnesses),
            'num_active_neurons': len(self.nucleus.active_subset)
        }
    
    def save_model(self, filepath: str):
        """Salva o modelo completo"""
        state = {
            'model_state_dict': self.state_dict(),
            'graph_state': self.neural_graph.save_state(),
            'plasticity_state': self.plasticity.save_state(),
            'manager_state': self.nucleus.save_state(),
            'step_count': self.step_count,
            'training_history': self.training_history
        }
        torch.save(state, filepath)
    
    def load_model(self, filepath: str):
        """Carrega o modelo completo"""
        state = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(state['model_state_dict'])
        self.neural_graph.load_state(state['graph_state'])
        self.plasticity.load_state(state['plasticity_state'])
        self.nucleus.load_state(state['manager_state'])
        self.step_count = state['step_count']
        self.training_history = state.get('training_history', [])

