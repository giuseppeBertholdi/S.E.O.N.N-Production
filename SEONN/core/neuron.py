"""
Neurônios Autônomos e Inteligentes
===================================

Cada neurônio é concebido como uma unidade autônoma e inteligente,
capaz de:
- Processar informações de forma descentralizada
- Analisar padrões
- Memorizar contextos
- Ajustar dinamicamente seu comportamento

Inspirado em arquiteturas descentralizadas sem pesos e Graph Neural Networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Tuple
from collections import deque
from .dna_neural import NeuralDNA


class AdaptiveActivation(nn.Module):
    """
    Função de ativação adaptativa que ajusta dinamicamente sua forma.
    """
    
    def __init__(self, base_activation: str = 'relu', adaptive=True):
        super().__init__()
        self.base_activation = base_activation
        self.adaptive = adaptive
        
        if adaptive:
            # Parâmetros adaptativos da função de ativação
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(0.0))
            self.gamma = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica função de ativação adaptativa.
        
        Parâmetros:
        - alpha: controla a inclinação
        - beta: controla o deslocamento
        - gamma: controla a escala
        """
        if self.adaptive:
            if self.base_activation == 'relu':
                return self.gamma * F.relu(self.alpha * x + self.beta)
            elif self.base_activation == 'sigmoid':
                return self.gamma * torch.sigmoid(self.alpha * x + self.beta)
            elif self.base_activation == 'tanh':
                return self.gamma * torch.tanh(self.alpha * x + self.beta)
            else:
                return self.gamma * F.relu(self.alpha * x + self.beta)
        else:
            return F.relu(x)


class AutonomicNeuron(nn.Module):
    """
    Neurônio Autônomo e Inteligente.
    
    Cada neurônio possui:
    - DNA Neural (identidade evolutiva)
    - Memória local
    - Capacidade de decisão autônoma
    - Adaptação contextual
    """
    
    def __init__(self, 
                 neuron_id: str,
                 input_dim: int,
                 output_dim: int,
                 dna: Optional[NeuralDNA] = None,
                 memory_size: int = 10):
        super().__init__()
        
        self.neuron_id = neuron_id
        self.dna = dna or NeuralDNA(neuron_id=neuron_id, 
                                   generation=0, 
                                   birth_timestamp=0.0)
        
        # Módulos de processamento
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Pesos adaptativos (out_features x in_features para F.linear)
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Função de ativação adaptativa
        self.activation = AdaptiveActivation('relu', adaptive=True)
        
        # Memória local (buffer circular)
        self.memory = deque(maxlen=memory_size)
        self.memory_buffer = torch.zeros(memory_size, output_dim)
        
        # Estado interno
        self.internal_state = torch.zeros(output_dim)
        self.last_input = None
        self.last_output = None
        
        # Adaptação contextual
        self.context_weight = nn.Parameter(torch.ones(output_dim))
        self.attention_weights = nn.Parameter(torch.ones(input_dim))
        
        # Processamento de padrões
        self.pattern_detector = nn.Linear(output_dim, 5)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processa a entrada e produz saída.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            context: Contexto adicional para adaptação
        """
        batch_size = x.size(0)
        
        # Atenção contextual
        if context is not None:
            context_gate = torch.sigmoid(context)
            x = x * self.attention_weights * context_gate
        
        # Processamento principal (x @ weight.T + bias onde x é batch x input_dim, weight é output_dim x input_dim)
        out = F.linear(x, self.weight, self.bias)
        
        # Ativação adaptativa
        out = self.activation(out)
        
        # Aplicação de contexto
        if context is not None:
            out = out * self.context_weight
        
        # Atualização de estado interno
        self.internal_state = self.internal_state * 0.9 + out.mean(dim=0) * 0.1
        
        # Detecção de padrões
        patterns = self.pattern_detector(self.internal_state.unsqueeze(0))
        
        # Registro na memória (apenas o primeiro elemento do batch para consistência)
        self.memory.append(out[0].clone().detach().unsqueeze(0))
        if len(self.memory) > 0:
            try:
                self.memory_buffer = torch.stack(list(self.memory))
            except RuntimeError:
                # Se tamanhos diferentes, usa apenas os últimos
                self.memory = list(self.memory)[-5:]
                if len(self.memory) > 0:
                    self.memory_buffer = torch.stack(list(self.memory))
        
        # Registro de DNA
        self.dna.activate(success=True)
        self.dna.total_inputs_processed += batch_size
        
        self.last_input = x.clone().detach()
        self.last_output = out.clone().detach()
        
        return out
    
    def learn_from_memory(self):
        """
        Aprende da memória local usando a metaplasticidade.
        Processa padrões recorrentes.
        """
        if len(self.memory) < 2:
            return
        
        # Pega os últimos padrões
        recent_patterns = list(self.memory)[-5:] if len(self.memory) >= 5 else list(self.memory)
        
        if len(recent_patterns) > 1:
            # Calcula variância dos padrões
            patterns_tensor = torch.stack(recent_patterns)
            variance = patterns_tensor.var(dim=0)
            
            # Ajusta pesos baseado na variância
            adjustment = torch.sigmoid(variance) * self.dna.adaptation_rate
            self.weight.data += adjustment.unsqueeze(0).expand_as(self.weight) * 0.01
            
            # Registra na memória do DNA
            avg_variance = variance.mean().item()
            self.dna.memory_strength = min(1.0, avg_variance + 0.1)
    
    def adapt_to_context(self, context_signal: torch.Tensor):
        """
        Adapta comportamento ao contexto fornecido.
        
        Args:
            context_signal: Sinal contextual (batch_size, context_dim)
        """
        # Atualiza pesos de contexto
        context_encoded = torch.sigmoid(context_signal)
        self.context_weight.data = self.context_weight.data * 0.9 + context_encoded * 0.1
        
        # Adapta a função de ativação
        context_mean = context_encoded.mean().item()
        self.activation.alpha.data = self.activation.alpha.data * 0.95 + torch.tensor(context_mean) * 0.05
    
    def get_specialization_probability(self, task_type: str) -> float:
        """
        Retorna a probabilidade de especialização para uma tarefa.
        """
        if self.dna.specialization == task_type:
            return self.dna.confidence_level
        elif self.dna.specialization == "general":
            return 0.5
        else:
            return 0.2
    
    def should_be_activated(self, task_type: str, energy_cost: float) -> bool:
        """
        Decisão autônoma sobre ativação baseada em:
        - Especialização
        - Energia disponível
        - Confiança
        """
        specialization_prob = self.get_specialization_probability(task_type)
        energy_factor = min(1.0, self.dna.energy_efficiency / energy_cost)
        
        activation_probability = specialization_prob * energy_factor * self.dna.confidence_level
        
        return np.random.rand() < activation_probability
    
    def update_specialization(self, task_type: str, performance: float):
        """
        Atualiza especialização baseado em desempenho.
        """
        frequency = performance
        self.dna.update_specialization(task_type, frequency)
    
    def get_dna_fitness(self) -> float:
        """Retorna o fitness do DNA neural"""
        return self.dna.get_fitness_score()
    
    def evolve_dna(self, mutation_rate: float = 0.1):
        """Aplica mutação ao DNA"""
        self.dna.mutate(mutation_rate)
    
    def __repr__(self):
        return (f"AutonomicNeuron(id={self.neuron_id}, "
                f"spec={self.dna.specialization}, "
                f"fitness={self.get_dna_fitness():.2f})")


class NeuronCluster(nn.Module):
    """
    Agrupamento de neurônios que trabalham cooperativamente.
    Implementa processamento distribuído.
    """
    
    def __init__(self, 
                 neuron_ids: List[str],
                 neurons: List[AutonomicNeuron],
                 cluster_specialization: str = "general"):
        super().__init__()
        
        self.neuron_ids = neuron_ids
        self.neurons = nn.ModuleList(neurons)
        self.cluster_specialization = cluster_specialization
        
        # Coordenação entre neurônios
        self.coordination_layer = nn.Linear(
            sum(n.output_dim for n in neurons),
            len(neurons)
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processa input através de todos os neurônios do cluster.
        """
        outputs = []
        
        for neuron in self.neurons:
            out = neuron(x, context)
            outputs.append(out)
        
        # Concatena saídas
        combined = torch.cat(outputs, dim=-1)
        
        # Coordenação
        coordination_signal = torch.sigmoid(self.coordination_layer(combined))
        
        # Aplica pesos de coordenação
        weighted_outputs = []
        for i, neuron in enumerate(self.neurons):
            weight = coordination_signal[:, i:i+1]
            weighted_outputs.append(outputs[i] * weight)
        
        return torch.cat(weighted_outputs, dim=-1)
    
    def get_cluster_fitness(self) -> float:
        """Calcula fitness médio do cluster"""
        return np.mean([n.get_dna_fitness() for n in self.neurons])
    
    def __len__(self):
        return len(self.neurons)

