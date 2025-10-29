"""
Plasticidade Sináptica Virtual
==============================

Inspirado na plasticidade sináptica do cérebro humano.
Implementa:
- Formação e dissolução dinâmica de conexões
- LTP (Long-Term Potentiation) e LTD (Long-Term Depression)
- Reforço baseado em uso e sucesso
- Adaptação contextual em tempo real

Baseado em princípios de Hebb (1949) e Bliss & Lomo (1973).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SynapticConnection:
    """
    Representa uma conexão sináptica entre neurônios.
    """
    source_id: str
    target_id: str
    weight: float
    strength: float = 0.5  # Força da conexão (0 a 1)
    age: int = 0  # Idade da conexão
    
    # Histórico
    activation_count: int = 0
    last_activation: float = 0.0
    success_count: int = 0
    
    # Metadados
    connection_type: str = "standard"  # standard, inhibitory, modulatory
    
    def strengthen(self, factor: float = 1.1):
        """Fortalecimento da conexão (LTP)"""
        self.strength = min(1.0, self.strength * factor)
        self.age += 1
    
    def weaken(self, factor: float = 0.9):
        """Enfraquecimento da conexão (LTD)"""
        self.strength = max(0.0, self.strength * factor)
        self.age += 1
    
    def activate(self, success: bool = True):
        """Registra uma ativação da conexão"""
        self.activation_count += 1
        self.last_activation = datetime.now().timestamp()
        if success:
            self.success_count += 1


class PlasticityMechanism:
    """
    Mecanismo de plasticidade sináptica virtual.
    """
    
    def __init__(self, 
                 ltp_threshold: float = 0.7,
                 ltd_threshold: float = 0.3,
                 decay_rate: float = 0.01,
                 max_connections: int = 1000):
        
        self.connections: Dict[Tuple[str, str], SynapticConnection] = {}
        self.ltp_threshold = ltp_threshold
        self.ltd_threshold = ltd_threshold
        self.decay_rate = decay_rate
        self.max_connections = max_connections
        
        # Histórico de mudanças
        self.change_history = deque(maxlen=100)
    
    def add_connection(self, source_id: str, target_id: str, 
                      initial_weight: float = 0.5,
                      connection_type: str = "standard"):
        """
        Adiciona uma nova conexão sináptica.
        """
        key = (source_id, target_id)
        
        if key not in self.connections:
            if len(self.connections) >= self.max_connections:
                # Remove conexão mais fraca
                self._remove_weakest_connection()
            
            self.connections[key] = SynapticConnection(
                source_id=source_id,
                target_id=target_id,
                weight=initial_weight,
                connection_type=connection_type
            )
    
    def _remove_weakest_connection(self):
        """Remove a conexão mais fraca"""
        if not self.connections:
            return
        
        weakest_key = min(self.connections.keys(), 
                         key=lambda k: self.connections[k].strength)
        del self.connections[weakest_key]
    
    def update_connection(self, source_id: str, target_id: str, 
                         activation_level: float, success: bool = True):
        """
        Atualiza a força da conexão baseado em ativação e sucesso.
        
        Implementa LTP e LTD baseado no nível de ativação.
        """
        key = (source_id, target_id)
        
        if key not in self.connections:
            self.add_connection(source_id, target_id)
        
        conn = self.connections[key]
        conn.activate(success)
        
        # Hebbian Historical: "neurônios que disparam juntos, se conectam juntos"
        if activation_level > self.ltp_threshold and success:
            # Long-Term Potentiation (LTP)
            improvement_factor = 1.0 + activation_level * 0.2
            conn.strengthen(improvement_factor)
            
        elif activation_level < self.ltd_threshold or not success:
            # Long-Term Depression (LTD)
            decay_factor = 1.0 - (1.0 - activation_level) * 0.1
            conn.weaken(decay_factor)
            
            # Remove conexão muito fraca
            if conn.strength < 0.05:
                del self.connections[key]
        
        # Decaimento temporal
        conn.strength *= (1.0 - self.decay_rate)
        conn.strength = max(0.0, min(1.0, conn.strength))
    
    def get_connection_weight(self, source_id: str, target_id: str) -> float:
        """Retorna o peso atual da conexão"""
        key = (source_id, target_id)
        if key in self.connections:
            return self.connections[key].weight * self.connections[key].strength
        return 0.0
    
    def get_connection_strength(self, source_id: str, target_id: str) -> float:
        """Retorna a força da conexão"""
        key = (source_id, target_id)
        if key in self.connections:
            return self.connections[key].strength
        return 0.0
    
    def prune_connections(self, min_strength: float = 0.1):
        """
        Remove conexões abaixo de um limiar de força.
        Simula poda sináptica neuronal.
        """
        to_remove = [
            key for key, conn in self.connections.items()
            if conn.strength < min_strength
        ]
        
        for key in to_remove:
            del self.connections[key]
    
    def get_active_connections(self, min_activity: int = 5) -> List[Tuple[str, str]]:
        """Retorna conexões com atividade acima do mínimo"""
        return [
            (source, target) for (source, target), conn in self.connections.items()
            if conn.activation_count >= min_activity
        ]
    
    def reinforce_pathway(self, path: List[str], reward: float):
        """
        Reforça um caminho de conexões (reward-based learning).
        
        Args:
            path: Lista de IDs de neurônios formando um caminho
            reward: Valor do reforço (positivo ou negativo)
        """
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            key = (source_id, target_id)
            if key in self.connections:
                conn = self.connections[key]
                
                if reward > 0:
                    # Reforço positivo
                    conn.strengthen(1.0 + reward)
                else:
                    # Reforço negativo
                    conn.weaken(1.0 + abs(reward))
    
    def adapt_to_context(self, context_signal: torch.Tensor, 
                        neuron_ids: List[str]):
        """
        Adapta conexões baseado em sinal contextual.
        
        Args:
            context_signal: Tensor de contexto (batch_size, context_dim)
            neuron_ids: IDs dos neurônios ativos
        """
        # Calcula atenção contextual
        context_mean = context_signal.mean().item()
        
        # Ajusta todas as conexões relacionadas aos neurônios ativos
        for source_id in neuron_ids:
            for target_id in neuron_ids:
                if source_id != target_id:
                    key = (source_id, target_id)
                    if key in self.connections:
                        conn = self.connections[key]
                        # Multiplica força pela atenção contextual
                        conn.strength *= (0.9 + context_mean * 0.1)
    
    def get_topology_statistics(self) -> Dict:
        """Retorna estatísticas da topologia"""
        if not self.connections:
            return {
                'total_connections': 0,
                'avg_strength': 0.0,
                'avg_activation': 0,
                'strong_connections': 0
            }
        
        strengths = [conn.strength for conn in self.connections.values()]
        activations = [conn.activation_count for conn in self.connections.values()]
        
        return {
            'total_connections': len(self.connections),
            'avg_strength': np.mean(strengths),
            'avg_activation': np.mean(activations),
            'strong_connections': len([s for s in strengths if s > 0.7]),
            'weak_connections': len([s for s in strengths if s < 0.3])
        }
    
    def save_state(self) -> Dict:
        """Salva o estado das conexões"""
        return {
            'connections': {
                f"{src}_{tgt}": {
                    'weight': conn.weight,
                    'strength': conn.strength,
                    'age': conn.age,
                    'activation_count': conn.activation_count,
                    'success_count': conn.success_count,
                    'connection_type': conn.connection_type
                }
                for (src, tgt), conn in self.connections.items()
            }
        }
    
    def load_state(self, state: Dict):
        """Carrega o estado das conexões"""
        self.connections.clear()
        
        for key, data in state['connections'].items():
            src, tgt = key.split('_')
            conn = SynapticConnection(
                source_id=src,
                target_id=tgt,
                weight=data['weight'],
                strength=data['strength'],
                age=data['age'],
                activation_count=data['activation_count'],
                success_count=data['success_count'],
                connection_type=data['connection_type']
            )
            self.connections[(src, tgt)] = conn


class ContextualPlasticity(nn.Module):
    """
    Plasticidade contextual avançada.
    Adapta conexões baseado em contexto e atenção.
    """
    
    def __init__(self, context_dim: int, num_neurons: int):
        super().__init__()
        
        self.context_dim = context_dim
        self.num_neurons = num_neurons
        
        # Módulos de atenção contextual
        self.context_encoder = nn.Linear(context_dim, context_dim)
        self.attention_layer = nn.Linear(context_dim, num_neurons * num_neurons)
        self.plasticity_gate = nn.Linear(context_dim, 1)
        
    def forward(self, context: torch.Tensor, 
               connection_mask: torch.Tensor) -> torch.Tensor:
        """
        Gera máscara de plasticidade baseada em contexto.
        
        Args:
            context: Tensor contextual (batch_size, context_dim)
            connection_mask: Máscara de conexões existentes (num_neurons, num_neurons)
        """
        batch_size = context.size(0)
        
        # Encoda contexto
        encoded_context = torch.tanh(self.context_encoder(context))
        
        # Gera pesos de atenção
        attention_weights = torch.sigmoid(self.attention_layer(encoded_context))
        attention_weights = attention_weights.view(batch_size, 
                                                   self.num_neurons, 
                                                   self.num_neurons)
        
        # Gate de plasticidade
        plasticity_signal = torch.sigmoid(self.plasticity_gate(encoded_context))
        
        # Aplica máscara e gate
        adaptive_mask = connection_mask.unsqueeze(0) * attention_weights * plasticity_signal
        
        return adaptive_mask

