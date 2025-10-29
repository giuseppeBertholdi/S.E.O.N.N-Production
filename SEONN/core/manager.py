"""
Núcleo Gerenciador (udc Centrar de Auto-Organização)
=====================================================

Atua como um "gerenciador de cognição", observando o estado da rede e
ativando subconjuntos de neurônios com base em:
- Complexidade da tarefa
- Especialização funcional
- Auto-organização crítica
- Reconfiguração dinâmica

Inspirado em conceitos de auto-organização crítica e modulação flexível.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
from dataclasses import dataclass

from .neuron import AutonomicNeuron
from .plasticity import PlasticityMechanism, SynapticConnection


@dataclass
class TaskContext:
    """
    Contexto de uma tarefa sendo processada.
    """
    task_id: str
    task_type: str
    complexity: float
    required_specialization: str
    time_limit: Optional[float] = None
    energy_budget: float = 1.0


class NucleusManager(nn.Module):
    """
    Núcleo Gerenciador da SEONN.
    
    Responsabilidades:
    - Observação do estado global da rede
    - Ativação seletiva de neurônios
    - Modulação de plasticidade
    - Coordenação de especialização
    - Otimização energética
    """
    
    def __init__(self, 
                 total_neurons: int,
                 max_active_neurons: int = 100,
                 critical_state_threshold: float = 0.8):
        super().__init__()
        
        self.total_neurons = total_neurons
        self.max_active_neurons = max_active_neurons
        self.critical_state_threshold = critical_state_threshold
        
        # Ajusta dimensão para ser divisível por num_heads
        adjusted_dim = ((total_neurons + 3) // 4) * 4  # Próximo múltiplo de 4
        self.adjusted_dim = adjusted_dim
        
        # Estado global
        self.global_state = torch.zeros(total_neurons)
        self.active_subset: Set[int] = set()
        
        # Módulos de observação
        self.observation_layer = nn.Linear(total_neurons, total_neurons)
        self.activation_gate = nn.Linear(total_neurons, 1)
        self.complexity_estimator = nn.Linear(512, 10)
        
        # Sistema de atenção global
        self.global_attention = nn.MultiheadAttention(adjusted_dim, num_heads=4)
        
        # Coordenador de especialização
        self.specialization_router = nn.Linear(total_neurons, adjusted_dim)
        
        # Gerenciador de plasticidade
        self.plasticity_modulator = nn.Linear(10, total_neurons * total_neurons)
        
        # Memória de tarefas
        self.task_memory = deque(maxlen=50)
        
        # Estatísticas
        self.activation_history = deque(maxlen=1000)
        self.complexity_history = deque(maxlen=1000)
    
    def observe_network_state(self, neuron_activities: torch.Tensor) -> Dict:
        """
        Observa e analisa o estado atual da rede.
        
        Args:
            neuron_activities: Tensor de atividades neuronais (num_neurons,)
        
        Returns:
            Dict com análise do estado
        """
        # Observação filtrada
        observed_state = torch.tanh(self.observation_layer(neuron_activities))
        
        # Estimativa de estado crítico
        activity_variance = observed_state.var()
        activity_mean = observed_state.mean()
        critical_state = float(activity_variance / max(activity_mean, 0.01))
        
        # Identifica neurônios mais ativos
        top_k = self.max_active_neurons
        top_indices = torch.topk(observed_state, k=min(top_k, len(observed_state))).indices
        
        # Contagem por especialização (se disponível)
        specializations = self._get_specialization_counts(observed_state)
        
        return {
            'observed_state': observed_state,
            'critical_state': critical_state,
            'is_critical': critical_state > self.critical_state_threshold,
            'top_indices': top_indices.tolist(),
            'activity_variance': float(activity_variance),
            'activity_mean': float(activity_mean),
            'specializations': specializations
        }
    
    def _get_specialization_counts(self, activities: torch.Tensor) -> Dict[str, int]:
        """Extrai contagens de especialização (placeholder para implementação futura)"""
        return {'general': len(activities)}
    
    def select_active_subnet(self, 
                            task_context: TaskContext,
                            neuron_fitness: torch.Tensor) -> Set[int]:
        """
        Seleciona sub-rede ativa baseado em:
        - Complexidade da tarefa
        - Especialização necessária
        - Fitness dos neurônios
        - Orçamento de energia
        
        Args:
            task_context: Contexto da tarefa
            neuron_fitness: Fitness de cada neurônio
        
        Returns:
            Set com índices dos neurônios ativados
        """
        # Estimativa de número de neurônios necessários
        required_neurons = int(
            self.max_active_neurons * task_context.complexity * task_context.energy_budget
        )
        required_neurons = max(10, min(required_neurons, self.max_active_neurons))
        
        # Combina fitness com especialização
        combined_scores = neuron_fitness.clone()
        
        # Bonus para especialização correta
        # (implementação simplificada)
        specialization_bonus = torch.ones_like(neuron_fitness) * 0.1
        combined_scores += specialization_bonus
        
        # Seleciona top neurônios
        top_indices = torch.topk(combined_scores, k=required_neurons).indices
        
        active_subnet = set(top_indices.tolist())
        
        # Armazena história
        self.activation_history.append({
            'task_type': task_context.task_type,
            'num_neurons': len(active_subnet),
            'complexity': task_context.complexity
        })
        
        self.active_subset = active_subnet
        
        return active_subnet
    
    def modulate_plasticity(self, 
                           plasticity_mechanism: PlasticityMechanism,
                           task_context: TaskContext,
                           reward_signal: float) -> None:
        """
        Modula a plasticidade baseado em contexto e recompensa.
        
        Args:
            plasticity_mechanism: Mecanismo de plasticidade
            task_context: Contexto da tarefa
            reward_signal: Sinal de recompensa (-1 a 1)
        """
        # Calcula sinal de modulação
        modulation_signal = torch.tensor([
            task_context.complexity,
            abs(reward_signal),
            task_context.energy_budget,
            len(self.active_subset) / self.max_active_neurons,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # Placeholders
        ])
        
        # Modula plasticidade
        plasticity_weights = torch.sigmoid(
            self.plasticity_modulator(modulation_signal)
        )
        
        # Aplica modulação às conexões ativas
        # (implementação simplificada)
        pass
    
    def coordinate_functional_specialization(self,
                                            neuron_activities: torch.Tensor,
                                            task_results: Dict) -> None:
        """
        Coordena especialização funcional entre neurônios.
        
        Args:
            neuron_activities: Atividades neuronais
            task_results: Resultados das tarefas com sucesso
        """
        # Encaminha atividades com base em especialização
        coordinated = torch.tanh(
            self.specialization_router(neuron_activities)
        )
        
        # Atualiza especializações baseado em sucesso
        # (implementação simplificada)
        pass
    
    def optimize_energy_allocation(self, 
                                  current_energy: float,
                                  task_complexity: float) -> float:
        """
        Otimiza alocação de energia baseado em demanda.
        
        Args:
            current_energy: Energia atual disponível
            task_complexity: Complexidade da tarefa
        
        Returns:
            Energia alocada
        """
        # Cálculo de energia necessária
        base_energy = 0.3  # Energia mínima
        complexity_energy = task_complexity * 0.5
        allocation_energy = base_energy + complexity_energy
        
        # Ajusta baseado em energia disponível
        allocated = min(current_energy, allocation_energy)
        
        return allocated
    
    def is_in_critical_state(self) -> bool:
        """
        Verifica se a rede está em estado crítico de auto-organização.
        
        Baseado em conceitos de auto-organização crítica.
        """
        if len(self.activation_history) < 10:
            return False
        
        # Analisa variância de ativações recentes
        recent_complexities = [
            h['complexity'] for h in list(self.activation_history)[-10:]
        ]
        
        variance = np.var(recent_complexities)
        mean = np.mean(recent_complexities)
        
        criticality = variance / max(mean, 0.01)
        
        return criticality > self.critical_state_threshold
    
    def get_network_statistics(self) -> Dict:
        """
        Retorna estatísticas do gerenciador e da rede.
        """
        if not self.activation_history:
            return {}
        
        recent_activations = list(self.activation_history)[-20:]
        
        return {
            'num_active_neurons': len(self.active_subset),
            'avg_complexity': np.mean([h['complexity'] for h in recent_activations]),
            'avg_neurons_per_task': np.mean([h['num_neurons'] for h in recent_activations]),
            'is_critical_state': self.is_in_critical_state(),
            'network_health': np.mean([h['num_neurons'] for h in recent_activations]) / self.max_active_neurons
        }
    
    def save_state(self) -> Dict:
        """Salva o estado do gerenciador"""
        return {
            'active_subset': list(self.active_subset),
            'task_memory': list(self.task_memory),
            'state_dict': self.state_dict()
        }
    
    def load_state(self, state: Dict):
        """Carrega o estado do gerenciador"""
        self.active_subset = set(state['active_subset'])
        self.task_memory = deque(state['task_memory'], maxlen=50)
        self.load_state_dict(state['state_dict'])


class ActivityCoordinator:
    """
    Coordenador de atividades neuronais.
    Sincroniza e coordena ativação de neurônios.
    """
    
    def __init__(self, num_neurons: int):
        self.num_neurons = num_neurons
        self.activation_schedule = {}
        self.synchronization_buffer = []
    
    def schedule_activation(self, neuron_id: int, delay: float):
        """Agenda ativação de um neurônio"""
        self.activation_schedule[neuron_id] = delay
    
    def coordinate_synchronous_activation(self, neuron_ids: List[int]):
        """
        Coordena ativação síncrona de neurônios.
        Simula sincronização neural.
        """
        # Agrupa neurônios para ativação síncrona
        self.synchronization_buffer.extend(neuron_ids)
        
        if len(self.synchronization_buffer) >= 10:
            # Ativa em batch
            batch = self.synchronization_buffer[:10]
            self.synchronization_buffer = self.synchronization_buffer[10:]
            return batch
        
        return []

