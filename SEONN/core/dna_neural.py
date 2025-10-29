"""
DNA Neural: Identidade Evolutiva de Neurônios Artificiais
========================================================

O DNA Neural atua como um código interno de identidade para cada neurônio,
armazenando informações fundamentais como:
- Identidade única
- Histórico de aprendizado
- Parâmetros funcionais
- Domínio de especialização

Inspirado por conceitos de genômica computacional e aprendizagem evolutiva.
"""

import numpy as np
import torch
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class NeuralDNA:
    """
    Estrutura de DNA Neural que representa a identidade evolutiva de um neurônio.
    """
    # Identidade única
    neuron_id: str
    generation: int
    birth_timestamp: float
    
    # Histórico de aprendizado
    activation_count: int = 0
    successful_activations: int = 0
    total_inputs_processed: int = 0
    error_history: List[float] = None
    
    # Parâmetros funcionais
    specialization: str = "general"  # Domínio de especialização
    learning_rate: float = 0.01
    memory_strength: float = 0.5
    plasticity_factor: float = 0.8
    
    # Capacidades adaptativas
    adaptation_rate: float = 0.1
    confidence_level: float = 0.5
    energy_efficiency: float = 1.0
    
    # Metadados
    parent_ids: List[str] = None
    mutation_count: int = 0
    crossover_count: int = 0
    
    def __post_init__(self):
        """Inicializa listas vazias se None"""
        if self.error_history is None:
            self.error_history = []
        if self.parent_ids is None:
            self.parent_ids = []

    def activate(self, success: bool = True):
        """Registra uma ativação do neurônio"""
        self.activation_count += 1
        if success:
            self.successful_activations += 1
        self.total_inputs_processed += 1
        self._update_confidence()

    def record_error(self, error: float):
        """Registra um erro para histórico"""
        self.error_history.append(error)
        if len(self.error_history) > 100:  # Mantém apenas os últimos 100
            self.error_history.pop(0)

    def update_specialization(self, task_type: str, frequency: float):
        """Atualiza a especialização baseado na frequência de tarefas"""
        if frequency > 0.7:  # Mais de 70% das ativações
            self.specialization = task_type
            self.confidence_level = min(1.0, self.confidence_level + 0.01)

    def evolve_parameters(self, learning_rate: float = None, 
                         plasticity: float = None):
        """Evolui os parâmetros funcionais"""
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if plasticity is not None:
            self.plasticity_factor = plasticity
        self.mutation_count += 1

    def _update_confidence(self):
        """Atualiza o nível de confiança baseado no histórico"""
        if self.activation_count > 0:
            success_rate = self.successful_activations / self.activation_count
            self.confidence_level = success_rate

    def clone(self, new_id: str) -> 'NeuralDNA':
        """Clona o DNA com mutações controladas"""
        new_dna = NeuralDNA(
            neuron_id=new_id,
            generation=self.generation + 1,
            birth_timestamp=datetime.now().timestamp(),
            specialization=self.specialization,
            learning_rate=self.learning_rate,
            plasticity_factor=self.plasticity_factor,
            adaptation_rate=self.adaptation_rate,
            confidence_level=self.confidence_level * 0.9,  # Herda confiança reduzida
            parent_ids=[self.neuron_id],
            memory_strength=self.memory_strength
        )
        return new_dna

    def crossover(self, other: 'NeuralDNA', child_id: str) -> 'NeuralDNA':
        """
        Realiza crossover genético entre dois DNAs.
        Inspirado em algoritmos genéticos.
        """
        # Combina características dos pais
        specialization = self.specialization if np.random.rand() > 0.5 else other.specialization
        learning_rate = (self.learning_rate + other.learning_rate) / 2
        plasticity_factor = (self.plasticity_factor + other.plasticity_factor) / 2
        
        child = NeuralDNA(
            neuron_id=child_id,
            generation=max(self.generation, other.generation) + 1,
            birth_timestamp=datetime.now().timestamp(),
            specialization=specialization,
            learning_rate=learning_rate,
            plasticity_factor=plasticity_factor,
            parent_ids=[self.neuron_id, other.neuron_id],
            crossover_count=1
        )
        
        self.crossover_count += 1
        other.crossover_count += 1
        
        return child

    def mutate(self, mutation_rate: float = 0.1):
        """
        Aplica mutação ao DNA Neural.
        Inspirado em algoritmos evolutivos.
        """
        if np.random.rand() < mutation_rate:
            # Muta learning rate
            self.learning_rate *= np.random.uniform(0.8, 1.2)
            self.mutation_count += 1
        
        if np.random.rand() < mutation_rate:
            # Muta fator de plasticidade
            self.plasticity_factor = np.clip(
                self.plasticity_factor * np.random.uniform(0.9, 1.1),
                0.1, 1.0
            )
            self.mutation_count += 1

    def get_fitness_score(self) -> float:
        """Calcula um score de fitness do neurônio"""
        success_rate = self.successful_activations / max(1, self.activation_count)
        avg_error = np.mean(self.error_history) if self.error_history else 0.5
        error_score = 1.0 - min(avg_error, 1.0)
        
        fitness = (
            success_rate * 0.4 +
            error_score * 0.3 +
            self.confidence_level * 0.2 +
            self.energy_efficiency * 0.1
        )
        
        return fitness

    def to_dict(self) -> Dict[str, Any]:
        """Converte o DNA para dicionário"""
        return asdict(self)

    def from_dict(self, data: Dict[str, Any]) -> 'NeuralDNA':
        """Carrega DNA de um dicionário"""
        return NeuralDNA(**data)

    def __repr__(self):
        return (f"NeuralDNA(id={self.neuron_id}, gen={self.generation}, "
                f"spec={self.specialization}, confidence={self.confidence_level:.2f})")


class DNAManager:
    """
    Gerencia uma população de DNAs Neurais.”
    
    Funcionalidades:
    - Seleção por fitness
    - Operações de crossover
    - Mutação populacional
    - Análise de diversidade genética
    """
    
    def __init__(self, population_size: int = 100):
        self.population: List[NeuralDNA] = []
        self.population_size = population_size
        self.generation_counter = 0

    def add_dna(self, dna: NeuralDNA):
        """Adiciona um DNA à população"""
        self.population.append(dna)

    def select_by_fitness(self, n: int) -> List[NeuralDNA]:
        """Seleciona os N melhores DNAs por fitness"""
        sorted_pop = sorted(self.population, 
                          key=lambda x: x.get_fitness_score(), 
                          reverse=True)
        return sorted_pop[:n]

    def evolve_population(self, mutation_rate: float = 0.1,
                         crossover_rate: float = 0.3):
        """
        Evolui a população através de seleção, crossover e mutação.
        
        Inspirado em algoritmos genéticos clássicos.
        """
        # Seleção
        elite = self.select_by_fitness(int(self.population_size * 0.2))
        new_population = elite.copy()
        
        # Crossover entre os melhores
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(elite, 2, replace=False)
            
            if np.random.rand() < crossover_rate:
                child_id = f"neuron_{self.generation_counter}_{len(new_population)}"
                child = parent1.crossover(parent2, child_id)
                child.mutate(mutation_rate)
                new_population.append(child)
            else:
                # Clonagem com mutação
                child_id = f"neuron_{self.generation_counter}_{len(new_population)}"
                child = parent1.clone(child_id)
                child.mutate(mutation_rate)
                new_population.append(child)
        
        self.population = new_population
        self.generation_counter += 1
        
        return self.population

    def get_diversity_score(self) -> float:
        """Calcula a diversidade genética da população"""
        if len(self.population) < 2:
            return 1.0
        
        specializations = [dna.specialization for dna in self.population]
        unique_specs = len(set(specializations))
        
        return unique_specs / len(specializations)

    def __len__(self):
        return len(self.population)

