"""
SEONN Core - Componentes Principais
===================================

Componentes da Self-Evolving Organic Neural Network
"""

from .dna_neural import NeuralDNA, DNAManager
from .neuron import AutonomicNeuron, NeuronCluster, AdaptiveActivation
from .plasticity import PlasticityMechanism, SynapticConnection, ContextualPlasticity
from .manager import NucleusManager, TaskContext, ActivityCoordinator
from .graph import DynamicNeuralGraph
from .seonn_model import SEONN_Model

__all__ = [
    'NeuralDNA',
    'DNAManager',
    'AutonomicNeuron',
    'NeuronCluster',
    'AdaptiveActivation',
    'PlasticityMechanism',
    'SynapticConnection',
    'ContextualPlasticity',
    'NucleusManager',
    'TaskContext',
    'ActivityCoordinator',
    'DynamicNeuralGraph',
    'SEONN_Model'
]

