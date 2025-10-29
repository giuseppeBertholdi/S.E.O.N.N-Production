import torch
from torch.nn import Parameter
import torch.nn.functional as F

class DynamicNeuralGraph:
    """
    High-performance dynamic neural graph with biological mechanisms.
    Optimized for speed and memory efficiency.
    """
    def __init__(self, num_neurons: int, device='cpu'):
        self.num_neurons = num_neurons
        self.device = device
        self.indices = torch.empty([2, 0], dtype=torch.long, device=self.device)
        self.weights = Parameter(torch.empty(0), requires_grad=True)
        
        # Performance optimization
        self._cached_sparse = None
        self._cache_valid = False
        
        # Biological tracking
        self.connection_age = torch.empty(0, dtype=torch.long, device=self.device)
        self.connection_strength_history = torch.empty(0, device=self.device)

    def set_connections(self, connections: torch.Tensor, weights: Parameter):
        """
        Sets the connections and their weights with performance optimization.
        """
        # Coalesce to remove duplicate connections and sum their initial weights
        coalesced = torch.sparse_coo_tensor(
            connections, weights, 
            (self.num_neurons, self.num_neurons)
        ).coalesce()
        
        self.indices = coalesced.indices()
        self.weights = Parameter(coalesced.values(), requires_grad=True)
        
        # Update biological tracking
        self._update_biological_tracking()
        
        # Invalidate cache
        self._cache_valid = False

    def _update_biological_tracking(self):
        """Update biological tracking parameters"""
        num_connections = self.indices.shape[1]
        
        # Initialize or extend connection age
        if self.connection_age.numel() == 0:
            self.connection_age = torch.zeros(num_connections, dtype=torch.long, device=self.device)
        elif self.connection_age.numel() < num_connections:
            # Extend for new connections
            new_ages = torch.zeros(num_connections - self.connection_age.numel(), dtype=torch.long, device=self.device)
            self.connection_age = torch.cat([self.connection_age, new_ages])
        elif self.connection_age.numel() > num_connections:
            # Truncate for removed connections
            self.connection_age = self.connection_age[:num_connections]
        
        # Initialize or extend strength history
        if self.connection_strength_history.numel() == 0:
            self.connection_strength_history = torch.zeros(num_connections, device=self.device)
        elif self.connection_strength_history.numel() < num_connections:
            # Extend for new connections
            new_strengths = torch.zeros(num_connections - self.connection_strength_history.numel(), device=self.device)
            self.connection_strength_history = torch.cat([self.connection_strength_history, new_strengths])
        elif self.connection_strength_history.numel() > num_connections:
            # Truncate for removed connections
            self.connection_strength_history = self.connection_strength_history[:num_connections]

    def age_connections(self):
        """Age all connections (biological mechanism)"""
        if self.connection_age.numel() > 0:
            with torch.no_grad():
                self.connection_age += 1

    def update_strength_history(self):
        """Update connection strength history"""
        if self.connection_strength_history.numel() > 0:
            # Exponential moving average
            alpha = 0.1
            with torch.no_grad():
                self.connection_strength_history = (1 - alpha) * self.connection_strength_history + alpha * torch.abs(self.weights.data)

    def get_cached_sparse(self):
        """Get cached sparse matrix for performance"""
        if not self._cache_valid or self._cached_sparse is None:
            self._cached_sparse = torch.sparse_coo_tensor(
                self.indices,
                self.weights.data,  # Use .data to avoid gradient graph
                (self.num_neurons, self.num_neurons)
            )
            self._cache_valid = True
        return self._cached_sparse

    def get_connection_statistics(self):
        """Get statistics about connections for biological analysis"""
        if self.weights.numel() == 0:
            return {
                'num_connections': 0,
                'mean_strength': 0.0,
                'std_strength': 0.0,
                'mean_age': 0.0,
                'active_connections': 0
            }
        
        return {
            'num_connections': self.weights.numel(),
            'mean_strength': self.weights.abs().mean().item(),
            'std_strength': self.weights.abs().std().item(),
            'mean_age': self.connection_age.float().mean().item() if self.connection_age.numel() > 0 else 0.0,
            'active_connections': (self.weights.abs() > 0.01).sum().item()
        }

    def __repr__(self):
        stats = self.get_connection_statistics()
        return f"DynamicNeuralGraph(Neurons: {self.num_neurons}, Connections: {stats['num_connections']}, Active: {stats['active_connections']})"
