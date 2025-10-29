import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from core.graph import DynamicNeuralGraph

class SEONN_Model(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, initial_neurons: int = 1000, 
                 initial_connectivity: float = 0.01, learning_rate=1e-3, 
                 plasticity_rate=0.01, homeostasis_strength=0.1, competition_strength=0.05):
        super(SEONN_Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_neurons = initial_neurons
        self.plasticity_rate = plasticity_rate
        self.homeostasis_strength = homeostasis_strength
        self.competition_strength = competition_strength
        
        # --- Biological parameters ---
        self.activity_history = torch.zeros(initial_neurons)
        self.synaptic_strength = torch.ones(initial_neurons, initial_neurons) * 0.1
        self.neuron_health = torch.ones(initial_neurons)
        self.learning_rate_adaptive = torch.ones(initial_neurons) * learning_rate
        
        # --- Initialize organic sparse topology ---
        self.graph = DynamicNeuralGraph(num_neurons=initial_neurons)
        self._initialize_organic_connections(initial_connectivity, input_size, output_size)

        # --- Map input and output dimensions to neurons ---
        if initial_neurons < input_size + output_size:
            raise ValueError(f"initial_neurons ({initial_neurons}) must be >= input_size + output_size ({input_size + output_size})")
        
        self.input_neurons = torch.arange(0, input_size)
        self.output_neurons = torch.arange(input_size, input_size + output_size)

        # --- Organic activation with biological variability ---
        self.base_activation = nn.GELU()
        self.activation_variability = torch.randn(initial_neurons) * 0.1

        # --- Adaptive optimizer with biological learning ---
        self.optimizer = torch.optim.AdamW([
            {'params': self.weights, 'lr': learning_rate, 'weight_decay': 1e-5}
        ])
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # --- Performance optimization ---
        self._cached_adjacency = None
        self._cache_valid = False
        self.propagation_steps = 5  # Reduced for speed
        
    def _initialize_organic_connections(self, initial_connectivity, input_size, output_size):
        """Initialize connections with biological-like patterns"""
        # Create more structured initial connectivity
        num_connections = int(self.num_neurons * self.num_neurons * initial_connectivity)
        
        # Prefer connections between nearby neurons (biological realism)
        connections = []
        weights = []
        
        # Input to hidden connections (more dense)
        for i in range(input_size):
            for j in range(input_size, min(input_size + 200, self.num_neurons)):
                if torch.rand(1) < initial_connectivity * 3:  # 3x more likely
                    connections.append([i, j])
                    weights.append(torch.randn(1) * 0.1)
        
        # Hidden to hidden connections (sparse but structured)
        for i in range(input_size, self.num_neurons - output_size):
            for j in range(i + 1, min(i + 50, self.num_neurons - output_size)):
                if torch.rand(1) < initial_connectivity:
                    connections.append([i, j])
                    weights.append(torch.randn(1) * 0.05)
        
        # Hidden to output connections (more dense)
        for i in range(self.num_neurons - output_size, self.num_neurons):
            for j in range(max(0, i - 100), i):
                if torch.rand(1) < initial_connectivity * 2:  # 2x more likely
                    connections.append([j, i])
                    weights.append(torch.randn(1) * 0.1)
        
        # Add random connections to fill remaining
        remaining = num_connections - len(connections)
        if remaining > 0:
            random_conns = torch.randint(0, self.num_neurons, (2, remaining))
            random_weights = torch.randn(remaining) * 0.05
            connections.extend(random_conns.t().tolist())
            weights.extend(random_weights.tolist())
        
        # Convert to tensors and create sparse representation
        if connections:
            connections_tensor = torch.tensor(connections[:num_connections]).t()
            weights_tensor = torch.tensor(weights[:num_connections])
        else:
            connections_tensor = torch.empty(2, 0, dtype=torch.long)
            weights_tensor = torch.empty(0)
        
        # Coalesce and set connections
        coalesced_sparse_tensor = torch.sparse_coo_tensor(
            connections_tensor, weights_tensor, 
            (self.num_neurons, self.num_neurons)
        ).coalesce()
        
        self.weights = Parameter(coalesced_sparse_tensor.values(), requires_grad=True)
        self.graph.set_connections(coalesced_sparse_tensor.indices(), self.weights)

    def forward(self, x: torch.Tensor):
        """ 
        Performs the organic forward pass with biological mechanisms.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize state with biological variability
        state = torch.zeros(batch_size, self.num_neurons, device=device)
        state[:, self.input_neurons] = x
        
        # Move biological parameters to device if needed
        if self.activity_history.device != device:
            self.activity_history = self.activity_history.to(device)
            self.activation_variability = self.activation_variability.to(device)
            self.neuron_health = self.neuron_health.to(device)

        # Create adjacency matrix for this forward pass (not cached to avoid gradient issues)
        adjacency_matrix = torch.sparse_coo_tensor(
            self.graph.indices,
            self.weights,
            (self.num_neurons, self.num_neurons)
        )

        # Organic propagation with biological mechanisms
        for step in range(self.propagation_steps):
            # Sparse matrix multiplication (optimized)
            new_state = torch.sparse.mm(adjacency_matrix, state.t()).t()
            
            # Apply biological mechanisms
            state = self._apply_biological_mechanisms(state, new_state, step)
            
            # Update activity history for plasticity (detached)
            self._update_activity_history(state.mean(0).detach())

        output = state[:, self.output_neurons]
        return output
    
    def _apply_biological_mechanisms(self, state, new_state, step):
        """Apply biological mechanisms during forward pass"""
        # Simplified biological mechanisms to avoid gradient issues
        # Apply activation
        activated = self.base_activation(state + new_state)
        
        return activated
    
    def _update_activity_history(self, current_activity):
        """Update activity history for plasticity mechanisms"""
        # Exponential moving average (detached to avoid gradient issues)
        alpha = 0.1
        with torch.no_grad():
            self.activity_history = (1 - alpha) * self.activity_history + alpha * current_activity.detach()

    def train_step(self, data, targets):
        """
        Performs organic training step with biological learning mechanisms.
        """
        self.optimizer.zero_grad()
        output = self.forward(data)
        loss = self.loss_fn(output, targets)
        loss.backward()
        
        # Apply biological learning mechanisms (before optimizer step)
        self._apply_biological_learning(data, targets, output)
        
        self.optimizer.step()
        
        # Update biological parameters
        self._update_biological_parameters()
        
        # Update graph biological tracking
        self.graph.age_connections()
        self.graph.update_strength_history()
        
        return loss.item()
    
    def _apply_biological_learning(self, data, targets, output):
        """Apply biological learning mechanisms like LTP/LTD"""
        # Simplified biological learning without gradient manipulation
        with torch.no_grad():
            # Simple reward-based learning
            correct_mask = (output.argmax(1) == targets).float()
            reward = correct_mask.mean()
            
            # Update learning rate based on performance
            if reward > 0.8:  # Good performance
                self.learning_rate_adaptive *= 1.01
            elif reward < 0.3:  # Poor performance
                self.learning_rate_adaptive *= 0.99
            
            # Clamp learning rates
            self.learning_rate_adaptive = torch.clamp(self.learning_rate_adaptive, 1e-6, 1e-2)
    
    def _update_biological_parameters(self):
        """Update biological parameters based on network performance"""
        with torch.no_grad():
            # Update neuron health based on activity
            activity_threshold = 0.1
            low_activity = self.activity_history < activity_threshold
            high_activity = self.activity_history > 0.8
            
            # Neurons with low activity lose health, high activity gain health
            self.neuron_health[low_activity] *= 0.99
            self.neuron_health[high_activity] = torch.clamp(self.neuron_health[high_activity] * 1.01, 0, 1)
            
            # Update adaptive learning rates
            performance_factor = 1.0 - self.activity_history.std()
            self.learning_rate_adaptive *= (0.99 + 0.01 * performance_factor)
            self.learning_rate_adaptive = torch.clamp(self.learning_rate_adaptive, 1e-6, 1e-2)

    def evolve(self, reward_signal=None, pruning_threshold: float = 0.001):
        """
        Organic evolution with biological mechanisms and performance optimization.
        """
        pruned_indices_for_viz = torch.empty(2, 0, dtype=torch.long, device=self.weights.device)
        new_indices_for_viz = torch.empty(2, 0, dtype=torch.long, device=self.weights.device)

        if self.weights.numel() == 0:
            return pruned_indices_for_viz, new_indices_for_viz

        # --- Organic Pruning Phase ---
        # Prune based on weight magnitude AND biological factors
        weight_mask = torch.abs(self.weights) >= pruning_threshold
        
        # Prune connections to unhealthy neurons
        unhealthy_neurons = self.neuron_health < 0.3
        if unhealthy_neurons.any():
            # Find connections involving unhealthy neurons
            from_indices = self.graph.indices[0]
            to_indices = self.graph.indices[1]
            unhealthy_mask = unhealthy_neurons[from_indices] | unhealthy_neurons[to_indices]
            weight_mask = weight_mask & ~unhealthy_mask
        
        # Prune connections with low activity
        low_activity_neurons = self.activity_history < 0.05
        if low_activity_neurons.any():
            from_indices = self.graph.indices[0]
            to_indices = self.graph.indices[1]
            low_activity_mask = low_activity_neurons[from_indices] & low_activity_neurons[to_indices]
            weight_mask = weight_mask & ~low_activity_mask
        
        # Store pruned indices for visualization
        pruned_mask = ~weight_mask
        if pruned_mask.any():
            pruned_indices_for_viz = self.graph.indices[:, pruned_mask]

        # Update indices and weights after pruning
        new_indices_after_pruning = self.graph.indices[:, weight_mask]
        new_weights_after_pruning = Parameter(self.weights[weight_mask].detach().clone(), requires_grad=True)

        # --- Organic Growth Phase ---
        # Grow connections based on biological principles
        num_new_connections = self._calculate_organic_growth()
        new_connection_indices = self._generate_organic_connections(num_new_connections)
        new_connection_weights = self._generate_organic_weights(num_new_connections)

        # Store new indices for visualization
        new_indices_for_viz = new_connection_indices

        # Combine and coalesce
        if new_indices_after_pruning.numel() > 0 and new_connection_indices.numel() > 0:
            combined_indices = torch.cat([new_indices_after_pruning, new_connection_indices], dim=1)
            combined_weights = torch.cat([new_weights_after_pruning.data, new_connection_weights], dim=0)
        elif new_indices_after_pruning.numel() > 0:
            combined_indices = new_indices_after_pruning
            combined_weights = new_weights_after_pruning.data
        else:
            combined_indices = new_connection_indices
            combined_weights = new_connection_weights

        coalesced_sparse_tensor = torch.sparse_coo_tensor(
            combined_indices,
            combined_weights,
            (self.num_neurons, self.num_neurons)
        ).coalesce()

        self.graph.set_connections(coalesced_sparse_tensor.indices(), coalesced_sparse_tensor.values())
        self.weights = Parameter(coalesced_sparse_tensor.values(), requires_grad=True)

        # Update optimizer efficiently
        self.optimizer.param_groups[0]['params'] = [self.weights]

        print(f"Organic evolution: Pruned {pruned_indices_for_viz.shape[1] if pruned_indices_for_viz.numel() > 0 else 0} connections")
        print(f"Organic growth: Added {new_indices_for_viz.shape[1] if new_indices_for_viz.numel() > 0 else 0} connections")
        print(f"Total connections: {self.weights.numel()}")

        return pruned_indices_for_viz, new_indices_for_viz
    
    def _calculate_organic_growth(self):
        """Calculate number of new connections based on biological principles"""
        # Base growth rate
        base_growth = 50
        
        # Increase growth if network performance is poor
        activity_variance = self.activity_history.var()
        if activity_variance < 0.01:  # Low variance = poor performance
            base_growth *= 2
        
        # Increase growth if many neurons are unhealthy
        unhealthy_ratio = (self.neuron_health < 0.5).float().mean()
        base_growth = int(base_growth * (1 + unhealthy_ratio))
        
        return min(base_growth, 200)  # Cap at 200 new connections
    
    def _generate_organic_connections(self, num_connections):
        """Generate new connections based on biological principles"""
        if num_connections == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.weights.device)
        
        # Prefer connections between active neurons
        active_neurons = self.activity_history > 0.1
        if active_neurons.sum() < 2:
            # Fallback to random if not enough active neurons
            return torch.randint(0, self.num_neurons, (2, num_connections), device=self.weights.device)
        
        active_indices = torch.where(active_neurons)[0]
        
        # Generate connections between active neurons
        from_neurons = active_indices[torch.randint(0, len(active_indices), (num_connections,))]
        to_neurons = active_indices[torch.randint(0, len(active_indices), (num_connections,))]
        
        return torch.stack([from_neurons, to_neurons])
    
    def _generate_organic_weights(self, num_connections):
        """Generate weights for new connections based on biological principles"""
        if num_connections == 0:
            return torch.empty(0, device=self.weights.device)
        
        # Use current weight statistics for initialization
        if self.weights.numel() > 0:
            weight_mean = self.weights.mean().item()
            weight_std = self.weights.std().item()
        else:
            weight_mean = 0.0
            weight_std = 0.1
        
        # Generate weights with biological variability
        new_weights = torch.normal(weight_mean, weight_std, (num_connections,), device=self.weights.device)
        
        # Scale by neuron health
        from_indices = torch.randint(0, self.num_neurons, (num_connections,), device=self.weights.device)
        health_factors = self.neuron_health[from_indices]
        new_weights *= health_factors
        
        return new_weights


