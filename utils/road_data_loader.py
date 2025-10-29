import torch

def generate_road_pattern(pattern_type: str, noise_level: float = 0.0):
    """
    Generates a simple 5x5 pixel road pattern.
    0: background, 1: road, 2: obstacle
    """
    pattern = torch.zeros(25) # 5x5 grid

    if pattern_type == "straight":
        pattern[7:18:5] = 1.0 # Middle column
    elif pattern_type == "left_turn":
        pattern[[6, 11, 17, 23]] = 1.0 # Simple curve
    elif pattern_type == "right_turn":
        pattern[[8, 12, 18, 22]] = 1.0 # Simple curve
    elif pattern_type == "obstacle":
        pattern[12] = 2.0 # Center pixel is an obstacle

    # Add noise for "rainy" conditions
    if noise_level > 0:
        noise = torch.randn(25) * noise_level
        pattern = pattern + noise
        pattern = torch.clamp(pattern, 0.0, 2.0) # Keep values within range

    return pattern

def get_road_condition_loader(batch_size: int = 16, num_samples_per_type: int = 100, noise_level: float = 0.0):
    """
    Generates a simple dataset of road conditions.
    Labels: 0: straight, 1: left_turn, 2: right_turn, 3: obstacle
    """
    patterns = []
    labels = []

    pattern_types = {"straight": 0, "left_turn": 1, "right_turn": 2, "obstacle": 3}

    for p_type, label_idx in pattern_types.items():
        for _ in range(num_samples_per_type):
            patterns.append(generate_road_pattern(p_type, noise_level))
            labels.append(label_idx)

    dataset = torch.utils.data.TensorDataset(torch.stack(patterns), torch.tensor(labels, dtype=torch.long))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
