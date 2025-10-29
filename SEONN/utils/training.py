"""
Utilidades de Treinamento
=========================
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable
from ..core import SEONN_Model, TaskContext


def train_seonn(model: SEONN_Model,
                train_loader,
                optimizer,
                criterion,
                num_epochs: int = 10,
                device: str = 'cuda'):
    """
    Treina a SEONN em um dataset.
    
    Args:
        model: Modelo SEONN
        train_loader: DataLoader de treino
        optimizer: Otimizador
        criterion: Função de loss
        num_epochs: Número de épocas
        device: Dispositivo
    
    Returns:
        Histórico de treinamento
    """
    history = {
        'loss': [],
        'accuracy': [],
        'active_neurons': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            task_context = TaskContext(
                task_id=f"epoch_{epoch}",
                task_type="training",
                complexity=0.5
            )
            
            result = model.train_step(
                x=x,
                y=y,
                task_context=task_context,
                optimizer=optimizer,
                criterion=criterion
            )
            
            epoch_loss += result['loss']
            epoch_acc += result['accuracy']
            batches += 1
        
        avg_loss = epoch_loss / batches
        avg_acc = epoch_acc / batches
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)
        history['active_neurons'].append(result['num_active_neurons'])
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss={avg_loss:.4f}, Acc={avg_acc*100:.2f}%")
    
    return history


def evaluate_seonn(model: SEONN_Model,
                  test_loader,
                  criterion,
                  device: str = 'cuda'):
    """
    Avalia a SEONN em dados de teste.
    
    Args:
        model: Modelo SEONN
        test_loader: DataLoader de teste
        criterion: Função de loss
        device: Dispositivo
    
    Returns:
        Métricas de avaliação
    """
    model.eval()
    
    total_loss = 0
    total_acc = 0
    batches = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
            loss = criterion(output, y)
            accuracy = (output.argmax(dim=1) == y).float().mean()
            
            total_loss += loss.item()
            total_acc += accuracy.item()
            batches += 1
    
    avg_loss = total_loss / batches
    avg_acc = total_acc / batches
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc
    }

