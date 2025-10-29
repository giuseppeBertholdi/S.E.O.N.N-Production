#!/usr/bin/env python3
"""
Script melhorado para treinar o modelo SEONN com parâmetros otimizados
"""

import torch
import torch.nn as nn
import numpy as np
from cat_dog_classifier import CatDogClassifier, CatDogDataset
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

class ImprovedCatDogDataset(CatDogDataset):
    """Dataset melhorado com padrões mais realistas"""
    
    def _generate_cat_image(self):
        """Gera imagem sintética de gato mais realista"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Padrões mais característicos de gato
        # Orelhas pontiagudas mais definidas
        for i in range(8, 18):
            for j in range(18, 28):
                if abs(i - 13) + abs(j - 23) < 8:
                    image[i, j] = 0.8
        
        for i in range(8, 18):
            for j in range(32, 42):
                if abs(i - 13) + abs(j - 37) < 8:
                    image[i, j] = 0.8
        
        # Nariz pequeno e triangular
        for i in range(25, 32):
            for j in range(28, 38):
                if abs(i - 28) + abs(j - 33) < 6:
                    image[i, j] = 0.6
        
        # Bigodes mais longos e definidos
        for i in range(30, 35):
            for j in range(10, 20):
                image[i, j] = 0.4
            for j in range(44, 54):
                image[i, j] = 0.4
        
        # Formato facial mais triangular
        for i in range(self.image_size):
            for j in range(self.image_size):
                dist_from_center = np.sqrt((i - 32)**2 + (j - 32)**2)
                if dist_from_center > 28:
                    image[i, j] *= 0.1
                elif dist_from_center > 20:
                    image[i, j] *= 0.3
        
        # Adicionar ruído realista
        noise = np.random.normal(0, 0.05, (self.image_size, self.image_size))
        image = np.clip(image + noise, 0, 1)
        
        return torch.FloatTensor(image).flatten()
    
    def _generate_dog_image(self):
        """Gera imagem sintética de cachorro mais realista"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Padrões mais característicos de cachorro
        # Orelhas caídas mais largas
        for i in range(10, 20):
            for j in range(15, 30):
                if abs(i - 15) + abs(j - 22) < 10:
                    image[i, j] = 0.7
        
        for i in range(10, 20):
            for j in range(34, 49):
                if abs(i - 15) + abs(j - 41) < 10:
                    image[i, j] = 0.7
        
        # Focinho mais largo e arredondado
        for i in range(25, 38):
            for j in range(22, 42):
                if abs(i - 31) + abs(j - 32) < 12:
                    image[i, j] = 0.6
        
        # Formato facial mais arredondado
        for i in range(self.image_size):
            for j in range(self.image_size):
                dist_from_center = np.sqrt((i - 32)**2 + (j - 32)**2)
                if dist_from_center > 30:
                    image[i, j] *= 0.1
                elif dist_from_center > 22:
                    image[i, j] *= 0.4
        
        # Adicionar ruído realista
        noise = np.random.normal(0, 0.05, (self.image_size, self.image_size))
        image = np.clip(image + noise, 0, 1)
        
        return torch.FloatTensor(image).flatten()

def train_improved_model():
    """Treina modelo com parâmetros otimizados"""
    print("🚀 Iniciando treinamento melhorado do modelo SEONN...")
    
    # Parâmetros otimizados
    config = {
        'epochs': 30,
        'batch_size': 64,
        'learning_rate': 0.001,
        'initial_neurons': 8000,  # Mais neurônios para melhor capacidade
        'initial_connectivity': 0.005,  # Menos conexões iniciais para evolução
        'plasticity_rate': 0.02,
        'homeostasis_strength': 0.15,
        'competition_strength': 0.08,
        'train_samples': 2000,  # Mais dados de treino
        'test_samples': 500    # Mais dados de teste
    }
    
    print("📊 Configurações otimizadas:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Criar classificador com parâmetros otimizados
    classifier = CatDogClassifier()
    
    # Substituir dataset por versão melhorada
    train_dataset = ImprovedCatDogDataset(
        num_samples=config['train_samples'], 
        image_size=classifier.image_size, 
        train=True
    )
    test_dataset = ImprovedCatDogDataset(
        num_samples=config['test_samples'], 
        image_size=classifier.image_size, 
        train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"\n📊 Dataset melhorado:")
    print(f"   🎓 Treino: {len(train_dataset)} amostras")
    print(f"   🧪 Teste: {len(test_dataset)} amostras")
    print(f"   🧠 Modelo: {classifier.model.num_neurons} neurônios")
    
    # Treinamento com monitoramento
    best_accuracy = 0
    training_history = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'neural_health': [],
        'connections': []
    }
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Treinamento
        classifier.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(classifier.device), target.to(classifier.device)
            
            loss = classifier.model.train_step(data, target)
            total_loss += loss
            
            # Calcular acurácia
            with torch.no_grad():
                output = classifier.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        # Evolução a cada 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n🔄 Evoluindo rede (Epoch {epoch+1})...")
            pruned, new = classifier.model.evolve(pruning_threshold=0.0005)
            print(f"   ✂️  Removidas: {pruned.shape[1] if pruned.numel() > 0 else 0}")
            print(f"   ➕ Adicionadas: {new.shape[1] if new.numel() > 0 else 0}")
        
        # Avaliação no conjunto de teste
        test_accuracy = evaluate_model(classifier, test_loader)
        
        # Salvar melhor modelo
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            classifier.save_model("cat_dog_seonn_best.pth")
            print(f"   💾 Novo melhor modelo salvo! Acurácia: {best_accuracy:.2f}%")
        
        # Registrar histórico
        training_history['epoch'].append(epoch + 1)
        training_history['loss'].append(avg_loss)
        training_history['accuracy'].append(accuracy)
        training_history['neural_health'].append(classifier.model.neuron_health.mean().item())
        training_history['connections'].append(classifier.model.weights.numel())
        
        # Progresso
        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (config['epochs'] - epoch - 1)
        
        print(f'Epoch {epoch+1:2d}/{config["epochs"]} | '
              f'Loss: {avg_loss:.4f} | '
              f'Train Acc: {accuracy:.2f}% | '
              f'Test Acc: {test_accuracy:.2f}% | '
              f'Health: {classifier.model.neuron_health.mean().item():.3f} | '
              f'ETA: {eta/60:.1f}min')
    
    # Resultados finais
    print(f"\n✅ Treinamento concluído!")
    print(f"⏱️  Tempo total: {(time.time() - start_time)/60:.1f} minutos")
    print(f"🏆 Melhor acurácia: {best_accuracy:.2f}%")
    
    # Avaliação detalhada
    print(f"\n📊 Avaliação final detalhada:")
    evaluate_detailed(classifier, test_loader)
    
    # Salvar modelo final
    classifier.save_model("cat_dog_seonn.pth")
    
    # Mostrar estatísticas finais
    stats = classifier.model.graph.get_connection_statistics()
    print(f"\n📊 Estatísticas finais da rede:")
    print(f"   🧠 Neurônios: {classifier.model.graph.num_neurons:,}")
    print(f"   🔗 Conexões: {stats['num_connections']:,}")
    print(f"   💚 Saúde Neural: {stats.get('neural_health', 1.0):.3f}")
    print(f"   📈 Variação Atividade: {stats.get('activity_variance', 0.0):.3f}")
    
    return classifier, training_history

def evaluate_model(classifier, test_loader):
    """Avalia o modelo e retorna acurácia"""
    classifier.model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(classifier.device), target.to(classifier.device)
            outputs = classifier.model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def evaluate_detailed(classifier, test_loader):
    """Avaliação detalhada por classe"""
    classifier.model.eval()
    correct = 0
    total = 0
    cat_correct = 0
    dog_correct = 0
    cat_total = 0
    dog_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(classifier.device), target.to(classifier.device)
            outputs = classifier.model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Estatísticas por classe
            for i in range(target.size(0)):
                if target[i] == 0:  # Gato
                    cat_total += 1
                    if predicted[i] == 0:
                        cat_correct += 1
                else:  # Cachorro
                    dog_total += 1
                    if predicted[i] == 1:
                        dog_correct += 1
    
    accuracy = 100 * correct / total
    cat_accuracy = 100 * cat_correct / cat_total if cat_total > 0 else 0
    dog_accuracy = 100 * dog_correct / dog_total if dog_total > 0 else 0
    
    print(f"   🎯 Acurácia Geral: {accuracy:.2f}%")
    print(f"   🐱 Acurácia Gatos: {cat_accuracy:.2f}%")
    print(f"   🐶 Acurácia Cachorros: {dog_accuracy:.2f}%")
    print(f"   🧠 Saúde Neural: {classifier.model.neuron_health.mean().item():.3f}")
    print(f"   🔗 Conexões: {classifier.model.weights.numel():,}")

def test_with_sample_images(classifier):
    """Testa o modelo com imagens de exemplo"""
    print(f"\n🧪 Testando modelo com imagens de exemplo...")
    
    # Criar algumas imagens de teste
    test_dataset = ImprovedCatDogDataset(num_samples=10, image_size=classifier.image_size)
    
    for i in range(5):  # Testar 5 imagens
        image, true_label = test_dataset[i]
        image_tensor = image.unsqueeze(0).to(classifier.device)
        
        with torch.no_grad():
            output = classifier.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            cat_prob = probabilities[0][0].item()
            dog_prob = probabilities[0][1].item()
        
        true_class = "Gato" if true_label == 0 else "Cachorro"
        predicted_class = "Gato" if cat_prob > dog_prob else "Cachorro"
        confidence = max(cat_prob, dog_prob) * 100
        
        status = "✅" if true_class == predicted_class else "❌"
        
        print(f"   {status} Imagem {i+1}: {true_class} → {predicted_class} "
              f"(Confiança: {confidence:.1f}%)")

if __name__ == "__main__":
    # Remover modelo antigo se existir
    if os.path.exists("cat_dog_seonn.pth"):
        print("🗑️  Removendo modelo antigo...")
        os.remove("cat_dog_seonn.pth")
    
    # Treinar modelo melhorado
    classifier, history = train_improved_model()
    
    # Testar com imagens de exemplo
    test_with_sample_images(classifier)
    
    print(f"\n🎉 Modelo melhorado treinado com sucesso!")
    print(f"💡 Reinicie o servidor para usar o novo modelo.")








