#!/usr/bin/env python3
"""
Script otimizado para treinar modelo SEONN até loss próximo de zero
"""

import torch
import numpy as np
from cat_dog_classifier import CatDogClassifier, CatDogDataset
import os
import time

class OptimizedCatDogDataset(CatDogDataset):
    """Dataset otimizado com padrões mais distintos"""
    
    def _generate_cat_image(self):
        """Gera imagem sintética de gato com padrões muito distintos"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Orelhas pontiagudas muito definidas
        for i in range(6, 20):
            for j in range(16, 26):
                if abs(i - 13) + abs(j - 21) < 6:
                    image[i, j] = 0.9
        
        for i in range(6, 20):
            for j in range(38, 48):
                if abs(i - 13) + abs(j - 43) < 6:
                    image[i, j] = 0.9
        
        # Nariz pequeno e triangular
        for i in range(26, 34):
            for j in range(30, 36):
                if abs(i - 30) + abs(j - 33) < 4:
                    image[i, j] = 0.8
        
        # Bigodes longos e definidos
        for i in range(32, 36):
            for j in range(8, 16):
                image[i, j] = 0.6
            for j in range(48, 56):
                image[i, j] = 0.6
        
        # Formato facial triangular
        for i in range(self.image_size):
            for j in range(self.image_size):
                dist_from_center = np.sqrt((i - 32)**2 + (j - 32)**2)
                if dist_from_center > 26:
                    image[i, j] *= 0.05
                elif dist_from_center > 18:
                    image[i, j] *= 0.2
        
        # Ruído mínimo
        noise = np.random.normal(0, 0.02, (self.image_size, self.image_size))
        image = np.clip(image + noise, 0, 1)
        
        return torch.FloatTensor(image).flatten()
    
    def _generate_dog_image(self):
        """Gera imagem sintética de cachorro com padrões muito distintos"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Orelhas caídas muito largas
        for i in range(8, 22):
            for j in range(12, 28):
                if abs(i - 15) + abs(j - 20) < 12:
                    image[i, j] = 0.8
        
        for i in range(8, 22):
            for j in range(36, 52):
                if abs(i - 15) + abs(j - 44) < 12:
                    image[i, j] = 0.8
        
        # Focinho largo e arredondado
        for i in range(26, 40):
            for j in range(20, 44):
                if abs(i - 33) + abs(j - 32) < 14:
                    image[i, j] = 0.7
        
        # Formato facial arredondado
        for i in range(self.image_size):
            for j in range(self.image_size):
                dist_from_center = np.sqrt((i - 32)**2 + (j - 32)**2)
                if dist_from_center > 28:
                    image[i, j] *= 0.05
                elif dist_from_center > 20:
                    image[i, j] *= 0.3
        
        # Ruído mínimo
        noise = np.random.normal(0, 0.02, (self.image_size, self.image_size))
        image = np.clip(image + noise, 0, 1)
        
        return torch.FloatTensor(image).flatten()

def train_until_zero_loss():
    """Treina modelo até loss próximo de zero"""
    print("🚀 Treinando modelo SEONN até loss próximo de zero...")
    
    # Configurações otimizadas para convergência rápida
    config = {
        'batch_size': 128,
        'learning_rate': 0.002,
        'initial_neurons': 6000,
        'initial_connectivity': 0.01,
        'plasticity_rate': 0.03,
        'homeostasis_strength': 0.2,
        'competition_strength': 0.1,
        'train_samples': 1000,
        'test_samples': 200,
        'target_loss': 0.0001,  # Parar quando loss < 0.0001
        'max_epochs': 50
    }
    
    print("📊 Configurações para convergência rápida:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Criar classificador
    classifier = CatDogClassifier()
    
    # Usar dataset otimizado
    train_dataset = OptimizedCatDogDataset(
        num_samples=config['train_samples'], 
        image_size=classifier.image_size, 
        train=True
    )
    test_dataset = OptimizedCatDogDataset(
        num_samples=config['test_samples'], 
        image_size=classifier.image_size, 
        train=False
    )
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"\n📊 Dataset otimizado:")
    print(f"   🎓 Treino: {len(train_dataset)} amostras")
    print(f"   🧪 Teste: {len(test_dataset)} amostras")
    print(f"   🧠 Modelo: {classifier.model.num_neurons} neurônios")
    
    # Treinamento com parada automática
    best_accuracy = 0
    start_time = time.time()
    
    for epoch in range(config['max_epochs']):
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
        
        # Avaliação no conjunto de teste
        test_accuracy = evaluate_model(classifier, test_loader)
        
        # Salvar melhor modelo
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            classifier.save_model("cat_dog_seonn_best.pth")
        
        # Progresso
        elapsed = time.time() - start_time
        
        print(f'Epoch {epoch+1:2d} | '
              f'Loss: {avg_loss:.6f} | '
              f'Train Acc: {accuracy:.2f}% | '
              f'Test Acc: {test_accuracy:.2f}% | '
              f'Health: {classifier.model.neuron_health.mean().item():.3f} | '
              f'Time: {elapsed/60:.1f}min')
        
        # Parar se loss for muito baixo
        if avg_loss < config['target_loss']:
            print(f"\n🎯 Meta atingida! Loss: {avg_loss:.6f} < {config['target_loss']}")
            break
        
        # Evolução a cada 3 epochs
        if (epoch + 1) % 3 == 0:
            print(f"   🔄 Evoluindo rede...")
            pruned, new = classifier.model.evolve(pruning_threshold=0.0001)
            print(f"   ✂️  Removidas: {pruned.shape[1] if pruned.numel() > 0 else 0}")
            print(f"   ➕ Adicionadas: {new.shape[1] if new.numel() > 0 else 0}")
    
    # Resultados finais
    print(f"\n✅ Treinamento concluído!")
    print(f"⏱️  Tempo total: {(time.time() - start_time)/60:.1f} minutos")
    print(f"🏆 Melhor acurácia: {best_accuracy:.2f}%")
    
    # Avaliação detalhada
    print(f"\n📊 Avaliação final:")
    evaluate_detailed(classifier, test_loader)
    
    # Salvar modelo final
    classifier.save_model("cat_dog_seonn.pth")
    
    # Mostrar estatísticas finais
    stats = classifier.model.graph.get_connection_statistics()
    print(f"\n📊 Estatísticas finais da rede:")
    print(f"   🧠 Neurônios: {classifier.model.graph.num_neurons:,}")
    print(f"   🔗 Conexões: {stats['num_connections']:,}")
    print(f"   💚 Saúde Neural: {stats.get('neural_health', 1.0):.3f}")
    
    return classifier

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

def test_model_performance(classifier):
    """Testa o modelo com várias imagens"""
    print(f"\n🧪 Testando performance do modelo...")
    
    test_dataset = OptimizedCatDogDataset(num_samples=20, image_size=classifier.image_size)
    correct = 0
    total = 0
    
    for i in range(20):
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
        
        if true_class == predicted_class:
            correct += 1
        total += 1
        
        status = "✅" if true_class == predicted_class else "❌"
        
        print(f"   {status} Teste {i+1:2d}: {true_class} → {predicted_class} "
              f"(Confiança: {confidence:.1f}%)")
    
    accuracy = 100 * correct / total
    print(f"\n📊 Performance final: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy

if __name__ == "__main__":
    # Remover modelo antigo
    if os.path.exists("cat_dog_seonn.pth"):
        print("🗑️  Removendo modelo antigo...")
        os.remove("cat_dog_seonn.pth")
    
    # Treinar modelo até loss próximo de zero
    classifier = train_until_zero_loss()
    
    # Testar performance
    test_model_performance(classifier)
    
    print(f"\n🎉 Modelo otimizado pronto!")
    print(f"💡 Reinicie o servidor para usar o novo modelo.")








