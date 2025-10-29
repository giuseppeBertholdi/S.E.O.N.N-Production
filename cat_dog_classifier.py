import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import requests
from PIL import Image
import numpy as np
from seonn_model import SEONN_Model
import json
import base64
from io import BytesIO

class CatDogDataset(Dataset):
    """Dataset simples para gatos e cachorros usando imagens sintéticas"""
    def __init__(self, num_samples=1000, image_size=64, train=True):
        self.num_samples = num_samples
        self.image_size = image_size
        self.train = train
        
        # Gerar dados sintéticos baseados em padrões
        self.data = []
        self.labels = []
        
        for i in range(num_samples):
            # Gerar imagem sintética (simulando gato ou cachorro)
            if i % 2 == 0:  # Gato
                image = self._generate_cat_image()
                label = 0
            else:  # Cachorro
                image = self._generate_dog_image()
                label = 1
            
            self.data.append(image)
            self.labels.append(label)
    
    def _generate_cat_image(self):
        """Gera imagem sintética de gato"""
        # Padrão mais vertical e pontiagudo (orelhas de gato)
        image = np.random.randn(self.image_size, self.image_size) * 0.1
        
        # Adicionar padrões característicos de gato
        # Orelhas pontiagudas
        image[5:15, 20:25] += 0.8
        image[5:15, 35:40] += 0.8
        
        # Nariz pequeno
        image[25:30, 30:35] += 0.6
        
        # Bigodes
        image[30:35, 15:20] += 0.4
        image[30:35, 40:45] += 0.4
        
        # Formato mais triangular
        for i in range(self.image_size):
            for j in range(self.image_size):
                if abs(i - 32) + abs(j - 32) > 25:
                    image[i, j] *= 0.3
        
        return torch.FloatTensor(image).flatten()
    
    def _generate_dog_image(self):
        """Gera imagem sintética de cachorro"""
        # Padrão mais arredondado e largo
        image = np.random.randn(self.image_size, self.image_size) * 0.1
        
        # Adicionar padrões característicos de cachorro
        # Orelhas caídas
        image[8:18, 18:28] += 0.6
        image[8:18, 32:42] += 0.6
        
        # Focinho mais largo
        image[25:35, 25:40] += 0.5
        
        # Formato mais arredondado
        for i in range(self.image_size):
            for j in range(self.image_size):
                if (i - 32)**2 + (j - 32)**2 > 400:
                    image[i, j] *= 0.2
        
        return torch.FloatTensor(image).flatten()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CatDogClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 64
        self.input_size = self.image_size * self.image_size  # 4096
        self.output_size = 2  # Gato ou Cachorro
        
        # Inicializar modelo SEONN
        self.model = SEONN_Model(
            input_size=self.input_size,
            output_size=self.output_size,
            initial_neurons=5000,  # Aumentado para acomodar input_size + output_size
            initial_connectivity=0.01,  # Reduzido para performance
            learning_rate=1e-3,
            plasticity_rate=0.01,
            homeostasis_strength=0.1,
            competition_strength=0.05
        ).to(self.device)
        
        # Não treinar automaticamente
        # if model_path and os.path.exists(model_path):
        #     if not self.load_model(model_path):
        #         # Se não conseguiu carregar, treinar novo modelo
        #         print("🔄 Treinando novo modelo...")
        #         self.train(epochs=5, batch_size=32)
        
        # Transformações para imagens
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def train(self, epochs=10, batch_size=32):
        """Treina o classificador"""
        print("🐱🐶 Treinando classificador Gato vs Cachorro...")
        
        # Criar datasets
        train_dataset = CatDogDataset(num_samples=800, image_size=self.image_size, train=True)
        test_dataset = CatDogDataset(num_samples=200, image_size=self.image_size, train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"📊 Dataset: {len(train_dataset)} treino, {len(test_dataset)} teste")
        print(f"🧠 Modelo: {self.model.num_neurons} neurônios, {self.model.weights.numel()} conexões")
        
        # Treinamento
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                loss = self.model.train_step(data, target)
                total_loss += loss
                
                # Calcular acurácia
                with torch.no_grad():
                    output = self.model(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            print(f'Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Acurácia: {accuracy:.2f}%')
            
            # Evolução a cada 3 epochs
            if (epoch + 1) % 3 == 0:
                print("🔄 Evoluindo rede...")
                pruned, new = self.model.evolve(pruning_threshold=0.001)
                print(f"   ✂️  Removidas: {pruned.shape[1] if pruned.numel() > 0 else 0}")
                print(f"   ➕ Adicionadas: {new.shape[1] if new.numel() > 0 else 0}")
                
                # Mostrar estatísticas após evolução
                stats = self.model.graph.get_connection_statistics()
                print(f"   🔗 Conexões: {stats['num_connections']:,}")
                print(f"   💚 Saúde Neural: {stats.get('neural_health', 1.0):.3f}")
        
        # Avaliação final
        self.evaluate(test_loader)
        
        # Salvar modelo
        self.save_model("cat_dog_seonn.pth")
        
        # Mostrar estatísticas finais
        stats = self.model.graph.get_connection_statistics()
        print(f"\n📊 Estatísticas finais da rede:")
        print(f"   🧠 Neurônios: {self.model.graph.num_neurons:,}")
        print(f"   🔗 Conexões: {stats['num_connections']:,}")
        print(f"   💚 Saúde Neural: {stats.get('neural_health', 1.0):.3f}")
        print(f"   📈 Variação Atividade: {stats.get('activity_variance', 0.0):.3f}")
        print(f"   🎓 Learning Rate: {0.001:.6f}")
    
    def evaluate(self, test_loader):
        """Avalia o modelo"""
        self.model.eval()
        correct = 0
        total = 0
        cat_correct = 0
        dog_correct = 0
        cat_total = 0
        dog_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
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
        
        print(f"\n📊 Resultados Finais:")
        print(f"   🎯 Acurácia Geral: {accuracy:.2f}%")
        print(f"   🐱 Acurácia Gatos: {cat_accuracy:.2f}%")
        print(f"   🐶 Acurácia Cachorros: {dog_accuracy:.2f}%")
        print(f"   🧠 Saúde Neural: {self.model.neuron_health.mean().item():.3f}")
        print(f"   🔗 Conexões: {self.model.weights.numel()}")
    
    def predict_image(self, image_path_or_data):
        """Prediz se a imagem é gato ou cachorro"""
        self.model.eval()
        
        # Processar imagem
        if isinstance(image_path_or_data, str):
            # Verificar se é base64 ou caminho do arquivo
            if len(image_path_or_data) > 100 and not os.path.exists(image_path_or_data):
                # É base64
                try:
                    image_data = base64.b64decode(image_path_or_data)
                    image = Image.open(BytesIO(image_data))
                except Exception as e:
                    raise ValueError(f"Erro ao decodificar imagem base64: {e}")
            else:
                # Caminho do arquivo
                image = Image.open(image_path_or_data)
        else:
            # Dados da imagem (bytes)
            image = Image.open(BytesIO(image_path_or_data))
        
        # Aplicar transformações
        image_tensor = self.transform(image).flatten().unsqueeze(0).to(self.device)
        
        # Predição
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            cat_prob = probabilities[0][0].item()
            dog_prob = probabilities[0][1].item()
        
        return {
            'cat_probability': cat_prob,
            'dog_probability': dog_prob,
            'prediction': 'Gato' if cat_prob > dog_prob else 'Cachorro',
            'confidence': max(cat_prob, dog_prob)
        }
    
    def save_model(self, path):
        """Salva o modelo"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'image_size': self.image_size
        }, path)
        print(f"💾 Modelo salvo em: {path}")
    
    def load_model(self, path):
        """Carrega o modelo"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Verificar se o modelo tem a mesma arquitetura
        if (checkpoint.get('input_size') == self.input_size and 
            checkpoint.get('output_size') == self.output_size):
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📂 Modelo carregado de: {path}")
        else:
            print(f"⚠️  Arquitetura do modelo salvo não corresponde. Treinando novo modelo...")
            return False
        return True

def main():
    """Função principal para treinar o classificador"""
    classifier = CatDogClassifier()
    classifier.train(epochs=15, batch_size=64)
    
    # Teste com imagem sintética
    print("\n🧪 Testando predição...")
    result = classifier.predict_image("test_image.jpg")  # Será gerada uma imagem sintética
    print(f"Predição: {result}")

if __name__ == "__main__":
    main()
