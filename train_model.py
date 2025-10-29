#!/usr/bin/env python3
"""
Script para treinar o modelo SEONN para classificação gato/cachorro
"""

from cat_dog_classifier import CatDogClassifier, CatDogDataset
import torch
import os
from torch.utils.data import DataLoader

def main():
    print("🚀 Iniciando treinamento do modelo SEONN...")
    
    # Verificar se já existe um modelo
    model_path = "cat_dog_seonn.pth"
    if os.path.exists(model_path):
        print(f"📂 Modelo existente encontrado: {model_path}")
        print("🔄 Continuando treinamento...")
        classifier = CatDogClassifier(model_path)
    else:
        print("🆕 Criando novo modelo...")
        classifier = CatDogClassifier()
    
    print("\n📊 Configurações do modelo:")
    print(f"   🧠 Neurônios: {classifier.model.graph.num_neurons:,}")
    print(f"   📏 Tamanho da imagem: {classifier.image_size}x{classifier.image_size}")
    print(f"   🎯 Classes: Gato, Cachorro")
    
    # Mostrar estatísticas iniciais
    stats = classifier.model.graph.get_connection_statistics()
    print(f"   🔗 Conexões iniciais: {stats['num_connections']:,}")
    print(f"   💚 Saúde Neural: {stats.get('neural_health', 1.0):.3f}")
    
    # Treinar por mais épocas
    print("\n🎓 Iniciando treinamento...")
    classifier.train(
        epochs=25,  # Mais épocas para melhor precisão
        batch_size=32  # Batch menor para melhor convergência
    )
    
    print("\n✅ Treinamento concluído!")
    print(f"💾 Modelo salvo em: {model_path}")
    
    # Testar o modelo
    print("\n🧪 Testando o modelo...")
    # Criar dataset de teste
    test_dataset = CatDogDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_accuracy = classifier.evaluate(test_loader)
    print(f"📈 Precisão final: {test_accuracy:.2%}")
    
    # Mostrar estatísticas finais
    stats = classifier.model.graph.get_connection_statistics()
    print(f"\n📊 Estatísticas finais da rede:")
    print(f"   🧠 Neurônios: {classifier.model.graph.num_neurons:,}")
    print(f"   🔗 Conexões: {stats['num_connections']:,}")
    print(f"   💚 Saúde Neural: {stats.get('neural_health', 1.0):.3f}")
    print(f"   📈 Variação Atividade: {stats.get('activity_variance', 0.0):.3f}")
    print(f"   🎓 Learning Rate: {0.001:.6f}")

if __name__ == "__main__":
    main()
