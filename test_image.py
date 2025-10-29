#!/usr/bin/env python3
import requests
import base64
from PIL import Image
import io
import numpy as np

def create_test_image():
    """Cria uma imagem de teste sintética"""
    # Criar imagem 64x64
    img = np.random.rand(64, 64) * 255
    img = img.astype(np.uint8)
    
    # Adicionar padrão de gato (orelhas pontiagudas)
    img[5:15, 20:25] = 200
    img[5:15, 35:40] = 200
    img[25:30, 30:35] = 150  # nariz
    
    # Converter para PIL Image
    pil_img = Image.fromarray(img, mode='L')
    
    # Converter para base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    
    return img_base64

def test_prediction():
    """Testa a predição da API"""
    print("🧪 Testando predição da SEONN...")
    
    # Criar imagem de teste
    img_base64 = create_test_image()
    
    # Fazer requisição
    try:
        response = requests.post('http://localhost:5001/api/predict', 
                               json={'image': img_base64},
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Predição realizada com sucesso!")
            print(f"   🐱 Probabilidade Gato: {result['cat_probability']:.3f}")
            print(f"   🐶 Probabilidade Cachorro: {result['dog_probability']:.3f}")
            print(f"   🎯 Predição: {result['prediction']}")
            print(f"   📊 Confiança: {result['confidence']:.3f}")
        else:
            print(f"❌ Erro na API: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro de conexão: {e}")

def test_network_stats():
    """Testa as estatísticas da rede"""
    print("\n📊 Testando estatísticas da rede...")
    
    try:
        response = requests.get('http://localhost:5001/api/network-stats', timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            stats = result['stats']
            print("✅ Estatísticas obtidas com sucesso!")
            print(f"   🧠 Neurônios: {stats['neurons']:,}")
            print(f"   🔗 Conexões: {stats['connections']:,}")
            print(f"   💚 Saúde Neural: {stats['neural_health']:.3f}")
            print(f"   📈 Variação Atividade: {stats['activity_variance']:.3f}")
            print(f"   🎓 Learning Rate: {stats['learning_rate']:.6f}")
        else:
            print(f"❌ Erro na API: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro de conexão: {e}")

if __name__ == "__main__":
    test_network_stats()
    test_prediction()


