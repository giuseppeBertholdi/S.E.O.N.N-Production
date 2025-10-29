#!/usr/bin/env python3
"""
Exemplo de uso da API Renault Detector.
Demonstra como integrar a API em aplicações Python.
"""

import requests
import json
from pathlib import Path
import base64
from PIL import Image
import io

class RenaultDetectorClient:
    def __init__(self, api_url="http://127.0.0.1:8000"):
        self.api_url = api_url
        
    def is_renault(self, image_path):
        """
        Verifica se uma imagem contém um carro Renault.
        
        Args:
            image_path (str): Caminho para a imagem
            
        Returns:
            dict: Resultado da predição
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.api_url}/predict", files=files)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, image_paths):
        """
        Faz predições em lote de múltiplas imagens.
        
        Args:
            image_paths (list): Lista de caminhos para imagens
            
        Returns:
            dict: Resultados das predições
        """
        try:
            files = []
            for path in image_paths:
                files.append(('files', open(path, 'rb')))
            
            response = requests.post(f"{self.api_url}/predict_batch", files=files)
            response.raise_for_status()
            
            # Fechar arquivos
            for _, file_obj in files:
                file_obj.close()
            
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_from_base64(self, image_base64):
        """
        Faz predição a partir de uma imagem em base64.
        
        Args:
            image_base64 (str): Imagem codificada em base64
            
        Returns:
            dict: Resultado da predição
        """
        try:
            # Decodificar base64 para bytes
            image_bytes = base64.b64decode(image_base64)
            
            # Converter para arquivo temporário
            files = {'file': ('image.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
            
            response = requests.post(f"{self.api_url}/predict", files=files)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_info(self):
        """Obtém informações sobre o modelo carregado."""
        try:
            response = requests.get(f"{self.api_url}/model_info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def health_check(self):
        """Verifica se a API está funcionando."""
        try:
            response = requests.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def example_usage():
    """Exemplo de uso da API."""
    print("🚗 Exemplo de uso da API Renault Detector")
    
    # Inicializar cliente
    client = RenaultDetectorClient()
    
    # Verificar saúde da API
    print("\n1. Verificando saúde da API...")
    health = client.health_check()
    if "error" not in health:
        print(f"✅ API está funcionando: {health['status']}")
    else:
        print(f"❌ Erro na API: {health['error']}")
        return
    
    # Obter informações do modelo
    print("\n2. Informações do modelo...")
    model_info = client.get_model_info()
    if "error" not in model_info:
        print(f"✅ Modelo: {model_info['model_path']}")
        print(f"✅ Classes: {model_info['class_names']}")
    else:
        print(f"❌ Erro ao obter informações: {model_info['error']}")
    
    # Exemplo com imagem de teste
    print("\n3. Testando predição...")
    
    # Criar imagem de teste se não existir
    test_image = Path("test_car.jpg")
    if not test_image.exists():
        print("Criando imagem de teste...")
        img = Image.new('RGB', (400, 300), color='blue')
        img.save(test_image)
        print(f"Imagem de teste criada: {test_image}")
    
    # Fazer predição
    result = client.is_renault(str(test_image))
    if "error" not in result:
        print(f"✅ Resultado: {result['class_name']}")
        print(f"✅ Confiança: {result['confidence']:.3f}")
        print(f"✅ É Renault: {result['is_renault']}")
    else:
        print(f"❌ Erro na predição: {result['error']}")
    
    # Exemplo com múltiplas imagens
    print("\n4. Testando predição em lote...")
    test_images = [str(test_image)] * 3  # Usar a mesma imagem 3 vezes
    batch_result = client.predict_batch(test_images)
    
    if "error" not in batch_result:
        print(f"✅ Processadas {len(batch_result['results'])} imagens")
        for i, result in enumerate(batch_result['results']):
            if 'error' not in result:
                print(f"  Imagem {i+1}: {result['class_name']} "
                      f"(confiança: {result['confidence']:.3f})")
    else:
        print(f"❌ Erro na predição em lote: {batch_result['error']}")

def integration_example():
    """Exemplo de integração em aplicação web."""
    print("\n🌐 Exemplo de integração web...")
    
    # Simular recebimento de imagem via upload
    def process_uploaded_image(image_file):
        """Processa imagem enviada via upload."""
        client = RenaultDetectorClient()
        
        # Salvar arquivo temporariamente
        temp_path = Path("temp_upload.jpg")
        with open(temp_path, 'wb') as f:
            f.write(image_file.read())
        
        # Fazer predição
        result = client.is_renault(str(temp_path))
        
        # Limpar arquivo temporário
        temp_path.unlink()
        
        return result
    
    print("✅ Função de integração criada")
    print("📝 Use esta função para processar uploads em sua aplicação web")

if __name__ == "__main__":
    example_usage()
    integration_example()
    
    print("\n📖 Para mais exemplos, consulte:")
    print("- README.md")
    print("- test_api.py")
    print("- Documentação da API em http://127.0.0.1:8000/docs")
