#!/usr/bin/env python3
"""
Script de exemplo para testar a API do Renault Detector.
Demonstra como usar a API para predições.
"""

import requests
import json
import argparse
from pathlib import Path
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APITester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def test_health(self):
        """Testa o endpoint de health check."""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            logger.info("✅ Health check passou!")
            logger.info(f"Status: {data['status']}")
            logger.info(f"Modelo carregado: {data['model_loaded']}")
            logger.info(f"Dispositivo: {data['device']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check falhou: {e}")
            return False
    
    def test_model_info(self):
        """Testa o endpoint de informações do modelo."""
        try:
            response = requests.get(f"{self.base_url}/model_info")
            response.raise_for_status()
            
            data = response.json()
            logger.info("✅ Informações do modelo obtidas!")
            logger.info(f"Caminho do modelo: {data['model_path']}")
            logger.info(f"Classes: {data['class_names']}")
            logger.info(f"Versão PyTorch: {data['torch_version']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter informações do modelo: {e}")
            return False
    
    def test_prediction(self, image_path):
        """Testa o endpoint de predição com uma imagem."""
        try:
            if not Path(image_path).exists():
                logger.error(f"❌ Arquivo não encontrado: {image_path}")
                return False
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.base_url}/predict", files=files)
                response.raise_for_status()
            
            data = response.json()
            logger.info("✅ Predição realizada com sucesso!")
            logger.info(f"É Renault: {data['is_renault']}")
            logger.info(f"Confiança: {data['confidence']:.3f}")
            logger.info(f"Classe: {data['class_name']}")
            logger.info(f"Mensagem: {data['message']}")
            
            if data['bbox']:
                logger.info(f"Bounding box: {data['bbox']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na predição: {e}")
            return False
    
    def test_batch_prediction(self, image_paths):
        """Testa o endpoint de predição em lote."""
        try:
            files = []
            for path in image_paths:
                if not Path(path).exists():
                    logger.error(f"❌ Arquivo não encontrado: {path}")
                    continue
                files.append(('files', open(path, 'rb')))
            
            if not files:
                logger.error("❌ Nenhum arquivo válido encontrado")
                return False
            
            response = requests.post(f"{self.base_url}/predict_batch", files=files)
            response.raise_for_status()
            
            # Fechar arquivos
            for _, file_obj in files:
                file_obj.close()
            
            data = response.json()
            logger.info("✅ Predição em lote realizada com sucesso!")
            
            for result in data['results']:
                if 'error' in result:
                    logger.error(f"❌ Erro em {result['filename']}: {result['error']}")
                else:
                    logger.info(f"✅ {result['filename']}: {result['class_name']} "
                              f"(confiança: {result['confidence']:.3f})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na predição em lote: {e}")
            return False
    
    def run_all_tests(self, test_image_path=None):
        """Executa todos os testes."""
        logger.info("🧪 Iniciando testes da API...")
        
        tests = [
            ("Health Check", lambda: self.test_health()),
            ("Model Info", lambda: self.test_model_info())
        ]
        
        if test_image_path:
            tests.append(("Prediction", lambda: self.test_prediction(test_image_path)))
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Testando: {test_name}")
            if test_func():
                passed += 1
            else:
                logger.error(f"❌ Teste falhou: {test_name}")
        
        logger.info(f"\n📊 Resultados: {passed}/{total} testes passaram")
        
        if passed == total:
            logger.info("🎉 Todos os testes passaram!")
        else:
            logger.error("❌ Alguns testes falharam!")
        
        return passed == total

def create_sample_image():
    """Cria uma imagem de exemplo para teste."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Criar imagem de exemplo
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Desenhar um carro simples
    draw.rectangle([50, 150, 350, 200], fill='blue', outline='black', width=2)
    draw.rectangle([100, 120, 300, 150], fill='lightblue', outline='black', width=2)
    draw.ellipse([80, 180, 120, 220], fill='black')
    draw.ellipse([280, 180, 320, 220], fill='black')
    
    # Adicionar texto
    try:
        font = ImageFont.load_default()
        draw.text((150, 250), "Carro de Teste", fill='black', font=font)
    except:
        draw.text((150, 250), "Carro de Teste", fill='black')
    
    # Salvar imagem
    sample_path = Path("sample_car.jpg")
    img.save(sample_path)
    logger.info(f"Imagem de exemplo criada: {sample_path}")
    
    return str(sample_path)

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Testar API do Renault Detector')
    parser.add_argument('--url', default='http://127.0.0.1:8000',
                       help='URL da API')
    parser.add_argument('--image', help='Caminho para imagem de teste')
    parser.add_argument('--create-sample', action='store_true',
                       help='Criar imagem de exemplo')
    parser.add_argument('--batch-images', nargs='+',
                       help='Lista de imagens para teste em lote')
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    # Criar imagem de exemplo se solicitado
    if args.create_sample:
        sample_path = create_sample_image()
        args.image = sample_path
    
    # Executar testes
    if args.batch_images:
        logger.info("🧪 Testando predição em lote...")
        tester.test_batch_prediction(args.batch_images)
    else:
        tester.run_all_tests(args.image)

if __name__ == "__main__":
    main()

