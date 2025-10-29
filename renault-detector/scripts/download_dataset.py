#!/usr/bin/env python3
"""
Script para baixar imagens de carros Renault e outras marcas para treinamento do modelo.
Usa DuckDuckGo Search para buscar imagens automaticamente.
"""

import os
import requests
import time
import random
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
import io
from duckduckgo_search import DDGS
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CarImageDownloader:
    def __init__(self, base_dir="data/images"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar diretórios para train, val, test
        for split in ['train', 'val', 'test']:
            (self.base_dir / split).mkdir(exist_ok=True)
        
        # Configurações de busca
        self.renault_queries = [
            "Renault car front view",
            "Renault Clio car",
            "Renault Megane car",
            "Renault Captur SUV",
            "Renault Kadjar car",
            "Renault Twingo car",
            "Renault Logan car",
            "Renault Sandero car",
            "Renault Duster SUV",
            "Renault Koleos SUV"
        ]
        
        self.other_brands_queries = [
            "Fiat car front view",
            "Volkswagen car front view", 
            "Toyota car front view",
            "Chevrolet car front view",
            "Ford car front view",
            "Honda car front view",
            "Nissan car front view",
            "Hyundai car front view",
            "Kia car front view",
            "Peugeot car front view"
        ]
        
        # Headers para evitar bloqueios
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def download_image(self, url, save_path):
        """Baixa uma imagem de uma URL e salva no caminho especificado."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Verificar se é uma imagem válida
            image = Image.open(io.BytesIO(response.content))
            image = image.convert('RGB')
            
            # Redimensionar se muito grande
            if image.width > 2000 or image.height > 2000:
                image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            
            # Salvar imagem
            image.save(save_path, 'JPEG', quality=95)
            return True
            
        except Exception as e:
            logger.warning(f"Erro ao baixar imagem {url}: {e}")
            return False
    
    def search_and_download(self, queries, label, num_images_per_query=50):
        """Busca e baixa imagens para uma categoria específica."""
        logger.info(f"Iniciando download para {label}...")
        
        downloaded_count = 0
        ddgs = DDGS()
        
        for query in queries:
            logger.info(f"Buscando: {query}")
            
            try:
                # Buscar imagens
                results = ddgs.images(
                    query,
                    max_results=num_images_per_query,
                    safesearch="moderate"
                )
                
                for i, result in enumerate(results):
                    if downloaded_count >= num_images_per_query * len(queries):
                        break
                    
                    url = result.get('image')
                    if not url:
                        continue
                    
                    # Determinar split (80% train, 10% val, 10% test)
                    rand = random.random()
                    if rand < 0.8:
                        split = 'train'
                    elif rand < 0.9:
                        split = 'val'
                    else:
                        split = 'test'
                    
                    # Nome do arquivo
                    filename = f"{label}_{downloaded_count:04d}.jpg"
                    save_path = self.base_dir / split / filename
                    
                    # Baixar imagem
                    if self.download_image(url, save_path):
                        downloaded_count += 1
                        logger.info(f"Baixada: {save_path} ({downloaded_count})")
                    
                    # Delay para evitar bloqueios
                    time.sleep(random.uniform(0.5, 1.5))
                
            except Exception as e:
                logger.error(f"Erro na busca por {query}: {e}")
                continue
        
        logger.info(f"Download concluído para {label}: {downloaded_count} imagens")
        return downloaded_count
    
    def download_all(self):
        """Baixa todas as imagens necessárias."""
        logger.info("Iniciando download do dataset completo...")
        
        # Download imagens Renault
        renault_count = self.search_and_download(
            self.renault_queries, 
            "renault", 
            num_images_per_query=30
        )
        
        # Download imagens outras marcas
        other_count = self.search_and_download(
            self.other_brands_queries, 
            "other", 
            num_images_per_query=30
        )
        
        logger.info(f"Download finalizado!")
        logger.info(f"Renault: {renault_count} imagens")
        logger.info(f"Outras marcas: {other_count} imagens")
        logger.info(f"Total: {renault_count + other_count} imagens")
        
        # Mostrar distribuição por split
        for split in ['train', 'val', 'test']:
            count = len(list((self.base_dir / split).glob('*.jpg')))
            logger.info(f"{split}: {count} imagens")
    
    def create_sample_dataset(self):
        """Cria um dataset de exemplo menor para testes rápidos."""
        logger.info("Criando dataset de exemplo...")
        
        # Download menor para testes
        renault_count = self.search_and_download(
            self.renault_queries[:3], 
            "renault", 
            num_images_per_query=10
        )
        
        other_count = self.search_and_download(
            self.other_brands_queries[:3], 
            "other", 
            num_images_per_query=10
        )
        
        logger.info(f"Dataset de exemplo criado: {renault_count + other_count} imagens")

def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download dataset de carros Renault')
    parser.add_argument('--sample', action='store_true', 
                       help='Criar apenas um dataset de exemplo menor')
    parser.add_argument('--base-dir', default='data/images',
                       help='Diretório base para salvar as imagens')
    
    args = parser.parse_args()
    
    downloader = CarImageDownloader(args.base_dir)
    
    if args.sample:
        downloader.create_sample_dataset()
    else:
        downloader.download_all()

if __name__ == "__main__":
    main()

