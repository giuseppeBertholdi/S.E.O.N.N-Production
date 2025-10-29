#!/usr/bin/env python3
"""
Script para preparar o dataset para treinamento YOLOv8.
Converte imagens para formato YOLO e aplica aumentos de dados.
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import logging
from typing import Tuple, List

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreparer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
        # Criar diretórios de labels
        for split in ['train', 'val', 'test']:
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Configurações de aumento de dados
        self.augmentation_config = {
            'flip_prob': 0.5,
            'rotation_range': 15,
            'brightness_range': 0.2,
            'contrast_range': 0.2,
            'blur_prob': 0.3,
            'noise_prob': 0.2,
            'crop_prob': 0.3
        }
    
    def create_yolo_labels(self):
        """Cria arquivos de label YOLO para todas as imagens."""
        logger.info("Criando arquivos de label YOLO...")
        
        for split in ['train', 'val', 'test']:
            split_dir = self.images_dir / split
            label_dir = self.labels_dir / split
            
            for img_path in split_dir.glob('*.jpg'):
                # Determinar classe baseada no nome do arquivo
                if 'renault' in img_path.stem.lower():
                    class_id = 0  # Renault
                else:
                    class_id = 1  # Other brands
                
                # Criar arquivo de label
                label_path = label_dir / f"{img_path.stem}.txt"
                
                # Para detecção de objetos, assumimos que o carro ocupa toda a imagem
                # Formato YOLO: class_id center_x center_y width height (normalizado)
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                
                logger.debug(f"Criado label: {label_path}")
        
        logger.info("Arquivos de label YOLO criados com sucesso!")
    
    def apply_data_augmentation(self, image: Image.Image) -> List[Image.Image]:
        """Aplica aumentos de dados a uma imagem."""
        augmented_images = [image]  # Incluir imagem original
        
        # Flip horizontal
        if random.random() < self.augmentation_config['flip_prob']:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(flipped)
        
        # Rotação
        if random.random() < 0.5:
            angle = random.uniform(-self.augmentation_config['rotation_range'], 
                                 self.augmentation_config['rotation_range'])
            rotated = image.rotate(angle, expand=True)
            # Redimensionar para manter proporção
            rotated = rotated.resize(image.size, Image.Resampling.LANCZOS)
            augmented_images.append(rotated)
        
        # Ajuste de brilho
        if random.random() < 0.5:
            factor = random.uniform(1 - self.augmentation_config['brightness_range'],
                                  1 + self.augmentation_config['brightness_range'])
            enhancer = ImageEnhance.Brightness(image)
            brightened = enhancer.enhance(factor)
            augmented_images.append(brightened)
        
        # Ajuste de contraste
        if random.random() < 0.5:
            factor = random.uniform(1 - self.augmentation_config['contrast_range'],
                                  1 + self.augmentation_config['contrast_range'])
            enhancer = ImageEnhance.Contrast(image)
            contrasted = enhancer.enhance(factor)
            augmented_images.append(contrasted)
        
        # Blur
        if random.random() < self.augmentation_config['blur_prob']:
            blurred = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
            augmented_images.append(blurred)
        
        # Adicionar ruído
        if random.random() < self.augmentation_config['noise_prob']:
            noisy = self.add_noise(image)
            augmented_images.append(noisy)
        
        return augmented_images
    
    def add_noise(self, image: Image.Image) -> Image.Image:
        """Adiciona ruído gaussiano à imagem."""
        img_array = np.array(image)
        noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
        noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    def augment_dataset(self):
        """Aplica aumentos de dados ao dataset de treinamento."""
        logger.info("Aplicando aumentos de dados ao dataset de treinamento...")
        
        train_dir = self.images_dir / 'train'
        train_label_dir = self.labels_dir / 'train'
        
        # Lista de imagens originais
        original_images = list(train_dir.glob('*.jpg'))
        
        for img_path in original_images:
            try:
                # Carregar imagem
                image = Image.open(img_path)
                
                # Aplicar aumentos
                augmented_images = self.apply_data_augmentation(image)
                
                # Salvar imagens aumentadas
                for i, aug_img in enumerate(augmented_images[1:], 1):  # Pular a original
                    aug_name = f"{img_path.stem}_aug_{i}.jpg"
                    aug_path = train_dir / aug_name
                    aug_img.save(aug_path, 'JPEG', quality=95)
                    
                    # Copiar label correspondente
                    label_path = train_label_dir / f"{img_path.stem}.txt"
                    aug_label_path = train_label_dir / f"{img_path.stem}_aug_{i}.txt"
                    if label_path.exists():
                        shutil.copy2(label_path, aug_label_path)
                
                logger.debug(f"Processada: {img_path.name}")
                
            except Exception as e:
                logger.error(f"Erro ao processar {img_path}: {e}")
        
        logger.info("Aumentos de dados aplicados com sucesso!")
    
    def create_data_yaml(self):
        """Cria arquivo data.yaml para YOLOv8."""
        logger.info("Criando arquivo data.yaml...")
        
        data_config = {
            'train': str(self.images_dir / 'train'),
            'val': str(self.images_dir / 'val'),
            'test': str(self.images_dir / 'test'),
            'nc': 2,  # Número de classes
            'names': {
                0: 'renault',
                1: 'other'
            }
        }
        
        yaml_path = self.data_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"Arquivo data.yaml criado em: {yaml_path}")
        return yaml_path
    
    def validate_dataset(self):
        """Valida o dataset criado."""
        logger.info("Validando dataset...")
        
        total_images = 0
        total_labels = 0
        
        for split in ['train', 'val', 'test']:
            img_count = len(list((self.images_dir / split).glob('*.jpg')))
            label_count = len(list((self.labels_dir / split).glob('*.txt')))
            
            logger.info(f"{split}: {img_count} imagens, {label_count} labels")
            total_images += img_count
            total_labels += label_count
        
        logger.info(f"Total: {total_images} imagens, {total_labels} labels")
        
        if total_images != total_labels:
            logger.warning("Número de imagens e labels não coincide!")
        else:
            logger.info("Dataset validado com sucesso!")
    
    def prepare_dataset(self, apply_augmentation=True):
        """Prepara o dataset completo."""
        logger.info("Iniciando preparação do dataset...")
        
        # Criar labels YOLO
        self.create_yolo_labels()
        
        # Aplicar aumentos de dados (apenas no treinamento)
        if apply_augmentation:
            self.augment_dataset()
        
        # Criar arquivo de configuração
        self.create_data_yaml()
        
        # Validar dataset
        self.validate_dataset()
        
        logger.info("Preparação do dataset concluída!")

def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preparar dataset para YOLOv8')
    parser.add_argument('--data-dir', default='data',
                       help='Diretório do dataset')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Não aplicar aumentos de dados')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.data_dir)
    preparer.prepare_dataset(apply_augmentation=not args.no_augmentation)

if __name__ == "__main__":
    main()

