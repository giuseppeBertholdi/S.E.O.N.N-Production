#!/usr/bin/env python3
"""
Script de treinamento YOLOv8 para detecção de carros Renault.
Configurações avançadas com wandb, early stopping e EMA.
"""

import os
import torch
import logging
from pathlib import Path
from ultralytics import YOLO
import wandb
from ultralytics.utils import LOGGER

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RenaultTrainer:
    def __init__(self, data_yaml="data/data.yaml", model_dir="models"):
        self.data_yaml = Path(data_yaml)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Configurações de treinamento
        self.training_config = {
            'epochs': 200,
            'imgsz': 1024,
            'batch': 16,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.1,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 10,
            'cache': False,
            'device': 'auto',
            'workers': 8,
            'project': 'renault-detector',
            'name': 'yolov8x_renault',
            'exist_ok': True,
            'pretrained': True,
            'augment': True,
            'mosaic': 1.0,
            'mixup': 0.2,
            'copy_paste': 0.0,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'erasing': 0.4,
            'crop_fraction': 1.0,
            'patience': 50,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 10,
            'cache': False,
            'device': 'auto',
            'workers': 8,
            'project': 'renault-detector',
            'name': 'yolov8x_renault',
            'exist_ok': True,
            'pretrained': True,
            'augment': True,
            'mosaic': 1.0,
            'mixup': 0.2,
            'copy_paste': 0.0,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'erasing': 0.4,
            'crop_fraction': 1.0,
            'patience': 50,
            'freeze': None,
            'multi_scale': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'deterministic': True,
            'single_cls': False,
            'dnn': False,
            'half': False,
            'verbose': True,
            'seed': 0,
            'local_rank': -1,
            'upload_dataset': False,
            'bbox_interval': -1,
            'artifact_alias': 'latest'
        }
    
    def setup_wandb(self):
        """Configura Weights & Biases para monitoramento."""
        try:
            wandb.init(
                project="renault-detector",
                name="yolov8x_renault_training",
                config=self.training_config,
                tags=["yolov8", "renault", "object-detection"]
            )
            logger.info("Wandb inicializado com sucesso!")
            return True
        except Exception as e:
            logger.warning(f"Erro ao inicializar wandb: {e}")
            return False
    
    def check_gpu(self):
        """Verifica disponibilidade de GPU."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU disponível: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            logger.warning("GPU não disponível, usando CPU")
            return False
    
    def train_model(self):
        """Treina o modelo YOLOv8."""
        logger.info("Iniciando treinamento do modelo YOLOv8...")
        
        # Verificar GPU
        self.check_gpu()
        
        # Configurar wandb
        wandb_enabled = self.setup_wandb()
        
        try:
            # Carregar modelo YOLOv8x
            model = YOLO('yolov8x.pt')
            logger.info("Modelo YOLOv8x carregado com sucesso!")
            
            # Treinar modelo
            results = model.train(
                data=str(self.data_yaml),
                **self.training_config
            )
            
            # Salvar melhor modelo
            best_model_path = self.model_dir / 'best.pt'
            if results.save_dir:
                best_path = Path(results.save_dir) / 'weights' / 'best.pt'
                if best_path.exists():
                    import shutil
                    shutil.copy2(best_path, best_model_path)
                    logger.info(f"Melhor modelo salvo em: {best_model_path}")
            
            logger.info("Treinamento concluído com sucesso!")
            
            # Log final do wandb
            if wandb_enabled:
                wandb.finish()
            
            return results
            
        except Exception as e:
            logger.error(f"Erro durante o treinamento: {e}")
            if wandb_enabled:
                wandb.finish()
            raise
    
    def validate_dataset(self):
        """Valida se o dataset está pronto para treinamento."""
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Arquivo data.yaml não encontrado: {self.data_yaml}")
        
        logger.info("Dataset validado com sucesso!")
        return True

def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Treinar modelo YOLOv8 para detecção Renault')
    parser.add_argument('--data-yaml', default='data/data.yaml',
                       help='Caminho para arquivo data.yaml')
    parser.add_argument('--model-dir', default='models',
                       help='Diretório para salvar modelos')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamanho do batch')
    parser.add_argument('--img-size', type=int, default=1024,
                       help='Tamanho da imagem')
    
    args = parser.parse_args()
    
    trainer = RenaultTrainer(args.data_yaml, args.model_dir)
    
    # Atualizar configurações se especificado
    trainer.training_config['epochs'] = args.epochs
    trainer.training_config['batch'] = args.batch_size
    trainer.training_config['imgsz'] = args.img_size
    
    try:
        # Validar dataset
        trainer.validate_dataset()
        
        # Treinar modelo
        results = trainer.train_model()
        
        logger.info("Treinamento finalizado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        exit(1)

if __name__ == "__main__":
    main()

