#!/usr/bin/env python3
"""
Script principal para executar o pipeline completo do Renault Detector.
Facilita a execução de todas as etapas do projeto.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RenaultDetectorPipeline:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.scripts_dir = self.base_dir / "scripts"
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        
    def run_command(self, command, description):
        """Executa um comando e trata erros."""
        logger.info(f"Executando: {description}")
        logger.info(f"Comando: {command}")
        
        try:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
            logger.info(f"✅ {description} - Concluído com sucesso!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} - Erro: {e}")
            logger.error(f"Saída de erro: {e.stderr}")
            return False
    
    def check_dependencies(self):
        """Verifica se as dependências estão instaladas."""
        logger.info("Verificando dependências...")
        
        required_packages = [
            ('ultralytics', 'ultralytics'), 
            ('torch', 'torch'), 
            ('fastapi', 'fastapi'), 
            ('opencv-python', 'cv2'), 
            ('pillow', 'PIL'), 
            ('numpy', 'numpy'), 
            ('pandas', 'pandas')
        ]
        
        missing_packages = []
        
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            logger.error(f"Pacotes faltando: {missing_packages}")
            logger.info("Execute: pip install -r requirements.txt")
            return False
        
        logger.info("✅ Todas as dependências estão instaladas!")
        return True
    
    def download_dataset(self, sample=False):
        """Baixa o dataset de imagens."""
        cmd = f"python {self.scripts_dir}/download_dataset.py"
        if sample:
            cmd += " --sample"
        
        return self.run_command(cmd, "Download do dataset")
    
    def prepare_dataset(self, no_augmentation=False):
        """Prepara o dataset para treinamento."""
        cmd = f"python {self.scripts_dir}/prepare_dataset.py"
        if no_augmentation:
            cmd += " --no-augmentation"
        
        return self.run_command(cmd, "Preparação do dataset")
    
    def train_model(self, epochs=200, batch_size=16, img_size=1024):
        """Treina o modelo YOLOv8."""
        cmd = f"python {self.scripts_dir}/train_yolo.py"
        cmd += f" --epochs {epochs} --batch-size {batch_size} --img-size {img_size}"
        
        return self.run_command(cmd, "Treinamento do modelo")
    
    def evaluate_model(self):
        """Avalia o modelo treinado."""
        cmd = f"python {self.scripts_dir}/evaluate_model.py"
        return self.run_command(cmd, "Avaliação do modelo")
    
    def start_api(self, host="127.0.0.1", port=8000):
        """Inicia a API FastAPI."""
        cmd = f"python {self.scripts_dir}/deploy_api.py --host {host} --port {port}"
        logger.info(f"Iniciando API em http://{host}:{port}")
        
        try:
            subprocess.run(cmd, shell=True)
        except KeyboardInterrupt:
            logger.info("API interrompida pelo usuário")
    
    def run_full_pipeline(self, sample=False, epochs=200):
        """Executa o pipeline completo."""
        logger.info("🚀 Iniciando pipeline completo do Renault Detector")
        
        steps = [
            ("Verificação de dependências", lambda: self.check_dependencies()),
            ("Download do dataset", lambda: self.download_dataset(sample)),
            ("Preparação do dataset", lambda: self.prepare_dataset()),
            ("Treinamento do modelo", lambda: self.train_model(epochs=epochs)),
            ("Avaliação do modelo", lambda: self.evaluate_model())
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n📋 {step_name}")
            if not step_func():
                logger.error(f"❌ Pipeline interrompido na etapa: {step_name}")
                return False
        
        logger.info("\n🎉 Pipeline completo executado com sucesso!")
        logger.info("📁 Modelo salvo em: models/best.pt")
        logger.info("📊 Métricas salvas em: models/metrics.json")
        logger.info("\n🚀 Para iniciar a API:")
        logger.info("python scripts/deploy_api.py")
        
        return True
    
    def quick_test(self):
        """Executa um teste rápido com dataset de exemplo."""
        logger.info("🧪 Executando teste rápido...")
        
        return self.run_full_pipeline(sample=True, epochs=10)
    
    def check_project_structure(self):
        """Verifica se a estrutura do projeto está correta."""
        logger.info("Verificando estrutura do projeto...")
        
        required_dirs = [
            self.scripts_dir,
            self.data_dir,
            self.models_dir
        ]
        
        required_files = [
            self.scripts_dir / "download_dataset.py",
            self.scripts_dir / "prepare_dataset.py",
            self.scripts_dir / "train_yolo.py",
            self.scripts_dir / "evaluate_model.py",
            self.scripts_dir / "deploy_api.py",
            self.base_dir / "requirements.txt",
            self.base_dir / "README.md"
        ]
        
        missing_items = []
        
        for item in required_dirs + required_files:
            if not item.exists():
                missing_items.append(str(item))
        
        if missing_items:
            logger.error("Itens faltando na estrutura do projeto:")
            for item in missing_items:
                logger.error(f"  - {item}")
            return False
        
        logger.info("✅ Estrutura do projeto está correta!")
        return True

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Pipeline do Renault Detector')
    parser.add_argument('--action', choices=[
        'check', 'download', 'prepare', 'train', 'evaluate', 
        'api', 'full', 'quick-test'
    ], default='check', help='Ação a executar')
    parser.add_argument('--sample', action='store_true', 
                       help='Usar dataset de exemplo')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Número de épocas para treinamento')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamanho do batch')
    parser.add_argument('--img-size', type=int, default=1024,
                       help='Tamanho da imagem')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Host da API')
    parser.add_argument('--port', type=int, default=8000,
                       help='Porta da API')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Não aplicar aumentos de dados')
    
    args = parser.parse_args()
    
    pipeline = RenaultDetectorPipeline()
    
    try:
        if args.action == 'check':
            pipeline.check_project_structure()
            pipeline.check_dependencies()
            
        elif args.action == 'download':
            pipeline.download_dataset(args.sample)
            
        elif args.action == 'prepare':
            pipeline.prepare_dataset(args.no_augmentation)
            
        elif args.action == 'train':
            pipeline.train_model(args.epochs, args.batch_size, args.img_size)
            
        elif args.action == 'evaluate':
            pipeline.evaluate_model()
            
        elif args.action == 'api':
            pipeline.start_api(args.host, args.port)
            
        elif args.action == 'full':
            pipeline.run_full_pipeline(args.sample, args.epochs)
            
        elif args.action == 'quick-test':
            pipeline.quick_test()
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Operação interrompida pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

