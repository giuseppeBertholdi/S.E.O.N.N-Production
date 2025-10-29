#!/usr/bin/env python3
"""
Script para avaliar o modelo YOLOv8 treinado.
Calcula métricas detalhadas e gera relatório de performance.
"""

import os
import json
import logging
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path="models/best.pt", data_yaml="data/data.yaml"):
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.model = None
        self.results = {}
        
    def load_model(self):
        """Carrega o modelo treinado."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        logger.info(f"Modelo carregado: {self.model_path}")
    
    def evaluate_on_test_set(self):
        """Avalia o modelo no conjunto de teste."""
        logger.info("Avaliando modelo no conjunto de teste...")
        
        # Avaliação usando ultralytics
        metrics = self.model.val(data=str(self.data_yaml), split='test')
        
        # Extrair métricas principais
        self.results = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1_score': float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-16)),
            'class_precision': {
                'renault': float(metrics.box.p[0]) if len(metrics.box.p) > 0 else 0.0,
                'other': float(metrics.box.p[1]) if len(metrics.box.p) > 1 else 0.0
            },
            'class_recall': {
                'renault': float(metrics.box.r[0]) if len(metrics.box.r) > 0 else 0.0,
                'other': float(metrics.box.r[1]) if len(metrics.box.r) > 1 else 0.0
            }
        }
        
        logger.info("Avaliação no conjunto de teste concluída!")
        return self.results
    
    def detailed_evaluation(self):
        """Avaliação detalhada com predições individuais."""
        logger.info("Realizando avaliação detalhada...")
        
        test_images_dir = Path("data/images/test")
        test_labels_dir = Path("data/labels/test")
        
        if not test_images_dir.exists():
            logger.warning("Diretório de teste não encontrado, pulando avaliação detalhada")
            return
        
        true_labels = []
        pred_labels = []
        confidences = []
        
        for img_path in test_images_dir.glob('*.jpg'):
            # Carregar imagem
            image = Image.open(img_path)
            
            # Fazer predição
            results = self.model(str(img_path))
            
            # Determinar classe verdadeira
            label_path = test_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    true_class = int(f.read().strip().split()[0])
                true_labels.append(true_class)
                
                # Determinar classe predita
                if len(results) > 0 and len(results[0].boxes) > 0:
                    pred_class = int(results[0].boxes.cls[0])
                    confidence = float(results[0].boxes.conf[0])
                    pred_labels.append(pred_class)
                    confidences.append(confidence)
                else:
                    pred_labels.append(1)  # Assumir "other" se não detectar
                    confidences.append(0.0)
        
        # Calcular métricas detalhadas
        if true_labels and pred_labels:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average='weighted'
            )
            
            cm = confusion_matrix(true_labels, pred_labels)
            
            self.results.update({
                'detailed_precision': float(precision),
                'detailed_recall': float(recall),
                'detailed_f1': float(f1),
                'confusion_matrix': cm.tolist(),
                'average_confidence': float(np.mean(confidences)),
                'total_samples': len(true_labels)
            })
            
            logger.info(f"Avaliação detalhada concluída: {len(true_labels)} amostras")
    
    def generate_confusion_matrix_plot(self):
        """Gera plot da matriz de confusão."""
        if 'confusion_matrix' not in self.results:
            return
        
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Renault', 'Other'],
                   yticklabels=['Renault', 'Other'])
        plt.title('Matriz de Confusão - Detecção Renault')
        plt.xlabel('Predição')
        plt.ylabel('Verdadeiro')
        
        plot_path = Path("models/confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Matriz de confusão salva em: {plot_path}")
    
    def test_on_sample_images(self):
        """Testa o modelo em algumas imagens de exemplo."""
        logger.info("Testando modelo em imagens de exemplo...")
        
        test_images_dir = Path("data/images/test")
        if not test_images_dir.exists():
            logger.warning("Diretório de teste não encontrado")
            return
        
        sample_images = list(test_images_dir.glob('*.jpg'))[:5]
        
        for img_path in sample_images:
            # Fazer predição
            results = self.model(str(img_path))
            
            # Mostrar resultado
            if len(results) > 0 and len(results[0].boxes) > 0:
                pred_class = int(results[0].boxes.cls[0])
                confidence = float(results[0].boxes.conf[0])
                class_name = 'Renault' if pred_class == 0 else 'Other'
                
                logger.info(f"{img_path.name}: {class_name} (confiança: {confidence:.3f})")
            else:
                logger.info(f"{img_path.name}: Nenhuma detecção")
    
    def save_metrics_report(self):
        """Salva relatório de métricas em JSON."""
        report_path = Path("models/metrics.json")
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Relatório de métricas salvo em: {report_path}")
    
    def print_summary(self):
        """Imprime resumo das métricas."""
        logger.info("=== RESUMO DE AVALIAÇÃO ===")
        logger.info(f"mAP@0.5: {self.results.get('mAP50', 0):.4f}")
        logger.info(f"mAP@0.5:0.95: {self.results.get('mAP50-95', 0):.4f}")
        logger.info(f"Precisão: {self.results.get('precision', 0):.4f}")
        logger.info(f"Recall: {self.results.get('recall', 0):.4f}")
        logger.info(f"F1-Score: {self.results.get('f1_score', 0):.4f}")
        
        if 'class_precision' in self.results:
            logger.info("Precisão por classe:")
            logger.info(f"  Renault: {self.results['class_precision']['renault']:.4f}")
            logger.info(f"  Other: {self.results['class_precision']['other']:.4f}")
        
        if 'average_confidence' in self.results:
            logger.info(f"Confiança média: {self.results['average_confidence']:.4f}")
        
        if 'total_samples' in self.results:
            logger.info(f"Total de amostras: {self.results['total_samples']}")
    
    def evaluate(self):
        """Executa avaliação completa do modelo."""
        logger.info("Iniciando avaliação completa do modelo...")
        
        # Carregar modelo
        self.load_model()
        
        # Avaliação no conjunto de teste
        self.evaluate_on_test_set()
        
        # Avaliação detalhada
        self.detailed_evaluation()
        
        # Gerar visualizações
        self.generate_confusion_matrix_plot()
        
        # Testar em imagens de exemplo
        self.test_on_sample_images()
        
        # Salvar relatório
        self.save_metrics_report()
        
        # Imprimir resumo
        self.print_summary()
        
        logger.info("Avaliação completa finalizada!")

def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Avaliar modelo YOLOv8')
    parser.add_argument('--model-path', default='models/best.pt',
                       help='Caminho para o modelo treinado')
    parser.add_argument('--data-yaml', default='data/data.yaml',
                       help='Caminho para arquivo data.yaml')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model_path, args.data_yaml)
    
    try:
        evaluator.evaluate()
    except Exception as e:
        logger.error(f"Erro na avaliação: {e}")
        exit(1)

if __name__ == "__main__":
    main()

