#!/usr/bin/env python3
"""
API FastAPI para detecção de carros Renault usando modelo YOLOv8 treinado.
Endpoint para upload de imagem e predição em tempo real.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import io
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
from PIL import Image
import torch
import numpy as np
import cv2

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RenaultDetectorAPI:
    def __init__(self, model_path="models/best.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = {0: 'renault', 1: 'other'}
        
    def load_model(self):
        """Carrega o modelo YOLOv8 treinado."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        logger.info(f"Modelo carregado: {self.model_path}")
        
        # Verificar se está usando GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Usando dispositivo: {device}")
    
    def preprocess_image(self, image_bytes: bytes) -> Image.Image:
        """Preprocessa a imagem para predição."""
        try:
            # Carregar imagem
            image = Image.open(io.BytesIO(image_bytes))
            
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionar se muito grande
            max_size = 2048
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao processar imagem: {e}")
    
    def predict(self, image: Image.Image) -> Dict:
        """Faz predição na imagem."""
        try:
            # Fazer predição
            results = self.model(image)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return {
                    "is_renault": False,
                    "confidence": 0.0,
                    "bbox": None,
                    "message": "Nenhum carro detectado"
                }
            
            # Pegar primeira detecção (maior confiança)
            box = results[0].boxes[0]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Coordenadas da bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            # Determinar se é Renault
            is_renault = class_id == 0
            class_name = self.class_names[class_id]
            
            return {
                "is_renault": is_renault,
                "confidence": confidence,
                "bbox": bbox,
                "class_name": class_name,
                "message": f"Carro detectado: {class_name}"
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

# Inicializar API
app = FastAPI(
    title="Renault Car Detector API",
    description="API para detecção de carros Renault usando YOLOv8",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar detector
detector = RenaultDetectorAPI()

@app.on_event("startup")
async def startup_event():
    """Inicializa o modelo na inicialização da API."""
    try:
        detector.load_model()
        logger.info("API inicializada com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao inicializar API: {e}")
        raise

@app.get("/")
async def root():
    """Endpoint raiz."""
    return {
        "message": "Renault Car Detector API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Endpoint de health check."""
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/predict")
async def predict_car(file: UploadFile = File(...)):
    """
    Endpoint para predição de carro Renault.
    
    Args:
        file: Arquivo de imagem (JPG, PNG, etc.)
    
    Returns:
        Dict com resultado da predição
    """
    try:
        # Verificar tipo de arquivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
        
        # Ler arquivo
        image_bytes = await file.read()
        
        # Preprocessar imagem
        image = detector.preprocess_image(image_bytes)
        
        # Fazer predição
        result = detector.predict(image)
        
        logger.info(f"Predição realizada: {result['class_name']} (confiança: {result['confidence']:.3f})")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no endpoint predict: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Endpoint para predição em lote de múltiplas imagens.
    
    Args:
        files: Lista de arquivos de imagem
    
    Returns:
        Lista com resultados das predições
    """
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Máximo de 10 imagens por lote")
        
        results = []
        
        for file in files:
            try:
                # Verificar tipo de arquivo
                if not file.content_type.startswith('image/'):
                    results.append({
                        "filename": file.filename,
                        "error": "Arquivo deve ser uma imagem"
                    })
                    continue
                
                # Ler arquivo
                image_bytes = await file.read()
                
                # Preprocessar imagem
                image = detector.preprocess_image(image_bytes)
                
                # Fazer predição
                result = detector.predict(image)
                result["filename"] = file.filename
                results.append(result)
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return JSONResponse(content={"results": results})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no endpoint predict_batch: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")

@app.get("/model_info")
async def model_info():
    """Retorna informações sobre o modelo carregado."""
    if detector.model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")
    
    return {
        "model_path": str(detector.model_path),
        "model_loaded": True,
        "class_names": detector.class_names,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__
    }

def main():
    """Função principal para executar a API."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Executar API de detecção Renault')
    parser.add_argument('--host', default='127.0.0.1', help='Host da API')
    parser.add_argument('--port', type=int, default=8000, help='Porta da API')
    parser.add_argument('--model-path', default='models/best.pt', help='Caminho do modelo')
    parser.add_argument('--reload', action='store_true', help='Recarregar automaticamente')
    
    args = parser.parse_args()
    
    # Atualizar caminho do modelo
    detector.model_path = Path(args.model_path)
    
    logger.info(f"Iniciando API em http://{args.host}:{args.port}")
    
    uvicorn.run(
        "deploy_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()

