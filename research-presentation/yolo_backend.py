from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import torch
from ultralytics import YOLO
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permitir CORS para o frontend React

# Carregar modelo YOLO
model_path = "/home/giuseppe/seonn_torch/yolov8n.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
    logger.info(f"Modelo YOLO carregado de: {model_path}")
else:
    logger.error(f"Modelo YOLO não encontrado em: {model_path}")
    model = None

# Classes COCO
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def base64_to_image(base64_string):
    """Converte string base64 para imagem PIL"""
    try:
        # Remover prefixo data:image se presente
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decodificar base64
        image_data = base64.b64decode(base64_string)
        
        # Converter para PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Converter para RGB se necessário
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        logger.error(f"Erro ao converter base64 para imagem: {e}")
        return None

def image_to_base64(image):
    """Converte imagem PIL para string base64"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        logger.error(f"Erro ao converter imagem para base64: {e}")
        return None

@app.route('/')
def index():
    return jsonify({
        "message": "YOLO Webcam API",
        "status": "running",
        "model_loaded": model is not None
    })

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """Endpoint para detecção de objetos em imagens"""
    try:
        if model is None:
            return jsonify({
                "error": "Modelo YOLO não carregado",
                "detections": []
            }), 500

        # Obter dados da requisição
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                "error": "Imagem não fornecida",
                "detections": []
            }), 400

        # Converter base64 para imagem
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({
                "error": "Erro ao processar imagem",
                "detections": []
            }), 400

        # Converter PIL para numpy array (formato que o YOLO espera)
        image_np = np.array(image)
        
        # Executar detecção
        results = model(image_np, conf=0.5)  # Confiança mínima de 50%
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # Obter coordenadas da bounding box
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    
                    # Obter classe e confiança
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Obter nome da classe
                    class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                    
                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": {
                            "x": float(x1),
                            "y": float(y1),
                            "width": float(x2 - x1),
                            "height": float(y2 - y1)
                        }
                    })

        return jsonify({
            "detections": detections,
            "total_detections": len(detections),
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        })

    except Exception as e:
        logger.error(f"Erro na detecção: {e}")
        return jsonify({
            "error": f"Erro interno: {str(e)}",
            "detections": []
        }), 500

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Endpoint para verificar status do modelo"""
    return jsonify({
        "model_loaded": model is not None,
        "model_path": model_path,
        "model_exists": os.path.exists(model_path),
        "classes_count": len(COCO_CLASSES)
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Endpoint para obter lista de classes"""
    return jsonify({
        "classes": COCO_CLASSES,
        "total_classes": len(COCO_CLASSES)
    })

if __name__ == '__main__':
    logger.info("Iniciando servidor YOLO Webcam API...")
    logger.info(f"Modelo carregado: {model is not None}")
    app.run(host='0.0.0.0', port=5000, debug=True)





