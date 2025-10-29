from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import io
from PIL import Image
import torch
import numpy as np
from cat_dog_classifier import CatDogClassifier
from pretrained_classifier import ImageNetCatDogClassifier
import os
import json
import logging
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Permitir CORS para o frontend

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('seonn_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicializar os classificadores
classifier = None
pretrained_classifier = None
prediction_count = 0
use_pretrained = True  # Usar modelo pré-treinado por padrão

def initialize_classifier():
    """Inicializa os classificadores"""
    global classifier, pretrained_classifier
    
    try:
        # Inicializar modelo pré-treinado (mais preciso)
        logger.info("🔄 Iniciando carregamento do classificador pré-treinado...")
        start_time = time.time()
        
        pretrained_classifier = ImageNetCatDogClassifier()
        
        load_time = time.time() - start_time
        logger.info(f"✅ Classificador pré-treinado carregado com sucesso! Tempo: {load_time:.2f}s")
        
        # Tentar carregar modelo SEONN como backup
        try:
            logger.info("🔄 Carregando modelo SEONN como backup...")
            classifier = CatDogClassifier("cat_dog_seonn.pth")
            
            if classifier and classifier.model:
                stats = classifier.model.graph.get_connection_statistics()
                logger.info(f"📊 Estatísticas do modelo SEONN:")
                logger.info(f"   🧠 Neurônios: {classifier.model.graph.num_neurons:,}")
                logger.info(f"   🔗 Conexões: {stats['num_connections']:,}")
                logger.info(f"   💚 Saúde Neural: {stats.get('neural_health', 1.0):.3f}")
        except Exception as e:
            logger.warning(f"⚠️ Modelo SEONN não disponível: {e}")
            classifier = None
        
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao carregar classificadores: {e}")
        return False

@app.route('/')
def index():
    """Servir a página principal"""
    return send_from_directory('web_demo', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Servir arquivos estáticos"""
    return send_from_directory('web_demo', filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint para predição de imagens"""
    global prediction_count, use_pretrained
    
    try:
        prediction_count += 1
        logger.info(f"🔮 Predição #{prediction_count} iniciada")
        
        # Escolher qual modelo usar
        active_classifier = None
        model_type = "none"
        
        if use_pretrained and pretrained_classifier:
            active_classifier = pretrained_classifier
            model_type = "pretrained_imagenet"
            logger.info("🎯 Usando modelo pré-treinado ImageNet")
        elif classifier:
            active_classifier = classifier
            model_type = "seonn"
            logger.info("🧬 Usando modelo SEONN")
        else:
            logger.error("❌ Nenhum classificador disponível")
            return jsonify({'error': 'Nenhum classificador disponível'}), 500
        
        # Obter dados da imagem
        data = request.get_json()
        if not data or 'image' not in data:
            logger.warning("⚠️ Imagem não fornecida na requisição")
            return jsonify({'error': 'Imagem não fornecida'}), 400
        
        # Decodificar imagem base64
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remover prefixo data:image/...;base64,
            image_data = image_data.split(',')[1]
        
        logger.info(f"📸 Processando imagem (tamanho: {len(image_data)} chars)")
        
        # Fazer predição
        start_time = time.time()
        result = active_classifier.predict_image(image_data)
        prediction_time = time.time() - start_time
        
        logger.info(f"🎯 Predição #{prediction_count} concluída:")
        logger.info(f"   📊 Resultado: {result['prediction']}")
        logger.info(f"   🎯 Confiança: {result['confidence']:.3f}")
        logger.info(f"   🐱 Prob. Gato: {result['cat_probability']:.3f}")
        logger.info(f"   🐶 Prob. Cachorro: {result['dog_probability']:.3f}")
        logger.info(f"   ⏱️ Tempo: {prediction_time:.3f}s")
        logger.info(f"   🔧 Modelo: {model_type}")
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'cat_probability': result['cat_probability'],
            'dog_probability': result['dog_probability'],
            'prediction_id': prediction_count,
            'processing_time': prediction_time,
            'model_type': model_type
        })
        
    except Exception as e:
        logger.error(f"❌ Erro na predição #{prediction_count}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/network-stats', methods=['GET'])
def get_network_stats():
    """Endpoint para obter estatísticas da rede"""
    try:
        logger.info("📊 Requisição de estatísticas da rede")
        
        if classifier is None:
            logger.error("❌ Classificador não inicializado para estatísticas")
            return jsonify({'error': 'Classificador não inicializado'}), 500
        
        model = classifier.model
        
        # Calcular estatísticas
        stats = {
            'neurons': model.num_neurons,
            'connections': model.weights.numel(),
            'neural_health': float(model.neuron_health.mean().item()),
            'activity_variance': float(model.activity_history.var().item()),
            'learning_rate': float(model.learning_rate_adaptive.mean().item()),
            'total_predictions': prediction_count
        }
        
        logger.info(f"📈 Estatísticas enviadas: {stats['neurons']} neurônios, {stats['connections']} conexões")
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao obter estatísticas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evolve', methods=['POST'])
def evolve_network():
    """Endpoint para evoluir a rede"""
    try:
        if classifier is None:
            return jsonify({'error': 'Classificador não inicializado'}), 500
        
        # Obter threshold do request
        data = request.get_json() or {}
        threshold = data.get('threshold', 0.001)
        
        # Evoluir a rede
        pruned_indices, new_indices = classifier.model.evolve(pruning_threshold=threshold)
        
        return jsonify({
            'success': True,
            'pruned_connections': pruned_indices.shape[1] if pruned_indices.numel() > 0 else 0,
            'new_connections': new_indices.shape[1] if new_indices.numel() > 0 else 0,
            'total_connections': classifier.model.weights.numel()
        })
        
    except Exception as e:
        print(f"Erro na evolução: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_network():
    """Endpoint para treinar a rede"""
    try:
        if classifier is None:
            return jsonify({'error': 'Classificador não inicializado'}), 500
        
        # Obter parâmetros do request
        data = request.get_json() or {}
        epochs = data.get('epochs', 5)
        batch_size = data.get('batch_size', 32)
        
        # Treinar
        classifier.train(epochs=epochs, batch_size=batch_size)
        
        return jsonify({
            'success': True,
            'message': f'Rede treinada por {epochs} epochs'
        })
        
    except Exception as e:
        print(f"Erro no treinamento: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/switch-model', methods=['POST'])
def switch_model():
    """Endpoint para alternar entre modelos"""
    global use_pretrained
    
    try:
        data = request.get_json() or {}
        model_type = data.get('model', 'pretrained')
        
        if model_type == 'pretrained':
            use_pretrained = True
            logger.info("🔄 Alternando para modelo pré-treinado ImageNet")
        elif model_type == 'seonn':
            use_pretrained = False
            logger.info("🔄 Alternando para modelo SEONN")
        else:
            return jsonify({'error': 'Tipo de modelo inválido'}), 400
        
        return jsonify({
            'success': True,
            'active_model': 'pretrained_imagenet' if use_pretrained else 'seonn',
            'message': f'Modelo alterado para {model_type}'
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao alternar modelo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint para verificar status da API"""
    logger.info("🔍 Verificação de status da API")
    
    status_info = {
        'status': 'online',
        'seonn_classifier_loaded': classifier is not None,
        'pretrained_classifier_loaded': pretrained_classifier is not None,
        'active_model': 'pretrained_imagenet' if use_pretrained else 'seonn',
        'model_path': 'cat_dog_seonn.pth' if os.path.exists('cat_dog_seonn.pth') else None,
        'total_predictions': prediction_count,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"📊 Status: SEONN {'✅' if classifier else '❌'}, Pretrained {'✅' if pretrained_classifier else '❌'}, Predições: {prediction_count}")
    
    return jsonify(status_info)

if __name__ == '__main__':
    logger.info("🚀 Iniciando API da SEONN...")
    
    # Inicializar classificador
    if initialize_classifier():
        logger.info("✅ API pronta para uso!")
        logger.info("🌐 Servidor iniciando em http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("❌ Falha ao inicializar classificador")
        logger.error("💡 Execute primeiro: python3 quick_train_model.py")


