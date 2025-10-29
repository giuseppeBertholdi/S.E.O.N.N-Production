#!/usr/bin/env python3
import http.server
import socketserver
import json
import base64
import io
from PIL import Image
import torch
import numpy as np
from cat_dog_classifier import CatDogClassifier
import os
import urllib.parse

class SEONNHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Inicializar classificador
        self.classifier = None
        try:
            # Tentar carregar modelo existente
            if os.path.exists("cat_dog_seonn.pth"):
                self.classifier = CatDogClassifier("cat_dog_seonn.pth")
                print("✅ Classificador SEONN carregado com sucesso!")
            else:
                # Criar novo classificador sem treinar
                self.classifier = CatDogClassifier()
                print("✅ Classificador SEONN criado (sem modelo pré-treinado)")
        except Exception as e:
            print(f"❌ Erro ao inicializar classificador: {e}")
            self.classifier = None
        
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/api/status':
            self.handle_status()
        elif self.path == '/api/network-stats':
            self.handle_network_stats()
        else:
            # Servir arquivos estáticos
            self.serve_static()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/predict':
            self.handle_predict()
        elif self.path == '/api/evolve':
            self.handle_evolve()
        elif self.path == '/api/train':
            self.handle_train()
        else:
            self.send_error(404, "Endpoint not found")
    
    def serve_static(self):
        """Serve static files from web_demo directory"""
        if self.path == '/':
            self.path = '/index.html'
        
        # Mapear para web_demo
        self.path = '/web_demo' + self.path
        
        # Chamar o método pai para servir arquivos
        super().do_GET()
    
    def handle_status(self):
        """Handle status endpoint"""
        response = {
            'status': 'online',
            'classifier_loaded': self.classifier is not None,
            'model_path': 'cat_dog_seonn.pth' if os.path.exists('cat_dog_seonn.pth') else None
        }
        self.send_json_response(response)
    
    def handle_network_stats(self):
        """Handle network stats endpoint"""
        if self.classifier is None:
            self.send_error(500, "Classificador não inicializado")
            return
        
        try:
            model = self.classifier.model
            
            # Calcular estatísticas
            stats = {
                'neurons': model.num_neurons,
                'connections': model.weights.numel(),
                'neural_health': float(model.neuron_health.mean().item()),
                'activity_variance': float(model.activity_history.var().item()),
                'learning_rate': float(model.learning_rate_adaptive.mean().item())
            }
            
            response = {
                'success': True,
                'stats': stats
            }
            self.send_json_response(response)
            
        except Exception as e:
            try:
                self.send_error(500, f"Erro ao obter estatísticas: {str(e)}")
            except BrokenPipeError:
                pass
    
    def handle_predict(self):
        """Handle prediction endpoint"""
        if self.classifier is None:
            self.send_error(500, "Classificador não inicializado")
            return
        
        try:
            # Ler dados do POST
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            if 'image' not in data:
                self.send_error(400, "Imagem não fornecida")
                return
            
            # Decodificar imagem base64
            image_data = data['image']
            if image_data.startswith('data:image'):
                # Remover prefixo data:image/...;base64,
                image_data = image_data.split(',')[1]
            
            # Fazer predição
            result = self.classifier.predict_image(image_data)
            
            response = {
                'success': True,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'cat_probability': result['cat_probability'],
                'dog_probability': result['dog_probability']
            }
            self.send_json_response(response)
            
        except Exception as e:
            try:
                self.send_error(500, f"Erro na predição: {str(e)}")
            except BrokenPipeError:
                pass
    
    def handle_evolve(self):
        """Handle evolution endpoint"""
        if self.classifier is None:
            self.send_error(500, "Classificador não inicializado")
            return
        
        try:
            # Ler dados do POST
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8')) if post_data else {}
            
            threshold = data.get('threshold', 0.001)
            
            # Evoluir a rede
            pruned_indices, new_indices = self.classifier.model.evolve(pruning_threshold=threshold)
            
            response = {
                'success': True,
                'pruned_connections': pruned_indices.shape[1] if pruned_indices.numel() > 0 else 0,
                'new_connections': new_indices.shape[1] if new_indices.numel() > 0 else 0,
                'total_connections': self.classifier.model.weights.numel()
            }
            self.send_json_response(response)
            
        except Exception as e:
            self.send_error(500, f"Erro na evolução: {str(e)}")
    
    def handle_train(self):
        """Handle training endpoint"""
        if self.classifier is None:
            self.send_error(500, "Classificador não inicializado")
            return
        
        try:
            # Ler dados do POST
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8')) if post_data else {}
            
            epochs = data.get('epochs', 5)
            batch_size = data.get('batch_size', 32)
            
            # Treinar
            self.classifier.train(epochs=epochs, batch_size=batch_size)
            
            response = {
                'success': True,
                'message': f'Rede treinada por {epochs} epochs'
            }
            self.send_json_response(response)
            
        except Exception as e:
            self.send_error(500, f"Erro no treinamento: {str(e)}")
    
    def send_json_response(self, data):
        """Send JSON response with CORS headers"""
        try:
            response = json.dumps(data).encode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            self.wfile.flush()
        except BrokenPipeError:
            # Cliente desconectou, ignorar
            pass
        except Exception as e:
            print(f"Erro ao enviar resposta: {e}")

def main():
    PORT = 5001  # Mudar porta para evitar conflito
    
    print("🚀 Iniciando servidor SEONN...")
    print(f"📡 Servidor rodando na porta {PORT}")
    print("🌐 Acesse: http://localhost:5001")
    
    with socketserver.TCPServer(("", PORT), SEONNHandler) as httpd:
        print("✅ Servidor iniciado com sucesso!")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Servidor interrompido pelo usuário")
            httpd.shutdown()

if __name__ == "__main__":
    main()
