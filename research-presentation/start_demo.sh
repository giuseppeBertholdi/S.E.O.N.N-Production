#!/bin/bash

# Script para iniciar o backend YOLO e o frontend React

echo "🚀 Iniciando YOLO Webcam Demo..."

# Verificar se o modelo YOLO existe
if [ ! -f "/home/giuseppe/seonn_torch/yolov8n.pt" ]; then
    echo "❌ Modelo YOLO não encontrado em /home/giuseppe/seonn_torch/yolov8n.pt"
    echo "📥 Baixando modelo YOLO..."
    cd /home/giuseppe/seonn_torch
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

# Instalar dependências do backend
echo "📦 Instalando dependências do backend..."
pip install -r requirements_backend.txt

# Iniciar backend em background
echo "🔧 Iniciando backend YOLO..."
cd /home/giuseppe/seonn_torch/research-presentation
python yolo_backend.py &
BACKEND_PID=$!

# Aguardar backend inicializar
echo "⏳ Aguardando backend inicializar..."
sleep 5

# Verificar se backend está rodando
if curl -s http://localhost:5000/api/model/status > /dev/null; then
    echo "✅ Backend YOLO iniciado com sucesso!"
else
    echo "❌ Erro ao iniciar backend"
    exit 1
fi

# Instalar dependências do frontend
echo "📦 Instalando dependências do frontend..."
npm install

# Iniciar frontend
echo "🌐 Iniciando frontend React..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🎉 Demo iniciado com sucesso!"
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend: http://localhost:5000"
echo ""
echo "Para parar os serviços, pressione Ctrl+C"

# Função para limpar processos ao sair
cleanup() {
    echo ""
    echo "🛑 Parando serviços..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Capturar Ctrl+C
trap cleanup SIGINT

# Aguardar indefinidamente
wait





