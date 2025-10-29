# YOLO Webcam Demo - Detecção em Tempo Real

Demonstração de detecção de objetos em tempo real usando YOLO v8 com webcam integrada ao React.

## 📋 Requisitos

- Python 3.8+
- Node.js 16+
- npm ou yarn
- Modelo YOLO (yolov8n.pt)

## 🚀 Instalação

### Backend (Flask + YOLO)

```bash
# Instalar dependências
pip install -r requirements_backend.txt

# Verificar se o modelo YOLO está disponível
ls /home/giuseppe/seonn_torch/yolov8n.pt
```

### Frontend (React)

```bash
# Navegar para a pasta do projeto
cd research-presentation

# Instalar dependências
npm install
```

## 💻 Execução

### Opção 1: Script Automatizado

```bash
# Executar o script que inicia backend e frontend
./start_demo.sh
```

### Opção 2: Manual

#### Terminal 1 - Backend:
```bash
cd research-presentation
python yolo_backend.py
```

#### Terminal 2 - Frontend:
```bash
cd research-presentation
npm run dev
```

## 🌐 Acessos

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000

## 📝 Uso

1. Acesse http://localhost:5173
2. Role até a seção "YOLO + Webcam"
3. Clique em "Iniciar Webcam"
4. Permita o acesso à câmera
5. Aponte para objetos (cadeiras, pessoas, laptops, etc.)
6. Veja as detecções em tempo real com bounding boxes coloridos

## 🎯 Funcionalidades

- ✅ Webcam em tempo real
- ✅ Detecção YOLO v8
- ✅ Bounding boxes coloridos
- ✅ Lista de objetos detectados
- ✅ Estatísticas em tempo real (FPS, contagem, etc.)
- ✅ Modo simulado (se backend não estiver disponível)

## 🔧 Estrutura

```
research-presentation/
├── src/
│   ├── components/
│   │   └── WebcamYOLODemo.jsx    # Componente principal da webcam
│   └── App.jsx                    # App principal
├── yolo_backend.py               # Backend Flask com YOLO
├── requirements_backend.txt       # Dependências Python
├── start_demo.sh                  # Script de inicialização
└── package.json                   # Configuração React
```

## 📦 Dependências

### Backend
- flask
- flask-cors
- opencv-python
- numpy
- pillow
- torch
- ultralytics

### Frontend
- react
- react-dom
- framer-motion
- lucide-react

## 🐛 Troubleshooting

### Erro: "Erro ao conectar com o backend YOLO"
- Certifique-se de que o backend está rodando: `python yolo_backend.py`
- Verifique se está em http://localhost:5000

### Erro: "Modelo YOLO não carregado"
- Verifique se yolov8n.pt existe em /home/giuseppe/seonn_torch/
- Baixe o modelo se necessário

### Webcam não funciona
- Verifique as permissões do navegador
- Tente usar HTTPS (alguns navegadores exigem HTTPS para webcam)

## 📊 Objetos Detectáveis

O YOLO v8 pode detectar 80 classes COCO, incluindo:
- Pessoas (person)
- Veículos (car, bus, truck, bicycle, motorcycle)
- Animais (cat, dog, horse, bird, etc.)
- Mobiliário (chair, couch, bed, dining table)
- Eletrônicos (laptop, mouse, keyboard, tv, cell phone)
- E muito mais!

## 📝 Notas

- A detecção funciona a ~10-15 FPS dependendo do hardware
- Para melhor performance, use GPU
- A simulação funciona mesmo sem backend para demonstração

## 🎓 Desenvolvido por

Pesquisa Acadêmica - IA e Machine Learning




