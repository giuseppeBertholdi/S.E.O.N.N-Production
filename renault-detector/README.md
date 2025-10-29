# Renault Car Detector

Um projeto completo de inteligência artificial para detectar carros da marca Renault em imagens usando YOLOv8. O objetivo é alcançar alta precisão (acima de 95%) na classificação de carros Renault vs outras marcas.

## 🚗 Características

- **Modelo**: YOLOv8x com configurações otimizadas
- **Precisão**: Meta de >95% de precisão
- **API**: FastAPI para deploy em produção
- **Monitoramento**: Integração com Weights & Biases
- **Aumentos**: Data augmentation avançado
- **Compatibilidade**: Google Colab e GPU CUDA

## 📁 Estrutura do Projeto

```
renault-detector/
├── data/
│   ├── images/
│   │   ├── train/          # Imagens de treinamento
│   │   ├── val/            # Imagens de validação
│   │   └── test/           # Imagens de teste
│   ├── labels/
│   │   ├── train/          # Labels YOLO para treinamento
│   │   ├── val/            # Labels YOLO para validação
│   │   └── test/           # Labels YOLO para teste
│   └── data.yaml           # Configuração do dataset
├── scripts/
│   ├── download_dataset.py # Download automático de imagens
│   ├── prepare_dataset.py  # Preparação e aumento de dados
│   ├── train_yolo.py       # Treinamento do modelo
│   ├── evaluate_model.py   # Avaliação e métricas
│   └── deploy_api.py       # API FastAPI
├── models/
│   └── best.pt             # Melhor modelo treinado
├── requirements.txt        # Dependências Python
└── README.md              # Este arquivo
```

## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone <repository-url>
cd renault-detector
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Configure o ambiente (opcional)
```bash
# Para usar Weights & Biases
wandb login

# Para usar GPU CUDA (recomendado)
# Certifique-se de ter CUDA instalado
```

## 📊 Uso do Projeto

### 1. Download do Dataset

Baixe automaticamente imagens de carros Renault e outras marcas:

```bash
# Dataset completo
python scripts/download_dataset.py

# Dataset de exemplo (menor, para testes)
python scripts/download_dataset.py --sample
```

### 2. Preparação do Dataset

Converta o dataset para formato YOLOv8 e aplique aumentos de dados:

```bash
# Com aumentos de dados
python scripts/prepare_dataset.py

# Sem aumentos de dados
python scripts/prepare_dataset.py --no-augmentation
```

### 3. Treinamento do Modelo

Treine o modelo YOLOv8 com configurações otimizadas:

```bash
# Treinamento padrão
python scripts/train_yolo.py

# Treinamento customizado
python scripts/train_yolo.py --epochs 100 --batch-size 8 --img-size 512
```

**Configurações do treinamento:**
- **Épocas**: 200
- **Tamanho da imagem**: 1024x1024
- **Batch size**: 16
- **Otimizador**: AdamW
- **Aumentos**: Flip, rotação, brilho, blur, etc.
- **Early stopping**: 50 épocas
- **EMA**: Exponential Moving Average habilitado

### 4. Avaliação do Modelo

Avalie o modelo treinado e gere métricas detalhadas:

```bash
python scripts/evaluate_model.py
```

**Métricas geradas:**
- mAP@0.5 e mAP@0.5:0.95
- Precisão, Recall e F1-Score
- Matriz de confusão
- Relatório JSON (`models/metrics.json`)

### 5. Deploy da API

Execute a API FastAPI para predições em tempo real:

```bash
# API local
uvicorn scripts.deploy_api:app --reload

# API com configurações customizadas
python scripts/deploy_api.py --host 0.0.0.0 --port 8000
```

## 🔧 API Endpoints

### Predição de Imagem Única
```bash
curl -X POST -F "file=@carro.jpg" http://127.0.0.1:8000/predict
```

**Resposta:**
```json
{
  "is_renault": true,
  "confidence": 0.95,
  "bbox": [100, 150, 300, 250],
  "class_name": "renault",
  "message": "Carro detectado: renault"
}
```

### Predição em Lote
```bash
curl -X POST -F "files=@carro1.jpg" -F "files=@carro2.jpg" http://127.0.0.1:8000/predict_batch
```

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Informações do Modelo
```bash
curl http://127.0.0.1:8000/model_info
```

## 📈 Monitoramento

O projeto inclui integração com **Weights & Biases** para monitoramento do treinamento:

- Métricas em tempo real
- Visualizações de loss e accuracy
- Comparação de experimentos
- Logs detalhados

## 🎯 Classes do Modelo

- **Classe 0**: Renault (carros da marca Renault)
- **Classe 1**: Other (outras marcas: Fiat, Volkswagen, Toyota, Chevrolet, etc.)

## 🔧 Configurações Avançadas

### Treinamento Customizado

```python
# Exemplo de configuração personalizada
trainer = RenaultTrainer()
trainer.training_config.update({
    'epochs': 300,
    'batch': 32,
    'imgsz': 1280,
    'lr0': 0.005,
    'patience': 100
})
```

### Aumentos de Dados

```python
# Configuração de aumentos personalizada
augmentation_config = {
    'flip_prob': 0.7,
    'rotation_range': 20,
    'brightness_range': 0.3,
    'contrast_range': 0.3,
    'blur_prob': 0.5
}
```

## 🐛 Solução de Problemas

### Erro de GPU
```bash
# Verificar disponibilidade de CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Instalar PyTorch com CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Erro de Memória
```bash
# Reduzir batch size
python scripts/train_yolo.py --batch-size 8

# Reduzir tamanho da imagem
python scripts/train_yolo.py --img-size 512
```

### Erro de Download
```bash
# Usar dataset de exemplo
python scripts/download_dataset.py --sample
```

## 📊 Resultados Esperados

Com o dataset completo e configurações otimizadas, você deve alcançar:

- **mAP@0.5**: >0.95
- **Precisão**: >0.95
- **Recall**: >0.90
- **F1-Score**: >0.92

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Suporte

Para dúvidas ou problemas:

1. Abra uma issue no GitHub
2. Consulte a documentação do YOLOv8: https://docs.ultralytics.com/
3. Verifique os logs de erro para diagnóstico

## 🔄 Atualizações Futuras

- [ ] Suporte a mais marcas de carros
- [ ] Interface web para upload de imagens
- [ ] Deploy em Docker
- [ ] Integração com banco de dados
- [ ] Cache de predições
- [ ] Métricas de performance em tempo real

---

**Desenvolvido com ❤️ para detecção inteligente de carros Renault**

