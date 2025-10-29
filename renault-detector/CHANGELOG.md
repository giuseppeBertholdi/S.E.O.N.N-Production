# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [1.0.0] - 2024-10-25

### Adicionado
- Projeto completo de detecção de carros Renault usando YOLOv8
- Script de download automático de imagens (`download_dataset.py`)
- Script de preparação de dataset com aumentos de dados (`prepare_dataset.py`)
- Script de treinamento com configurações otimizadas (`train_yolo.py`)
- Script de avaliação com métricas detalhadas (`evaluate_model.py`)
- API FastAPI para predições em tempo real (`deploy_api.py`)
- Pipeline automatizado (`run_pipeline.py`)
- Script de teste da API (`test_api.py`)
- Exemplo de uso da API (`example_usage.py`)
- Configuração para Google Colab (`colab_setup.ipynb`)
- Dockerfile e docker-compose para deploy
- Script de instalação (`setup.sh`)
- Documentação completa (README.md)
- Arquivo de configuração centralizada (`config.py`)
- Suporte a Weights & Biases para monitoramento
- Aumentos de dados avançados (flip, rotação, brilho, blur, ruído)
- Early stopping e Exponential Moving Average (EMA)
- Compatibilidade com GPU CUDA e Google Colab
- Endpoints de API para predição única e em lote
- Health check e informações do modelo
- Validação automática de dataset
- Geração de relatórios de métricas em JSON
- Matriz de confusão e visualizações
- Suporte a múltiplos formatos de imagem
- Configurações otimizadas para alta precisão (>95%)

### Características Técnicas
- **Modelo**: YOLOv8x
- **Precisão**: Meta de >95%
- **Tamanho da imagem**: 1024x1024
- **Batch size**: 16
- **Épocas**: 200
- **Otimizador**: AdamW
- **Classes**: Renault (0) e Other (1)
- **Aumentos**: Flip, rotação, brilho, contraste, blur, ruído
- **Monitoramento**: Weights & Biases
- **API**: FastAPI com endpoints RESTful
- **Deploy**: Docker e docker-compose

### Estrutura do Projeto
```
renault-detector/
├── data/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── train_yolo.py
│   ├── evaluate_model.py
│   └── deploy_api.py
├── models/
│   └── best.pt
├── requirements.txt
├── README.md
├── Dockerfile
├── docker-compose.yml
└── setup.sh
```

### Comandos Principais
```bash
# Instalação
pip install -r requirements.txt

# Pipeline completo
python run_pipeline.py --action full

# Treinamento
python scripts/train_yolo.py

# API
uvicorn scripts.deploy_api:app --reload

# Teste
python test_api.py
```

### Próximas Versões
- [ ] Suporte a mais marcas de carros
- [ ] Interface web para upload
- [ ] Deploy em cloud (AWS, GCP, Azure)
- [ ] Cache de predições
- [ ] Métricas em tempo real
- [ ] Suporte a vídeo
- [ ] Modelo quantizado para edge devices
