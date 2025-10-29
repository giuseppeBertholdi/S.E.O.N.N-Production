# Dockerfile para Renault Detector
# Imagem otimizada para treinamento e deploy

FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código do projeto
COPY . .

# Criar diretórios necessários
RUN mkdir -p data/images/{train,val,test} \
    data/labels/{train,val,test} \
    models \
    logs

# Tornar scripts executáveis
RUN chmod +x scripts/*.py run_pipeline.py test_api.py

# Expor porta da API
EXPOSE 8000

# Comando padrão (iniciar API)
CMD ["python", "scripts/deploy_api.py", "--host", "0.0.0.0", "--port", "8000"]

# Labels para metadados
LABEL maintainer="Renault Detector Team"
LABEL description="API para detecção de carros Renault usando YOLOv8"
LABEL version="1.0.0"
