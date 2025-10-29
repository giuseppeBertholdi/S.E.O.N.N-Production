#!/usr/bin/env bash
# Script de instalação e configuração do projeto Renault Detector
# Executa todas as etapas necessárias para configurar o ambiente

set -e  # Parar em caso de erro

echo "🚗 Configurando projeto Renault Detector..."

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale Python 3.8+ primeiro."
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"

# Criar ambiente virtual (opcional)
read -p "Deseja criar um ambiente virtual? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "📦 Criando ambiente virtual..."
    python3 -m venv renault_env
    source renault_env/bin/activate
    echo "✅ Ambiente virtual criado e ativado"
fi

# Instalar dependências
echo "📦 Instalando dependências..."
pip install -r requirements.txt

# Verificar instalação do PyTorch
echo "🔍 Verificando PyTorch..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}')"

# Verificar outras dependências importantes
echo "🔍 Verificando dependências..."
python3 -c "
import ultralytics
import fastapi
import cv2
import PIL
print('✅ Todas as dependências principais estão instaladas!')
"

# Criar diretórios necessários
echo "📁 Criando estrutura de diretórios..."
mkdir -p data/images/{train,val,test}
mkdir -p data/labels/{train,val,test}
mkdir -p models
mkdir -p logs

echo "✅ Estrutura de diretórios criada"

# Verificar estrutura do projeto
echo "🔍 Verificando estrutura do projeto..."
python3 run_pipeline.py --action check

echo ""
echo "🎉 Configuração concluída com sucesso!"
echo ""
echo "📋 Próximos passos:"
echo "1. Baixar dataset: python run_pipeline.py --action download --sample"
echo "2. Preparar dataset: python run_pipeline.py --action prepare"
echo "3. Treinar modelo: python run_pipeline.py --action train"
echo "4. Avaliar modelo: python run_pipeline.py --action evaluate"
echo "5. Iniciar API: python run_pipeline.py --action api"
echo ""
echo "📖 Para mais informações, consulte o README.md"
