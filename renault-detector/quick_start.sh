#!/usr/bin/env bash
# Script de início rápido para Renault Detector
# Executa o pipeline completo com configurações otimizadas

set -e

echo "🚗 Renault Detector - Início Rápido"
echo "=================================="

# Verificar se estamos no diretório correto
if [ ! -f "requirements.txt" ]; then
    echo "❌ Execute este script no diretório raiz do projeto"
    exit 1
fi

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado"
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"

# Instalar dependências se necessário
if [ ! -d "venv" ] && [ ! -d "renault_env" ]; then
    echo "📦 Instalando dependências..."
    pip install -r requirements.txt
else
    echo "✅ Dependências já instaladas"
fi

# Verificar estrutura
echo "🔍 Verificando estrutura do projeto..."
python3 run_pipeline.py --action check

# Executar pipeline completo
echo "🚀 Iniciando pipeline completo..."
python3 run_pipeline.py --action full --sample

echo ""
echo "🎉 Pipeline concluído com sucesso!"
echo ""
echo "📋 Próximos passos:"
echo "1. Verificar métricas: cat models/metrics.json"
echo "2. Iniciar API: python scripts/deploy_api.py"
echo "3. Testar API: python test_api.py --create-sample"
echo ""
echo "📖 Para mais informações, consulte o README.md"