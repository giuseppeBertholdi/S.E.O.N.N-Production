# SEONN Web Demo

## Como executar a demonstração completa

### 1. Iniciar o Backend (API SEONN)
```bash
# No diretório raiz do projeto
python3 backend_api.py
```

O backend irá:
- Carregar o modelo SEONN treinado (`cat_dog_seonn.pth`)
- Iniciar a API Flask na porta 5000
- Disponibilizar endpoints para predição e estatísticas da rede

### 2. Abrir o Frontend
```bash
# Abrir o arquivo index.html no navegador
# Ou usar um servidor local simples:
python3 -m http.server 8000
# E acessar: http://localhost:8000
```

### 3. Funcionalidades Disponíveis

#### ✅ Com Backend Ativo:
- **Predições Reais**: Usa o modelo SEONN treinado para classificar gatos/cachorros
- **Estatísticas Dinâmicas**: Mostra dados reais da rede neural
- **Indicador de Status**: 🟢 Verde quando o modelo está carregado

#### ⚠️ Sem Backend:
- **Modo Simulação**: Usa dados simulados para demonstração
- **Indicador de Status**: 🔴 Vermelho indicando modo simulação

### 4. Endpoints da API

- `GET /api/status` - Status da API e modelo
- `POST /api/predict` - Predição de imagem
- `GET /api/network-stats` - Estatísticas da rede
- `POST /api/evolve` - Evolução da rede
- `POST /api/train` - Treinamento adicional

### 5. Modelo SEONN

O modelo já está treinado e salvo em `cat_dog_seonn.pth`. Ele foi treinado com:
- **Arquitetura**: Self-Evolving Optimized Neural Network
- **Dataset**: Imagens sintéticas de gatos e cachorros
- **Classes**: 2 (Gato, Cachorro)
- **Tamanho da Imagem**: 64x64 pixels
- **Neurônios**: ~5000 neurônios iniciais
- **Conexões**: Evolutivas (pruning e crescimento)

### 6. Troubleshooting

Se a API não estiver funcionando:
1. Verifique se o arquivo `cat_dog_seonn.pth` existe
2. Execute `python3 train_model.py` para treinar um novo modelo
3. Verifique se a porta 5000 está livre
4. Consulte os logs do console para erros

### 7. Desenvolvimento

Para modificar o modelo ou treinar novamente:
```bash
python3 train_model.py
```

Para testar apenas o classificador:
```bash
python3 cat_dog_classifier.py
```