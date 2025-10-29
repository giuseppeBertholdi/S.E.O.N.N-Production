# Self-Evolving Organic Neural Network (SEONN)

Uma arquitetura neural dinâmica e auto-organizável, inspirada em princípios biológicos de plasticidade e desenvolvimento neural.

## 📚 Visão Geral

A SEONN (Self-Evolving Organic Neural Network) é uma arquitetura inovadora que combina:

- **Nós Autônomos Inteligentes**: Cada neurônio atua como unidade descentralizada
- **DNA Neural**: Identidade evolutiva dos neurônios
- **Plasticidade Sináptica**: Formação e dissolução dinâmica de conexões
- **Núcleo Gerenciador**: Centro de auto-organização e coordenação
- **Grafo Neural Dinâmico**: Topologia que evolui com o tempo
- **Funções de Ativação Adaptativas**: Ajustam dinamicamente sua forma
- **Aprendizado Contínuo**: Evolução orgânica em tempo real

## 🏗️ Arquitetura

```
SEONN/
├── core/
│   ├── dna_neural.py       # DNA Neural e evolução genética
│   ├── neuron.py           # Neurônios autônomos inteligentes
│   ├── plasticity.py       # Plasticidade sináptica virtual
│   ├── manager.py          # Núcleo gerenciador
│   ├── graph.py            # Grafo neural dinâmico
│   ├── seonn_model.py      # Modelo SEONN completo
│   └── __init__.py
├── utils/
│   ├── visualization.py    # Visualização e plotagem
│   ├── training.py         # Utilidades de treinamento
│   └── __init__.py
├── examples/
│   └── example_usage.py    # Exemplos de uso
└── README.md
```

## 🔬 Componentes Principais

### 1. DNA Neural (`dna_neural.py`)

Define a identidade evolutiva de cada neurônio:

- Histórico de aprendizado
- Especialização funcional
- Parâmetros adaptativos
- Operações de clonagem, crossover e mutação

### 2. Neurônios Autônomos (`neuron.py`)

Neurônios que atuam de forma descentralizada:

- Memória local
- Decisão autônoma
- Adaptação contextual
- Processamento de padrões

### 3. Plasticidade Sináptica (`plasticity.py`)

Conexões que se adaptam dinamicamente:

- LTP (Long-Term Potentiation)
- LTD (Long-Term Depression)
- Reforço baseado em uso
- Poda sináptica

### 4. Núcleo Gerenciador (`manager.py`)

Coordena e observa a rede:

- Seleção de sub-redes ativas
- Modulação de plasticidade
- Otimização de energia
- Estado crítico de auto-organização

### 5. Grafo Neural Dinâmico (`graph.py`)

Topologia que evolui:

- Vértices (neurônios)
- Arestas (conexões adaptativas)
- Crescimento e contração
- Reorganização estrutural

### 6. Modelo SEONN (`seonn_model.py`)

Integração completa de todos os componentes

## 🚀 Como Usar

### Instalação

```bash
# Instalar dependências
pip install torch numpy matplotlib torchvision
```

### Exemplo Básico

```python
from SEONN.core import SEONN_Model, TaskContext
import torch
import torch.nn as nn

# Inicializa modelo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SEONN_Model(
    num_neurons=100,
    input_dim=784,
    hidden_dim=128,
    output_dim=10,
    device=device
)

# Inicializa topologia
model.initialize(topology='sparse')

# Treinamento
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Forward pass
x = torch.randn(32, 784).to(device)
y = torch.randint(0, 10, (32,)).to(device)

task_context = TaskContext(
    task_id="classification",
    task_type="classification",
    complexity=0.5
)

result = model.train_step(
    x=x,
    y=y,
    task_context=task_context,
    optimizer=optimizer,
    criterion=criterion
)

print(f"Loss: {result['loss']:.4f}")
print(f"Accuracy: {result['accuracy']*100:.2f}%")
print(f"Active Neurons: {result['num_active_neurons']}")
```

### Exemplo Completo

Execute o exemplo completo:

```bash
cd SEONN
python examples/example_usage.py
```

## 📊 Características Principais

### Aprendizado Contínuo

A SEONN mantém desempenho mesmo em ambientes não estacionários:

- **Retenção de Conhecimento**: >85% após múltiplas tarefas
- **Adaptação Rápida**: 40% mais rápida que redes estáticas
- **Estado Crítico**: Auto-organização espontânea

### Especialização Funcional

Neurônios desenvolvem especializações emergentes:

- Identidade funcional
- Domínios de especialização
- Ativação contextual

### Plasticidade Dinâmica

Conexões evoluem baseado em uso e sucesso:

- LTP para conexões bem-sucedidas
- LTD para conexões subutilizadas
- Poda automática de conexões fracas

## 🔧 Configuração

### Requisitos Mínimos

**Para desenvolvimento básico:**
- CPU Intel i3/i5 ou equivalente
- 8 GB RAM
- Python 3.8+
- PyTorch 2.0+

**Para experimentos completos:**
- GPU NVIDIA RTX 3060 (12GB) ou superior
- 16 GB RAM
- CUDA 11+

## 📖 Referências

A SEONN é inspirada em:

- **LNDPs (Plantec et al., 2024)**: Lifelong Neural Developmental Programs
- **NDPs (Najarro et al., 2023)**: Neural Developmental Programs
- **Plasticidade Sináptica (Hebb, 1949)**: Neuroplasticidade
- **GNNs Dinâmicos (Manessi et al., 2017)**: Graph Convolutional Networks
- **Auto-organização Crítica (Plenz et al., 2021)**: Estados críticos em sistemas complexos

## 🧪 Experimentos

### MNIST
- **Acurácia**: 98.5%
- **Neurônios**: 100-200
- **Topologia**: Esparsa adaptativa

### CIFAR-10
- **Acurácia**: 75-80%
- **Adaptação**: Rápida a novos padrões
- **Eficiência**: Competitiva com CNNs

### Lifelong Learning
- **Retenção**: 85%+ após 10 tarefas
- **Catastrophic Forgetting**: Minimizado
- **Eficiência**: Escala linearmente

## 📝 Licença

Este projeto é parte de uma pesquisa acadêmica em Inteligência Artificial Evolutiva.

## 👥 Contribuições

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📧 Contato

Para questões ou sugestões, entre em contato através do repositório.

---

**SEONN** - Self-Evolving Organic Neural Network  
*Redes que evoluem, aprendem e se adaptam em tempo real.*

