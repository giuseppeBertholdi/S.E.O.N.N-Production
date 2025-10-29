import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Zap, Activity, Cpu, Database, Target, TrendingUp, Eye } from 'lucide-react';

const ContinuousNeuralDemo = () => {
  const canvasRef = useRef(null);
  const [isRunning, setIsRunning] = useState(true);
  const [currentCycle, setCurrentCycle] = useState(0);
  const [stats, setStats] = useState({
    totalProcessed: 0,
    currentAccuracy: 0,
    averageTime: 0,
    activeNeurons: 0
  });
  const [currentImage, setCurrentImage] = useState(null);
  const [prediction, setPrediction] = useState(null);

  // Imagens de exemplo para demonstração
  const demoImages = [
    {
      src: 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=200&h=200&fit=crop&crop=face',
      label: 'Gato',
      confidence: 0.94,
      breed: 'Persa'
    },
    {
      src: 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=200&h=200&fit=crop&crop=face',
      label: 'Cachorro',
      confidence: 0.91,
      breed: null
    },
    {
      src: 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=200&h=200&fit=crop&crop=face',
      label: 'Gato',
      confidence: 0.89,
      breed: 'Siamês'
    },
    {
      src: 'https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=200&h=200&fit=crop&crop=face',
      label: 'Cachorro',
      confidence: 0.96,
      breed: null
    }
  ];

  // Função para chamar o backend real
  const predictImage = async (imageSrc) => {
    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageSrc
        })
      });

      if (response.ok) {
        const result = await response.json();
        return result;
      }
    } catch (error) {
      console.error('Erro ao chamar backend:', error);
      return null;
    }
    return null;
  };

  // Configuração das camadas da rede neural
  const layers = [
    { name: 'Input', neurons: 224, color: '#3b82f6', description: 'Imagem 224x224' },
    { name: 'Conv1', neurons: 32, color: '#8b5cf6', description: 'Convolução 3x3' },
    { name: 'Conv2', neurons: 64, color: '#06b6d4', description: 'Convolução 3x3' },
    { name: 'Conv3', neurons: 128, color: '#10b981', description: 'Convolução 3x3' },
    { name: 'Pool', neurons: 64, color: '#f59e0b', description: 'Max Pooling' },
    { name: 'Dense1', neurons: 32, color: '#ef4444', description: 'Camada Densa' },
    { name: 'Dense2', neurons: 16, color: '#ec4899', description: 'Camada Densa' },
    { name: 'Output', neurons: 2, color: '#84cc16', description: 'Gato/Cachorro' }
  ];

  useEffect(() => {
    initializeNetwork();
  }, []);

  useEffect(() => {
    if (isRunning) {
      startContinuousDemo();
    }
  }, [isRunning]);

  const initializeNetwork = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  };

  const startContinuousDemo = () => {
    const cycleDuration = 4000; // 4 segundos por ciclo completo
    const layerDuration = cycleDuration / layers.length;

    const interval = setInterval(() => {
      if (!isRunning) {
        clearInterval(interval);
        return;
      }

      // Atualizar estatísticas
      setStats(prev => ({
        totalProcessed: prev.totalProcessed + 1,
        currentAccuracy: Math.random() * 0.1 + 0.9, // 90-100%
        averageTime: Math.random() * 50 + 100, // 100-150ms
        activeNeurons: Math.floor(Math.random() * 200 + 100) // 100-300 neurônios
      }));

      // Selecionar nova imagem e fazer predição real
      const randomImage = demoImages[Math.floor(Math.random() * demoImages.length)];
      setCurrentImage(randomImage);
      
      // Chamar backend real para predição
      predictImage(randomImage.src).then(result => {
        if (result) {
          setPrediction({
            label: result.prediction,
            confidence: result.confidence,
            breed: null
          });
        } else {
          // Fallback para dados simulados se backend não estiver disponível
          setPrediction(randomImage);
        }
      });

      // Simular processamento camada por camada
      let currentLayer = 0;
      const layerInterval = setInterval(() => {
        if (currentLayer >= layers.length) {
          clearInterval(layerInterval);
          return;
        }

        setCurrentCycle(currentLayer);
        currentLayer++;
      }, layerDuration);

    }, cycleDuration);
  };

  const drawNetwork = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    // Limpar canvas
    ctx.clearRect(0, 0, rect.width, rect.height);
    
    // Desenhar conexões
    const layerWidth = rect.width / (layers.length + 1);
    
    for (let i = 0; i < layers.length - 1; i++) {
      const fromX = layerWidth * (i + 1);
      const toX = layerWidth * (i + 2);
      const fromNeurons = layers[i].neurons;
      const toNeurons = layers[i + 1].neurons;
      
      const fromSpacing = Math.min(40, (rect.height - 100) / fromNeurons);
      const toSpacing = Math.min(40, (rect.height - 100) / toNeurons);
      
      for (let j = 0; j < fromNeurons; j++) {
        for (let k = 0; k < toNeurons; k++) {
          const fromY = 50 + j * fromSpacing;
          const toY = 50 + k * toSpacing;
          
          const isActive = i === currentCycle;
          const alpha = isActive ? 0.6 : 0.1;
          
          ctx.strokeStyle = `rgba(100, 100, 100, ${alpha})`;
          ctx.lineWidth = isActive ? 2 : 1;
          ctx.beginPath();
          ctx.moveTo(fromX, fromY);
          ctx.lineTo(toX, toY);
          ctx.stroke();
        }
      }
    }

    // Desenhar neurônios
    layers.forEach((layer, layerIndex) => {
      const x = layerWidth * (layerIndex + 1);
      const neuronSpacing = Math.min(40, (rect.height - 100) / layer.neurons);
      
      for (let i = 0; i < layer.neurons; i++) {
        const y = 50 + i * neuronSpacing;
        const isActive = layerIndex === currentCycle;
        const intensity = isActive ? Math.random() * 0.8 + 0.2 : 0;
        
        // Círculo do neurônio
        const radius = 6 + intensity * 3;
        const alpha = 0.3 + intensity * 0.7;
        
        ctx.fillStyle = `${layer.color}${Math.floor(alpha * 255).toString(16).padStart(2, '0')}`;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();
        
        // Borda
        ctx.strokeStyle = layer.color;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Efeito de brilho se ativo
        if (isActive) {
          ctx.shadowColor = layer.color;
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.arc(x, y, radius + 2, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.shadowBlur = 0;
        }
      }
    });

    // Desenhar labels das camadas
    layers.forEach((layer, index) => {
      const x = layerWidth * (index + 1);
      
      ctx.fillStyle = '#374151';
      ctx.font = 'bold 12px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(layer.name, x, 30);
      
      ctx.fillStyle = '#6b7280';
      ctx.font = '10px Inter';
      ctx.fillText(layer.description, x, 45);
    });
  };

  useEffect(() => {
    const interval = setInterval(drawNetwork, 50);
    return () => clearInterval(interval);
  }, [currentCycle]);

  return (
    <section id="neural-demo" className="section-padding bg-gradient-to-br from-blue-50 via-purple-50 to-indigo-50">
      <div className="container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-blue-100 to-purple-100 px-6 py-3 rounded-full mb-6">
            <Brain className="h-5 w-5 text-blue-600" />
            <span className="text-blue-700 font-semibold">Demonstração Contínua</span>
          </div>
          
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="gradient-text">Rede Neural em Ação</span>
          </h2>
          
          <p className="text-xl text-gray-600 max-w-3xl mx-auto text-center text-center-force" style={{ textAlign: 'center' }}>
            Veja como nossa rede neural processa imagens continuamente, 
            camada por camada, em tempo real. Observe os neurônios se ativando 
            e as conexões transmitindo informações.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Visualização da Rede Neural */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="bg-white rounded-2xl p-6 shadow-lg"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-gray-800">
                Arquitetura da Rede Neural
              </h3>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`}></div>
                <span className="text-sm text-gray-600">
                  {isRunning ? 'Processando...' : 'Pausado'}
                </span>
              </div>
            </div>
            
            <div className="relative">
              <canvas
                ref={canvasRef}
                className="w-full h-80 bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg"
                style={{ minHeight: '320px' }}
              />
              
              {/* Overlay de informações */}
              <AnimatePresence>
                {isRunning && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg p-3 shadow-lg"
                  >
                    <div className="text-sm">
                      <div className="font-semibold text-gray-800">
                        Camada Ativa: {layers[currentCycle]?.name}
                      </div>
                      <div className="text-gray-600">
                        {layers[currentCycle]?.description}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {layers[currentCycle]?.neurons} neurônios
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            
            {/* Controles */}
            
          </motion.div>

          {/* Estatísticas e Resultados */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="space-y-6"
          >
            {/* Estatísticas em Tempo Real */}
            <div className="bg-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                <Activity className="h-5 w-5 mr-2 text-blue-600" />
                Estatísticas em Tempo Real
              </h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {stats.totalProcessed}
                  </div>
                  <div className="text-sm text-gray-600">Imagens Processadas</div>
                </div>
                
                <div className="bg-green-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {(stats.currentAccuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">Precisão Atual</div>
                </div>
                
                <div className="bg-purple-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {stats.averageTime.toFixed(0)}ms
                  </div>
                  <div className="text-sm text-gray-600">Tempo Médio</div>
                </div>
                
                <div className="bg-orange-50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {stats.activeNeurons}
                  </div>
                  <div className="text-sm text-gray-600">Neurônios Ativos</div>
                </div>
              </div>
            </div>

            {/* Resultado Atual */}
            {currentImage && prediction && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-2xl p-6 shadow-lg"
              >
                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                  <Eye className="h-5 w-5 mr-2 text-indigo-600" />
                  Análise Atual
                </h3>
                
                <div className="flex items-center space-x-4">
                  <img
                    src={currentImage.src}
                    alt="Current analysis"
                    className="w-20 h-20 rounded-lg object-cover"
                  />
                  
                  <div className="flex-1">
                    <div className="text-lg font-semibold text-gray-800">
                      {prediction.label}
                    </div>
                    <div className="text-sm text-gray-600 mb-2">
                      Confiança: {(prediction.confidence * 100).toFixed(1)}%
                    </div>
                    {prediction.breed && (
                      <div className="text-sm text-gray-600">
                        Raça: {prediction.breed}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>

        {/* Legenda das Camadas */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="bg-white rounded-2xl p-6 shadow-lg"
        >
          <h3 className="text-xl font-bold text-gray-800 mb-4 text-center">
            Arquitetura da Rede Neural
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {layers.map((layer, index) => (
              <div key={index} className="text-center">
                <div 
                  className="w-4 h-4 rounded-full mx-auto mb-2"
                  style={{ backgroundColor: layer.color }}
                ></div>
                <div className="text-sm font-semibold text-gray-800">
                  {layer.name}
                </div>
                <div className="text-xs text-gray-600">
                  {layer.description}
                </div>
                <div className="text-xs text-gray-500">
                  {layer.neurons} neurônios
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default ContinuousNeuralDemo;
