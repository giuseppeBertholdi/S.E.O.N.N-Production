import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const NeuralNetworkVisualization = ({ isProcessing, processingStep, imageSrc }) => {
  const canvasRef = useRef(null);
  const [neurons, setNeurons] = useState([]);
  const [connections, setConnections] = useState([]);
  const [activations, setActivations] = useState({});
  const [currentLayer, setCurrentLayer] = useState(0);

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
    if (isProcessing) {
      animateProcessing();
    }
  }, [isProcessing, processingStep]);

  const initializeNetwork = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Calcular posições dos neurônios
    const neuronPositions = [];
    const layerWidth = canvas.width / window.devicePixelRatio / (layers.length + 1);
    
    layers.forEach((layer, layerIndex) => {
      const x = layerWidth * (layerIndex + 1);
      const neuronSpacing = Math.min(40, (canvas.height / window.devicePixelRatio - 100) / layer.neurons);
      const startY = 50;
      
      for (let i = 0; i < layer.neurons; i++) {
        const y = startY + i * neuronSpacing;
        neuronPositions.push({
          id: `${layerIndex}-${i}`,
          x,
          y,
          layer: layerIndex,
          neuron: i,
          color: layer.color,
          name: layer.name,
          description: layer.description
        });
      }
    });

    setNeurons(neuronPositions);

    // Criar conexões entre camadas
    const newConnections = [];
    for (let i = 0; i < layers.length - 1; i++) {
      const currentLayerNeurons = neuronPositions.filter(n => n.layer === i);
      const nextLayerNeurons = neuronPositions.filter(n => n.layer === i + 1);
      
      currentLayerNeurons.forEach(fromNeuron => {
        nextLayerNeurons.forEach(toNeuron => {
          newConnections.push({
            id: `${fromNeuron.id}-${toNeuron.id}`,
            from: fromNeuron,
            to: toNeuron,
            weight: Math.random() * 2 - 1,
            active: false
          });
        });
      });
    }

    setConnections(newConnections);
  };

  const animateProcessing = () => {
    const stepDuration = 1000;
    let currentStep = 0;

    const interval = setInterval(() => {
      if (currentStep >= layers.length) {
        clearInterval(interval);
        return;
      }

      setCurrentLayer(currentStep);
      
      // Ativar neurônios da camada atual
      const newActivations = {};
      neurons.forEach(neuron => {
        if (neuron.layer === currentStep) {
          newActivations[neuron.id] = {
            intensity: Math.random() * 0.8 + 0.2,
            timestamp: Date.now()
          };
        }
      });
      setActivations(newActivations);

      // Ativar conexões
      setConnections(prev => prev.map(conn => ({
        ...conn,
        active: conn.from.layer === currentStep
      })));

      currentStep++;
    }, stepDuration);
  };

  const drawNetwork = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    // Limpar canvas
    ctx.clearRect(0, 0, rect.width, rect.height);
    
    // Desenhar conexões
    connections.forEach(conn => {
      const alpha = conn.active ? 0.6 : 0.1;
      const width = conn.active ? 2 : 1;
      
      ctx.strokeStyle = `rgba(100, 100, 100, ${alpha})`;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(conn.from.x, conn.from.y);
      ctx.lineTo(conn.to.x, conn.to.y);
      ctx.stroke();
    });

    // Desenhar neurônios
    neurons.forEach(neuron => {
      const activation = activations[neuron.id];
      const isActive = activation && (Date.now() - activation.timestamp) < 1000;
      const intensity = isActive ? activation.intensity : 0;
      
      // Círculo do neurônio
      const radius = 8 + intensity * 4;
      const alpha = 0.3 + intensity * 0.7;
      
      ctx.fillStyle = `${neuron.color}${Math.floor(alpha * 255).toString(16).padStart(2, '0')}`;
      ctx.beginPath();
      ctx.arc(neuron.x, neuron.y, radius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Borda
      ctx.strokeStyle = neuron.color;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Efeito de brilho se ativo
      if (isActive) {
        ctx.shadowColor = neuron.color;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, radius + 2, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    });

    // Desenhar labels das camadas
    layers.forEach((layer, index) => {
      const x = (rect.width / (layers.length + 1)) * (index + 1);
      
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
  }, [neurons, connections, activations]);

  return (
    <div className="neural-network-container">
      <div className="bg-white rounded-2xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-800">
            Visualização da Rede Neural
          </h3>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isProcessing ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`}></div>
            <span className="text-sm text-gray-600">
              {isProcessing ? 'Processando...' : 'Aguardando'}
            </span>
          </div>
        </div>
        
        <div className="relative">
          <canvas
            ref={canvasRef}
            className="w-full h-64 bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg"
            style={{ minHeight: '256px' }}
          />
          
          {/* Overlay de informações */}
          <AnimatePresence>
            {isProcessing && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg p-3 shadow-lg"
              >
                <div className="text-sm">
                  <div className="font-semibold text-gray-800">
                    Camada Ativa: {layers[currentLayer]?.name}
                  </div>
                  <div className="text-gray-600">
                    {layers[currentLayer]?.description}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {layers[currentLayer]?.neurons} neurônios
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        {/* Legenda */}
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
          {layers.map((layer, index) => (
            <div key={index} className="flex items-center space-x-2">
              <div 
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: layer.color }}
              ></div>
              <span className="text-xs text-gray-600">{layer.name}</span>
            </div>
          ))}
        </div>
        
        {/* Estatísticas em tempo real */}
        <div className="mt-4 grid grid-cols-3 gap-4 text-center">
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-lg font-bold text-blue-600">
              {Object.keys(activations).length}
            </div>
            <div className="text-xs text-gray-600">Neurônios Ativos</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-lg font-bold text-purple-600">
              {connections.filter(c => c.active).length}
            </div>
            <div className="text-xs text-gray-600">Conexões Ativas</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-lg font-bold text-green-600">
              {currentLayer + 1}/{layers.length}
            </div>
            <div className="text-xs text-gray-600">Camada Atual</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NeuralNetworkVisualization;

