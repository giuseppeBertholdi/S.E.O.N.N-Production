import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart3, 
  TrendingUp, 
  Target, 
  Award,
  CheckCircle,
  Clock,
  Zap,
  Brain,
  Cpu,
  Database
} from 'lucide-react';

const Results = () => {
  const [activeMetric, setActiveMetric] = useState(0);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: 'easeOut',
      },
    },
  };

  const metrics = [
    {
      id: 0,
      title: 'Retenção de Conhecimento',
      icon: Target,
      value: '85%+',
      description: 'Capacidade de manter conhecimento em aprendizado contínuo',
      color: 'text-green-600',
      bgColor: 'bg-green-100',
      details: [
        { label: 'MNIST', value: '98.5%', status: 'excellent' },
        { label: 'CIFAR-10', value: '89.7%', status: 'good' },
        { label: 'Cat vs Dog', value: '94.2%', status: 'excellent' },
      ],
    },
    {
      id: 1,
      title: 'Tempo de Adaptação',
      icon: Clock,
      value: '-40%',
      description: 'Redução no tempo de adaptação em ambientes dinâmicos',
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
      details: [
        { label: 'MNIST', value: '45min', status: 'fast' },
        { label: 'CIFAR-10', value: '3.2h', status: 'moderate' },
        { label: 'Cat vs Dog', value: '2.8h', status: 'moderate' },
      ],
    },
    {
      id: 2,
      title: 'Capacidade de Reorganização',
      icon: Zap,
      value: 'Alta',
      description: 'Flexibilidade estrutural e auto-organização',
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
      details: [
        { label: 'GPU Utilization', value: '92%', status: 'excellent' },
        { label: 'Memory Usage', value: '78%', status: 'good' },
        { label: 'CPU Usage', value: '45%', status: 'good' },
      ],
    },
  ];

  const achievements = [
    {
      title: 'Arquitetura Dinâmica',
      description: 'Redes neurais que se reorganizam estruturalmente em tempo real',
      icon: Brain,
      status: 'completed',
    },
    {
      title: 'Aprendizado Contínuo',
      description: 'Mecanismos de plasticidade sináptica virtual para adaptação contínua',
      icon: Cpu,
      status: 'completed',
    },
    {
      title: 'Evolução Orgânica',
      description: 'Sistemas que evoluem autonomamente como organismos biológicos',
      icon: Database,
      status: 'completed',
    },
    {
      title: 'Comparação com Redes Convencionais',
      description: 'Desempenho superior em métricas essenciais de adaptabilidade',
      icon: BarChart3,
      status: 'in_progress',
    },
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'excellent':
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'good':
        return 'text-blue-600 bg-blue-100';
      case 'moderate':
        return 'text-yellow-600 bg-yellow-100';
      case 'in_progress':
        return 'text-purple-600 bg-purple-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <section id="results" className="section-padding bg-white">
      <div className="container">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <motion.h2
            variants={itemVariants}
            className="text-4xl md:text-5xl font-bold mb-6"
          >
            <span className="gradient-text">Resultados da SEONN</span>
          </motion.h2>
          <motion.p
            variants={itemVariants}
            className="text-xl text-gray-600 max-w-3xl mx-auto text-center section-description"
          >
            Os resultados demonstram a eficácia dos modelos desenvolvidos e 
            o progresso significativo alcançado durante a pesquisa.
          </motion.p>
        </motion.div>

        {/* Metrics Navigation */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="flex flex-wrap justify-center gap-4 mb-12"
        >
          {metrics.map((metric) => (
            <motion.button
              key={metric.id}
              variants={itemVariants}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setActiveMetric(metric.id)}
              className={`flex items-center space-x-3 px-6 py-4 rounded-lg font-medium transition-all ${
                activeMetric === metric.id
                  ? 'bg-primary text-white shadow-lg'
                  : 'bg-white text-secondary hover:bg-gray border'
              }`}
            >
              <metric.icon className="h-6 w-6" />
              <span>{metric.title}</span>
            </motion.button>
          ))}
        </motion.div>

        {/* Metrics Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeMetric}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="card p-8 md:p-12 mb-16"
          >
            <div className="text-center mb-8">
              <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full ${metrics[activeMetric].bgColor} mb-4`}>
                {React.createElement(metrics[activeMetric].icon, { className: `h-10 w-10 ${metrics[activeMetric].color}` })}
              </div>
              <h3 className="text-3xl font-bold text-gray-800 mb-2">
                {metrics[activeMetric].title}
              </h3>
              <div className="text-5xl font-bold gradient-text mb-2">
                {metrics[activeMetric].value}
              </div>
              <p className="text-lg text-gray-600">
                {metrics[activeMetric].description}
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              {metrics[activeMetric].details.map((detail, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  className="text-center p-6 bg-gray-50 rounded-lg"
                >
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-3 ${getStatusColor(detail.status)}`}>
                    {detail.status}
                  </div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-1">
                    {detail.label}
                  </h4>
                  <p className="text-2xl font-bold text-primary-600">
                    {detail.value}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Achievements */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mb-16"
        >
          <motion.h3
            variants={itemVariants}
            className="text-3xl font-bold text-center mb-12"
          >
            <span className="gradient-text">Principais Avanços</span>
          </motion.h3>
          
          <div className="grid md:grid-cols-2 gap-8">
            {achievements.map((achievement, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -5 }}
                className="card p-8"
              >
                <div className="flex items-start space-x-4">
                  <div className={`p-3 rounded-full ${achievement.status === 'completed' ? 'bg-green-100' : 'bg-purple-100'}`}>
                    <achievement.icon className={`h-6 w-6 ${achievement.status === 'completed' ? 'text-green-600' : 'text-purple-600'}`} />
                  </div>
                  <div className="flex-grow">
                    <div className="flex items-center space-x-2 mb-2">
                      <h4 className="text-xl font-semibold text-gray-800">
                        {achievement.title}
                      </h4>
                      {achievement.status === 'completed' ? (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      ) : (
                        <Clock className="h-5 w-5 text-purple-500" />
                      )}
                    </div>
                    <p className="text-gray-600">
                      {achievement.description}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Performance Chart Placeholder */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="card p-8 md:p-12"
        >
          <div className="text-center mb-8">
            <TrendingUp className="h-12 w-12 text-primary-600 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-800 mb-2">
              Comparação de Performance
            </h3>
            <p className="text-gray-600">
              Gráfico comparativo dos resultados obtidos vs métodos baseline
            </p>
          </div>
          
          <div className="bg-gradient-to-r from-primary-50 to-secondary-50 rounded-lg p-8 text-center">
            <div className="text-6xl font-bold gradient-text mb-4">
              +15.3%
            </div>
            <p className="text-xl text-gray-700 mb-2">
              Melhoria média em relação aos métodos tradicionais
            </p>
            <div className="flex justify-center space-x-8 text-sm text-gray-600">
              <span>MNIST: +12.1%</span>
              <span>CIFAR-10: +18.7%</span>
              <span>Cat vs Dog: +15.1%</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Results;
