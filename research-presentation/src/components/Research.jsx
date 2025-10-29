import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  Cpu, 
  Database, 
  BarChart3, 
  Code, 
  Zap,
  CheckCircle,
  Clock,
  Target
} from 'lucide-react';

const Research = () => {
  const [activeTab, setActiveTab] = useState(0);

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

  const tabs = [
    {
      id: 0,
      title: 'Componentes',
      icon: Brain,
      content: {
        title: 'Componentes Fundamentais',
        description: 'Os principais componentes que constituem a arquitetura Self-Evolving Organic Neural Network.',
        steps: [
          {
            title: 'Nós Autônomos',
            description: 'Neurônios inteligentes capazes de processar informações e ajustar comportamento dinamicamente.',
            status: 'completed',
            icon: Brain,
          },
          {
            title: 'DNA Neural',
            description: 'Estrutura de identidade evolutiva que armazena histórico de aprendizado e especialização.',
            status: 'completed',
            icon: Database,
          },
          {
            title: 'Plasticidade Sináptica',
            description: 'Formação e dissolução dinâmica de conexões baseada em relevância contextual.',
            status: 'in_progress',
            icon: Zap,
          },
          {
            title: 'Núcleo Gerenciador',
            description: 'Centro de auto-organização que observa padrões e ativa subconjuntos especializados.',
            status: 'pending',
            icon: BarChart3,
          },
          {
            title: 'Grafo Dinâmico',
            description: 'Estrutura física que evolui com o tempo influenciada pelo fluxo de dados.',
            status: 'pending',
            icon: Target,
          },
        ],
      },
    },
    {
      id: 1,
      title: 'Implementação',
      icon: Cpu,
      content: {
        title: 'Tecnologias Utilizadas',
        description: 'Frameworks e bibliotecas utilizadas na implementação da arquitetura SEONN.',
        technologies: [
          {
            name: 'PyTorch',
            description: 'Framework principal para redes neurais dinâmicas e grafos',
            category: 'Deep Learning',
            icon: Brain,
          },
          {
            name: 'PyTorch Geometric',
            description: 'Extensão para aprendizado em grafos e estruturas dinâmicas',
            category: 'Graph Learning',
            icon: Cpu,
          },
          {
            name: 'Python',
            description: 'Linguagem de programação principal',
            category: 'Programação',
            icon: Code,
          },
          {
            name: 'NumPy',
            description: 'Computação numérica eficiente',
            category: 'Computação',
            icon: Database,
          },
          {
            name: 'Matplotlib',
            description: 'Visualização de dados e resultados',
            category: 'Visualização',
            icon: BarChart3,
          },
          {
            name: 'Scikit-learn',
            description: 'Algoritmos de machine learning clássico',
            category: 'ML',
            icon: Target,
          },
        ],
      },
    },
    {
      id: 2,
      title: 'Aplicações',
      icon: Database,
      content: {
        title: 'Aplicações Práticas',
        description: 'Potencial de aplicação da SEONN em diversos setores industriais e tecnológicos.',
        datasets: [
          {
            name: 'Robótica Autônoma',
            description: 'Adaptação dinâmica a ambientes imprevisíveis e operações flexíveis',
            samples: 'Produção',
            classes: 'Agricultura',
            type: 'Espacial',
          },
          {
            name: 'Diagnóstico Médico',
            description: 'Sistemas que evoluem com novos dados clínicos e detectam doenças complexas',
            samples: 'Detecção',
            classes: 'Precoce',
            type: 'Clínica',
          },
          {
            name: 'Cibersegurança',
            description: 'Sistemas de defesa proativos que identificam e neutralizam ameaças emergentes',
            samples: 'Defesa',
            classes: 'Proativa',
            type: 'Cibernética',
          },
        ],
      },
    },
  ];

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'in_progress':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'pending':
        return <Target className="h-5 w-5 text-gray-400" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'border-green-200 bg-green-50';
      case 'in_progress':
        return 'border-yellow-200 bg-yellow-50';
      case 'pending':
        return 'border-gray-200 bg-gray-50';
      default:
        return 'border-gray-200 bg-white';
    }
  };

  return (
    <section id="research" className="section-padding bg-gray-50">
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
            <span className="gradient-text">Arquitetura da SEONN</span>
          </motion.h2>
          <motion.p
            variants={itemVariants}
            className="text-xl text-gray-600 max-w-3xl mx-auto text-center section-description"
          >
            Explore os componentes fundamentais da Self-Evolving Organic Neural Network
            e sua implementação técnica.
          </motion.p>
        </motion.div>

        {/* Tab Navigation */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="flex flex-wrap justify-center gap-4 mb-12"
        >
          {tabs.map((tab) => (
            <motion.button
              key={tab.id}
              variants={itemVariants}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-primary text-white shadow-lg'
                  : 'bg-white text-secondary hover:bg-gray border'
              }`}
            >
              <tab.icon className="h-5 w-5" />
              <span>{tab.title}</span>
            </motion.button>
          ))}
        </motion.div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="card p-8 md:p-12"
          >
            <div className="mb-8">
              <h3 className="text-3xl font-bold text-gray-800 mb-4">
                {tabs[activeTab].content.title}
              </h3>
              <p className="text-lg text-gray-600">
                {tabs[activeTab].content.description}
              </p>
            </div>

            {/* Methodology Steps */}
            {activeTab === 0 && (
              <div className="space-y-6">
                {tabs[activeTab].content.steps.map((step, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`flex items-start space-x-4 p-6 rounded-lg border-2 ${getStatusColor(step.status)}`}
                  >
                    <div className="flex-shrink-0">
                      {getStatusIcon(step.status)}
                    </div>
                    <div className="flex-grow">
                      <div className="flex items-center space-x-3 mb-2">
                        <step.icon className="h-6 w-6 text-primary-600" />
                        <h4 className="text-xl font-semibold text-gray-800">
                          {step.title}
                        </h4>
                      </div>
                      <p className="text-gray-600">{step.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}

            {/* Technologies Grid */}
            {activeTab === 1 && (
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {tabs[activeTab].content.technologies.map((tech, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ y: -5 }}
                    className="bg-white p-6 rounded-lg border border-gray-200 hover:shadow-lg transition-all duration-300"
                  >
                    <div className="flex items-center space-x-3 mb-4">
                      <tech.icon className="h-8 w-8 text-primary-600" />
                      <h4 className="text-lg font-semibold text-gray-800">
                        {tech.name}
                      </h4>
                    </div>
                    <p className="text-gray-600 mb-3">{tech.description}</p>
                    <span className="inline-block px-3 py-1 bg-primary-100 text-primary-700 text-sm font-medium rounded-full">
                      {tech.category}
                    </span>
                  </motion.div>
                ))}
              </div>
            )}

            {/* Datasets */}
            {activeTab === 2 && (
              <div className="space-y-6">
                {tabs[activeTab].content.datasets.map((dataset, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-white p-6 rounded-lg border border-gray-200 hover:shadow-lg transition-all duration-300"
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <h4 className="text-xl font-semibold text-gray-800 mb-2">
                          {dataset.name}
                        </h4>
                        <p className="text-gray-600 mb-4">{dataset.description}</p>
                        <div className="flex space-x-6 text-sm text-gray-500">
                          <span><strong>Amostras:</strong> {dataset.samples}</span>
                          <span><strong>Classes:</strong> {dataset.classes}</span>
                          <span><strong>Tipo:</strong> {dataset.type}</span>
                        </div>
                      </div>
                      <Database className="h-12 w-12 text-primary-600 opacity-20" />
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
};

export default Research;
