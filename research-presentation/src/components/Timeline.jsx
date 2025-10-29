import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle2, Calendar, Code, GitBranch, TrendingUp, FileText, Presentation } from 'lucide-react';

const Timeline = () => {
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

  const milestones = [
    {
      id: 1,
      phase: 'Fase 1',
      title: 'Fundamentação Teórica',
      description: 'Revisão bibliográfica e definição da arquitetura SEONN',
      date: 'Jan - Mar 2024',
      icon: Calendar,
      status: 'completed',
    },
    {
      id: 2,
      phase: 'Fase 2',
      title: 'Implementação Inicial',
      description: 'Desenvolvimento do modelo base e estrutura dinâmica',
      date: 'Abr - Jun 2024',
      icon: Code,
      status: 'completed',
    },
    {
      id: 3,
      phase: 'Fase 3',
      title: 'Testes e Validação',
      description: 'Experimentação com MNIST, CIFAR-10 e Cat vs Dog',
      date: 'Jul - Set 2024',
      icon: GitBranch,
      status: 'completed',
    },
    {
      id: 4,
      phase: 'Fase 4',
      title: 'Otimização',
      description: 'Refinamento dos mecanismos de plasticidade e evolução',
      date: 'Abr - Jun 2024',
      icon: TrendingUp,
      status: 'completed',
    },
    {
      id: 5,
      phase: 'Fase 5',
      title: 'Documentação',
      description: 'Preparação de materiais para feira de ciências',
      date: 'Jul - Set 2024',
      icon: FileText,
      status: 'completed',
    },
    {
      id: 6,
      phase: 'Fase 6',
      title: 'Apresentação',
      description: 'Exposição dos resultados na feira de iniciação científica',
      date: 'Set 2024',
      icon: Presentation,
      status: 'completed',
    },
  ];

  const getIconColor = (status) => {
    return status === 'completed' 
      ? 'bg-green-100 text-green-600' 
      : 'bg-blue-100 text-blue-600';
  };

  return (
    <section id="timeline" className="section-padding bg-white">
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
            <span className="gradient-text">Linha do Tempo</span>
          </motion.h2>
          <motion.p
            variants={itemVariants}
            className="text-xl text-gray-600 max-w-3xl mx-auto text-center section-description"
          >
            Evolução do projeto SEONN até Setembro de 2024
          </motion.p>
        </motion.div>

        {/* Timeline Grid 3x2 */}
        <div className="max-w-6xl mx-auto">
          <div className="relative py-8">
            {/* Timeline Cards Grid - 3 colunas, 2 linhas */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {milestones.map((milestone, index) => (
                <motion.div
                  key={milestone.id}
                  variants={itemVariants}
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true }}
                  className="relative"
                >
                  {/* Card */}
                  <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl p-6 shadow-sm border border-gray-100 hover:shadow-md hover:border-blue-200 transition-all text-center">
                    {/* Phase Badge */}
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-xs font-semibold text-blue-600 uppercase tracking-wide">
                        {milestone.phase}
                      </span>
                      {milestone.status === 'completed' && (
                        <CheckCircle2 className="h-5 w-5 text-green-500" />
                      )}
                    </div>
                    
                    {/* Icon */}
                    <div className={`inline-flex w-16 h-16 ${getIconColor(milestone.status)} rounded-2xl flex items-center justify-center mb-4 shadow-lg`}>
                      <milestone.icon className="h-8 w-8" />
                    </div>

                    {/* Content */}
                    <div className="mb-3">
                      <h3 className="text-lg font-semibold text-gray-800 mb-2">
                        {milestone.title}
                      </h3>
                      <p className="text-sm text-gray-600 leading-relaxed">
                        {milestone.description}
                      </p>
                    </div>

                    {/* Date */}
                    <span className="text-xs text-gray-500 font-medium">
                      {milestone.date}
                    </span>
                  </div>

                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Timeline;
