import React from 'react';
import { motion } from 'framer-motion';
import { Target, CheckCircle, Lightbulb, ArrowRight, Telescope } from 'lucide-react';

const Conclusions = () => {
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

  const conclusions = [
    {
      icon: Target,
      title: 'Arquitetura Inovadora',
      description: 'A SEONN demonstrou eficacia superior em ambientes dinamicos, com capacidade de adaptacao estrutural em tempo real.',
    },
    {
      icon: CheckCircle,
      title: 'Alto Desempenho',
      description: 'Resultados comprovam melhoria de 15% na performance geral e 85% de retencao de conhecimento em aprendizado continuo.',
    },
    {
      icon: Lightbulb,
      title: 'Abordagem Biologica',
      description: 'Mecanismos inspirados em plasticidade neural provaram-se eficazes para evolucao e auto-organizacao de redes.',
    },
  ];

  const futureWork = [
    {
      title: 'Escalabilidade',
      description: 'Investigar otimizacoes para redes de grande porte com milhares de nos',
    },
    {
      title: 'Aplicacoes Praticas',
      description: 'Adaptar a SEONN para problemas reais: robotica, diagnostico medico e ciberseguranca',
    },
    {
      title: 'Benchmarks Abrangentes',
      description: 'Comparar com mais modelos state-of-the-art em diferentes dominios',
    },
    {
      title: 'Metricas Avancadas',
      description: 'Desenvolver indicadores especificos para medir saude neural e eficiencia evolutiva',
    },
  ];

  return (
    <section id="conclusions" className="section-padding bg-white">
      <div className="container">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mb-16"
        >
          <motion.div
            variants={itemVariants}
            className="text-center mb-12"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="gradient-text">Conclusoes</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto section-description">
              Principais descobertas e contribuicoes cientificas do projeto
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {conclusions.map((conclusion, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -5 }}
                className="bg-gradient-to-br from-blue-50 to-white rounded-xl p-7 shadow-sm border border-blue-100 hover:shadow-md hover:border-blue-200 transition-all duration-300"
              >
                <div className="w-14 h-14 bg-blue-100 rounded-xl flex items-center justify-center mb-6">
                  <conclusion.icon className="h-7 w-7 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-800 mb-4">
                  {conclusion.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {conclusion.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mt-12"
        >
          <motion.div
            variants={itemVariants}
            className="text-center mb-10"
          >
            <div className="inline-flex items-center gap-3 mb-4">
              <Telescope className="h-8 w-8 text-blue-600" />
              <h2 className="text-4xl md:text-5xl font-bold">
                <span className="gradient-text">Trabalhos Futuros</span>
              </h2>
            </div>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto section-description">
              Direcoes futuras de pesquisa e desenvolvimento
            </p>
          </motion.div>

            <div className="max-w-5xl mx-auto grid md:grid-cols-2 gap-5">
            {futureWork.map((work, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ x: 4 }}
                className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 hover:shadow-md hover:border-blue-200 transition-all group"
              >
                <div className="flex items-start gap-4">
                  <ArrowRight className="h-6 w-6 text-blue-600 flex-shrink-0 mt-1 group-hover:translate-x-2 transition-transform" />
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">
                      {work.title}
                    </h3>
                    <p className="text-gray-600 leading-relaxed">
                      {work.description}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Conclusions;
