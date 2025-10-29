import React from 'react';
import { motion } from 'framer-motion';
import { 
  AlertCircle, 
  Lightbulb, 
  TrendingUp, 
  Rocket, 
  Code, 
  Zap,
  Brain,
  Network,
  Database,
  Cpu
} from 'lucide-react';

const Innovation = () => {
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

  const problems = [
    {
      icon: Network,
      title: 'Redes Fixas',
      description: 'Arquiteturas estáticas que não se adaptam a mudanças no ambiente',
      color: 'bg-red-100 text-red-600',
    },
    {
      icon: Database,
      title: 'Aprendizado Estático',
      description: 'Limitação em ambientes dinâmicos com dados em constante evolução',
      color: 'bg-orange-100 text-orange-600',
    },
    {
      icon: Cpu,
      title: 'Overhead Computacional',
      description: 'Ineficiência computacional em modelos de grande escala',
      color: 'bg-yellow-100 text-yellow-600',
    },
  ];

  const innovations = [
    {
      icon: Brain,
      title: 'Evolução Orgânica',
      description: 'Redes que crescem e se reorganizam automaticamente',
      color: 'bg-green-500',
      gradient: 'from-green-500 to-emerald-500',
    },
    {
      icon: Zap,
      title: 'Plasticidade Sináptica',
      description: 'Conexões que se formam e dissolvem dinamicamente',
      color: 'bg-blue-500',
      gradient: 'from-blue-500 to-cyan-500',
    },
    {
      icon: Rocket,
      title: 'Aprendizado Contínuo',
      description: 'Adaptação em tempo real sem esquecer conhecimento anterior',
      color: 'bg-purple-500',
      gradient: 'from-purple-500 to-pink-500',
    },
  ];

  return (
    <section id="innovation" className="section-padding bg-gray-50">
      <div className="container">
        {/* Header */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <motion.div variants={itemVariants} className="inline-block mb-4">
            <span className="px-4 py-2 bg-purple-100 text-purple-700 rounded-full text-sm font-semibold">
              🚀 Inovação e Diferenciação
            </span>
          </motion.div>
          <motion.h2
            variants={itemVariants}
            className="text-4xl md:text-5xl font-bold mb-6"
          >
            <span className="gradient-text">O Problema & A Solução</span>
          </motion.h2>
          <motion.p
            variants={itemVariants}
            className="text-xl text-gray-600 max-w-3xl mx-auto section-description"
          >
            Identificamos as limitações das redes neurais convencionais e desenvolvemos 
            uma arquitetura revolucionária que supera esses desafios.
          </motion.p>
        </motion.div>

        {/* Problems Section */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mb-12"
        >
          <motion.h3
            variants={itemVariants}
            className="text-3xl font-bold text-center mb-12"
          >
            Desafios das Redes Neurais Tradicionais
          </motion.h3>
          
          <div className="grid md:grid-cols-3 gap-8">
            {problems.map((problem, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -10, scale: 1.02 }}
                className="card p-8 bg-white hover:shadow-xl transition-all duration-300"
              >
                <div className={`w-16 h-16 ${problem.color} rounded-2xl flex items-center justify-center mb-6`}>
                  <problem.icon className="h-8 w-8" />
                </div>
                <h4 className="text-xl font-semibold text-gray-800 mb-4">
                  {problem.title}
                </h4>
                <p className="text-gray-600 leading-relaxed">
                  {problem.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Solutions Section */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mt-16"
        >
          <motion.h3
            variants={itemVariants}
            className="text-3xl font-bold text-center mb-8"
          >
            A Solução SEONN
          </motion.h3>
          
          <div className="grid md:grid-cols-3 gap-8">
            {innovations.map((innovation, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -10, scale: 1.02 }}
                className="relative overflow-hidden group"
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${innovation.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300`}></div>
                <div className="card p-8 bg-white relative z-10 border-2 border-transparent group-hover:border-transparent">
                  <div className={`w-16 h-16 ${innovation.color} rounded-2xl flex items-center justify-center mb-6 transform group-hover:scale-110 transition-transform`}>
                    <innovation.icon className="h-8 w-8 text-white" />
                  </div>
                  <h4 className="text-xl font-semibold text-gray-800 mb-4">
                    {innovation.title}
                  </h4>
                  <p className="text-gray-600 leading-relaxed">
                    {innovation.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Impact Section */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mt-20"
        >
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-3xl p-8 md:p-12 text-white relative overflow-hidden">
            <div className="absolute top-0 right-0 w-96 h-96 bg-white/10 rounded-full -mr-48 -mt-48"></div>
            <div className="relative z-10">
              <motion.div
                variants={itemVariants}
                className="flex items-center justify-center gap-3 mb-6"
              >
                <TrendingUp className="h-10 w-10" />
                <h3 className="text-3xl font-bold">Impacto Esperado</h3>
              </motion.div>
              <motion.div
                variants={itemVariants}
                className="grid md:grid-cols-3 gap-6 mt-8"
              >
                <div className="text-center p-6 bg-white/10 rounded-2xl backdrop-blur-sm">
                  <div className="text-4xl font-bold mb-2">+15%</div>
                  <p className="text-white/90">Performance</p>
                </div>
                <div className="text-center p-6 bg-white/10 rounded-2xl backdrop-blur-sm">
                  <div className="text-4xl font-bold mb-2">85%</div>
                  <p className="text-white/90">Retenção</p>
                </div>
                <div className="text-center p-6 bg-white/10 rounded-2xl backdrop-blur-sm">
                  <div className="text-4xl font-bold mb-2">-40%</div>
                  <p className="text-white/90">Tempo Adaptação</p>
                </div>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Innovation;

