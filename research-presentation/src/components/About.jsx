import React from 'react';
import { motion } from 'framer-motion';
import { User, Calendar, MapPin, Award, BookOpen, Target } from 'lucide-react';

const About = () => {
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

  const stats = [
    { label: 'Período', value: '2024-2025', icon: Calendar, color: 'bg-blue-100 text-blue-600' },
    { label: 'Área', value: 'IA / Deep Learning', icon: MapPin, color: 'bg-purple-100 text-purple-600' },
    { label: 'Tipo', value: 'Acadêmica', icon: BookOpen, color: 'bg-green-100 text-green-600' },
    { label: 'Status', value: 'Ativa', icon: Award, color: 'bg-orange-100 text-orange-600' },
  ];

  const objectives = [
    {
      title: 'Arquitetura Dinâmica',
      description: 'Desenvolver redes neurais que se reorganizam estruturalmente em tempo real.',
      icon: Target,
    },
    {
      title: 'Aprendizado Contínuo',
      description: 'Implementar mecanismos de plasticidade sináptica virtual para adaptação contínua.',
      icon: BookOpen,
    },
    {
      title: 'Evolução Orgânica',
      description: 'Criar sistemas que evoluem autonomamente como organismos biológicos.',
      icon: Award,
    },
  ];

  return (
    <section id="about" className="section-padding bg-white">
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
            <span className="gradient-text">Sobre a SEONN</span>
          </motion.h2>
          <motion.p
            variants={itemVariants}
            className="text-xl text-gray-600 max-w-3xl mx-auto text-center section-description"
          >
            A Self-Evolving Organic Neural Network (SEONN) é uma arquitetura neural dinâmica 
            que rompe com os modelos tradicionais de redes fixas, promovendo aprendizado 
            contínuo e adaptação contextual em tempo real.
          </motion.p>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              whileHover={{ scale: 1.05 }}
              className="card p-6 text-center"
            >
              <stat.icon className="h-8 w-8 text-blue mx-auto mb-3" />
              <h3 className="text-lg font-semibold text-gray-800 mb-1">
                {stat.value}
              </h3>
              <p className="text-sm text-gray-600">{stat.label}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Objectives */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid md:grid-cols-3 gap-8"
        >
          {objectives.map((objective, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              whileHover={{ y: -5 }}
              className="card p-8 text-center"
            >
              <div className="p-4 bg-blue-light rounded-full w-fit mx-auto mb-6">
                <objective.icon className="h-8 w-8 text-blue" />
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                {objective.title}
              </h3>
              <p className="text-gray-600 leading-relaxed">
                {objective.description}
              </p>
            </motion.div>
          ))}
        </motion.div>

        {/* Research Description */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mt-16"
        >
          <motion.div
            variants={itemVariants}
            className="gradient-bg rounded-2xl p-8 md:p-12"
          >
            <div className="flex items-center mb-6">
              <User className="h-8 w-8 text-blue mr-3" />
              <h3 className="text-2xl font-bold text-gray-800">
                Fundamentação Teórica
              </h3>
            </div>
            <div className="prose prose-lg max-w-none text-gray-700">
              <p className="mb-4">
                A SEONN representa um avanço significativo no campo da inteligência artificial, 
                ao propor uma arquitetura neural dinâmica e auto-organizável, inspirada em 
                princípios biológicos de plasticidade e desenvolvimento neural.
              </p>
              <p className="mb-4">
                Ao integrar elementos como nós autônomos inteligentes, DNA Neural, plasticidade 
                sináptica virtual e grafos dinâmicos, a SEONN transcende as limitações das redes 
                neurais convencionais, promovendo aprendizado contínuo e adaptação contextual.
              </p>
              <p>
                Os resultados evidenciam desempenho superior em métricas essenciais como 
                retenção de conhecimento, acurácia em ambientes dinâmicos e capacidade de 
                reorganização estrutural, inaugurando um novo paradigma em redes neurais evolutivas.
              </p>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default About;
