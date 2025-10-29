import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown, Brain, Zap, Target, Award, TrendingUp, Bot } from 'lucide-react';

function Hero() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
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

  const iconVariants = {
    hidden: { scale: 0, rotate: -180 },
    visible: {
      scale: 1,
      rotate: 0,
      transition: {
        duration: 0.8,
        ease: 'easeOut',
      },
    },
  };

  const stats = [
    { icon: Brain, label: 'Dinamico', value: 'Adaptacao Real', color: 'text-blue-600' },
    { icon: Zap, label: 'Eficiencia', value: '+15% Performance', color: 'text-green-600' },
    { icon: Target, label: 'Precisao', value: '98.5% MNIST', color: 'text-purple-600' },
    { icon: Award, label: 'Inovacao', value: 'Organico', color: 'text-orange-600' },
  ];

  return (
    <section id="home" className="min-h-screen flex items-center justify-center gradient-bg relative overflow-hidden">
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          animate={{
            rotate: 360,
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: 'linear',
          }}
          className="absolute -top-40 -right-40 w-80 h-80 bg-blue-light rounded-full opacity-20 blur-3xl"
        />
        <motion.div
          animate={{
            rotate: -360,
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: 'linear',
          }}
          className="absolute -bottom-40 -left-40 w-96 h-96 bg-gray-light rounded-full opacity-20 blur-3xl"
        />
      </div>

      <div className="container relative z-10 py-20">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="text-center"
        >
          <motion.div variants={itemVariants} className="mb-8">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="inline-block mb-6 px-6 py-2 bg-blue-light rounded-full border border-blue-200"
            >
              <span className="text-sm font-semibold text-blue-700">Pesquisa em Inteligencia Artificial Evolutiva</span>
            </motion.div>
            
            <div className="flex flex-col md:flex-row items-center justify-center gap-6 mb-6">
              <motion.h1 className="text-5xl md:text-7xl font-bold text-center">
                <span className="gradient-text">Self-Evolving Organic</span>
                <br />
                <span className="text-primary">Neural Network</span>
              </motion.h1>
              
            </div>
            
            <motion.p
              variants={itemVariants}
              className="text-xl md:text-2xl text-secondary max-w-3xl mx-auto text-center section-description leading-relaxed"
            >
              Uma arquitetura neural dinamica e auto-organizavel, inspirada em principios biologicos 
              de plasticidade e desenvolvimento neural. Redes que evoluem, aprendem e se adaptam em tempo real.
            </motion.p>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12 max-w-4xl mx-auto"
          >
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                variants={iconVariants}
                whileHover={{ scale: 1.05, y: -5 }}
                className="bg-white/80 backdrop-blur-lg rounded-2xl p-4 shadow-lg border border-gray-200"
              >
                <stat.icon className={`h-6 w-6 ${stat.color} mx-auto mb-2`} />
                <p className="text-xs text-gray-600 mb-1">{stat.label}</p>
                <p className="text-sm font-bold text-gray-800">{stat.value}</p>
              </motion.div>
            ))}
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12"
          >
            <motion.button
              whileHover={{ scale: 1.05, boxShadow: "0 10px 25px rgba(59, 130, 246, 0.3)" }}
              whileTap={{ scale: 0.95 }}
              onClick={() => document.getElementById('research').scrollIntoView({ behavior: 'smooth' })}
              className="btn-primary text-lg px-8 py-4 flex items-center gap-2"
            >
              <Brain className="h-5 w-5" />
              Explorar Pesquisa
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => document.getElementById('results').scrollIntoView({ behavior: 'smooth' })}
              className="btn-secondary text-lg px-8 py-4 flex items-center gap-2"
            >
              <TrendingUp className="h-5 w-5" />
              Ver Resultados
            </motion.button>
          </motion.div>

          <motion.div
            variants={itemVariants}
            initial={{ y: 0 }}
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2 cursor-pointer"
            onClick={() => document.getElementById('about').scrollIntoView({ behavior: 'smooth' })}
          >
            <ArrowDown className="h-8 w-8 text-gray-400" />
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}

export default Hero;

