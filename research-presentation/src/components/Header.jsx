import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Menu, X, GraduationCap, Home, User, BookOpen, BarChart3, Mail, Brain, Cpu } from 'lucide-react';

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { name: 'Início', href: '#home', icon: Home },
    { name: 'Sobre', href: '#about', icon: User },
    { name: 'Inovação', href: '#innovation', icon: Brain },
    { name: 'Pesquisa', href: '#research', icon: BookOpen },
    { name: 'Resultados', href: '#results', icon: BarChart3 },
    { name: 'Timeline', href: '#timeline', icon: BookOpen },
    { name: 'Conclusões', href: '#conclusions', icon: BookOpen },
  ];

  const scrollToSection = (href) => {
    const element = document.querySelector(href);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
    setIsMenuOpen(false);
  };

  return (
    <motion.header
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
      className={`fixed top-0 left-0 right-0 z-50 navbar ${
        isScrolled ? 'scrolled' : ''
      }`}
    >
      <nav className="navbar-container">
        {/* Logo */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="flex items-center space-x-3"
        >
          <GraduationCap className="h-7 w-7 text-blue" />
          <span className="text-lg font-semibold gradient-text">
            S.E.O.N.N.
          </span>
        </motion.div>

        {/* Desktop Navigation */}
        <div className="hidden md:flex">
          <ul className="nav-links">
            {navItems.map((item) => (
              <li key={item.name}>
                <motion.button
                  onClick={() => scrollToSection(item.href)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="nav-link"
                >
                  <item.icon className="h-4 w-4 btn-icon" />
                  <span>{item.name}</span>
                </motion.button>
              </li>
            ))}
          </ul>
        </div>

        {/* Mobile Menu Button */}
        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          className="md:hidden p-2 rounded-full hover:bg-gray-100 transition-colors"
        >
          {isMenuOpen ? (
            <X className="h-5 w-5 text-gray-600" />
          ) : (
            <Menu className="h-5 w-5 text-gray-600" />
          )}
        </motion.button>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="mobile-menu"
          >
            {navItems.map((item) => (
              <motion.button
                key={item.name}
                onClick={() => scrollToSection(item.href)}
                whileHover={{ x: 5 }}
                whileTap={{ scale: 0.98 }}
                className="mobile-nav-link"
              >
                <item.icon className="h-5 w-5 btn-icon" />
                <span>{item.name}</span>
              </motion.button>
            ))}
          </motion.div>
        )}
      </nav>
    </motion.header>
  );
};

export default Header;
