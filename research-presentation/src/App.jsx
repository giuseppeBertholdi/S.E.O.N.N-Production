import React, { useState } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import About from './components/About';
import Innovation from './components/Innovation';
import Research from './components/Research';
import Results from './components/Results';
import Timeline from './components/Timeline';
import Conclusions from './components/Conclusions';
import ContinuousNeuralDemo from './components/ContinuousNeuralDemo';
import Footer from './components/Footer';
import Chat from './components/Chat';

function App() {
  const [showChat, setShowChat] = useState(false);

  if (showChat || window.location.pathname === '/chat') {
    return <Chat onBack={() => { setShowChat(false); window.history.pushState({}, '', '/'); }} />;
  }

  return (
    <div className="App">
      <Header />
      <main>
        <Hero />
        <About />
        <Innovation />
        <Research />
        <Results />
        <Timeline />
        <Conclusions />
        <ContinuousNeuralDemo />
      </main>
    </div>
  );
}

export default App;
