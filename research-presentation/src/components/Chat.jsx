import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, ArrowLeft, Bot, User, Brain } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const GEMINI_API_KEY = 'AIzaSyBjDD7u_Th5FOxFb61WTLjwpnnqC6WWLRg';
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`;

// Endpoint do backend como fallback (para contornar problemas de certificado SSL/Fortinet)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
const CHAT_API_URL = `${API_BASE_URL}/api/chat`;

const SEONN_CONTEXT = 'Você é um assistente sobre Self-Evolving Organic Neural Network (SEONN). A SEONN é uma arquitetura neural dinâmica e auto-organizável, inspirada em princípios biológicos. Componentes: Nós Autônomos, DNA Neural, Plasticidade Sináptica, Núcleo Gerenciador, Grafo Neural Dinâmico. Resultados: 85%+ retenção, 40% redução no tempo de adaptação. Aplicações: Robótica, Diagnóstico Médico, Cibersegurança.';

function Chat({ onBack }) {
  const [messages, setMessages] = useState([{ 
    role: 'assistant', 
    content: 'Olá! Sou seu assistente sobre a SEONN. Como posso ajudar?' 
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMessage = { role: 'user', content: input };
    const userInput = input;
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const prompt = `${SEONN_CONTEXT}\n\nPergunta: ${userInput}\n\nResposta:`;
      
      // Tentar primeiro chamar diretamente a API do Gemini
      let response;
      let usingProxy = false;
      
      try {
        console.log('🔗 Tentando conexão direta com Gemini API...');
        response = await fetch(GEMINI_API_URL, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { 
              temperature: 0.7, 
              maxOutputTokens: 1024 
            }
          }),
          mode: 'cors',
          credentials: 'omit'
        });

        if (!response.ok) {
          throw new Error(`Erro HTTP ${response.status}`);
        }
        
        console.log('✅ Conexão direta bem-sucedida!');
      } catch (directError) {
        // Se falhar com erro de certificado/SSL, usar backend como proxy
        if (directError.message.includes('Failed to fetch') || 
            directError.message.includes('ERR_CERT') || 
            directError.message.includes('certificate') ||
            directError.name === 'TypeError') {
          
          console.warn('⚠️ Erro na conexão direta, usando backend como proxy...');
          console.warn('Erro:', directError.message);
          
          usingProxy = true;
          
          // Tentar via backend proxy
          response = await fetch(CHAT_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userInput })
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('❌ Erro do backend:', errorData);
            throw new Error(errorData.error || errorData.details || `Erro HTTP: ${response.status}`);
          }
          
          console.log('✅ Conexão via proxy bem-sucedida!');
        } else {
          throw directError;
        }
      }

      const data = await response.json();
      
      // Formato diferente dependendo se veio direto ou via proxy
      const aiResponse = usingProxy 
        ? (data.response || 'Desculpe, não consegui processar sua pergunta.')
        : (data.candidates?.[0]?.content?.parts?.[0]?.text || 'Desculpe, não consegui processar sua pergunta.');
      
      setMessages(prev => [...prev, { role: 'assistant', content: aiResponse }]);
    } catch (error) {
      console.error('Erro completo:', error);
      console.error('Tipo do erro:', error.constructor.name);
      console.error('Stack trace:', error.stack);
      
      let errorMessage = '❌ Erro ao processar mensagem.\n\n';
      
      // Detectar tipo específico de erro
      if (error.message.includes('backend') || error.message.includes('localhost:5000')) {
        errorMessage += '🔄 **Erro ao usar proxy do backend**\n\n';
        errorMessage += 'O sistema tentou usar o backend como proxy, mas falhou.\n\n';
        errorMessage += '💡 **Verifique:**\n';
        errorMessage += '1. O backend está rodando? Execute: `python backend_api.py`\n';
        errorMessage += '2. O servidor está na porta 5000?\n';
        errorMessage += '3. Verifique os logs do backend\n\n';
        errorMessage += `Detalhes: ${error.message}`;
      } else if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        errorMessage += '🔍 **Erro de Conexão - Fortinet Detectado**\n\n';
        errorMessage += 'O Fortinet está bloqueando a conexão HTTPS.\n';
        errorMessage += 'Tentando usar backend como proxy...\n\n';
        errorMessage += '💡 **Soluções:**\n';
        errorMessage += '1. ✅ **Inicie o backend**: `python backend_api.py` (resolve o problema)\n';
        errorMessage += '2. Instale o certificado do Fortinet no sistema\n';
        errorMessage += '3. Desative temporariamente o Fortinet\n';
        errorMessage += '4. Use outra rede sem firewall corporativo\n\n';
        errorMessage += '📋 **Modo automático**: O sistema tentará usar o backend automaticamente se disponível.\n\n';
        errorMessage += `Detalhes: ${error.message}`;
      } else if (error.message.includes('NetworkError') || error.message.includes('Network request failed')) {
        errorMessage += '🌐 **Erro de Rede**\n\n';
        errorMessage += 'Não foi possível conectar à API do Gemini.\n';
        errorMessage += 'Verifique sua conexão com a internet.\n\n';
        errorMessage += `Detalhes: ${error.message}`;
      } else if (error.message.includes('ERR_CERT') || error.message.includes('SSL') || error.message.includes('certificate')) {
        errorMessage += '🔒 **Erro de Certificado SSL**\n\n';
        errorMessage += 'Fortinet interceptando conexão HTTPS.\n\n';
        errorMessage += '✅ **Solução Rápida**: Execute `python backend_api.py` para usar proxy.\n\n';
        errorMessage += `Detalhes: ${error.message}`;
      } else {
        errorMessage += `**Erro:** ${error.message || 'Erro desconhecido'}\n\n`;
        errorMessage += `Tipo: ${error.name || 'Unknown'}`;
      }
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: errorMessage 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(135deg, #dbeafe 0%, #ffffff 50%, #f3e8ff 100%)',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {/* Header */}
      <div style={{ 
        background: 'white', 
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)', 
        position: 'sticky', 
        top: 0, 
        zIndex: 50,
        padding: '1.5rem'
      }}>
        <div style={{ 
          maxWidth: '1200px', 
          margin: '0 auto', 
          display: 'flex', 
          alignItems: 'center', 
          gap: '1rem'
        }}>
          <button 
            onClick={onBack} 
            style={{ 
              padding: '0.5rem', 
              border: 'none', 
              background: 'transparent', 
              cursor: 'pointer',
              borderRadius: '8px'
            }}
          >
            <ArrowLeft size={24} color="#4b5563" />
          </button>
          
          <div style={{ 
            width: '56px', 
            height: '56px', 
            background: 'linear-gradient(135deg, #3b82f6, #2563eb, #9333ea)', 
            borderRadius: '12px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(59,130,246,0.3)'
          }}>
            <Brain size={28} color="white" />
          </div>
          
          <div>
            <h1 style={{ 
              margin: 0, 
              fontSize: '1.5rem', 
              fontWeight: 'bold', 
              background: 'linear-gradient(135deg, #2563eb, #9333ea)', 
              WebkitBackgroundClip: 'text', 
              WebkitTextFillColor: 'transparent'
            }}>
              Assistente SEONN
            </h1>
            <p style={{ 
              margin: 0, 
              fontSize: '0.875rem', 
              color: '#6b7280' 
            }}>
              Inteligência Artificial especializada em redes neurais evolutivas
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div style={{ 
        maxWidth: '1200px', 
        margin: '0 auto', 
        padding: '2rem', 
        paddingBottom: '200px' 
      }}>
        {messages.map((msg, idx) => (
          <div 
            key={idx} 
            style={{ 
              display: 'flex', 
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start', 
              marginBottom: '1.5rem'
            }}
          >
            <div style={{
              display: 'flex', 
              gap: '1rem', 
              maxWidth: '80%', 
              flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
            }}>
              {/* Avatar */}
              <div style={{ 
                width: '40px', 
                height: '40px', 
                borderRadius: '50%', 
                background: msg.role === 'user' 
                  ? 'linear-gradient(135deg, #3b82f6, #2563eb)' 
                  : 'white', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center', 
                border: msg.role === 'bot' ? '2px solid #dbeafe' : 'none',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
              }}>
                {msg.role === 'user' ? (
                  <User size={20} color="white" />
                ) : (
                  <Bot size={20} color="#2563eb" />
                )}
              </div>
              
              {/* Message */}
              <div style={{ 
                background: msg.role === 'user' 
                  ? 'linear-gradient(135deg, #3b82f6, #2563eb)' 
                  : 'white', 
                color: msg.role === 'user' ? 'white' : '#1f2937', 
                padding: '1rem 1.5rem', 
                borderRadius: '24px', 
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)', 
                fontSize: '0.9375rem', 
                lineHeight: '1.6',
                whiteSpace: 'pre-wrap'
              }}>
                {msg.role === 'assistant' ? (
                  <ReactMarkdown 
                    components={{
                      p: ({node, ...props}) => <p style={{margin: '0 0 0.5rem 0'}} {...props} />,
                      h1: ({node, ...props}) => <h3 style={{margin: '0 0 0.5rem 0', fontSize: '1rem', fontWeight: 'bold'}} {...props} />,
                      h2: ({node, ...props}) => <h3 style={{margin: '0 0 0.5rem 0', fontSize: '0.95rem', fontWeight: 'bold'}} {...props} />,
                      h3: ({node, ...props}) => <h3 style={{margin: '0 0 0.5rem 0', fontSize: '0.9rem', fontWeight: 'bold'}} {...props} />,
                      ul: ({node, ...props}) => <ul style={{margin: '0 0 0.5rem 0', paddingLeft: '1.25rem'}} {...props} />,
                      ol: ({node, ...props}) => <ol style={{margin: '0 0 0.5rem 0', paddingLeft: '1.25rem'}} {...props} />,
                      li: ({node, ...props}) => <li style={{marginBottom: '0.25rem'}} {...props} />,
                      code: ({node, ...props}) => <code style={{background: 'rgba(0,0,0,0.1)', padding: '0.125rem 0.25rem', borderRadius: '4px', fontSize: '0.875rem'}} {...props} />,
                      blockquote: ({node, ...props}) => <blockquote style={{margin: '0.5rem 0', paddingLeft: '1rem', borderLeft: '3px solid rgba(59,130,246,0.3)'}} {...props} />
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div style={{ 
            display: 'flex', 
            gap: '1rem', 
            maxWidth: '80%' 
          }}>
            <div style={{ 
              width: '40px', 
              height: '40px', 
              borderRadius: '50%', 
              background: 'white', 
              border: '2px solid #dbeafe', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
            }}>
              <Bot size={20} color="#2563eb" />
            </div>
            <div style={{ 
              background: 'white', 
              padding: '1rem 1.5rem', 
              borderRadius: '24px', 
              boxShadow: '0 4px 12px rgba(0,0,0,0.1)' 
            }}>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <div style={{ 
                  width: '8px', 
                  height: '8px', 
                  background: '#3b82f6', 
                  borderRadius: '50%' 
                }}></div>
                <div style={{ 
                  width: '8px', 
                  height: '8px', 
                  background: '#3b82f6', 
                  borderRadius: '50%' 
                }}></div>
                <div style={{ 
                  width: '8px', 
                  height: '8px', 
                  background: '#3b82f6', 
                  borderRadius: '50%' 
                }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div style={{ 
        position: 'fixed', 
        bottom: 0, 
        left: 0, 
        right: 0, 
        background: 'rgba(255,255,255,0.95)', 
        padding: '1.5rem', 
        boxShadow: '0 -4px 20px rgba(0,0,0,0.1)',
        backdropFilter: 'blur(10px)'
      }}>
        <div style={{ 
          maxWidth: '1200px', 
          margin: '0 auto', 
          display: 'flex', 
          gap: '1rem' 
        }}>
          <textarea 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Digite sua pergunta sobre a SEONN..." 
            disabled={isLoading} 
            style={{ 
              flex: 1, 
              padding: '1rem 1.5rem', 
              borderRadius: '16px', 
              border: '2px solid #bfdbfe', 
              fontSize: '0.9375rem', 
              outline: 'none', 
              resize: 'none', 
              maxHeight: '120px',
              fontFamily: 'inherit'
            }} 
          />
          
          <button 
            onClick={sendMessage} 
            disabled={isLoading || !input.trim()} 
            style={{ 
              width: '64px', 
              height: '64px', 
              borderRadius: '16px', 
              border: 'none', 
              background: isLoading || !input.trim() 
                ? '#e5e7eb' 
                : 'linear-gradient(135deg, #3b82f6, #2563eb, #9333ea)', 
              cursor: 'pointer', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              boxShadow: isLoading || !input.trim() ? 'none' : '0 4px 12px rgba(59,130,246,0.4)',
              transition: 'all 0.2s'
            }}
          >
            <Send size={24} color={isLoading || !input.trim() ? '#9ca3af' : 'white'} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default Chat;