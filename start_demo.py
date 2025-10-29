#!/usr/bin/env python3
"""
Script para iniciar a demonstração completa da SEONN
"""

import subprocess
import sys
import time
import webbrowser
import os
from threading import Thread

def start_backend():
    """Inicia o backend Flask"""
    print("🚀 Iniciando backend SEONN...")
    try:
        subprocess.run([sys.executable, "backend_api.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao iniciar backend: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Backend interrompido")

def start_frontend():
    """Inicia servidor frontend"""
    print("🌐 Iniciando servidor frontend...")
    try:
        os.chdir("web_demo")
        subprocess.run([sys.executable, "-m", "http.server", "8001"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao iniciar frontend: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Frontend interrompido")

def main():
    print("🧬 SEONN - Self-Evolving Optimized Neural Network")
    print("=" * 50)
    
    # Verificar se o modelo existe
    if not os.path.exists("cat_dog_seonn.pth"):
        print("⚠️ Modelo não encontrado. Treinando novo modelo...")
        try:
            subprocess.run([sys.executable, "train_model.py"], check=True)
            print("✅ Modelo treinado com sucesso!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao treinar modelo: {e}")
            return
    
    print("📋 Instruções:")
    print("1. O backend será iniciado na porta 5000")
    print("2. O frontend será iniciado na porta 8001")
    print("3. Acesse: http://localhost:8001")
    print("4. Pressione Ctrl+C para parar")
    print()
    
    # Aguardar um pouco antes de abrir o navegador
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8001")
    
    # Iniciar thread para abrir navegador
    browser_thread = Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Iniciar backend em thread separada
        backend_thread = Thread(target=start_backend)
        backend_thread.daemon = True
        backend_thread.start()
        
        # Aguardar um pouco para o backend inicializar
        time.sleep(2)
        
        # Iniciar frontend (bloqueante)
        start_frontend()
        
    except KeyboardInterrupt:
        print("\n🛑 Demonstração interrompida")
        print("👋 Obrigado por usar a SEONN!")

if __name__ == "__main__":
    main()

