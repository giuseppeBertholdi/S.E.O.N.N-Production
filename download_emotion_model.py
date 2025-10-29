"""
Script para baixar o modelo de reconhecimento de emoções FER+
"""
import urllib.request
import os

def download_emotion_model():
    """Baixa o modelo emotion-ferplus-8.onnx"""
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    model_filename = "emotion-ferplus-8.onnx"
    
    print(f"📥 Baixando modelo de reconhecimento de emoções...")
    print(f"URL: {model_url}")
    
    try:
        urllib.request.urlretrieve(model_url, model_filename)
        print(f"✅ Modelo baixado com sucesso: {model_filename}")
        print(f"📊 Tamanho do arquivo: {os.path.getsize(model_filename) / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        print(f"❌ Erro ao baixar modelo: {e}")
        return False

if __name__ == "__main__":
    if os.path.exists("emotion-ferplus-8.onnx"):
        print("⚠️ O arquivo emotion-ferplus-8.onnx já existe!")
        resposta = input("Deseja baixar novamente? (s/n): ")
        if resposta.lower() != 's':
            print("Operação cancelada.")
            exit(0)
    
    success = download_emotion_model()
    
    if success:
        print("\n🎉 Você pode agora executar analise.py!")
    else:
        print("\n❌ Falha ao baixar o modelo.")
        print("Você pode baixar manualmente de:")
        print("https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus")




