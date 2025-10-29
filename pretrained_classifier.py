#!/usr/bin/env python3
"""
Classificador Gato vs Cachorro usando modelos pré-treinados do PyTorch
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import logging

class PretrainedCatDogClassifier:
    """Classificador usando modelos pré-treinados"""
    
    def __init__(self, model_name='resnet18'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Carregar modelo pré-treinado
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes: gato/cachorro
        elif model_name == 'efficientnet':
            try:
                # Tentar usar EfficientNet se disponível
                self.model = models.efficientnet_b0(pretrained=True)
                self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
            except:
                # Fallback para ResNet18
                self.model = models.resnet18(pretrained=True)
                self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        elif model_name == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transformações para imagens
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Mapeamento de classes do ImageNet para gato/cachorro
        self.imagenet_cats = [
            'tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat', 'Egyptian_cat',
            'lynx', 'leopard', 'snow_leopard', 'jaguar', 'lion', 'tiger', 'cheetah'
        ]
        
        self.imagenet_dogs = [
            'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu',
            'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback',
            'Afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
            'black-and-tan_coonhound', 'Walker_hound', 'English_foxhound',
            'redbone', 'borzoi', 'Irish_wolfhound', 'Italian_greyhound',
            'whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound',
            'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier',
            'American_Staffordshire_terrier', 'Bedlington_terrier', 'Border_terrier',
            'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
            'Norwich_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier',
            'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn',
            'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer',
            'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier',
            'silky_terrier', 'soft-coated_wheaten_terrier', 'West_Highland_white_terrier',
            'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever', 'golden_retriever',
            'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer',
            'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_spaniel',
            'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel',
            'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke',
            'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog',
            'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres',
            'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher',
            'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller',
            'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog',
            'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky',
            'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great_Pyrenees',
            'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon',
            'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle',
            'Mexican_hairless', 'timber_wolf', 'white_wolf', 'red_wolf', 'coyote',
            'dingo', 'dhole', 'African_hunting_dog', 'hyena', 'red_fox', 'kit_fox',
            'Arctic_fox', 'grey_fox', 'tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat',
            'Egyptian_cat', 'lion', 'tiger', 'jaguar', 'leopard', 'snow_leopard',
            'lynx', 'bobcat', 'cheetah'
        ]
        
        logging.info(f"✅ Modelo {model_name} carregado com sucesso!")
        logging.info(f"📱 Dispositivo: {self.device}")
    
    def predict_image(self, image_data):
        """Prediz se a imagem é gato ou cachorro"""
        try:
            # Processar imagem - sempre tratar como base64 se for string longa
            if isinstance(image_data, str):
                if len(image_data) > 100:
                    # É base64 - decodificar diretamente
                    image_data = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_data))
                else:
                    # É caminho de arquivo
                    image = Image.open(image_data)
            else:
                # É dados binários
                image = Image.open(BytesIO(image_data))
            
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Aplicar transformações
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predição
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Para modelos pré-treinados, vamos usar uma abordagem diferente
                # Vamos usar as classes do ImageNet para determinar se é gato ou cachorro
                _, predicted_class = torch.max(outputs, 1)
                
                # Se o modelo foi treinado especificamente para gato/cachorro
                if predicted_class.item() == 0:
                    cat_prob = probabilities[0][0].item()
                    dog_prob = probabilities[0][1].item()
                else:
                    cat_prob = probabilities[0][0].item()
                    dog_prob = probabilities[0][1].item()
            
            return {
                'cat_probability': cat_prob,
                'dog_probability': dog_prob,
                'prediction': 'Gato' if cat_prob > dog_prob else 'Cachorro',
                'confidence': max(cat_prob, dog_prob),
                'model_type': 'pretrained_' + self.model_name
            }
            
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            # Fallback para predição aleatória
            cat_prob = np.random.random()
            dog_prob = 1 - cat_prob
            
            return {
                'cat_probability': cat_prob,
                'dog_probability': dog_prob,
                'prediction': 'Gato' if cat_prob > dog_prob else 'Cachorro',
                'confidence': max(cat_prob, dog_prob),
                'model_type': 'fallback',
                'error': str(e)
            }

class ImageNetCatDogClassifier:
    """Classificador usando classes do ImageNet"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carregar modelo pré-treinado do ImageNet
        self.model = models.resnet18(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Carregar labels do ImageNet
        import urllib.request
        try:
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            urllib.request.urlretrieve(url, "imagenet_classes.txt")
            with open("imagenet_classes.txt", "r") as f:
                self.imagenet_classes = [line.strip() for line in f.readlines()]
        except:
            # Fallback para classes básicas
            self.imagenet_classes = [f"class_{i}" for i in range(1000)]
        
        # Transformações
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Classes de gatos e cachorros no ImageNet
        cat_keywords = ['tiger cat', 'persian cat', 'siamese cat', 'egyptian cat', 'lynx', 'leopard', 'jaguar', 'lion', 'tiger']
        dog_keywords = ['maltese dog', 'bulldog', 'chihuahua', 'terrier', 'retriever', 'spaniel', 'shepherd', 'poodle', 'hound']
        
        self.cat_classes = [i for i, cls in enumerate(self.imagenet_classes) 
                          if any(cat_keyword in cls.lower() for cat_keyword in cat_keywords)]
        
        self.dog_classes = [i for i, cls in enumerate(self.imagenet_classes) 
                           if any(dog_keyword in cls.lower() for dog_keyword in dog_keywords)]
        
        logging.info(f"✅ Classificador ImageNet carregado!")
        logging.info(f"🐱 Classes de gatos: {len(self.cat_classes)}")
        logging.info(f"🐶 Classes de cachorros: {len(self.dog_classes)}")
    
    def predict_image(self, image_data):
        """Prediz usando classes do ImageNet"""
        try:
            # Processar imagem
            if isinstance(image_data, str):
                if image_data.startswith('http://') or image_data.startswith('https://'):
                    # É uma URL - baixar imagem
                    import requests
                    response = requests.get(image_data, timeout=10)
                    image = Image.open(BytesIO(response.content))
                elif len(image_data) > 100:
                    # É base64 - decodificar diretamente
                    image_data = base64.b64decode(image_data)
                    image = Image.open(BytesIO(image_data))
                else:
                    # É caminho de arquivo
                    image = Image.open(image_data)
            else:
                # É dados binários
                image = Image.open(BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Aplicar transformações
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predição
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Pegar as top 5 predições
                top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)
                
                # Verificar nas top predições
                cat_score = 0
                dog_score = 0
                
                for idx, prob in zip(top5_indices[0], top5_probs[0]):
                    class_name = self.imagenet_classes[idx.item()].lower()
                    prob_value = prob.item()
                    
                    # Verificar se é um gato
                    if any(cat_word in class_name for cat_word in ['cat', 'tiger', 'lion', 'leopard', 'jaguar', 'lynx', 'persian', 'siamese', 'egyptian']):
                        cat_score += prob_value
                    # Verificar se é um cachorro
                    elif any(dog_word in class_name for dog_word in ['dog', 'hound', 'terrier', 'retriever', 'spaniel', 'shepherd', 'mastiff', 'bulldog', 'poodle', 'maltese', 'chihuahua']):
                        dog_score += prob_value
                
                # Se não encontrou nas top 5, verificar todas
                if cat_score == 0 and dog_score == 0:
                    for idx in range(len(self.imagenet_classes)):
                        class_name = self.imagenet_classes[idx].lower()
                        prob_value = probabilities[0][idx].item()
                        
                        if any(cat_word in class_name for cat_word in ['cat', 'tiger', 'lion', 'leopard', 'jaguar', 'lynx', 'persian', 'siamese', 'egyptian']):
                            cat_score += prob_value
                        elif any(dog_word in class_name for dog_word in ['dog', 'hound', 'terrier', 'retriever', 'spaniel', 'shepherd', 'mastiff', 'bulldog', 'poodle', 'maltese', 'chihuahua']):
                            dog_score += prob_value
                
                # Normalizar
                total_score = cat_score + dog_score
                if total_score > 0:
                    cat_prob = cat_score / total_score
                    dog_prob = dog_score / total_score
                else:
                    cat_prob = 0.5
                    dog_prob = 0.5
            
            return {
                'cat_probability': cat_prob,
                'dog_probability': dog_prob,
                'prediction': 'Gato' if cat_prob > dog_prob else 'Cachorro',
                'confidence': max(cat_prob, dog_prob),
                'model_type': 'imagenet_resnet18'
            }
            
        except Exception as e:
            logging.error(f"Erro na predição ImageNet: {e}")
            return {
                'cat_probability': 0.5,
                'dog_probability': 0.5,
                'prediction': 'Indefinido',
                'confidence': 0.5,
                'model_type': 'error',
                'error': str(e)
            }

def test_models():
    """Testa diferentes modelos"""
    print("🧪 Testando diferentes modelos de classificação...")
    
    # Criar imagem de teste
    test_image = Image.new('RGB', (224, 224), color='red')
    
    models_to_test = [
        ('ImageNet ResNet18', ImageNetCatDogClassifier()),
        ('ResNet18 Custom', PretrainedCatDogClassifier('resnet18')),
        ('MobileNet Custom', PretrainedCatDogClassifier('mobilenet'))
    ]
    
    for name, model in models_to_test:
        print(f"\n📊 Testando {name}:")
        try:
            result = model.predict_image(test_image)
            print(f"   🎯 Predição: {result['prediction']}")
            print(f"   🎲 Confiança: {result['confidence']:.3f}")
            print(f"   🐱 Prob. Gato: {result['cat_probability']:.3f}")
            print(f"   🐶 Prob. Cachorro: {result['dog_probability']:.3f}")
            print(f"   🔧 Modelo: {result['model_type']}")
        except Exception as e:
            print(f"   ❌ Erro: {e}")

if __name__ == "__main__":
    test_models()
