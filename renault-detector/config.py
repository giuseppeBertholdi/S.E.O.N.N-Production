# Configurações do projeto Renault Detector
# Arquivo de configuração centralizada

# Configurações de treinamento
TRAINING_CONFIG = {
    'epochs': 200,
    'imgsz': 1024,
    'batch': 16,
    'optimizer': 'AdamW',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'patience': 50,
    'label_smoothing': 0.1,
    'augment': True,
    'mosaic': 1.0,
    'mixup': 0.2,
    'fliplr': 0.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'erasing': 0.4,
    'crop_fraction': 1.0
}

# Configurações de aumento de dados
AUGMENTATION_CONFIG = {
    'flip_prob': 0.5,
    'rotation_range': 15,
    'brightness_range': 0.2,
    'contrast_range': 0.2,
    'blur_prob': 0.3,
    'noise_prob': 0.2,
    'crop_prob': 0.3
}

# Configurações de download
DOWNLOAD_CONFIG = {
    'renault_queries': [
        "Renault car front view",
        "Renault Clio car",
        "Renault Megane car",
        "Renault Captur SUV",
        "Renault Kadjar car",
        "Renault Twingo car",
        "Renault Logan car",
        "Renault Sandero car",
        "Renault Duster SUV",
        "Renault Koleos SUV"
    ],
    'other_brands_queries': [
        "Fiat car front view",
        "Volkswagen car front view", 
        "Toyota car front view",
        "Chevrolet car front view",
        "Ford car front view",
        "Honda car front view",
        "Nissan car front view",
        "Hyundai car front view",
        "Kia car front view",
        "Peugeot car front view"
    ],
    'images_per_query': 30,
    'sample_images_per_query': 10
}

# Configurações da API
API_CONFIG = {
    'host': '127.0.0.1',
    'port': 8000,
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'max_batch_size': 10
}

# Configurações de paths
PATHS = {
    'data_dir': 'data',
    'images_dir': 'data/images',
    'labels_dir': 'data/labels',
    'models_dir': 'models',
    'scripts_dir': 'scripts',
    'data_yaml': 'data/data.yaml',
    'model_path': 'models/best.pt',
    'metrics_path': 'models/metrics.json'
}

# Configurações de classes
CLASSES = {
    0: 'renault',
    1: 'other'
}

# Configurações de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'renault_detector.log'
}

# Configurações de GPU
GPU_CONFIG = {
    'use_gpu': True,
    'device': 'auto',
    'workers': 8
}

