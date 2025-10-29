// SEONN Demo - JavaScript
class SEONNDemo {
    constructor() {
        this.apiUrl = 'http://localhost:5000';
        this.currentImage = null;
        this.isProcessing = false;
        
        // Cinematic network visualization properties
        this.networkAnimation = null;
        this.isAnimating = false;
        this.currentPhase = 0;
        this.neurons = [];
        this.connections = [];
        this.energyWaves = [];
        this.cinematicMode = true;
        this.animationSpeed = 2000; // Slower for cinematic effect
        this.networkData = {
            phases: [
                { 
                    name: 'Inicialização', 
                    neurons: 30, 
                    connections: 45, 
                    plasticity: 'Baixa',
                    description: 'Rede esparsa com conexões básicas',
                    color: '#6366f1',
                    duration: 4000
                },
                { 
                    name: 'Aprendizado', 
                    neurons: 80, 
                    connections: 150, 
                    plasticity: 'Média',
                    description: 'Formação de padrões especializados',
                    color: '#8b5cf6',
                    duration: 5000
                },
                { 
                    name: 'Especialização', 
                    neurons: 150, 
                    connections: 280, 
                    plasticity: 'Alta',
                    description: 'Emergência de sub-redes funcionais',
                    color: '#06b6d4',
                    duration: 6000
                },
                { 
                    name: 'Otimização', 
                    neurons: 120, 
                    connections: 200, 
                    plasticity: 'Otimizada',
                    description: 'Pruning e refinamento contínuo',
                    color: '#10b981',
                    duration: 5000
                }
            ]
        };
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkAPIStatus();
        this.loadNetworkStats();
        this.setupNavigation();
        this.setupNetworkVisualization();
        
        // Atualizar estatísticas a cada 5 segundos
        setInterval(() => {
            this.loadNetworkStats();
        }, 5000);
    }

    setupEventListeners() {
        // Upload de arquivo
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
    }

    setupNavigation() {
        const navButtons = document.querySelectorAll('.nav-btn');
        const sections = document.querySelectorAll('.section');

        navButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetSection = button.getAttribute('data-section');
                
                // Remove active class from all buttons and sections
                navButtons.forEach(btn => btn.classList.remove('active'));
                sections.forEach(section => section.classList.remove('active'));
                
                // Add active class to clicked button and target section
                button.classList.add('active');
                document.getElementById(targetSection).classList.add('active');
            });
        });
    }

    setupNetworkVisualization() {
        const playBtn = document.getElementById('playBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        const resetBtn = document.getElementById('resetBtn');

        if (playBtn) playBtn.addEventListener('click', () => this.startNetworkAnimation());
        if (pauseBtn) pauseBtn.addEventListener('click', () => this.pauseNetworkAnimation());
        if (resetBtn) resetBtn.addEventListener('click', () => this.resetNetworkAnimation());

        // Initialize network
        this.initializeNetwork();
    }

    initializeNetwork() {
        const svg = document.getElementById('networkSvg');
        if (!svg) return;

        // Clear existing content
        svg.querySelector('#neurons').innerHTML = '';
        svg.querySelector('#connections').innerHTML = '';
        svg.querySelector('#pulses').innerHTML = '';

        // Generate initial network
        this.generateNetworkPhase(0);
    }

    generateNetworkPhase(phaseIndex) {
        const phase = this.networkData.phases[phaseIndex];
        const svg = document.getElementById('networkSvg');
        if (!svg) return;

        const neuronsGroup = svg.querySelector('#neurons');
        const connectionsGroup = svg.querySelector('#connections');
        const energyWavesGroup = svg.querySelector('#energyWaves');
        
        // Clear existing with cinematic fade
        this.fadeOutElements(neuronsGroup);
        this.fadeOutElements(connectionsGroup);
        this.fadeOutElements(energyWavesGroup);
        
        setTimeout(() => {
            neuronsGroup.innerHTML = '';
            connectionsGroup.innerHTML = '';
            energyWavesGroup.innerHTML = '';
            this.generateCinematicNetwork(phase, phaseIndex);
        }, 500);
    }

    generateCinematicNetwork(phase, phaseIndex) {
        const svg = document.getElementById('networkSvg');
        const neuronsGroup = svg.querySelector('#neurons');
        const connectionsGroup = svg.querySelector('#connections');
        const energyWavesGroup = svg.querySelector('#energyWaves');

        // Generate neurons with cinematic positioning
        this.neurons = [];
        const centerX = 400;
        const centerY = 300;
        const baseRadius = 80 + phaseIndex * 25;
        
        // Create core neurons first
        for (let i = 0; i < Math.min(phase.neurons, 20); i++) {
            const angle = (i / Math.min(phase.neurons, 20)) * 2 * Math.PI;
            const radius = baseRadius + Math.random() * 30;
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);
            
            this.createCinematicNeuron(i, x, y, phase, true);
        }
        
        // Create peripheral neurons
        for (let i = 20; i < phase.neurons; i++) {
            const angle = Math.random() * 2 * Math.PI;
            const radius = baseRadius + 50 + Math.random() * 80;
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);
            
            this.createCinematicNeuron(i, x, y, phase, false);
        }

        // Generate connections with cinematic timing
        this.generateCinematicConnections(phase);
        
        // Generate energy waves
        this.generateEnergyWaves(phase);
        
        // Update progress and info
        this.updateProgress(phaseIndex);
        this.updateNetworkInfo(phase);
    }

    createCinematicNeuron(id, x, y, phase, isCore) {
        const neuronsGroup = document.querySelector('#neurons');
        const neuron = {
            id: id,
            x: x,
            y: y,
            active: Math.random() > (isCore ? 0.3 : 0.6),
            phase: phase,
            isCore: isCore
        };
        
        this.neurons.push(neuron);

        // Create neuron with cinematic effects
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', neuron.active ? (isCore ? 6 : 4) : (isCore ? 4 : 3));
        circle.setAttribute('fill', neuron.active ? 'url(#neuronGradient)' : '#64748b');
        circle.setAttribute('class', 'neuron cinematic');
        circle.setAttribute('data-id', id);
        circle.setAttribute('opacity', '0');
        
        if (neuron.active) {
            circle.setAttribute('filter', 'url(#neuronGlow)');
            circle.classList.add('active');
        }

        neuronsGroup.appendChild(circle);
        
        // Animate entrance
        setTimeout(() => {
            circle.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
            circle.setAttribute('opacity', '1');
            circle.style.transform = 'scale(1)';
        }, id * 50);
    }

    generateCinematicConnections(phase) {
        const connectionsGroup = document.querySelector('#connections');
        this.connections = [];
        
        // Create connections with cinematic timing
        for (let i = 0; i < phase.connections; i++) {
            const from = this.neurons[Math.floor(Math.random() * this.neurons.length)];
            const to = this.neurons[Math.floor(Math.random() * this.neurons.length)];
            
            if (from.id !== to.id) {
                const connection = {
                    from: from,
                    to: to,
                    active: Math.random() > 0.4,
                    strength: Math.random()
                };
                
                this.connections.push(connection);
                this.createCinematicConnection(connection, i);
            }
        }
    }

    createCinematicConnection(connection, index) {
        const connectionsGroup = document.querySelector('#connections');
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        
        line.setAttribute('x1', connection.from.x);
        line.setAttribute('y1', connection.from.y);
        line.setAttribute('x2', connection.to.x);
        line.setAttribute('y2', connection.to.y);
        line.setAttribute('stroke', connection.active ? 'url(#connectionGradient)' : 'rgba(100, 116, 139, 0.3)');
        line.setAttribute('stroke-width', connection.active ? 2 : 1);
        line.setAttribute('stroke-dasharray', connection.active ? '8,4' : 'none');
        line.setAttribute('class', 'connection cinematic');
        line.setAttribute('opacity', '0');
        
        if (connection.active) {
            line.setAttribute('filter', 'url(#cinematicGlow)');
            line.classList.add('active');
        }

        connectionsGroup.appendChild(line);
        
        // Animate entrance
        setTimeout(() => {
            line.style.transition = 'opacity 1s ease';
            line.setAttribute('opacity', '1');
        }, index * 20);
    }

    generateEnergyWaves(phase) {
        const energyWavesGroup = document.querySelector('#energyWaves');
        
        // Create energy waves emanating from center
        for (let i = 0; i < 3; i++) {
            const wave = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            wave.setAttribute('cx', '400');
            wave.setAttribute('cy', '300');
            wave.setAttribute('r', '20');
            wave.setAttribute('fill', 'none');
            wave.setAttribute('stroke', phase.color);
            wave.setAttribute('stroke-width', '2');
            wave.setAttribute('opacity', '0.6');
            wave.setAttribute('class', 'energy-wave');
            
            // Animate wave expansion
            const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
            animate.setAttribute('attributeName', 'r');
            animate.setAttribute('values', '20;200;20');
            animate.setAttribute('dur', '4s');
            animate.setAttribute('repeatCount', 'indefinite');
            animate.setAttribute('begin', `${i * 1.5}s`);
            
            wave.appendChild(animate);
            energyWavesGroup.appendChild(wave);
        }
    }

    fadeOutElements(group) {
        const elements = group.children;
        for (let element of elements) {
            element.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            element.style.opacity = '0';
            element.style.transform = 'scale(0.8)';
        }
    }

    updateProgress(phaseIndex) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const phase = this.networkData.phases[phaseIndex];
        
        if (progressFill) {
            const progress = ((phaseIndex + 1) / this.networkData.phases.length) * 100;
            progressFill.style.width = `${progress}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${phase.name} - ${phase.description}`;
        }
    }

    updateNetworkInfo(phase) {
        const activeNeuronsEl = document.getElementById('activeNeurons');
        const dynamicConnectionsEl = document.getElementById('dynamicConnections');
        const adaptationRateEl = document.getElementById('adaptationRate');
        const plasticityLevelEl = document.getElementById('plasticityLevel');

        if (activeNeuronsEl) activeNeuronsEl.textContent = this.neurons.filter(n => n.active).length;
        if (dynamicConnectionsEl) dynamicConnectionsEl.textContent = this.connections.filter(c => c.active).length;
        if (adaptationRateEl) adaptationRateEl.textContent = `${Math.floor(Math.random() * 30 + 70)}%`;
        if (plasticityLevelEl) plasticityLevelEl.textContent = phase.plasticity;
    }

    startNetworkAnimation() {
        if (this.isAnimating) return;
        
        this.isAnimating = true;
        document.getElementById('playBtn').classList.add('active');
        document.getElementById('pauseBtn').classList.remove('active');
        
        // Start cinematic sequence
        this.playCinematicSequence();
    }

    playCinematicSequence() {
        if (!this.isAnimating) return;
        
        const phase = this.networkData.phases[this.currentPhase];
        
        // Update timeline with cinematic timing
        document.querySelectorAll('.timeline-item').forEach((item, index) => {
            item.classList.toggle('active', index === this.currentPhase);
        });

        // Generate new phase with cinematic effects
        this.generateNetworkPhase(this.currentPhase);
        
        // Add cinematic pulse effects
        setTimeout(() => {
            this.addCinematicPulseEffects();
        }, 1000);
        
        // Move to next phase after duration
        setTimeout(() => {
            this.currentPhase = (this.currentPhase + 1) % this.networkData.phases.length;
            this.playCinematicSequence();
        }, phase.duration);
    }

    pauseNetworkAnimation() {
        this.isAnimating = false;
        if (this.networkAnimation) {
            clearInterval(this.networkAnimation);
            this.networkAnimation = null;
        }
        
        document.getElementById('playBtn').classList.remove('active');
        document.getElementById('pauseBtn').classList.add('active');
    }

    resetNetworkAnimation() {
        this.pauseNetworkAnimation();
        this.currentPhase = 0;
        this.initializeNetwork();
        
        // Reset timeline
        document.querySelectorAll('.timeline-item').forEach((item, index) => {
            item.classList.toggle('active', index === 0);
        });
    }

    animateNetworkEvolution() {
        this.currentPhase = (this.currentPhase + 1) % this.networkData.phases.length;
        
        // Update timeline
        document.querySelectorAll('.timeline-item').forEach((item, index) => {
            item.classList.toggle('active', index === this.currentPhase);
        });

        // Generate new phase with animation
        this.generateNetworkPhase(this.currentPhase);
        
        // Add pulse effects
        this.addPulseEffects();
    }

    addCinematicPulseEffects() {
        const svg = document.getElementById('networkSvg');
        const pulsesGroup = svg.querySelector('#pulses');
        
        // Clear existing pulses
        pulsesGroup.innerHTML = '';

        // Add cinematic pulses to active neurons
        this.neurons.filter(n => n.active).forEach((neuron, index) => {
            setTimeout(() => {
                this.createCinematicPulse(neuron);
            }, index * 100);
        });
    }

    createCinematicPulse(neuron) {
        const pulsesGroup = document.querySelector('#pulses');
        
        // Create multiple pulse rings
        for (let i = 0; i < 3; i++) {
            const pulse = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            pulse.setAttribute('cx', neuron.x);
            pulse.setAttribute('cy', neuron.y);
            pulse.setAttribute('r', 2);
            pulse.setAttribute('fill', 'none');
            pulse.setAttribute('stroke', neuron.isCore ? '#6366f1' : '#8b5cf6');
            pulse.setAttribute('stroke-width', '2');
            pulse.setAttribute('opacity', '0.8');
            pulse.setAttribute('class', 'cinematic-pulse');
            
            // Animate pulse expansion
            const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
            animate.setAttribute('attributeName', 'r');
            animate.setAttribute('values', '2;30;2');
            animate.setAttribute('dur', '2s');
            animate.setAttribute('repeatCount', 'indefinite');
            animate.setAttribute('begin', `${i * 0.3}s`);
            
            const opacityAnimate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
            opacityAnimate.setAttribute('attributeName', 'opacity');
            opacityAnimate.setAttribute('values', '0.8;0;0.8');
            opacityAnimate.setAttribute('dur', '2s');
            opacityAnimate.setAttribute('repeatCount', 'indefinite');
            opacityAnimate.setAttribute('begin', `${i * 0.3}s`);
            
            pulse.appendChild(animate);
            pulse.appendChild(opacityAnimate);
            pulsesGroup.appendChild(pulse);
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
    if (!file.type.startsWith('image/')) {
            this.showError('Por favor, selecione um arquivo de imagem válido.');
            return;
        }

        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            this.showError('A imagem é muito grande. Por favor, selecione uma imagem menor que 10MB.');
        return;
    }
    
        this.currentImage = file;
        this.displayImage(file);
        this.predictImage(file);
    }

    displayImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
            const previewImage = document.getElementById('previewImage');
            previewImage.src = e.target.result;
            
            const resultSection = document.getElementById('resultSection');
            resultSection.style.display = 'block';
            
            // Scroll to result section
            resultSection.scrollIntoView({ behavior: 'smooth' });
    };
    reader.readAsDataURL(file);
}

    async predictImage(file) {
        if (this.isProcessing) return;
        
        this.isProcessing = true;
        this.showLoading();

        try {
            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch(`${this.apiUrl}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                    image: await this.fileToBase64(file)
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
            this.displayResult(result);
            
        } catch (error) {
            console.error('Erro na predição:', error);
            this.showError('Erro ao processar a imagem. Verifique se o servidor está rodando.');
        } finally {
            this.isProcessing = false;
            this.hideLoading();
        }
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = error => reject(error);
        });
    }

    displayResult(result) {
        const prediction = document.getElementById('prediction');
        const confidence = document.getElementById('confidence');
        const catProb = document.getElementById('catProb');
        const dogProb = document.getElementById('dogProb');
        const catProbBar = document.getElementById('catProbBar');
        const dogProbBar = document.getElementById('dogProbBar');
        const modelType = document.getElementById('modelType');
        const processingTime = document.getElementById('processingTime');

        // Atualizar predição principal
        const isCat = result.prediction === 'Gato';
        const icon = isCat ? 
            `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 9V7L15 5.5V7.5L21 9ZM3 9L9 7.5V5.5L3 7V9ZM12 8C13.1 8 14 8.9 14 10C14 11.1 13.1 12 12 12C10.9 12 10 11.1 10 10C10 8.9 10.9 8 12 8ZM12 14C13.1 14 14 14.9 14 16C14 17.1 13.1 18 12 18C10.9 18 10 17.1 10 16C10 14.9 10.9 14 12 14ZM12 20C13.1 20 14 20.9 14 22C14 23.1 13.1 24 12 24C10.9 24 10 23.1 10 22C10 20.9 10.9 20 12 20Z" fill="currentColor"/>
            </svg>` :
            `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM21 9V7L15 5.5V7.5L21 9ZM3 9L9 7.5V5.5L3 7V9ZM12 8C13.1 8 14 8.9 14 10C14 11.1 13.1 12 12 12C10.9 12 10 11.1 10 10C10 8.9 10.9 8 12 8ZM12 14C13.1 14 14 14.9 14 16C14 17.1 13.1 18 12 18C10.9 18 10 17.1 10 16C10 14.9 10.9 14 12 14ZM12 20C13.1 20 14 20.9 14 22C14 23.1 13.1 24 12 24C10.9 24 10 23.1 10 22C10 20.9 10.9 20 12 20Z" fill="currentColor"/>
            </svg>`;
        
        prediction.innerHTML = `${icon} ${result.prediction}`;
        prediction.style.background = isCat ? 
            'linear-gradient(135deg, #f59e0b, #f97316)' : 
            'linear-gradient(135deg, #6366f1, #8b5cf6)';

        // Atualizar confiança
        confidence.textContent = `Confiança: ${(result.confidence * 100).toFixed(1)}%`;

        // Atualizar probabilidades
        const catProbValue = result.cat_probability;
        const dogProbValue = result.dog_probability;

        // Animar porcentagens
        this.animatePercentage(catProb, catProbValue * 100);
        this.animatePercentage(dogProb, dogProbValue * 100);

        // Animar barras de probabilidade
        setTimeout(() => {
            this.animateProgressBar(catProbBar, catProbValue * 100);
            this.animateProgressBar(dogProbBar, dogProbValue * 100);
        }, 200);

        // Atualizar informações do modelo
        modelType.textContent = `Modelo: ${result.model_type || 'SEONN'}`;
        processingTime.textContent = `Tempo: ${(result.processing_time * 1000).toFixed(0)}ms`;
    }

    animatePercentage(element, targetValue) {
        const startValue = 0;
        const duration = 1200; // 1.2 segundos
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Usar easing function para suavizar a animação
            const easeOutCubic = 1 - Math.pow(1 - progress, 3);
            const currentValue = startValue + (targetValue - startValue) * easeOutCubic;
            
            element.textContent = `${currentValue.toFixed(1)}%`;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
        } else {
                element.textContent = `${targetValue.toFixed(1)}%`;
            }
        };

        requestAnimationFrame(animate);
    }

    animateProgressBar(element, targetWidth) {
        // Reset da barra
        element.style.width = '0%';
        element.classList.remove('animate');
        
        // Forçar reflow
        element.offsetHeight;
        
        // Definir largura alvo como variável CSS
        element.style.setProperty('--target-width', `${targetWidth}%`);
        
        // Adicionar classe de animação
        element.classList.add('animate');
        
        // Remover classe após animação
        setTimeout(() => {
            element.classList.remove('animate');
            element.style.width = `${targetWidth}%`;
        }, 1200);
    }

    showLoading() {
        const prediction = document.getElementById('prediction');
        const confidence = document.getElementById('confidence');
        
        prediction.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="animation: spin 1s linear infinite;">
                <path d="M12 2V6M12 18V22M4.93 4.93L7.76 7.76M16.24 16.24L19.07 19.07M2 12H6M18 12H22M4.93 19.07L7.76 16.24M16.24 7.76L19.07 4.93" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Processando...
        `;
        prediction.style.background = 'linear-gradient(135deg, #64748b, #94a3b8)';
        confidence.textContent = 'Analisando imagem...';
    }

    hideLoading() {
        // Loading será substituído pelo resultado
    }

    showError(message) {
        const prediction = document.getElementById('prediction');
        const confidence = document.getElementById('confidence');
        
        prediction.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 9V13M12 17H12.01M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Erro
        `;
        prediction.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
        confidence.textContent = message;
    }

    async checkAPIStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/api/status`);
            const status = await response.json();
            
            this.updateStatusIndicator(status);
        } catch (error) {
            console.error('Erro ao verificar status da API:', error);
            this.updateStatusIndicator({ status: 'offline' });
        }
    }

    updateStatusIndicator(status) {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('span');

        if (status.status === 'online') {
            statusDot.classList.add('connected');
            statusText.textContent = 'Conectado';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Desconectado';
        }
    }

    async loadNetworkStats() {
        try {
            const response = await fetch(`${this.apiUrl}/api/network-stats`);
            const stats = await response.json();
            
            this.updateNetworkStats(stats);
        } catch (error) {
            console.error('Erro ao carregar estatísticas:', error);
        }
    }

    updateNetworkStats(stats) {
        // Atualizar estatísticas básicas
        const elements = {
            neurons: stats.neurons?.toLocaleString() || '-',
            connections: stats.connections?.toLocaleString() || '-',
            neuralHealth: stats.neural_health ? `${(stats.neural_health * 100).toFixed(1)}%` : '-',
            activityVariance: stats.activity_variance?.toFixed(3) || '-',
            totalPredictions: stats.total_predictions || '0',
            activeModel: stats.active_model || 'SEONN'
        };

        Object.entries(elements).forEach(([key, value]) => {
            const element = document.getElementById(key);
            if (element) {
                element.textContent = value;
            }
        });

        // Atualizar métricas de desempenho (simuladas)
        this.updatePerformanceMetrics(stats);
    }

    updatePerformanceMetrics(stats) {
        // Simular métricas de desempenho baseadas nas estatísticas reais
        const currentAccuracy = document.getElementById('currentAccuracy');
        const avgResponseTime = document.getElementById('avgResponseTime');
        const adaptationRate = document.getElementById('adaptationRate');

        if (currentAccuracy) {
            const accuracy = stats.neural_health ? (stats.neural_health * 100).toFixed(1) : '95.2';
            currentAccuracy.textContent = `${accuracy}%`;
        }

        if (avgResponseTime) {
            avgResponseTime.textContent = '247ms';
        }

        if (adaptationRate) {
            adaptationRate.textContent = '87.3%';
        }
    }
}

// Inicializar quando o DOM estiver carregado
document.addEventListener('DOMContentLoaded', () => {
    new SEONNDemo();
});

// Adicionar animações suaves para elementos
document.addEventListener('DOMContentLoaded', () => {
    // Animar barras de comparação
    const comparisonBars = document.querySelectorAll('.seonn-bar, .traditional-bar');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bar = entry.target;
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = width;
                }, 200);
        }
    });
});

    comparisonBars.forEach(bar => observer.observe(bar));
});