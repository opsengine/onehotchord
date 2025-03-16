// Audio processing and model inference functionality
export class ChordRecognizer {
    constructor(CQTransform) {
        // Store the CQTransform class
        this.CQTransform = CQTransform;
        
        // Audio context and analyzer setup
        this.audioContext = null;
        this.analyzer = null;
        this.cqt = null;
        this.dataArray = null;
        this.isListening = false;
        this.updateInterval = null;
        
        // ONNX model setup
        this.session = null;
        this.modelLoaded = false;
        
        // Note names and chord types for display
        this.noteNames = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"];
        this.chordNames = ["dim", "min", "maj", "7", "maj7", "min7"];
        
        // Callbacks
        this.onModelLoaded = null;
        this.onModelError = null;
        this.onAudioError = null;
        this.onUpdate = null;
        this.onStatusChange = null;
    }
    
    // Load ONNX model
    async loadModel() {
        try {
            this.session = new onnx.InferenceSession();
            await this.session.loadModel("one_hot_chord.onnx");
            console.log("Model loaded successfully");
            this.modelLoaded = true;
            
            if (this.onModelLoaded) {
                this.onModelLoaded();
            }
        } catch (e) {
            console.error("Failed to load ONNX model:", e);
            
            if (this.onModelError) {
                this.onModelError(e);
            }
        }
    }
    
    // Initialize audio
    async initAudio() {
        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 22050 // Match the model's expected sample rate
            });
            
            // Get user media
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const source = this.audioContext.createMediaStreamSource(stream);
            
            // Create analyzer
            this.analyzer = this.audioContext.createAnalyser();
            this.analyzer.fftSize = 8192;
            this.analyzer.smoothingTimeConstant = 0.85;
            
            // Connect source to analyzer
            source.connect(this.analyzer);
            
            // Create CQT transform using the provided CQTransform class
            this.cqt = new this.CQTransform({
                minFreq: 65.41,  // C2
                maxFreq: 2093.0, // C7
                binsPerOctave: 12,
                sampleRate: this.audioContext.sampleRate,
                fftSize: this.analyzer.fftSize
            });
            
            // Create data array for frequency data
            this.dataArray = new Float32Array(this.analyzer.frequencyBinCount);
            
            // Start listening
            this.startListening();
            
            return true;
        } catch (error) {
            console.error('Error initializing audio:', error);
            
            if (this.onAudioError) {
                this.onAudioError(error);
            }
            
            return false;
        }
    }
    
    // Start listening and updating
    startListening() {
        this.isListening = true;
        
        if (this.onStatusChange) {
            this.onStatusChange('listening');
        }
        
        // Update every 100ms
        this.updateInterval = setInterval(() => this.updateAudioData(), 100);
    }
    
    // Stop listening
    stopListening() {
        this.isListening = false;
        
        if (this.onStatusChange) {
            this.onStatusChange('stopped');
        }
        
        clearInterval(this.updateInterval);
    }
    
    // Calculate volume from FFT data
    getVolumeFromFFT(frequencyData) {
        let totalEnergy = 0;
        
        for (let i = 0; i < frequencyData.length; i++) {
            // Convert from dB to magnitude
            const magnitude = Math.pow(10, frequencyData[i] / 20);
            totalEnergy += magnitude * magnitude;
        }
        
        // Root mean square (RMS)
        const rms = Math.sqrt(totalEnergy / frequencyData.length);

        // Convert to a 0-1 scale with some scaling to make it more visible
        return Math.min(1, rms * 5000);
    }
    
    // Calculate octave centroid
    calculateOctaveCentroid(cqtData, note) {
        let sum = 0;
        
        for (let oct = 0; oct < 5; oct++) {
            const idx = note + (oct * 12);
            if (idx < cqtData.length) {
                sum += cqtData[idx] * (oct + 2);
            }
        }
        
        return sum / 7;
    }
    
    // Implement softmax function
    softmax(logits) {
        // Find the maximum value to avoid numerical instability
        const maxLogit = Math.max(...logits);
        
        // Subtract max from each value and exponentiate
        const expValues = logits.map(logit => Math.exp(logit - maxLogit));
        
        // Calculate the sum of all exponentiated values
        const sumExp = expValues.reduce((acc, val) => acc + val, 0);
        
        // Normalize by the sum to get probabilities
        return expValues.map(exp => exp / sumExp);
    }

    // Implement sigmoid function for presence output
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    // Run model inference on the CQT data
    async runModelInference(chroma, octaveCentroid) {
        if (!this.modelLoaded) return null;
    
        try {
            // Prepare the input features (24 values: 12 for chroma, 12 for octave centroid)
            const features = new Float32Array(24);
            features.set(chroma, 0);
            features.set(octaveCentroid, 12);

            // Prepare the input tensor
            const inputTensor = new onnx.Tensor(features, 'float32', [1, 24]);
            
            // Run inference
            const outputMap = await this.session.run([inputTensor]);
            
            const rootOutput = outputMap.get("root_output");
            const chordOutput = outputMap.get("chord_output");
            const presenceOutput = outputMap.get("presence_output");
            
            if (!rootOutput || !chordOutput || !presenceOutput) {
                throw new Error("Could not get model outputs");
            }
            
            // Get the raw logits from the tensors
            const rootLogits = rootOutput.data;
            const chordLogits = chordOutput.data;
            const presenceLogit = presenceOutput.data[0];  // Single value
            
            // Apply softmax to convert logits to probabilities
            const rootProbs = this.softmax(Array.from(rootLogits));
            const chordProbs = this.softmax(Array.from(chordLogits));
            const presenceProb = this.sigmoid(presenceLogit);  // Apply sigmoid for presence
            
            // Find the predicted root note (highest probability)
            let maxRootProb = -Infinity;
            let predictedRoot = -1;
            for (let i = 0; i < rootProbs.length; i++) {
                if (rootProbs[i] > maxRootProb) {
                    maxRootProb = rootProbs[i];
                    predictedRoot = i;
                }
            }
            
            // Find the predicted chord type (highest probability)
            let maxChordProb = -Infinity;
            let predictedChord = -1;
            for (let i = 0; i < chordProbs.length; i++) {
                if (chordProbs[i] > maxChordProb) {
                    maxChordProb = chordProbs[i];
                    predictedChord = i;
                }
            }
            
            return {
                root: predictedRoot,
                rootProb: maxRootProb,
                chord: predictedChord,
                chordProb: maxChordProb,
                presence: presenceProb
            };
            
        } catch (e) {
            console.error("Error running inference:", e);
            return null;
        }
    }
    
    // Update audio data and visualization
    async updateAudioData() {
        // Get frequency data
        this.analyzer.getFloatFrequencyData(this.dataArray);
        
        // Calculate volume
        const volume = this.getVolumeFromFFT(this.dataArray);
        
        // Compute CQT
        const cqtData = this.cqt.compute(this.dataArray);
        
        // Calculate chroma vector (sum energy across octaves for each note)
        const chroma = new Float32Array(12);

        const minOctave = 2; // C2 is our lowest note
        for (let note = 0; note < 12; note++) {
            for (let octave = minOctave; octave <= minOctave + 5; octave++) {
                const binIndex = (octave - minOctave) * 12 + note;
                if (binIndex < cqtData.length) {
                    chroma[note] += cqtData[binIndex];
                }
            }
        }

        // Calculate octave centroid
        const octaveCentroid = new Float32Array(12);
        for (let note = 0; note < 12; note++) {
            let sum = 0;
            for (let oct = 0; oct < 5; oct++) {
                const idx = note + (oct * 12);
                if (idx < cqtData.length) {
                    sum += cqtData[idx] * (oct + 2);
                }
            }
            octaveCentroid[note] = sum / (7 * chroma[note]);
        }

        // Normalize chroma for visualization
        const maxChroma = Math.max(...chroma);
        const normalizedChroma = new Float32Array(12);
        
        if (maxChroma > 0) {
            for (let i = 0; i < 12; i++) {
                normalizedChroma[i] = chroma[i] / maxChroma;
            }
        }
        
        // Run model inference
        const prediction = await this.runModelInference(normalizedChroma, octaveCentroid);
        
        // Determine chord display text
        let chordText = "play a chord...";
        if (prediction && prediction.presence > 0.5 && volume > 0.1) {
            const rootNote = this.noteNames[prediction.root];
            const chordTypeName = this.chordNames[prediction.chord];
            chordText = rootNote + " " + chordTypeName;
        } else if (prediction && prediction.presence > 0.25 && prediction.presence < 0.75 && volume > 0.1) {
            chordText = "???";
        }
        
        // Call update callback with all the data
        if (this.onUpdate) {
            this.onUpdate({
                volume,
                normalizedChroma,
                prediction,
                chordText
            });
        }
    }
    
    // Toggle listening state
    toggleListening() {
        if (this.isListening) {
            this.stopListening();
            return false;
        } else {
            if (!this.audioContext) {
                return this.initAudio();
            } else {
                this.startListening();
                return true;
            }
        }
    }
} 