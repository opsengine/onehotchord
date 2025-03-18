/**
 * Constant-Q Transform implementation
 */
class CQTransform {
    static TONNETZ_MATRIX = (() => {
        const T = new Array(6).fill().map(() => new Array(12).fill(0));

        const r1 = 1.0;
        const r2 = 1.0;
        const r3 = 0.5;

        // Generate the transformation matrix
        for (let l = 0; l < 12; l++) {
            // Perfect fifth (7 semitones) - rows 0-1
            T[0][l] = r1 * Math.sin(l * 7 * Math.PI / 6);
            T[1][l] = r1 * Math.cos(l * 7 * Math.PI / 6);
            // Minor third (3 semitones) - rows 2-3
            T[2][l] = r2 * Math.sin(l * 3 * Math.PI / 2);
            T[3][l] = r2 * Math.cos(l * 3 * Math.PI / 2);
            // Major third (4 semitones) - rows 4-5
            T[4][l] = r3 * Math.sin(l * 2 * Math.PI / 3);
            T[5][l] = r3 * Math.cos(l * 2 * Math.PI / 3);
        }

        return T;
    })();

    constructor(options = {}) {
        // Default parameters
        this.minFreq = options.minFreq || 65.41;  // C2
        this.maxFreq = options.maxFreq || 2093.0; // C7
        this.binsPerOctave = options.binsPerOctave || 12;
        this.sampleRate = options.sampleRate || 44100;
        this.debug = options.debug || false;
        this.fftSize = options.fftSize || 8192;
        
        // Calculate number of octaves and total bins
        this.numOctaves = Math.log2(this.maxFreq / this.minFreq);
        this.totalBins = Math.ceil(this.numOctaves * this.binsPerOctave);
        
        // Calculate Q factor (constant ratio of frequency to resolution)
        this.Q = 1 / (Math.pow(2, 1/this.binsPerOctave) - 1);
        
        // Initialize center frequencies and kernels
        this.centerFreqs = new Array(this.totalBins);
        this.kernels = new Array(this.totalBins);
        
        // Calculate center frequencies
        for (let k = 0; k < this.totalBins; k++) {
            this.centerFreqs[k] = this.minFreq * Math.pow(2, k / this.binsPerOctave);
        }
        
        // Initialize kernels
        this.initKernels();
    }
    
    /**
     * Initialize CQT kernels
     */
    initKernels() {
        for (let k = 0; k < this.totalBins; k++) {
            const centerFreq = this.centerFreqs[k];
            
            // Calculate window length based on Q factor
            const windowLength = Math.ceil(this.Q * this.sampleRate / centerFreq);
            
            // Create kernel
            this.kernels[k] = this.createKernel(centerFreq, windowLength);
        }
    }
    
    /**
     * Create a frequency-domain kernel for a specific center frequency
     */
    createKernel(centerFreq, windowLength) {
        // Create a kernel
        const kernel = {
            centerFreq: centerFreq,
            weights: []
        };
        
        // Calculate bin width in Hz
        const binWidth = this.sampleRate / this.fftSize;
        
        // Use the window length to determine bandwidth
        // This is more theoretically correct than using a fixed percentage
        const bandwidth = this.sampleRate / windowLength;
        
        // Calculate the range of FFT bins that contribute to this kernel
        const minBin = Math.max(0, Math.floor((centerFreq - bandwidth/2) / binWidth));
        const maxBin = Math.min(this.fftSize / 2, Math.ceil((centerFreq + bandwidth/2) / binWidth));
        
        // Calculate weights for each FFT bin
        for (let bin = minBin; bin <= maxBin; bin++) {
            const freq = bin * binWidth;
            const distance = Math.abs(freq - centerFreq);
            
            // Use a moderate filter response
            let response = 0;
            if (distance < bandwidth/2) {
                // Gaussian filter with moderate steepness
                response = Math.exp(-0.5 * Math.pow(distance / (bandwidth * 0.3), 2));
            }
            
            if (response > 0.001) { // Only store significant weights
                kernel.weights.push({
                    bin: bin,
                    freq: freq,
                    response: response
                });
            }
        }
        
        // Normalize kernel weights
        let sumSquared = 0;
        for (let i = 0; i < kernel.weights.length; i++) {
            sumSquared += kernel.weights[i].response * kernel.weights[i].response;
        }
        
        const normFactor = 1 / Math.sqrt(sumSquared);
        for (let i = 0; i < kernel.weights.length; i++) {
            kernel.weights[i].response *= normFactor;
        }
        
        return kernel;
    }
    
    /**
     * Compute CQT from analyzer frequency data
     * @param {Float32Array} frequencyData - FFT data from analyzer
     * @returns {Object} - Contains cqt and chroma arrays
     */
    compute(frequencyData) {
        const cqt = new Float32Array(this.totalBins);
        
        // Convert dB to magnitude with a high scaling factor
        const magnitudes = new Float32Array(frequencyData.length);
        for (let i = 0; i < frequencyData.length; i++) {
            // Add a floor to avoid very small values
            const dB = Math.max(frequencyData[i], -100);
            // Apply a high scaling factor to the magnitudes
            magnitudes[i] = Math.pow(10, dB / 20) * 1000000;
        }
        
        // Store bin contributions for debugging
        let binContributions = null;
        if (this.debug) {
            binContributions = new Array(this.totalBins);
        }
        
        // Apply CQT kernels
        for (let k = 0; k < this.totalBins; k++) {
            const kernel = this.kernels[k];
            let energy = 0;
            
            // Store contributions for this bin if debugging
            if (this.debug) {
                binContributions[k] = [];
            }
            
            // Sum the frequency contributions
            for (let i = 0; i < kernel.weights.length; i++) {
                const weight = kernel.weights[i];
                if (weight.bin < magnitudes.length) {
                    const contribution = magnitudes[weight.bin] * weight.response;
                    energy += contribution;
                    
                    // Store this contribution for debugging
                    if (this.debug) {
                        binContributions[k].push({
                            fftBin: weight.bin,
                            fftFreq: weight.freq.toFixed(2),
                            magnitude: magnitudes[weight.bin].toFixed(4),
                            weight: weight.response.toFixed(4),
                            contribution: contribution.toFixed(4)
                        });
                    }
                }
            }
            
            // Apply contrast enhancement
            energy = Math.pow(energy, 1.5);
            
            cqt[k] = energy;
        }

        return cqt;
    }
    
    /**
     * Print detailed debug information
     */
    printDebugInfo(cqt, chroma, binContributions, noteNames, minOctave) {
        const maxOctave = minOctave + Math.floor(this.numOctaves);
        
        console.log("CQT Bins by Octave and Note with FFT Bin Contributions:");
        console.log("-----------------------------------------------------");
        
        // For each octave, then each note
        for (let octave = minOctave; octave <= maxOctave; octave++) {
            for (let note = 0; note < 12; note++) {
                // Calculate the bin index
                const binIndex = (octave - minOctave) * 12 + note;
                
                // Skip if bin is out of range
                if (binIndex >= this.totalBins) continue;
                
                // Get the note name and frequency
                const noteName = noteNames[note] + octave;
                const frequency = this.centerFreqs[binIndex].toFixed(2);
                const energy = cqt[binIndex].toFixed(4);
                
                console.log(`\n${noteName} (${frequency} Hz) - Energy: ${energy}`);
                console.log("FFT Bin Contributions:");
                console.log("FFT Bin\tFreq (Hz)\tMagnitude\tWeight\tContribution");
                console.log("-------\t--------\t---------\t------\t------------");
                
                // Print the top 10 contributing bins (or fewer if there are less)
                const contributions = binContributions[binIndex];
                
                // Sort contributions by their value (highest first)
                contributions.sort((a, b) => parseFloat(b.contribution) - parseFloat(a.contribution));
                
                // Print top 10 or all if less than 10
                const numToPrint = Math.min(10, contributions.length);
                for (let i = 0; i < numToPrint; i++) {
                    const c = contributions[i];
                    console.log(`${c.fftBin}\t${c.fftFreq}\t${c.magnitude}\t${c.weight}\t${c.contribution}`);
                }
                
                // Print total number of contributing bins
                if (contributions.length > numToPrint) {
                    console.log(`... and ${contributions.length - numToPrint} more bins`);
                }
                
                // Print total energy from all bins
                const totalContribution = contributions.reduce((sum, c) => sum + parseFloat(c.contribution), 0).toFixed(4);
                console.log(`Total contribution from all ${contributions.length} bins: ${totalContribution}`);
            }
            
            // Add a separator between octaves
            console.log("\n" + "-".repeat(50));
        }
        
        // Print chroma vector
        console.log("\nChroma Vector (Energy by Note):");
        console.log("Note\tEnergy");
        console.log("----\t------");
        
        for (let note = 0; note < 12; note++) {
            console.log(`${noteNames[note]}\t${chroma[note].toFixed(4)}`);
        }
        
        // Print normalized chroma vector
        const maxChroma = Math.max(...chroma);
        console.log("\nNormalized Chroma Vector:");
        console.log("Note\tNormalized Energy");
        console.log("----\t----------------");
        
        for (let note = 0; note < 12; note++) {
            console.log(`${noteNames[note]}\t${(chroma[note] / maxChroma).toFixed(4)}`);
        }
    }
    
    /**
     * Get the chroma vector (energy by note, summed across octaves)
     * @param {Float32Array} frequencyData - FFT data from analyzer
     * @returns {Float32Array} - 12-element array with energy for each note
     */
    getChroma(frequencyData) {
        return this.compute(frequencyData).chroma;
    }
    
    /**
     * Get the normalized chroma vector (values from 0 to 1)
     * @param {Float32Array} frequencyData - FFT data from analyzer
     * @returns {Float32Array} - 12-element array with normalized energy for each note
     */
    getNormalizedChroma(frequencyData) {
        const chroma = this.getChroma(frequencyData);
        const maxValue = Math.max(...chroma);
        
        if (maxValue > 0) {
            const normalizedChroma = new Float32Array(12);
            for (let i = 0; i < 12; i++) {
                normalizedChroma[i] = chroma[i] / maxValue;
            }
            return normalizedChroma;
        }
        
        return chroma;
    }

    /**
     * Compute the Tonnetz vector (6D)
     * @param {Float32Array} chromagram - 12-element array with energy for each note
     * @returns {Float32Array} - 6-element array with the Tonnetz vector
     */
    computeTonnetz(chromagram) {
        // Chromagram should be [C, C#, D, ..., B] (12 elements)
        if (chromagram.length !== 12) {
            throw new Error("Chromagram must have 12 pitch classes");
        }

        // Normalize the chromagram
        let norm = 0;
        for (let i = 0; i < chromagram.length; i++) {
            norm += Math.abs(chromagram[i]);
        }

        // Compute 6D Tonnetz vector using the pre-computed static matrix
        const tonnetz = new Array(6).fill(0);
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 12; j++) {
                tonnetz[i] += chromagram[j] * CQTransform.TONNETZ_MATRIX[i][j];
            }
            if (norm !== 0) {
                tonnetz[i] /= norm;
            }
        }

        return tonnetz;
    }
}

// Export the class
export default CQTransform;
