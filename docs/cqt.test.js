import CQTransform from "./cqt.js";

import { readFileSync } from 'fs';
import { test, expect, describe } from '@jest/globals';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import wav from 'node-wav';
import { fft } from 'fft-js';

const __dirname = dirname(fileURLToPath(import.meta.url));

describe("CQTransform", () => {
  describe("computeTonnetz", () => {
    let cqt;

    beforeEach(() => {
      // Create a new CQTransform instance before each test
      cqt = new CQTransform();
    });

    test("throws error if chromagram does not have 12 elements", () => {
      // Test with too few elements
      expect(() => {
        cqt.computeTonnetz([1, 0, 1, 0]);
      }).toThrow("Chromagram must have 12 pitch classes");

      // Test with too many elements
      expect(() => {
        cqt.computeTonnetz([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
      }).toThrow("Chromagram must have 12 pitch classes");
    });

    test("returns correct12D vector for a valid chromagram", () => {
      const chromagram = new Float32Array([
        0.07893735, 0.88358967, 0.39349285, 0.88152689, 0.47314087, 0.75323159,
        0.41434758, 0.07315281, 0.74576723, 0.28947901, 0.88042053, 0.59191402,
      ]);

      const result = cqt.computeTonnetz(chromagram);

      expect(result).toHaveLength(12);
      const expectedFirstValue = [
        -0.2687286086627805, -0.13470527223393206, 0.05878720685808317,
        -0.06044518761017612, -0.023343640279308842, -0.11349494942485312,
        -0.08081084842200784, -0.11970701453469644, -0.05878720685808327,
        -0.060445187610176344, -6.598214834035845e-16, -0.07536577780649374,
      ];
      expect(result).toStrictEqual(expectedFirstValue);
    });

    test("handles zero input correctly", () => {
      const chromagram = new Float32Array(12).fill(0);

      const result = cqt.computeTonnetz(chromagram);

      // All output values should be zero when input is zero
      for (let i = 0; i < result.length; i++) {
        expect(result[i]).toBe(0);
      }
    });

    test("handles normalization correctly", () => {
      // Test with different magnitudes but same relative values
      const chromagram1 = new Float32Array([
        1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
      ]);
      const chromagram2 = new Float32Array([
        2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0,
      ]);

      const result1 = cqt.computeTonnetz(chromagram1);
      const result2 = cqt.computeTonnetz(chromagram2);

      // Results should be the same since the relative proportions are identical
      for (let i = 0; i < result1.length; i++) {
        expect(result1[i]).toBeCloseTo(result2[i]);
      }
    });
  });

  describe("compute", () => {
    let cqt;

    beforeEach(() => {
      cqt = new CQTransform({
        minFreq: 65.41, // C2
        maxFreq: 2093.0, // C7
        binsPerOctave: 12,
        sampleRate: 44100,
        fftSize: 2048,
      });
    });

    test("returns correct number of bins", () => {
      // Create mock frequency data (dB scale)
      const freqDataSize = 1024; // Half of fftSize
      const mockFreqData = new Float32Array(freqDataSize).fill(-100);

      // Set some non-zero values
      mockFreqData[10] = -20;
      mockFreqData[20] = -30;

      const result = cqt.compute(mockFreqData);

      // Expected bins based on min/max frequency and bins per octave
      const expectedNumOctaves = Math.log2(2093.0 / 65.41); // ~5 octaves
      const expectedTotalBins = Math.ceil(expectedNumOctaves * 12); // ~60 bins

      expect(result.length).toBe(expectedTotalBins);
    });

    test("detects energy at specific frequencies", () => {
      // Create mock frequency data to simulate energy at middle C (261.63 Hz)
      const freqDataSize = 1024;
      const mockFreqData = new Float32Array(freqDataSize).fill(-100);

      // Calculate which FFT bin corresponds to middle C
      const binWidth = 44100 / 2048;
      const middleCBin = Math.round(261.63 / binWidth);

      // Set energy at middle C
      mockFreqData[middleCBin] = -20; // Higher value = more energy

      const result = cqt.compute(mockFreqData);

      // Find which CQT bin corresponds to middle C (C4)
      const c4Index = 2 * 12 + 0; // Assuming C2 is the base (minFreq)

      // Middle C should have significant energy compared to other bins
      let maxEnergyBin = 0;
      let maxEnergy = 0;

      for (let i = 0; i < result.length; i++) {
        if (result[i] > maxEnergy) {
          maxEnergy = result[i];
          maxEnergyBin = i;
        }
      }

      // Allowing some tolerance due to how CQT bins can spread energy
      expect(maxEnergyBin).toBeCloseTo(c4Index, 1);
    });

    // test("returns zeros for silent input", () => {
    //   // Create mock frequency data with all values at minimum
    //   const freqDataSize = 1024;
    //   const mockFreqData = new Float32Array(freqDataSize).fill(-120);

    //   const result = cqt.compute(mockFreqData);

    //   // Sum of energy should be very close to zero
    //   const totalEnergy = result.reduce((sum, val) => sum + val, 0);
    //   expect(totalEnergy).toBeLessThan(0.001);
    // });

    test("calculate correct cqt for test.wav", async () => {
      const wavPath = join(__dirname, '..', 'test.wav');
      
      try {
        // Read the WAV file
        const wavBuffer = readFileSync(wavPath);
        const audioData = wav.decode(wavBuffer);
        const sampleRate = audioData.sampleRate;
        const samples = audioData.channelData[0];
        
        // Get a frame of data (use fftSize samples)
        const fftSize = 2048;
        const frame = samples.slice(0, fftSize);
        
        // Prepare input for FFT (convert to complex format required by fft-js)
        const complexInput = frame.map(x => [x, 0]); // [real, imaginary]
        
        // Calculate FFT
        const fftResult = fft(complexInput);
        
        // Convert to magnitude (absolute values)
        const magnitudes = new Float32Array(fftSize/2);
        for (let i = 0; i < fftSize/2; i++) {
          const real = fftResult[i][0];
          const imag = fftResult[i][1];
          magnitudes[i] = Math.sqrt(real*real + imag*imag);
        }
        
        // Convert to dB scale for CQT input (roughly -100dB to 0dB)
        const frequencyData = new Float32Array(fftSize/2);
        for (let i = 0; i < fftSize/2; i++) {
          // Add small value to avoid log(0)
          const magnitude = magnitudes[i] + 1e-10;
          frequencyData[i] = 20 * Math.log10(magnitude);
        }
        
        // Compute CQT with our frequency data
        const result = cqt.compute(frequencyData);
        
        // Log first few bins to verify output
        console.log("CQT first 10 bins:", result.slice(0, 10));
        
        // Basic verification - result should have expected length
        expect(result.length).toBeGreaterThan(0);
        
        // Create a chromagram by summing across octaves
        const chromagram = new Float32Array(12).fill(0);
        for (let i = 0; i < result.length; i++) {
          const pitchClass = i % 12;
          chromagram[pitchClass] += result[i];
        }
        
        console.log("Chromagram:", Array.from(chromagram).map(v => v.toFixed(3)));
        
        // Calculate tonnetz
        const tonnetz = cqt.computeTonnetz(chromagram);
        console.log("Tonnetz:", Array.from(tonnetz).map(v => v.toFixed(3)));
        
      } catch (error) {
        console.error('Error processing audio:', error);
        throw error;
      }
    });
  });
});
