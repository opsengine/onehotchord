import CQTransform from './cqt.js';

// Sample chromagram (C major)
const chromagram = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0];

// Initialize CQTransform
const cqt = new CQTransform({
  minFreq: 65.41,
  maxFreq: 2093.0,
  binsPerOctave: 12,
  sampleRate: 44100,
  fftSize: 16384
});

// Calculate tonnetz
const tonnetz = cqt.computeTonnetz(chromagram);
console.log('Tonnetz for C major:', tonnetz);
// console.log(CQTransform.TONNETZ_MATRIX_EXTENDED);
console.log(CQTransform.TONNETZ_MATRIX_EXTENDED);
