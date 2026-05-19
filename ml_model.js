/*
  GFRP SHM — ML Model using TensorFlow.js
  
  Features used for classification:
  1. Frequency (Hz)
  2. Deviation from baseline (%)
  3. RMS acceleration magnitude
  4. Frequency ratio (freq/baseline)
  
  Classes:
  0 = HEALTHY   (no hole, M1 = 79-88 Hz)
  1 = WARNING   (minor damage, M1 = 71-79 Hz)
  2 = CRITICAL  (severe damage, M1 < 71 Hz)
  
  Training data from your Abaqus FEA + measurements
*/

const tf = require('@tensorflow/tfjs-node');

// ── Training data from Abaqus + your measurements ────────────────
// Format: [frequency, deviation%, rms_accel, freq_ratio]
const TRAINING_DATA = [
  // HEALTHY readings (no hole plate, actual measurements)
  [83.87, 0.0,   9.82, 1.000],
  [82.42, -1.7,  9.85, 0.983],
  [83.33, -0.6,  9.83, 0.994],
  [84.38,  0.6,  9.81, 1.007],
  [81.25, -3.1,  9.84, 0.969],
  [80.47, -4.1,  9.82, 0.960],
  [83.59, -0.3,  9.83, 0.997],
  [86.72,  3.4,  9.81, 1.034],
  [82.81, -1.3,  9.82, 0.988],
  [84.96,  1.3,  9.80, 1.013],
  [85.16,  1.5,  9.83, 1.016],
  [80.86, -3.6,  9.84, 0.964],

  // WARNING readings (borderline damage)
  [78.13, -6.8,  9.84, 0.932],
  [76.56, -8.7,  9.83, 0.913],
  [74.22, -11.5, 9.82, 0.885],
  [75.39, -10.1, 9.81, 0.899],
  [77.34, -7.8,  9.83, 0.922],
  [73.83, -11.9, 9.82, 0.881],
  [72.66, -13.4, 9.84, 0.867],
  [71.88, -14.3, 9.83, 0.857],
  [74.61, -11.0, 9.82, 0.890],
  [76.17, -9.2,  9.81, 0.909],

  // CRITICAL readings (60mm hole plate, actual measurements)
  [67.97, -18.9, 9.83, 0.811],
  [68.75, -18.0, 9.82, 0.820],
  [65.63, -21.7, 9.84, 0.783],
  [60.16, -28.3, 9.83, 0.718],
  [54.69, -34.8, 9.82, 0.652],
  [57.03, -32.0, 9.81, 0.680],
  [50.78, -39.4, 9.83, 0.606],
  [46.09, -45.0, 9.82, 0.550],
  [48.44, -42.2, 9.84, 0.578],
  [53.13, -36.6, 9.83, 0.634],
  [63.28, -24.5, 9.82, 0.755],
  [71.09, -15.2, 9.81, 0.848],
  [69.53, -17.1, 9.83, 0.829],
  [45.31, -46.0, 9.82, 0.540],
  [51.17, -38.9, 9.84, 0.611],
  [58.59, -30.1, 9.83, 0.699],
];

// Labels (0=HEALTHY, 1=WARNING, 2=CRITICAL)
const LABELS = [
  0,0,0,0,0,0,0,0,0,0,0,0,  // HEALTHY
  1,1,1,1,1,1,1,1,1,1,       // WARNING
  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2  // CRITICAL
];

// Damage level mapping
const CLASS_NAMES = ['HEALTHY', 'WARNING', 'CRITICAL'];

let model = null;

// ── Feature scaling parameters ────────────────────────────────────
const FEATURE_MIN = [45.0,  -50.0, 9.79, 0.50];
const FEATURE_MAX = [90.0,    7.0, 9.86, 1.08];

function normalizeFeatures(features) {
  return features.map((f, i) => 
    (f - FEATURE_MIN[i]) / (FEATURE_MAX[i] - FEATURE_MIN[i])
  );
}

// ── Build and train model ─────────────────────────────────────────
async function trainModel() {
  console.log('[ML] Building neural network...');

  model = tf.sequential({
    layers: [
      tf.layers.dense({
        inputShape: [4],
        units: 16,
        activation: 'relu',
        kernelInitializer: 'glorotUniform'
      }),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({
        units: 8,
        activation: 'relu'
      }),
      tf.layers.dense({
        units: 3,
        activation: 'softmax'
      })
    ]
  });

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });

  const xs = tf.tensor2d(
    TRAINING_DATA.map(d => normalizeFeatures(d))
  );
  const ys = tf.tensor1d(LABELS, 'int32');

  const history = await model.fit(xs, ys, {
    epochs: 200,
    batchSize: 8,
    validationSplit: 0.2,
    shuffle: true,
    verbose: 0
  });

  const finalAcc = history.history.acc[history.history.acc.length - 1];
  console.log(`[ML] Training complete! Accuracy: ${(finalAcc * 100).toFixed(1)}%`);

  xs.dispose();
  ys.dispose();

  return model;
}

// ── Predict damage level ──────────────────────────────────────────
async function predict(freq, deviation, accelMag) {
  if (!model) {
    console.log('[ML] Model not ready yet');
    return null;
  }

  const freqRatio  = freq / 83.87;
  const features   = normalizeFeatures([freq, deviation, accelMag, freqRatio]);
  const inputTensor = tf.tensor2d([features]);
  const prediction  = model.predict(inputTensor);
  const probabilities = await prediction.data();

  inputTensor.dispose();
  prediction.dispose();

  const maxIdx = probabilities.indexOf(Math.max(...probabilities));

  return {
    class:           CLASS_NAMES[maxIdx],
    confidence:      (probabilities[maxIdx] * 100).toFixed(1),
    probabilities: {
      HEALTHY:  (probabilities[0] * 100).toFixed(1),
      WARNING:  (probabilities[1] * 100).toFixed(1),
      CRITICAL: (probabilities[2] * 100).toFixed(1)
    },
    estimatedDamage: estimateDamage(freq)
  };
}

// ── Estimate damage level from frequency ─────────────────────────
function estimateDamage(freq) {
  if (freq >= 79.68) return { level: 'No damage', holeSize: '0 mm' };
  if (freq >= 74.45) return { level: 'Minor damage', holeSize: '~50 mm' };
  if (freq >= 67.64) return { level: 'Moderate damage', holeSize: '~60 mm' };
  if (freq >= 61.57) return { level: 'Significant damage', holeSize: '~80 mm' };
  return { level: 'Severe damage', holeSize: '>80 mm' };
}

module.exports = { trainModel, predict };