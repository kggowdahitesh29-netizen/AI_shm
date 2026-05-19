/*
  GFRP SHM ML Model
  Uses @tensorflow/tfjs (pure JS, works on Render free tier)
  3-class neural network: HEALTHY / WARNING / CRITICAL
*/

const tf = require('@tensorflow/tfjs');

const TRAINING_DATA = [
  // HEALTHY
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

  // WARNING
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

  // CRITICAL
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

const LABELS = [
  0,0,0,0,0,0,0,0,0,0,0,0,
  1,1,1,1,1,1,1,1,1,1,
  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
];

const CLASS_NAMES  = ['HEALTHY','WARNING','CRITICAL'];
const FEATURE_MIN  = [45.0, -50.0, 9.79, 0.50];
const FEATURE_MAX  = [90.0,   7.0, 9.86, 1.08];

let model = null;

function normalize(features) {
  return features.map((f, i) =>
    (f - FEATURE_MIN[i]) / (FEATURE_MAX[i] - FEATURE_MIN[i])
  );
}

async function trainModel() {
  console.log('[ML] Building neural network...');
  try {
    model = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape:[4], units:16, activation:'relu' }),
        tf.layers.dropout({ rate:0.2 }),
        tf.layers.dense({ units:8, activation:'relu' }),
        tf.layers.dense({ units:3, activation:'softmax' })
      ]
    });

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });

    const xs = tf.tensor2d(TRAINING_DATA.map(d => normalize(d)));
    const ys = tf.tensor1d(LABELS, 'int32');

    const history = await model.fit(xs, ys, {
      epochs: 200,
      batchSize: 8,
      shuffle: true,
      verbose: 0
    });

    const acc = history.history.acc;
    const finalAcc = acc ? acc[acc.length - 1] : 'N/A';
    console.log(`[ML] Training done! Accuracy: ${typeof finalAcc === 'number' ? (finalAcc*100).toFixed(1) : finalAcc}%`);

    xs.dispose();
    ys.dispose();
  } catch(err) {
    console.log('[ML] Train error:', err.message);
    model = null;
  }
}

async function predict(freq, deviation, accelMag) {
  if (!model) return null;

  try {
    const freqRatio = freq / 83.87;
    const features  = normalize([freq, deviation, accelMag, freqRatio]);
    const input     = tf.tensor2d([features]);
    const output    = model.predict(input);
    const probs     = Array.from(await output.data());

    input.dispose();
    output.dispose();

    const maxIdx = probs.indexOf(Math.max(...probs));

    return {
      class:       CLASS_NAMES[maxIdx],
      confidence:  (probs[maxIdx] * 100).toFixed(1),
      probabilities: {
        HEALTHY:  (probs[0] * 100).toFixed(1),
        WARNING:  (probs[1] * 100).toFixed(1),
        CRITICAL: (probs[2] * 100).toFixed(1)
      },
      estimatedDamage: estimateDamage(freq)
    };
  } catch(err) {
    console.log('[ML] Predict error:', err.message);
    return null;
  }
}

function estimateDamage(freq) {
  if (freq >= 79.68) return { level:'No damage',            holeSize:'0 mm' };
  if (freq >= 74.45) return { level:'Minor damage',         holeSize:'~50 mm' };
  if (freq >= 67.64) return { level:'Moderate damage',      holeSize:'~60 mm' };
  if (freq >= 61.57) return { level:'Significant damage',   holeSize:'~80 mm' };
  return               { level:'Severe damage',             holeSize:'>80 mm' };
}

module.exports = { trainModel, predict };