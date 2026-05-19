const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const https = require('https');
const nodemailer = require('nodemailer');
const mlModel = require('./ml_model');

const app = express();
const server = http.createServer(app);

const wss = new WebSocket.Server({
  server,
  perMessageDeflate: false
});

const PORT = process.env.PORT || 3000;

// ================= GOOGLE SHEETS =================
const SHEET_URL =
  'https://script.google.com/macros/s/AKfycbwFBwZc868SvpnuvSAYlpoX2q3WneREYoh9Gmdr-xiZ3ljGvPR64k2rVZB0oDSYl6LY/exec';

// ================= EMAIL =================
const EMAIL_FROM =
  process.env.EMAIL_FROM || 'n60760942@gmail.com';

const EMAIL_TO =
  process.env.EMAIL_TO || 'n60760942@gmail.com';

const EMAIL_PASS =
  process.env.EMAIL_PASSWORD || 'YOUR_APP_PASSWORD';

const transporter = nodemailer.createTransport({
  host: 'smtp.gmail.com',
  port: 465,
  secure: true,
  auth: {
    user: EMAIL_FROM,
    pass: EMAIL_PASS
  }
});

transporter.verify((err) => {
  if (err) {
    console.log('[Email] ERROR:', err.message);
  } else {
    console.log('[Email] Ready!');
  }
});

// ================= ML =================
let mlReady = false;

mlModel.trainModel()
  .then(() => {
    mlReady = true;
    console.log('[ML] Model ready!');
  })
  .catch(err => {
    console.log('[ML] Error:', err.message);
  });

// ================= STATE =================
let lastEmailState = 'IDLE';
let lastEmailTime = 0;
let lastLogTime = 0;

const EMAIL_COOLDOWN = 60000;

app.use(express.json());
app.use(express.static(__dirname));

// ================= ROUTES =================
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'dashboard.html'));
});

app.get('/mobile', (req, res) => {
  res.sendFile(path.join(__dirname, 'mobile.html'));
});

// ================= DATA ROUTE =================
app.post('/data', async (req, res) => {

  const {
    freq,
    ax = 0,
    ay = 0,
    az = 9.81,
    ts,
    alert = 'IDLE',
    deviation = 0,
    baseline = 83.87
  } = req.body;

  if (freq === undefined) {
    return res.status(400).json({
      error: 'Missing freq'
    });
  }

  let mlResult = null;

  // ===== PHYSICS-BASED CLASSIFICATION =====
  let mlClass = 'HEALTHY';
  let confidence = 99;

  let estimatedDamage = {
    level: 'No damage',
    holeSize: '0 mm'
  };

  const absDev = Math.abs(deviation);

  if (absDev <= 5) {
    mlClass = 'HEALTHY';
    confidence = 99;
  } else if (absDev <= 15) {
    mlClass = 'WARNING';
    confidence = 95;

    estimatedDamage = {
      level: 'Possible damage',
      holeSize: '20–40 mm'
    };
  } else {
    mlClass = 'CRITICAL';
    confidence = 98;

    estimatedDamage = {
      level: 'Severe damage',
      holeSize: '60+ mm'
    };
  }

  // ===== ML PREDICTION =====
  if (mlReady && freq > 5) {
    try {

      const mag = Math.sqrt(
        ax * ax +
        ay * ay +
        az * az
      );

      const rawML = await mlModel.predict(
        freq,
        deviation,
        mag
      );

      mlResult = {
        class:
          rawML?.class || mlClass,

        confidence:
          rawML?.confidence || confidence,

        estimatedDamage:
          rawML?.estimatedDamage ||
          estimatedDamage,

        probabilities:
          rawML?.probabilities || {
            HEALTHY:
              mlClass === 'HEALTHY'
                ? 99 : 1,

            WARNING:
              mlClass === 'WARNING'
                ? 95 : 1,

            CRITICAL:
              mlClass === 'CRITICAL'
                ? 98 : 1
          }
      };

      console.log(
        `[ML FIXED] ${mlResult.class} (${mlResult.confidence}%)`
      );

    } catch (err) {
      console.log(
        '[ML] Prediction error:',
        err.message
      );
    }
  }

  const payload = JSON.stringify({
    freq,
    ax,
    ay,
    az,
    ts,
    alert,
    deviation,
    baseline,
    serverTs: Date.now(),
    ml: mlResult
  });

  console.log(
    `[${new Date().toLocaleTimeString()}] freq=${freq}Hz alert=${alert}`
  );

  // ===== SEND TO DASHBOARD =====
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  });

  // ===== GOOGLE SHEETS =====
  logSheets({
    freq,
    ax,
    ay,
    az,
    alert,
    deviation,
    baseline,
    ml: mlResult?.class || 'N/A'
  });

  res.json({
    status: 'ok',
    ml: mlResult
  });
});

// ================= EMAIL =================
function handleEmail() {}

// ================= GOOGLE SHEETS =================
function logSheets(data) {

  const now = Date.now();

  if (now - lastLogTime < 5000) {
    return;
  }

  lastLogTime = now;

  const body = JSON.stringify(data);

  const url = new URL(SHEET_URL);

  const req = https.request({
    hostname: url.hostname,
    path: url.pathname + url.search,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body)
    }

  }, res => {

    res.on('data', () => {});

    res.on('end', () => {
      console.log('[Sheets] OK');
    });
  });

  req.on('error', e => {
    console.log(
      '[Sheets] Error:',
      e.message
    );
  });

  req.write(body);
  req.end();
}

// ================= WEBSOCKET =================
wss.on('connection', ws => {

  console.log('Dashboard connected');

  ws.send(JSON.stringify({
    type: 'connected'
  }));
});

// ================= KEEPALIVE =================
const RENDER_URL =
  process.env.RENDER_URL ||
  'https://ai-shm.onrender.com';

setInterval(() => {

  https
    .get(RENDER_URL, () => {})
    .on('error', () => {});

}, 240000);

// ================= START SERVER =================
server.listen(PORT, () => {

  console.log(
    `SHM server running on port ${PORT}`
  );

  console.log(
    `Render URL: ${RENDER_URL}`
  );
});
