const express    = require('express');
const http       = require('http');
const WebSocket  = require('ws');
const path       = require('path');
const https      = require('https');
const nodemailer = require('nodemailer');
const mlModel    = require('./ml_model');

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocket.Server({ server, perMessageDeflate: false });

const PORT = process.env.PORT || 3000;

const GOOGLE_SHEET_URL = 'https://script.google.com/macros/s/AKfycbwFBwZc868SvpnuvSAYlpoX2q3WneREYoh9Gmdr-xiZ3ljGvPR64k2rVZB0oDSYl6LY/exec';

const EMAIL_FROM     = process.env.EMAIL_FROM     || 'your_gmail@gmail.com';
const EMAIL_TO       = process.env.EMAIL_TO       || 'your_gmail@gmail.com';
const EMAIL_PASSWORD = process.env.EMAIL_PASSWORD || 'your16digitpassword';

// ── FIXED email transporter ───────────────────────────────────────
const transporter = nodemailer.createTransport({
  host: 'smtp.gmail.com',
  port: 587,
  secure: false,
  auth: {
    user: EMAIL_FROM,
    pass: EMAIL_PASSWORD
  },
  tls: {
    rejectUnauthorized: false
  }
});

// Verify email on startup
transporter.verify((error) => {
  if (error) {
    console.log('[Email] Setup error:', error.message);
    console.log('[Email] Check your Gmail app password!');
  } else {
    console.log('[Email] Ready to send alerts!');
  }
});

let lastEmailAlertState = 'IDLE';
let lastEmailTime       = 0;
const EMAIL_COOLDOWN_MS = 60000;

// ── ML model ──────────────────────────────────────────────────────
let mlReady = false;
mlModel.trainModel().then(() => {
  mlReady = true;
  console.log('[ML] Model trained and ready!');
}).catch(err => {
  console.log('[ML] Training error:', err.message);
});

app.use(express.json());

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'dashboard.html'));
});

app.get('/mobile', (req, res) => {
  res.sendFile(path.join(__dirname, 'mobile.html'));
});

app.post('/data', async (req, res) => {
  const body      = req.body;
  const freq      = body.freq;
  const ax        = body.ax      || 0;
  const ay        = body.ay      || 0;
  const az        = body.az      || 9.81;
  const ts        = body.ts;
  const alert     = body.alert   || 'IDLE';
  const deviation = body.deviation || 0;
  const baseline  = body.baseline  || 83.87;

  if (freq === undefined) {
    return res.status(400).json({ error: 'Missing freq field' });
  }

  // ── ML prediction ────────────────────────────────────────────────
  let mlResult = null;
  if (mlReady && freq > 5 && alert !== 'IDLE' && alert !== 'MEASURING') {
    const accelMag = Math.sqrt(ax*ax + ay*ay + az*az);
    mlResult = await mlModel.predict(freq, deviation, accelMag);
    if (mlResult) {
      console.log(`[ML] Prediction: ${mlResult.class} (${mlResult.confidence}% confidence)`);
      console.log(`[ML] Estimated damage: ${mlResult.estimatedDamage.level} — ${mlResult.estimatedDamage.holeSize}`);
    }
  }

  const payload = JSON.stringify({
    freq, ax, ay, az, ts,
    alert, deviation, baseline,
    serverTs: Date.now(),
    ml: mlResult
  });

  console.log(`[${new Date().toLocaleTimeString()}] freq=${freq} Hz  alert=${alert}  dev=${deviation}%${mlResult ? '  ML:' + mlResult.class : ''}`);

  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  });

  if (alert !== 'IDLE' && alert !== 'MEASURING') {
    checkAndSendEmail(alert, freq, deviation, ax, ay, az, mlResult);
    logToGoogleSheets({ freq, ax, ay, az, alert, deviation, baseline, ml: mlResult ? mlResult.class : 'N/A' });
  }

  res.json({ status: 'ok', ml: mlResult });
});

function checkAndSendEmail(alertState, freq, deviation, ax, ay, az, mlResult) {
  const now = Date.now();
  if (alertState === lastEmailAlertState) return;
  if (now - lastEmailTime < EMAIL_COOLDOWN_MS) return;

  if (alertState === 'CRITICAL' || alertState === 'WARNING') {
    lastEmailAlertState = alertState;
    lastEmailTime       = now;
    sendAlertEmail(alertState, freq, deviation, ax, ay, az, mlResult);
  } else if (alertState === 'NORMAL' &&
             lastEmailAlertState !== 'NORMAL' &&
             lastEmailAlertState !== 'IDLE') {
    lastEmailAlertState = alertState;
    sendRecoveryEmail(freq);
  }
}

function sendAlertEmail(alertState, freq, deviation, ax, ay, az, mlResult) {
  const isCritical = alertState === 'CRITICAL';
  const emoji      = isCritical ? '🔴' : '🟡';
  const subject    = `${emoji} GFRP SHM ${alertState} — Freq=${freq} Hz`;

  const mlSection = mlResult ? `
    <tr style="background:#e8f4f8;border-bottom:1px solid #eee;">
      <td style="padding:10px;color:#666;font-weight:bold;">AI Prediction</td>
      <td style="padding:10px;color:#0066cc;font-weight:bold;">
        ${mlResult.class} (${mlResult.confidence}% confidence)
      </td>
    </tr>
    <tr style="background:#fff;border-bottom:1px solid #eee;">
      <td style="padding:10px;color:#666;font-weight:bold;">Estimated Damage</td>
      <td style="padding:10px;color:#333;">
        ${mlResult.estimatedDamage.level} — ${mlResult.estimatedDamage.holeSize}
      </td>
    </tr>
    <tr style="background:#e8f4f8;">
      <td style="padding:10px;color:#666;font-weight:bold;">ML Probabilities</td>
      <td style="padding:10px;color:#333;font-size:12px;">
        Healthy: ${mlResult.probabilities.HEALTHY}% |
        Warning: ${mlResult.probabilities.WARNING}% |
        Critical: ${mlResult.probabilities.CRITICAL}%
      </td>
    </tr>` : '';

  const html = `
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
      <div style="background:${isCritical ? '#ff4a4a' : '#f5a623'};color:white;padding:20px;border-radius:8px 8px 0 0;">
        <h2 style="margin:0;">${emoji} ${alertState} — GFRP Plate SHM</h2>
        <p style="margin:5px 0 0 0;opacity:0.9;">${new Date().toLocaleString()}</p>
      </div>
      <div style="background:#f9f9f9;padding:20px;border-radius:0 0 8px 8px;border:1px solid #ddd;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#fff;border-bottom:1px solid #eee;">
            <td style="padding:10px;color:#666;font-weight:bold;">M1 Frequency</td>
            <td style="padding:10px;color:${isCritical ? '#ff4a4a' : '#f5a623'};font-size:20px;font-weight:bold;">${freq} Hz</td>
          </tr>
          <tr style="background:#f9f9f9;border-bottom:1px solid #eee;">
            <td style="padding:10px;color:#666;font-weight:bold;">Baseline M1</td>
            <td style="padding:10px;">83.87 Hz — FFFF No hole 0°</td>
          </tr>
          <tr style="background:#fff;border-bottom:1px solid #eee;">
            <td style="padding:10px;color:#666;font-weight:bold;">Deviation</td>
            <td style="padding:10px;color:${isCritical ? '#ff4a4a' : '#f5a623'};font-weight:bold;">${deviation}%</td>
          </tr>
          ${mlSection}
          <tr style="background:#fff;">
            <td style="padding:10px;color:#666;font-weight:bold;">Acceleration</td>
            <td style="padding:10px;">X=${ax} Y=${ay} Z=${az} m/s²</td>
          </tr>
        </table>
        <div style="margin-top:16px;padding:12px;background:#fff3cd;border-radius:6px;border:1px solid #ffc107;">
          <strong>Action required:</strong> Inspect GFRP plate immediately.
        </div>
        <p style="color:#999;font-size:12px;margin-top:16px;">
          GFRP SHM · MTech Project · ESP8266 + ADXL345 + AI/ML
        </p>
      </div>
    </div>`;

  transporter.sendMail({ from: EMAIL_FROM, to: EMAIL_TO, subject, html }, (err, info) => {
    if (err) {
      console.log('[Email] Error:', err.message);
    } else {
      console.log(`[Email] ${alertState} alert sent! ID: ${info.messageId}`);
    }
  });
}

function sendRecoveryEmail(freq) {
  transporter.sendMail({
    from: EMAIL_FROM,
    to: EMAIL_TO,
    subject: `✅ GFRP SHM — Returned to normal (${freq} Hz)`,
    html: `<p>Plate frequency returned to normal.</p>
           <p><strong>Current: ${freq} Hz</strong></p>
           <p>Baseline: 83.87 Hz · Normal: 79.68–88.06 Hz</p>`
  }, (err, info) => {
    if (err) console.log('[Email] Error:', err.message);
    else console.log('[Email] Recovery sent! ID:', info.messageId);
  });
}

let lastLogTime = 0;

function logToGoogleSheets(data) {
  const now = Date.now();
  if (now - lastLogTime < 5000) return;
  lastLogTime = now;

  const body = JSON.stringify(data);
  const url  = new URL(GOOGLE_SHEET_URL);

  const options = {
    hostname: url.hostname,
    path: url.pathname + url.search,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body)
    }
  };

  function makeRequest(opts, redirectCount) {
    if (redirectCount > 5) return;
    const req = https.request(opts, (res) => {
      let raw = '';
      res.on('data', chunk => raw += chunk);
      res.on('end', () => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          const redirectUrl = new URL(res.headers.location);
          makeRequest({
            hostname: redirectUrl.hostname,
            path: redirectUrl.pathname + redirectUrl.search,
            method: 'GET',
            headers: {}
          }, redirectCount + 1);
        } else {
          console.log('[Sheets] Logged OK');
        }
      });
    });
    req.on('error', err => console.log('[Sheets] Error:', err.message));
    if (opts.method === 'POST') req.write(body);
    req.end();
  }

  makeRequest(options, 0);
}

wss.on('connection', ws => {
  console.log('Dashboard connected.');

  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.ping();
  }, 30000);

  ws.on('pong', () => console.log('Pong received'));
  ws.on('close', () => {
    clearInterval(pingInterval);
    console.log('Dashboard disconnected.');
  });
  ws.on('error', (err) => {
    clearInterval(pingInterval);
    console.log('WS error:', err.message);
  });

  ws.send(JSON.stringify({ type: 'connected' }));
});

wss.on('error', (err) => console.log('WS server error:', err.message));

setInterval(() => {
  https.get('https://gfrp-shm.onrender.com', () => {}).on('error', () => {});
}, 4 * 60 * 1000);

server.listen(PORT, () => {
  console.log(`SHM server running on port ${PORT}`);
  console.log(`Email → ${EMAIL_TO}`);
  console.log(`Google Sheets logging active`);
});