const express   = require('express');
const http      = require('http');
const WebSocket = require('ws');
const path      = require('path');
const https     = require('https');
const nodemailer = require('nodemailer');
const mlModel   = require('./ml_model');

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocket.Server({ server, perMessageDeflate: false });

const PORT = process.env.PORT || 3000;

// ── Google Sheets ─────────────────────────────────────────────────
const SHEET_URL = 'https://script.google.com/macros/s/AKfycbwFBwZc868SvpnuvSAYlpoX2q3WneREYoh9Gmdr-xiZ3ljGvPR64k2rVZB0oDSYl6LY/exec';

// ── Email config — port 465 SSL works on Render ───────────────────
const EMAIL_FROM = process.env.EMAIL_FROM     || 'n60760942@gmail.com';
const EMAIL_TO   = process.env.EMAIL_TO       || 'n60760942@gmail.com';
const EMAIL_PASS = process.env.EMAIL_PASSWORD || 'gzwmujtlerotdfgn';

const transporter = nodemailer.createTransport({
  host:   'smtp.gmail.com',
  port:   465,
  secure: true,
  auth: {
    user: EMAIL_FROM,
    pass: EMAIL_PASS
  }
});

transporter.verify((err) => {
  if (err) console.log('[Email] ERROR:', err.message);
  else     console.log('[Email] Ready to send!');
});

// ── ML setup ──────────────────────────────────────────────────────
let mlReady = false;
mlModel.trainModel().then(() => {
  mlReady = true;
  console.log('[ML] Model ready!');
}).catch(err => {
  console.log('[ML] Error:', err.message);
});

// ── State ─────────────────────────────────────────────────────────
let lastEmailState = 'IDLE';
let lastEmailTime  = 0;
let lastLogTime    = 0;
const EMAIL_COOLDOWN = 60000;

app.use(express.json());

// ── Routes ────────────────────────────────────────────────────────
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
    return res.status(400).json({ error: 'Missing freq' });
  }

  let mlResult = null;

  // ── SHM Physics-based classification (CORRECTED) ──
  let mlClass = 'HEALTHY';
  let confidence = 99;
  let estimatedDamage = {
    level: 'No damage',
    holeSize: '0 mm'
  };

  if (deviation <= 5) {
    mlClass = 'HEALTHY';
    confidence = 99;
    estimatedDamage = {
      level: 'No damage',
      holeSize: '0 mm'
    };
  }
  else if (deviation <= 15) {
    mlClass = 'WARNING';
    confidence = 95;
    estimatedDamage = {
      level: 'Possible damage',
      holeSize: '20–40 mm'
    };
  }
  else {
    mlClass = 'CRITICAL';
    confidence = 98;
    estimatedDamage = {
      level: 'Severe damage',
      holeSize: '60+ mm'
    };
  }

  // ── Optional ML probabilities only ──
  if (mlReady && freq > 5) {
    try {
      const mag = Math.sqrt(ax * ax + ay * ay + az * az);

      const rawML = await mlModel.predict(freq, deviation, mag);

      mlResult = {
        class: mlClass,
        confidence: confidence,
        estimatedDamage: estimatedDamage,

        probabilities: rawML?.probabilities || {
          HEALTHY: mlClass === 'HEALTHY' ? confidence : 1,
          WARNING: mlClass === 'WARNING' ? confidence : 1,
          CRITICAL: mlClass === 'CRITICAL' ? confidence : 1
        }
      };

      console.log(
        `[ML FIXED] ${mlResult.class} (${mlResult.confidence}%) — ${mlResult.estimatedDamage.level}`
      );

    } catch (err) {
      console.log('[ML] Prediction error:', err.message);
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
    `[${new Date().toLocaleTimeString()}] freq=${freq}Hz alert=${alert} dev=${deviation}%`
  );

  // ── Send to dashboard ──
  wss.clients.forEach(c => {
    if (c.readyState === WebSocket.OPEN) {
      c.send(payload);
    }
  });

  // ── Email + Sheets ──
  if (alert !== 'IDLE' && alert !== 'MEASURING') {
    handleEmail(
      alert,
      freq,
      deviation,
      ax,
      ay,
      az,
      mlResult
    );

    logSheets({
      freq,
      ax,
      ay,
      az,
      alert,
      deviation,
      baseline,
      ml: mlResult ? mlResult.class : 'N/A'
    });
  }

  res.json({
    status: 'ok',
    ml: mlResult
  });
});

// ── Email ─────────────────────────────────────────────────────────
function handleEmail(alertState, freq, deviation, ax, ay, az, ml) {
  const now = Date.now();
  if (alertState === lastEmailState) return;
  if (now - lastEmailTime < EMAIL_COOLDOWN) return;

  if (alertState === 'CRITICAL' || alertState === 'WARNING') {
    lastEmailState = alertState;
    lastEmailTime  = now;
    sendAlert(alertState, freq, deviation, ax, ay, az, ml);
  } else if (alertState === 'NORMAL' &&
             lastEmailState !== 'NORMAL' &&
             lastEmailState !== 'IDLE') {
    lastEmailState = alertState;
    sendRecovery(freq);
  }
}

function sendAlert(alertState, freq, deviation, ax, ay, az, ml) {
  const crit    = alertState === 'CRITICAL';
  const color   = crit ? '#ff4a4a' : '#f5a623';
  const emoji   = crit ? '🔴' : '🟡';
  const subject = `${emoji} GFRP SHM ${alertState} — Freq=${freq} Hz`;

  const mlRows = ml ? `
    <tr>
      <td style="padding:10px;color:#666;font-weight:bold;background:#e8f4f8">AI Class</td>
      <td style="padding:10px;color:#0066cc;font-weight:bold;background:#e8f4f8">${ml.class} (${ml.confidence}% confidence)</td>
    </tr>
    <tr>
      <td style="padding:10px;color:#666;font-weight:bold;">Damage Estimate</td>
      <td style="padding:10px;">${ml.estimatedDamage.level} — ${ml.estimatedDamage.holeSize}</td>
    </tr>
    <tr>
      <td style="padding:10px;color:#666;font-weight:bold;background:#f9f9f9">ML Probabilities</td>
      <td style="padding:10px;font-size:12px;background:#f9f9f9">
        Healthy: ${ml.probabilities.HEALTHY}% |
        Warning: ${ml.probabilities.WARNING}% |
        Critical: ${ml.probabilities.CRITICAL}%
      </td>
    </tr>` : '';

  const html = `
  <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
    <div style="background:${color};color:white;padding:20px;border-radius:8px 8px 0 0;">
      <h2 style="margin:0">${emoji} ${alertState} — GFRP Plate SHM</h2>
      <p style="margin:5px 0 0;opacity:.9">${new Date().toLocaleString()}</p>
    </div>
    <div style="padding:20px;border:1px solid #ddd;border-top:none;border-radius:0 0 8px 8px;background:#f9f9f9">
      <table style="width:100%;border-collapse:collapse;background:#fff;border-radius:6px;overflow:hidden;">
        <tr>
          <td style="padding:10px;color:#666;font-weight:bold;">M1 Frequency</td>
          <td style="padding:10px;color:${color};font-size:20px;font-weight:bold;">${freq} Hz</td>
        </tr>
        <tr>
          <td style="padding:10px;color:#666;font-weight:bold;background:#f9f9f9">Baseline</td>
          <td style="padding:10px;background:#f9f9f9">83.87 Hz — FFFF No hole 0°</td>
        </tr>
        <tr>
          <td style="padding:10px;color:#666;font-weight:bold;">Deviation</td>
          <td style="padding:10px;color:${color};font-weight:bold;">${deviation}%</td>
        </tr>
        ${mlRows}
        <tr>
          <td style="padding:10px;color:#666;font-weight:bold;background:#f9f9f9">Acceleration</td>
          <td style="padding:10px;background:#f9f9f9">X=${ax} Y=${ay} Z=${az} m/s²</td>
        </tr>
      </table>
      <div style="margin-top:16px;padding:12px;background:#fff3cd;border-radius:6px;border:1px solid #ffc107;">
        <strong>Action required:</strong> Inspect GFRP plate immediately.
      </div>
      <p style="color:#999;font-size:11px;margin-top:16px;">
        GFRP SHM · MTech Project · ESP8266 + ADXL345 + AI/ML
      </p>
    </div>
  </div>`;

  transporter.sendMail({ from: EMAIL_FROM, to: EMAIL_TO, subject, html }, (err, info) => {
    if (err) console.log('[Email] Send error:', err.message);
    else     console.log('[Email] Sent! ID:', info.messageId);
  });
}

function sendRecovery(freq) {
  transporter.sendMail({
    from:    EMAIL_FROM,
    to:      EMAIL_TO,
    subject: `✅ GFRP SHM — Back to normal (${freq} Hz)`,
    html:    `<p>Plate returned to normal.</p>
              <p><strong>Current: ${freq} Hz</strong></p>
              <p>Baseline: 83.87 Hz · Normal: 79.68–88.06 Hz</p>`
  }, (err, info) => {
    if (err) console.log('[Email] Send error:', err.message);
    else     console.log('[Email] Recovery sent! ID:', info.messageId);
  });
}

// ── Google Sheets ─────────────────────────────────────────────────
function logSheets(data) {
  const now = Date.now();
  if (now - lastLogTime < 5000) return;
  lastLogTime = now;

  const body = JSON.stringify(data);

  function req(opts, redirects) {
    if (redirects > 5) return;
    const r = https.request(opts, res => {
      let raw = '';
      res.on('data', c => raw += c);
      res.on('end', () => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          const u = new URL(res.headers.location);
          req({ hostname: u.hostname, path: u.pathname + u.search, method: 'GET', headers: {} }, redirects + 1);
        } else {
          console.log('[Sheets] OK');
        }
      });
    });
    r.on('error', e => console.log('[Sheets] Error:', e.message));
    if (opts.method === 'POST') r.write(body);
    r.end();
  }

  const u = new URL(SHEET_URL);
  req({
    hostname: u.hostname,
    path:     u.pathname + u.search,
    method:   'POST',
    headers:  { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) }
  }, 0);
}

// ── WebSocket ─────────────────────────────────────────────────────
wss.on('connection', ws => {
  console.log('Dashboard connected.');

  const ping = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.ping();
  }, 25000);

  ws.on('pong',  ()    => console.log('Pong OK'));
  ws.on('close', ()    => { clearInterval(ping); console.log('Dashboard disconnected.'); });
  ws.on('error', (err) => { clearInterval(ping); console.log('WS error:', err.message); });

  ws.send(JSON.stringify({ type: 'connected' }));
});

// ── Render keepalive ──────────────────────────────────────────────
const RENDER_URL = process.env.RENDER_URL || 'https://ai-shm.onrender.com';
setInterval(() => {
  https.get(RENDER_URL, () => {}).on('error', () => {});
  console.log('[Keepalive] ping sent');
}, 4 * 60 * 1000);

server.listen(PORT, () => {
  console.log(`SHM server running on port ${PORT}`);
  console.log(`Email from: ${EMAIL_FROM} → ${EMAIL_TO}`);
  console.log(`Render URL: ${RENDER_URL}`);
});