const http = require('http');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const port = process.env.PORT || 3000;

function send(res, code, obj) {
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(obj));
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    return send(res, 200, { ok: true });
  }

  if (req.method === 'POST' && req.url === '/render') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 25 * 1024 * 1024) req.destroy(); // Limite de 25MB
    });

    req.on('end', async () => {
      try {
        const cfg = JSON.parse(body);

        if (!cfg || typeof cfg !== 'object')
          throw new Error('missing config body');
        if (!cfg.outPath)
          throw new Error('config.outPath is required');

        // Grava config num arquivo temporário e roda CLI sob xvfb-run
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'editly-'));
        const cfgPath = path.join(tmpDir, 'config.json');
        // Force no transitions at server level for stability and default to fast=false for quality
        const merged = { fast: false, ...cfg };
        fs.writeFileSync(cfgPath, JSON.stringify(merged));

        console.log('🎬 Renderizando vídeo...');
        // Match Xvfb screen to requested output resolution to avoid internal up/downscaling
        const w = Number(merged.width) > 0 ? Number(merged.width) : 1280;
        const h = Number(merged.height) > 0 ? Number(merged.height) : 720;
        const args = ['-a', '-s', `-screen 0 ${w}x${h}x24`, 'node', '/app/node_modules/editly/cli.js', cfgPath];
        const child = spawn('xvfb-run', args, { stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env } });
        child.stdout.on('data', (d) => process.stdout.write(d));
        child.stderr.on('data', (d) => process.stderr.write(d));
        child.on('close', (code) => {
          try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
          if (code === 0) {
            console.log('✅ Renderização concluída:', cfg.outPath);
            return send(res, 200, { ok: true, outPath: cfg.outPath });
          }
          return send(res, 500, { ok: false, error: `editly exit ${code}` });
        });
      } catch (err) {
        console.error('❌ Erro ao renderizar vídeo:', err);
        return send(res, 500, { ok: false, error: String(err.message || err) });
      }
    });

    return;
  }

  return send(res, 404, { ok: false, error: 'not-found' });
});

server.listen(port, () => {
  console.log(`[editly-api] listening on :${port}`);
});
