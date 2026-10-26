// Capture an HTML animation frame-by-frame and encode it as an Instagram Story MP4.
//
// The page is 1080x1920. Animations must be CSS animations / Web Animations that start
// at page load (and/or SVG SMIL, and/or a window.renderFrame(t) hook for JS-driven bits).
// Every frame, all animations are paused and seeked to the exact frame time, so the
// output is deterministic and never drops frames.
//
// Usage:
//   node capture.js <story.html> <out.mp4> [--fps 30] [--duration 15]
//   node capture.js <story.html> <sheet.png> --sheet [--every 0.5] [--safe]   contact sheet of frames
//   node capture.js <story.html> <frame.png> --at 7.25 [--safe]                 one full-size frame
// Duration defaults to window.STORY_DURATION (seconds) if the page defines it.
const path = require('path');
const { spawn } = require('child_process');
const { chromium } = require('playwright');

const W = 1080;
const H = 1920;
// Instagram Story UI overlays: profile bar at the top, reply bar at the bottom.
const SAFE_TOP = 250;
const SAFE_BOTTOM = 340;

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i === -1 ? fallback : process.argv[i + 1];
}
const flag = (name) => process.argv.includes(name);

async function seek(page, t) {
  await page.evaluate((ms) => {
    for (const a of document.getAnimations()) {
      a.pause();
      a.currentTime = ms;
    }
    for (const svg of document.querySelectorAll('svg')) {
      if (svg.pauseAnimations) {
        svg.pauseAnimations();
        svg.setCurrentTime(ms / 1000);
      }
    }
    if (typeof window.renderFrame === 'function') window.renderFrame(ms / 1000);
  }, t * 1000);
}

async function safeOverlay(page, on) {
  await page.evaluate(
    ({ on, top, bottom }) => {
      let el = document.getElementById('__safe');
      if (!on) return el && el.remove();
      if (el) return;
      el = document.createElement('div');
      el.id = '__safe';
      el.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483647';
      el.innerHTML =
        `<div style="position:absolute;left:0;right:0;top:0;height:${top}px;background:rgba(255,0,60,.28);border-bottom:2px dashed #ff2d55"></div>` +
        `<div style="position:absolute;left:0;right:0;bottom:0;height:${bottom}px;background:rgba(255,0,60,.28);border-top:2px dashed #ff2d55"></div>`;
      document.body.appendChild(el);
    },
    { on, top: SAFE_TOP, bottom: SAFE_BOTTOM }
  );
}

(async () => {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) {
    console.error('Usage: node capture.js <story.html> <out.mp4|png> [--fps 30] [--duration 15] [--sheet [--every 0.5]] [--at t] [--safe]');
    process.exit(1);
  }
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: W, height: H } });
  await page.goto('file://' + path.resolve(input));
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(200);
  const duration = Number(arg('--duration', 0)) || (await page.evaluate(() => window.STORY_DURATION)) || 15;
  const safe = flag('--safe');

  if (arg('--at') !== undefined) {
    await seek(page, Number(arg('--at')));
    await safeOverlay(page, safe);
    await page.screenshot({ path: output, clip: { x: 0, y: 0, width: W, height: H } });
    console.log(`Frame at ${arg('--at')}s -> ${output}`);
  } else if (flag('--sheet')) {
    const every = Number(arg('--every', 0.5));
    const times = [];
    for (let t = 0; t <= duration + 1e-9; t += every) times.push(Number(t.toFixed(3)));
    const cols = 8;
    const tw = 216;
    const th = 384;
    const shots = [];
    await safeOverlay(page, safe);
    for (const t of times) {
      await seek(page, t);
      const buf = await page.screenshot({ clip: { x: 0, y: 0, width: W, height: H } });
      shots.push({ t, src: 'data:image/png;base64,' + buf.toString('base64') });
    }
    const rows = Math.ceil(shots.length / cols);
    const sheet = await browser.newPage({ viewport: { width: cols * (tw + 8) + 8, height: rows * (th + 34) + 8 } });
    await sheet.setContent(
      `<body style="margin:0;padding:4px;background:#222;font:600 18px Inter,sans-serif;color:#eee;display:flex;flex-wrap:wrap">` +
        shots
          .map((s) => `<div style="margin:4px;width:${tw}px"><img src="${s.src}" style="width:${tw}px;height:${th}px;display:block"><div style="height:26px;line-height:26px">${s.t.toFixed(2)}s</div></div>`)
          .join('') +
        `</body>`
    );
    await sheet.waitForTimeout(300);
    await sheet.screenshot({ path: output, fullPage: true });
    console.log(`Contact sheet: ${shots.length} frames every ${every}s over ${duration}s -> ${output}`);
  } else {
    const fps = Number(arg('--fps', 30));
    const frames = Math.round(duration * fps);
    const ff = spawn(
      'ffmpeg',
      ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
        '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p',
        '-r', String(fps), '-movflags', '+faststart', output],
      { stdio: ['pipe', 'inherit', 'inherit'] }
    );
    const done = new Promise((res, rej) => ff.on('close', (c) => (c === 0 ? res() : rej(new Error('ffmpeg exited ' + c)))));
    for (let i = 0; i < frames; i++) {
      await seek(page, i / fps);
      const buf = await page.screenshot({ clip: { x: 0, y: 0, width: W, height: H } });
      if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
      if (i % fps === 0) process.stdout.write(`\r${(i / fps).toFixed(0)}s / ${duration}s`);
    }
    ff.stdin.end();
    await done;
    console.log(`\nEncoded ${frames} frames at ${fps}fps (${duration}s) -> ${output}`);
  }
  await browser.close();
})();
