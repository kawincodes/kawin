// Render a poster HTML file to a 1080x1350 PNG (Instagram 4:5 portrait).
// Usage: node render.js <input.html> <output.png> [scale]
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const [input, output, scale = '1'] = process.argv.slice(2);
  if (!input || !output) {
    console.error('Usage: node render.js <input.html> <output.png> [scale]');
    process.exit(1);
  }
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1080, height: 1350 },
    deviceScaleFactor: Number(scale),
  });
  await page.goto('file://' + path.resolve(input));
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(300);
  await page.screenshot({ path: output, clip: { x: 0, y: 0, width: 1080, height: 1350 } });
  await browser.close();
  console.log('Rendered', output);
})();
