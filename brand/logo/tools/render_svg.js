// Render an SVG to PNG at a given pixel width (height follows the viewBox).
// Usage: node render_svg.js <in.svg> <out.png> <width> [background]   (background default: transparent)
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const [input, output, widthArg, bg] = process.argv.slice(2);
  if (!input || !output || !widthArg) {
    console.error('Usage: node render_svg.js <in.svg> <out.png> <width> [background]');
    process.exit(1);
  }
  const svg = fs.readFileSync(input, 'utf8');
  const vb = svg.match(/viewBox="\s*([-\d.]+)[\s,]+([-\d.]+)[\s,]+([-\d.]+)[\s,]+([-\d.]+)\s*"/);
  if (!vb) throw new Error('SVG needs a viewBox');
  const width = Number(widthArg);
  const height = Math.round((width * Number(vb[4])) / Number(vb[3]));
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width, height } });
  const fontsCss = path.resolve(__dirname, '../../../posters/video-editor-hiring/assets/fonts.css');
  await page.setContent(
    `<!doctype html><html><head><link rel="stylesheet" href="file://${fontsCss}"><style>html,body{margin:0;background:${bg || 'transparent'}}svg{display:block;width:${width}px;height:${height}px}</style></head><body>${svg}</body></html>`
  );
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: output, omitBackground: !bg, clip: { x: 0, y: 0, width, height } });
  await browser.close();
  console.log(`Rendered ${output} ${width}x${height}`);
})();
