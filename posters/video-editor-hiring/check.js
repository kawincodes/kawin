// Layout sanity check for a 1080x1350 poster.
// Reports visible text that falls outside the canvas or the 48px safe margin,
// elements whose content overflows their box, and text smaller than 22px.
// Usage: node check.js <input.html>
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const input = process.argv[2];
  if (!input) {
    console.error('Usage: node check.js <input.html>');
    process.exit(1);
  }
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1350 } });
  await page.goto('file://' + path.resolve(input));
  await page.evaluate(() => document.fonts.ready);

  const report = await page.evaluate(() => {
    const SAFE = 48;
    const issues = [];
    const label = (el) => {
      const txt = (el.innerText || el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 40);
      return `<${el.tagName.toLowerCase()}${el.className ? '.' + String(el.className).split(' ').join('.') : ''}> "${txt}"`;
    };
    const doc = document.documentElement;
    if (doc.scrollWidth > 1080 || doc.scrollHeight > 1350) {
      issues.push(`page scroll size ${doc.scrollWidth}x${doc.scrollHeight} exceeds 1080x1350`);
    }
    const fonts = new Set();
    for (const el of document.body.querySelectorAll('*')) {
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) === 0) continue;
      const hasOwnText = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
      if (!hasOwnText) continue;
      fonts.add(cs.fontFamily.split(',')[0].trim());
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      if (r.left < 0 || r.top < 0 || r.right > 1080 || r.bottom > 1350) {
        issues.push(`OUTSIDE CANVAS ${label(el)} box=${Math.round(r.left)},${Math.round(r.top)},${Math.round(r.right)},${Math.round(r.bottom)}`);
      } else if (r.left < SAFE || r.top < SAFE || r.right > 1080 - SAFE || r.bottom > 1350 - SAFE) {
        issues.push(`inside ${SAFE}px margin ${label(el)} box=${Math.round(r.left)},${Math.round(r.top)},${Math.round(r.right)},${Math.round(r.bottom)}`);
      }
      if (el.scrollWidth > el.clientWidth + 2 && cs.overflowX !== 'visible') {
        issues.push(`text overflow-x ${label(el)} scrollWidth=${el.scrollWidth} clientWidth=${el.clientWidth}`);
      }
      if (parseFloat(cs.fontSize) < 22) {
        issues.push(`small text ${cs.fontSize} ${label(el)}`);
      }
    }
    for (const img of document.images) {
      if (!img.complete || img.naturalWidth === 0) issues.push(`broken image ${img.getAttribute('src')}`);
    }
    return { issues, fonts: [...fonts] };
  });

  const loaded = await page.evaluate(() =>
    [...document.fonts].filter((f) => f.status === 'loaded').map((f) => `${f.family} ${f.weight}`)
  );
  console.log('Fonts used by text:', report.fonts.join(', '));
  console.log('Webfonts loaded:', loaded.join(', ') || '(none)');
  if (report.issues.length === 0) console.log('OK: no layout issues found');
  else report.issues.forEach((i) => console.log('-', i));
  await browser.close();
})();
