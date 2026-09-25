import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";

const [url, screenshotPath] = process.argv.slice(2);
if (!url || !screenshotPath) {
  console.error("Usage: node scripts/validate_viewer_smoke.mjs <viewer-url> <screenshot.png>");
  process.exit(2);
}

const errors = [];
const browser = await chromium.launch({
  headless: true,
  args: [
    "--use-gl=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
  ],
});

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });

  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForFunction(
    () => document.querySelector("#overlay-status")?.textContent?.startsWith("Loaded"),
    null,
    { timeout: 60000 }
  );

  const result = await page.evaluate(() => {
    const canvas = document.querySelector("#canvas-container canvas");
    const status = document.querySelector("#overlay-status")?.textContent || "";
    return {
      status,
      hasCanvas: Boolean(canvas),
      canvasWidth: canvas?.width || 0,
      canvasHeight: canvas?.height || 0,
    };
  });

  if (!result.hasCanvas || result.canvasWidth <= 0 || result.canvasHeight <= 0) {
    throw new Error(`Viewer has no renderable canvas: ${JSON.stringify(result)}`);
  }
  if (!result.status.startsWith("Loaded")) {
    throw new Error(`Viewer did not report loaded model: ${JSON.stringify(result)}`);
  }

  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });

  // Ignore CDN deprecation noise, but fail on project/runtime errors.
  const blocking = errors.filter(
    (message) =>
      !message.includes("favicon") &&
      !message.includes("punycode") &&
      !message.includes("Failed to load resource: the server responded with a status of 404")
  );
  if (blocking.length) {
    throw new Error(`Viewer runtime errors: ${JSON.stringify(blocking)}`);
  }

  console.log("VIEWER_SMOKE", JSON.stringify({ url, screenshotPath, ...result }));
} finally {
  await browser.close();
}
