import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

globalThis.self = globalThis;
if (typeof globalThis.ProgressEvent === "undefined") {
  globalThis.ProgressEvent = class ProgressEvent {
    constructor(type, init = {}) {
      this.type = type;
      Object.assign(this, init);
    }
  };
}

const require = createRequire(import.meta.url);
const validator = require("gltf-validator");
const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");

const files = process.argv.slice(2);
if (!files.length) {
  console.error("Usage: node scripts/validate_exported_glbs.mjs <file.glb> [...]");
  process.exit(2);
}

let failed = false;

for (const file of files) {
  const data = fs.readFileSync(file);
  const validation = await validator.validateBytes(new Uint8Array(data), {
    uri: path.basename(file),
    maxIssues: 1000,
  });

  const errors = validation.issues?.numErrors ?? 0;
  const warnings = validation.issues?.numWarnings ?? 0;
  console.log(
    "GLTF_VALIDATOR",
    file,
    JSON.stringify({ errors, warnings, infos: validation.issues?.numInfos ?? 0 })
  );

  if (errors > 0) {
    failed = true;
    for (const message of validation.issues.messages || []) {
      if (message.severity === 0) console.error("validator_error", JSON.stringify(message));
    }
  }

  try {
    const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    const gltf = await new Promise((resolve, reject) => {
      new GLTFLoader().parse(arrayBuffer, "", resolve, reject);
    });
    let meshCount = 0;
    gltf.scene.traverse((object) => {
      if (object.isMesh) meshCount += 1;
    });
    console.log("THREE_GLTFLOADER", file, JSON.stringify({ meshCount }));
    if (meshCount === 0) {
      console.error("Three.js parsed the GLB but found no meshes:", file);
      failed = true;
    }
  } catch (error) {
    console.error("Three.js GLTFLoader failed:", file, error);
    failed = true;
  }
}

if (failed) process.exit(1);
