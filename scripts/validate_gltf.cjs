// Install: npm install --prefix output/gltf-validation gltf-validator
const fs = require('fs');
const path = require('path');
const validator = require(path.resolve('output/gltf-validation/node_modules/gltf-validator'));
(async () => {
  for (const file of process.argv.slice(2)) {
    const result = await validator.validateBytes(new Uint8Array(fs.readFileSync(file)), {uri: file, maxIssues: 500});
    fs.writeFileSync(file + '.validation.json', JSON.stringify(result, null, 2));
    console.log(file, JSON.stringify({errors: result.issues.numErrors, warnings: result.issues.numWarnings,
      infos: result.issues.numInfos, hints: result.issues.numHints}));
    if (result.issues.numErrors || result.issues.numWarnings) process.exitCode = 1;
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
