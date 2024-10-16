// This script does a few things before publishing the package to npm:
// 1. ensures that the "main" field in package.json is set to "augurs.js", which is
//    the same as checking that wasm-pack was run with the "--out-name augurs" flag.
// 2. adds the "snippets/" directory to the files array in package.json.
//    This is needed because of https://github.com/rustwasm/wasm-pack/issues/1206.
// 3. renames the npm package from "augurs-js" to "augurs".
//    Once https://github.com/rustwasm/wasm-pack/issues/949 is fixed we can remove this part.
const fs = require('fs');
const path = require('path');

try {
  const pkgPath = path.join(__dirname, "pkg/package.json");

  // Check if package.json exists
  if (!fs.existsSync(pkgPath)) {
    console.error(`Error: File ${pkgPath} not found.`);
    process.exit(1);
  }

  const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf-8'));

  // Check that the 'main' field is set to "augurs.js".
  if (pkg.main !== 'augurs.js') {
    console.error(`Error: The 'main' field in package.json is not set to "augurs.js". Make sure you passed '--out-name augurs' to 'wasm-pack build'.`);
    process.exit(1);
  }

  // Add snippets to the files array. If no files array exists, create one.
  pkg.files = pkg.files || [];
  if (!pkg.files.includes('snippets/')) {
    pkg.files.push('snippets/');
    console.log('Added "snippets/" to package.json.');
  } else {
    console.log('"snippets/" already exists in package.json.');
  }

  // Rename the npm package from "@bsull/augurs-js" to "@bsull/augurs".
  pkg.name = '@bsull/augurs';
  console.log('Renamed the npm package from "augurs-js" to "augurs".');

  fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
  console.log('Successfully updated package.json.');
} catch (error) {
  console.error(`An error occurred: ${error.message}`);
  process.exit(1);
}
