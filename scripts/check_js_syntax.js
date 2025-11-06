#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

function findJsFiles(dir) {
  const results = [];
  const items = fs.readdirSync(dir, { withFileTypes: true });
  for (const it of items) {
    const p = path.join(dir, it.name);
    if (it.isDirectory()) {
      if (it.name === 'node_modules' || it.name === 'vendor') continue;
      results.push(...findJsFiles(p));
    } else if (it.isFile() && it.name.endsWith('.js')) {
      results.push(p);
    }
  }
  return results;
}

function checkFile(file) {
  const src = fs.readFileSync(file, 'utf8');
  try {
    // Try to parse the file by wrapping in a Function
    // This detects syntax errors similar to evaluating the script
    new Function(src);
    return null;
  } catch (err) {
    return err;
  }
}

function main() {
  const base = path.resolve(__dirname, '..');
  const target = path.join(base, 'webapp', 'static', 'js');
  if (!fs.existsSync(target)) {
    console.error('Target directory not found:', target);
    process.exit(2);
  }
  const files = findJsFiles(target);
  let hasError = false;
  for (const f of files) {
    const err = checkFile(f);
    if (err) {
      console.error('Syntax error in', f);
      console.error(err && err.stack ? err.stack : String(err));
      console.error('---');
      hasError = true;
    }
  }
  if (hasError) {
    console.error('Syntax checks failed.');
    process.exit(1);
  }
  console.log('All JS files parse without syntax errors.');
}

main();
