import {readFileSync} from 'fs';
import {createRequire} from 'module';
const require = createRequire(import.meta.url);
const markdownlint = require('markdownlint');

const files = [
  'docs/ML_DEPLOYMENT_CHECKLIST.md',
  'docs/ML_OPTIMIZATION_METRICS.md',
  'docs/ML_OPTIMIZATION_SUMMARY.md',
  'docs/ML_QUALITY_METRICS_SUMMARY.md',
  'docs/ML_QUICKSTART.md',
  'docs/ml_training_data_export.md'
];

files.forEach(f => {
  const result = markdownlint.sync({files: [f]});
  const issues = result[f];
  if (issues && issues.length) {
    console.log(`${f}: ${issues.length} issues`);
    issues.forEach(i => console.log(`  Line ${i.lineNumber}: ${i.ruleNames.join('/')} - ${i.ruleDescription}`));
  } else {
    console.log(`${f}: ✓ CLEAN`);
  }
});
