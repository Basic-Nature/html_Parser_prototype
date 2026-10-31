'use strict';
// Off-repository contracts for a frozen candidate. No vote totals or invented public records.
const assert = require('node:assert/strict');
const test = require('node:test');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const root = process.env.EP_UI15A_ROOT || path.resolve(__dirname, '../../../..');
const geo = require(path.join(root, 'webapp/static/js/public_geo_coverage.js'));
const cross = require(path.join(root, 'webapp/static/js/cross_surface_context.js'));
const worklistCode = fs.readFileSync(path.join(root,'webapp/static/js/workflow_public.js'),'utf8');
const template = fs.readFileSync(path.join(root,'webapp/templates/worklist.html'),'utf8');
function linkFor(state, coverage = geo) {
  let href = '';
  const select={ value: state };
  const link={setAttribute(k,v){ assert.equal(k,'href');href=v; }};
  const window={ElectionPulsePublicCoverage:coverage};
  const document={readyState:'loading',addEventListener(){}};
  const ctx={window,document,URL,URLSearchParams,console,setTimeout,clearTimeout};
  vm.runInNewContext(worklistCode,ctx,{timeout:1500,filename:'workflow_public.js'});
  const proto=window.WorkflowPublicSurface?.prototype;
  assert.ok(proto);
  proto.updatePublicGeoLink.call({byId(id){return id==='workflow-filter-state'?select:id==='workflowPublicGeoBallotLink'?link:null;}});
  return href;
}
for(const [input,code] of [['TX','TX'],['Arizona','AZ'],['PR','PR'],['puerto rico','PR'],[' az ','AZ']]) {
  test(`Worklist -> Ballot Lens canonical ${input}`,()=>assert.equal(linkFor(input),'/ballot_lens?geo_state='+code));
}
for(const raw of ['ZZ','',null,'foobar','US','AA','foo/bar','TX&role=admin']) {
  test(`Unknown or untrusted ${String(raw)} has no geo_state`,()=>assert.equal(linkFor(raw),'/ballot_lens'));
}
test('Missing geography mapping fails closed',()=>assert.equal(linkFor('TX',null),'/ballot_lens'));
test('56 distinct code/label mappings',()=>{
 const codes=['TX','AZ','PR','GU','AS','MP','VI','DC','CA','NY'];
 for (const code of codes) {assert.equal(geo.canonicalStateCode(geo.canonicalStateLabel(code)),code);}
 assert.equal(geo.canonicalStateCode('ZZ'),null);
});
test('Data Framework -> Worklist canonical presentation only',()=>{
 const u=cross.buildContextHref('https://www.electionpulse.org/data_framework?state=Texas&year=2024&workflow_pass_id=secret','/worklist','/worklist','https://www.electionpulse.org');
 assert.equal(u,'/worklist?state=TX&year=2024');
});
test('Worklist -> Data Framework canonical presentation only',()=>{
 const u=cross.buildContextHref('https://www.electionpulse.org/worklist?state=PR&year=2026&registry_source_id=private','/data_framework','/data_framework','https://www.electionpulse.org');
 assert.equal(u,'/data_framework?state=Puerto+Rico&year=2026');
});
test('Cross-origin handoff fails closed',()=>assert.equal(cross.buildContextHref('https://attacker.invalid/worklist?state=TX','/data_framework','/data_framework','https://www.electionpulse.org'),null));
test('Worklist loads canonical geography before workflow script and maintains CSP nonce',()=>{
 let a=template.indexOf("filename='js/public_geo_coverage.js'");
 let b=template.indexOf("filename='js/workflow_public.js'");
 assert.ok(a>=0&&b>a);
 assert.match(template,/nonce="\{\{ g\.csp_nonce \}\}"/);
 assert.match(template,/w24ui15a-geo-link-[0-9a-f]{12}/);
});
test('Worklist remains public view only',()=>{
 assert.match(template,/public surface remains view-only/i);
 assert.match(worklistCode,/method:\s*'GET'/);
});
