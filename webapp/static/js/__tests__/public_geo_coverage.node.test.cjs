'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const geo = require('../public_geo_coverage.js');
function payload(state = 'Texas') {
  return {
    contract:'canonical_coverage_v1', authority:'canonical_production', data_source:'canonical',
    pagination:{limit:10,offset:0,total_races:1,has_more:false},
    active_filters:{state,year:null,contest:null},
    semantic_contract:{precinct_null:'preserved',lineage:'not_inferred'},
    items:[{state,year:2024,contest:'Test fixture — not real election data',
      result_count:3,jurisdiction_count:2,jurisdiction_scope_rows:3,
      precinct_scope_rows:0, verification_status:'approved',
      official_source_relationship:'not_established'}],
  };
}
test('valid state selector is one normalized 2-letter code',()=>{
  assert.equal(geo.normalizeState('tx'),'TX');
  assert.equal(geo.normalizeState(' TX '),'TX');
  for (const bad of ['', 'TEX', 'T2', '../TX', 'TX&role=admin', 'ZZ', null])
    assert.equal(geo.normalizeState(bad),null);
});
test('routes contain state only, no contributor or execution authority',()=>{
  assert.equal(geo.coverageUrl('TX'),'/api/data_framework/canonical_coverage?state=Texas&limit=10&offset=0');
  assert.equal(geo.worklistUrl('TX'),'/worklist?state=TX');
  assert.equal(geo.ballotLensUrl('TX'),'/ballot_lens?geo_state=TX');
  assert.equal(geo.ballotLensUrl('not-valid'),'/ballot_lens');
  assert.equal(geo.coverageUrl('TX#hack'),null);
});
test('accept valid jurisdiction-only coverage without asserting precinct data',()=>{
  const data = geo.parseCoverage(payload(),'TX');
  assert.equal(data.totalRaces,1);
  assert.equal(data.items[0].jurisdictionRows,3);
  assert.equal(data.items[0].precinctRows,0);
  assert.ok(!JSON.stringify(data).includes('source_url'));
});
test('reject cross-state substitution',()=>{
  assert.throws(()=>geo.parseCoverage(payload('AZ'),'TX'));
  const p=payload(); p.items[0].state='Arizona';
  assert.throws(()=>geo.parseCoverage(p,'TX'));
});
test('reject mutation/provenance claims and impossible pagination',()=>{
  for (const key of ['contract','authority','data_source']) {
    const p=payload(); p[key]='attacker-controlled'; assert.throws(()=>geo.parseCoverage(p,'TX'));
  }
  const p=payload(); p.pagination.total_races=0;
  assert.throws(()=>geo.parseCoverage(p,'TX'));
  const q=payload(); q.items[0].official_source_relationship='official_verified';
  assert.throws(()=>geo.parseCoverage(q,'TX'));
  const r=payload(); r.items[0].precinct_scope_rows=1;
  assert.throws(()=>geo.parseCoverage(r,'TX'));
});
test('handles true zero records distinctly from missing coverage',()=>{
  const p=payload();p.items=[];p.pagination.total_races=0;
  const parsed=geo.parseCoverage(p,'TX');assert.deepEqual(parsed.items,[]);
  assert.equal(parsed.totalRaces,0);
});
test('status does not elevate noncanonical or untrusted strings into a link',()=>{
  const p=payload();p.items[0].contest='<script>alert(1)</script>';
  const x=geo.parseCoverage(p,'TX');
  assert.equal(x.items[0].contest,'<script>alert(1)</script>'); // DOM must use textContent
});

function mockDOM(fetchImpl) {
  function element() {
    return {
      textContent:'', attrs: Object.create(null), hidden:false, children:[],
      setAttribute(name,value){this.attrs[name]=value;},
      replaceChildren(){this.children=[];},
      appendChild(child){this.children.push(child);},
      addEventListener(name,handler){this.listeners ??= {};this.listeners[name]=handler;},
    };
  }
  const selectors=['[data-ep-coverage-status]','[data-ep-coverage-list]',
    '[data-ep-worklist-link]','[data-ep-lens-link]','[data-ep-coverage-title]'];
  const nodes=Object.fromEntries(selectors.map(k=>[k,element()]));
  const section={dataset:{hideWhenUnselected:'false'},hidden:false,children:[],
    appendChild(child){this.children.push(child);},querySelector(k){return nodes[k]}};
  const win={document:{createElement:element},fetch:fetchImpl};
  return {section,nodes,win};
}
test('mount only GETs, renders genuine response as text, links preserve state',async()=>{
  const calls=[];
  const dom=mockDOM(async(url,options)=>{
    calls.push({url,options});
    return {ok:true,status:200,json:async()=>payload()};
  });
  const control=geo.mount(dom.section,dom.win);
  await control.select('TX');
  assert.equal(calls.length,1);
  assert.equal(calls[0].options.method,'GET');
  assert.equal(calls[0].options.credentials,'same-origin');
  assert.equal(dom.section.dataset.uiState,'ready');
  assert.equal(dom.nodes['[data-ep-coverage-list]'].children.length,1);
  assert.equal(dom.nodes['[data-ep-worklist-link]'].attrs.href,'/worklist?state=TX');
  assert.equal(dom.nodes['[data-ep-lens-link]'].attrs.href,'/ballot_lens?geo_state=TX');
  await control.select('TX');
  assert.equal(calls.length,1,'same-state cache avoids a repeat database read');
  control.clear();
  assert.equal(dom.section.dataset.uiState,'idle');
  assert.equal(dom.nodes['[data-ep-coverage-list]'].children.length,0);
});
test('restricted status is not mislabeled as no election coverage',async()=>{
  const dom=mockDOM(async()=>({ok:false,status:403}));
  const control=geo.mount(dom.section,dom.win);
  await control.select('AZ');
  assert.equal(dom.section.dataset.uiState,'restricted');
  assert.match(dom.nodes['[data-ep-coverage-status]'].textContent,/restricted/);
});
test('invalid geographic request makes zero API requests',async()=>{
  let calls=0;
  const dom=mockDOM(async()=>{calls++;throw new Error('should not run')});
  const control=geo.mount(dom.section,dom.win);
  await control.select('TX&privileged=true');
  assert.equal(calls,0);
  assert.equal(dom.section.dataset.uiState,'idle');
});
test('out-of-order response cannot overwrite later state',async()=>{
  let unblockTX;
  const txPending=new Promise(resolve=>{unblockTX=resolve});
  const dom=mockDOM(async(url)=>{
    if(url.includes('state=Texas')) return txPending;
    return {ok:true,status:200,json:async()=>payload('Arizona')};
  });
  const control=geo.mount(dom.section,dom.win);
  const first=control.select('TX');
  await control.select('AZ');
  unblockTX({ok:true,status:200,json:async()=>payload('Texas')});
  await first;
  assert.equal(dom.section.dataset.uiState,'ready');
  assert.equal(dom.nodes['[data-ep-coverage-title]'].textContent,'Published coverage — Arizona (AZ)');
});

test('S1R keeps canonical database labels separate from 56 geographic selector codes',()=>{
  assert.equal(geo.canonicalStateLabel('TX'), 'Texas');
  assert.equal(geo.canonicalStateLabel('az'), 'Arizona');
  assert.equal(geo.canonicalStateLabel('DC'), 'District of Columbia');
  assert.equal(geo.canonicalStateLabel('ZZ'), null);
  assert.equal(geo.coverageUrl('NY'), '/api/data_framework/canonical_coverage?state=New+York&limit=10&offset=0');
  assert.equal(geo.worklistUrl('TX'), '/worklist?state=TX');
  assert.equal(geo.ballotLensUrl('TX'), '/ballot_lens?geo_state=TX');
});
test('S1R fails closed on stale code-scoped response and other-state substitution',()=>{
  const wrongFilter=payload(); wrongFilter.active_filters.state='TX';
  assert.throws(()=>geo.parseCoverage(wrongFilter,'TX'));
  const wrongRecord=payload(); wrongRecord.items[0].state='Arizona';
  assert.throws(()=>geo.parseCoverage(wrongRecord,'TX'));
  const rawCode=payload(); rawCode.items[0].state='TX';
  assert.throws(()=>geo.parseCoverage(rawCode,'TX'));
  const correct=geo.parseCoverage(payload(),'TX');
  assert.equal(correct.state,'TX');
  assert.equal(correct.canonicalState,'Texas');
});

function facets(state = 'Texas', entries = [
  { name: 'Sample Reporting Scope Alpha', type: 'reporting_unit' },
  { name: 'Sample District Beta', type: 'district' },
]) {
  return { contract: 'canonical_facets_v1', authority: 'canonical_production',
    data_source: 'canonical', active_filters: { state, year: null, jurisdiction: null, contest: null },
    semantic_contract: {facet_mode:'self_excluding', lineage:'not_inferred'},
    states: ['Texas'], years: [2024], contests: ['fixture contest'], jurisdictions: entries };
}
test('S2 builds one same-origin, bounded state facets GET using canonical label', () => {
  assert.equal(geo.facetUrl('TX'), '/api/data_framework/canonical_facets?state=Texas');
  assert.equal(geo.facetUrl('NY'), '/api/data_framework/canonical_facets?state=New+York');
  assert.equal(geo.facetUrl('ZZ'), null);
});
test('S2 parses reporting_unit without inventing county or precinct type', () => {
  const parsed = geo.parseJurisdictionFacets(facets(), 'TX');
  assert.equal(parsed.items.length, 2);
  assert.equal(parsed.items[1].type, 'reporting_unit');
  assert.equal(parsed.canonicalState, 'Texas');
});
test('S2 rejects cross-state, authority, tampered type, oversized, and duplicate facets', () => {
  const variants = [];
  const a = facets('Arizona'); variants.push(a);
  const b = facets(); b.authority = 'attacker'; variants.push(b);
  const c = facets(); c.semantic_contract.lineage = 'verified'; variants.push(c);
  const d = facets(); d.jurisdictions[0].type = 'county<script>'; variants.push(d);
  const e = facets(); e.jurisdictions = Array(5001).fill({name:'X', type:'reporting_unit'}); variants.push(e);
  const f = facets(); f.jurisdictions.push({name:'Sample District Beta',type:'district'}); variants.push(f);
  for (const val of variants) assert.throws(() => geo.parseJurisdictionFacets(val,'TX'));
});
test('S2 deep link contains state and exact selected jurisdiction only', () => {
  assert.equal(geo.dataFrameworkJurisdictionUrl('TX','Sample Reporting Scope Alpha'),
    '/data_framework?state=Texas&jurisdiction=Sample+Reporting+Scope+Alpha');
  assert.equal(geo.dataFrameworkJurisdictionUrl('TX','Other & <type>'),
    '/data_framework?state=Texas&jurisdiction=Other+%26+%3Ctype%3E');
  assert.equal(geo.dataFrameworkJurisdictionUrl('TX','  leading'),null);
  assert.equal(geo.dataFrameworkJurisdictionUrl('ZZ','value'),null);
});
test('S2 lazy jurisdiction browser does not fetch until the visitor requests it',async() => {
  const calls=[];
  const dom=mockDOM(async(url,options)=>{
    calls.push({url,options});
    return {ok:true,status:200,json:async()=>url.includes('canonical_facets')?facets():payload()};
  });
  const control=geo.mount(dom.section,dom.win);
  await control.select('TX');
  assert.equal(calls.length,1);
  assert.ok(control.jurisdictions);
  assert.equal(control.jurisdictions.elements.browse.disabled,false);
  await control.jurisdictions.load();
  assert.equal(calls.length,2);
  assert.equal(calls[1].url,geo.facetUrl('TX'));
  assert.equal(calls[1].options.method,'GET');
  assert.equal(calls[1].options.credentials,'same-origin');
  assert.match(control.jurisdictions.elements.status.textContent,/2 jurisdiction entries/);
  const first=control.jurisdictions.elements.select.children[1];
  assert.match(first.textContent,/District/);
  const second=control.jurisdictions.elements.select.children[2];
  assert.match(second.textContent,/Reporting unit/);
  control.jurisdictions.choose('1');
  assert.equal(control.jurisdictions.elements.link.hidden,false);
  assert.match(control.jurisdictions.elements.link.attrs.href,/state=Texas/);
  control.jurisdictions.choose('123456');
  assert.equal(control.jurisdictions.elements.link.hidden,true);
  await control.jurisdictions.load();
  assert.equal(calls.length,2,'same-state facets cache');
});
test('S2 controls hide stale jurisdiction links immediately on geographic change',async()=>{
  const calls=[];
  const dom=mockDOM(async(url)=>{calls.push(url);return {ok:true,status:200,json:async()=>
    url.includes('canonical_facets')?facets():payload()};});
  const control=geo.mount(dom.section,dom.win);
  await control.select('TX');
  await control.jurisdictions.load();
  control.jurisdictions.choose('0');
  assert.equal(control.jurisdictions.elements.link.hidden,false);
  await control.select('AZ');
  assert.equal(control.jurisdictions.elements.link.hidden,true);
  assert.equal(control.jurisdictions.elements.select.disabled,true);
  assert.equal(calls.filter(v=>v.includes('canonical_facets')).length,1);
});
test('S2 malformed facets and HTTP restrictions are not misreported as zero',async()=>{
  const dom=mockDOM(async(url)=>({ok:url.includes('coverage'),status:url.includes('coverage')?200:403,
    json:async()=>payload()}));
  const control=geo.mount(dom.section,dom.win);
  await control.select('TX');
  await control.jurisdictions.load();
  assert.match(control.jurisdictions.elements.status.textContent,/restricted/);
  assert.equal(control.jurisdictions.elements.select.disabled,true);
  const second=mockDOM(async(url)=>({ok:true,status:200,json:async()=>
    url.includes('canonical_facets')?facets('Arizona'):payload()}));
  const control2=geo.mount(second.section,second.win);
  await control2.select('TX');
  await control2.jurisdictions.load();
  assert.match(control2.jurisdictions.elements.status.textContent,/could not be validated/);
});
test('S2 earlier facet response cannot overwrite later geographic context',async()=>{
  let resolve;
  const slow = new Promise(done=>{resolve=done});
  const dom=mockDOM(async(url)=>{
    if(url.includes('canonical_facets'))return slow;
    return {ok:true,status:200,json:async()=>payload('Texas')};
  });
  const control=geo.mount(dom.section,dom.win);
  await control.select('TX');
  const pending=control.jurisdictions.load();
  control.clear();
  resolve({ok:true,status:200,json:async()=>facets()});
  await pending;
  assert.equal(control.jurisdictions.elements.select.disabled,true);
  assert.equal(control.jurisdictions.elements.link.hidden,true);
  assert.match(control.jurisdictions.elements.status.textContent,/Select a state/);
});

// W24 S2R: live Texas canonical facets contain one case-insensitive name/type
// collision. Preserve both distinct published labels, reject exact replay.
test('S2R preserves case-variant reporting units without conflating types or erasing entries', () => {
  const rows = [
    {name: 'Sample Reporting Unit', type: 'reporting_unit'},
    {name: 'SAMPLE REPORTING UNIT', type: 'reporting_unit'},
    {name: 'Sample Reporting Unit', type: 'district'},
  ];
  const parsed = geo.parseJurisdictionFacets(facets('Texas', rows), 'TX');
  assert.equal(parsed.items.length, 3);
  assert.equal(parsed.caseVariantEntries, 1);
  assert.deepEqual(new Set(parsed.items.map(x => x.name)),
    new Set(['Sample Reporting Unit', 'SAMPLE REPORTING UNIT']));
  assert.equal(parsed.items.filter(x => x.type === 'reporting_unit').length, 2);
  assert.ok(parsed.items.every(x => x.type === 'district' || x.type === 'reporting_unit'));
});
test('S2R exact duplicate remains fail-closed and cross-state protection remains active', () => {
  const exact = facets('Texas', [
    {name: 'Sample Reporting Unit', type: 'reporting_unit'},
    {name: 'Sample Reporting Unit', type: 'reporting_unit'},
  ]);
  assert.throws(() => geo.parseJurisdictionFacets(exact,'TX'), /Duplicate exact/);
  const wrongState = facets('Arizona', [
    {name: 'Sample Reporting Unit', type: 'reporting_unit'},
    {name: 'SAMPLE REPORTING UNIT', type: 'reporting_unit'},
  ]);
  assert.throws(() => geo.parseJurisdictionFacets(wrongState,'TX'));
});
test('S2R shows an explicit capitalization-variant note and both live-style options', async () => {
  const aliases = facets('Texas', [
    {name: 'Sample Reporting Unit', type: 'reporting_unit'},
    {name: 'SAMPLE REPORTING UNIT', type: 'reporting_unit'},
  ]);
  const dom = mockDOM(async (url) => ({ok:true,status:200,
    json:async()=> url.includes('canonical_facets') ? aliases : payload()}));
  const control = geo.mount(dom.section, dom.win);
  await control.select('TX');
  await control.jurisdictions.load();
  assert.equal(control.jurisdictions.elements.select.disabled, false);
  assert.equal(control.jurisdictions.elements.select.children.length, 3);
  assert.match(control.jurisdictions.elements.status.textContent, /2 jurisdiction entries/);
  assert.match(control.jurisdictions.elements.status.textContent, /1 capitalization-variant entry is preserved/);
  assert.ok(control.jurisdictions.elements.select.children.some(x => x.textContent.startsWith('SAMPLE REPORTING UNIT')));
  assert.ok(control.jurisdictions.elements.select.children.some(x => x.textContent.startsWith('Sample Reporting Unit')));
  control.jurisdictions.choose('0');
  assert.equal(control.jurisdictions.elements.link.hidden, false);
  assert.match(control.jurisdictions.elements.link.attrs.href, /jurisdiction=/);
});

// W24 S2P: full source labels remain available behind an intentional disclosure.
test('S2P collapses all race details on state load, preserves every recorded row', async () => {
  const p = payload();
  p.items = [p.items[0], {...p.items[0], year: 2022, contest:'Test fixture second race'}];
  p.pagination.total_races = 2;
  const dom = mockDOM(async()=>({ok:true,status:200,json:async()=>p}));
  const ctrl = geo.mount(dom.section,dom.win);
  await ctrl.select('TX');
  const detail = dom.section.children.find(v=>v.className==='ep-public-coverage__details');
  assert.ok(detail);
  assert.equal(detail.hidden,false);
  assert.equal(detail.open,false);
  assert.match(detail.children[0].textContent,/View 2 published race summaries/);
  assert.equal(dom.nodes['[data-ep-coverage-list]'].children.length,2);
  assert.match(dom.nodes['[data-ep-coverage-status]'].textContent,/complete precinct coverage is not established/);
  assert.ok(!dom.nodes['[data-ep-coverage-status]'].textContent.includes('verified_production'));
  ctrl.clear();
  assert.equal(detail.hidden,true);
});

test('S2P recorded publication label never means independent source attestation', () => {
  assert.equal(geo.recordedStatusLabel('verified_production'),'marked verified in production');
  assert.equal(geo.recordedStatusLabel('approved'),'marked approved');
  assert.equal(geo.recordedStatusLabel('<img src=x>'),'not available');
});

test('S2P filters jurisdiction list without another GET and with correct original index', async()=>{
  const rows = Array.from({length:100},(_,i)=>({name:`Published Area ${String(i).padStart(3,'0')}`,type:'reporting_unit'}));
  rows.push({name:'Jeff Davis',type:'reporting_unit'}, {name:'JEFF DAVIS',type:'reporting_unit'});
  let facetsCalls=0;
  const dom=mockDOM(async(url)=>{
    if (url.includes('canonical_facets')) { facetsCalls++;return {ok:true,status:200,json:async()=>facets('Texas',rows)}; }
    return {ok:true,status:200,json:async()=>payload()};
  });
  const ctrl=geo.mount(dom.section,dom.win);
  await ctrl.select('TX');
  await ctrl.jurisdictions.load();
  const {search,select,countNote,link,selection}=ctrl.jurisdictions.elements;
  assert.equal(facetsCalls,1);
  assert.equal(select.children.length,41,'forty entries plus placeholder, not 102-entry dropdown');
  assert.match(countNote.textContent,/Showing 40 of 102/);
  search.value='jeFF daVis';
  search.listeners.input();
  assert.equal(select.children.length,3,'preserve capitalization variants');
  assert.match(countNote.textContent,/Showing 2 of 2/);
  assert.match(select.children[1].textContent,/Jeff Davis|JEFF DAVIS/);
  ctrl.jurisdictions.choose(select.children[1].value);
  assert.equal(link.hidden,false);
  assert.match(selection.textContent,/Selected:/);
  assert.match(link.attrs.href,/state=Texas/);
  search.value='Not a published name';
  search.listeners.input();
  assert.equal(link.hidden,true,'search changes invalidate prior selection/link');
  assert.equal(select.disabled,true);
  assert.match(countNote.textContent,/No matching published names/);
  assert.equal(facetsCalls,1,'client-side search never calls the API');
});

test('S2P preserves selected context only while entry remains selected', async()=>{
  const dom=mockDOM(async(url)=>({ok:true,status:200,json:async()=>url.includes('canonical_facets')?facets():payload()}));
  const ctrl=geo.mount(dom.section,dom.win);
  await ctrl.select('TX');
  await ctrl.jurisdictions.load();
  ctrl.jurisdictions.choose('0');
  assert.equal(ctrl.jurisdictions.elements.link.hidden,false);
  await ctrl.select('AZ');
  assert.equal(ctrl.jurisdictions.elements.link.hidden,true);
  assert.equal(ctrl.jurisdictions.elements.search.hidden,true);
  assert.equal(ctrl.jurisdictions.elements.search.value,'');
});
