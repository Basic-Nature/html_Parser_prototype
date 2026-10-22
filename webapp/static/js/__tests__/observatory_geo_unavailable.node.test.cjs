'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const sourcePath = process.env.W24UI4D_SOURCE || path.resolve(__dirname, '..', 'observatory.js');
if (!sourcePath || !fs.existsSync(sourcePath)) throw new Error('W24UI4D_SOURCE_MISSING');
const source = fs.readFileSync(sourcePath, 'utf8');

function element(tag = 'div') {
  const classNames = new Set();
  return {
    tagName:tag, dataset: {}, children: [], listeners: {}, textContent:'', value:'', disabled:false,
    classList:{ add(...xs){ xs.forEach(x=>classNames.add(x)); }, remove(...xs){ xs.forEach(x=>classNames.delete(x)); }, contains(x){return classNames.has(x);}, toggle(x,on){if(on)classNames.add(x);else classNames.delete(x);} },
    addEventListener(event,fn){(this.listeners[event] ||= []).push(fn);},
    setAttribute(name,v){(this.attrs ||= {})[name] = String(v);},
    replaceChildren(...nodes){this.children=nodes;},
    append(...nodes){this.children.push(...nodes);},
    appendChild(node){this.children.push(node);return node;},
    insertBefore(node){this.children.unshift(node);return node;},
    querySelector(){return null;}, querySelectorAll(){return[];},
  };
}

async function run({sourceName='configured', fetchCase='unavailable'}) {
  const nodes = Object.fromEntries([
    'observatoryShell','observatoryStage','observatoryMapStage','observatoryNationalMap',
    'observatoryMapPaths','observatoryMapLoading','observatoryJurisdictionPicker',
    'observatoryMapSignal','observatoryContextTitle','observatoryContextCopy',
    'observatoryContextCode','observatoryContextStatus','observatoryLensTitle',
    'observatoryLensCopy','observatoryPhrase','observatoryProjectLink',
  ].map(k=>[k,element()]));
  nodes.observatoryNationalMap.dataset.mapSource = sourceName === 'configured' ? '/static/assets/observatory/national_geography_2025.json' : '';
  nodes.observatoryJurisdictionPicker.disabled = true;
  const placeholder=element('option'); placeholder.textContent='Loading geographic choices…';
  nodes.observatoryJurisdictionPicker.children=[placeholder];
  const requests=[];
  const document = {
    readyState:'complete',
    getElementById:k=>nodes[k] ?? null,
    querySelector:()=>null, querySelectorAll:()=>[],
    createElement:element,
  };
  const window = {
    matchMedia:()=>({matches:true}),
    ElectionPulsePublicCoverage:{normalizeState:()=>null},
    location:{href:'http://127.0.0.1:5000/',search:'',pathname:'/',hash:''},
    history:{state:null,replaceState(){throw new Error('Unexpected URL mutation');}},
  };
  const fetch = async (url,options) => {
    requests.push({url,method:options?.method});
    if(fetchCase==='http-error') return {ok:false,status:503};
    if(fetchCase==='invalid-contract') return {ok:true,status:200,json:async()=>({jurisdiction_count:0,jurisdictions:[],semantics:{}})};
    throw new Error('simulated geography failure');
  };
  const console = {error:()=>{}};
  vm.runInNewContext(source,{document,window,fetch,console,URL,URLSearchParams}, {filename:path.basename(sourcePath), timeout:3000});
  // Flush async response/body parsing without launching a server.
  await new Promise(resolve=>setImmediate(resolve));
  return {nodes,requests};
}
function assertUnavailable({nodes,requests},expectedRequests) {
  const picker=nodes.observatoryJurisdictionPicker;
  const stage=nodes.observatoryMapStage;
  assert.equal(requests.length,expectedRequests,'geography GET must be bounded');
  assert.equal(picker.disabled,true,'unverified geography stays unselectable');
  assert.equal(picker.children.length,1,'no stale/jurisdiction options');
  assert.equal(picker.children[0].value,'','unavailable option cannot represent a state');
  assert.match(picker.children[0].textContent,/unavailable/i,'picker must not remain Loading');
  assert.equal(stage.classList.contains('map-error'),true,'map error state is explicit');
  assert.match(nodes.observatoryMapLoading.textContent,/could not be resolved|not configured/i);
  assert.equal(nodes.observatoryProjectLink.attrs?.href,undefined,'no state/project selection');
}

test('missing geography source updates BOTH map and disabled picker, no fetch',async()=>{
  assertUnavailable(await run({sourceName:'missing'}),0);
});
test('HTTP failure updates BOTH map and disabled picker, bounded GET',async()=>{
  const result=await run({fetchCase:'http-error'});
  assert.deepEqual(result.requests.map(v=>v.method),['GET']);
  assertUnavailable(result,1);
});
test('invalid jurisdiction contract updates BOTH map and disabled picker',async()=>{
  assertUnavailable(await run({fetchCase:'invalid-contract'}),1);
});
test('network error updates BOTH map and disabled picker',async()=>{
  assertUnavailable(await run({fetchCase:'network-error'}),1);
});
