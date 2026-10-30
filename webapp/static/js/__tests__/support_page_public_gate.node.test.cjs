/* W24DON2E: public support render must not use static private-preview values. */
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const js = fs.readFileSync(path.join(__dirname, '..', 'support_page.js'), 'utf8');
const contract = require('../support_snapshot_contract.js');
function launch({publicMode,financialFlag,fetchImpl,preview}){
  let handler;
  const values=new Map();
  const el=(id)=>{if(!values.has(id))values.set(id,{textContent:'—',value:'all',firstChild:null,appendChild(){},removeChild(){},setAttribute(){},getAttribute(){return 'gross_captured_usd'}});return values.get(id)};
  const document={
    body:{dataset:{supportPublic:publicMode?'true':'false',financialApproved:financialFlag?'true':'false'}},
    addEventListener(n,fn){if(n==='DOMContentLoaded')handler=fn},
    querySelector(){return el('hero-provider-total')},
    getElementById:el,querySelectorAll(){return[]},
  };
  const window={EP_SUPPORT_SNAPSHOT:preview,ElectionPulseSupportSnapshotContract:contract};
  const context={document,window,fetch:fetchImpl,Intl,Date,console,URL};
  vm.runInNewContext(js,context);
  return {fire:()=>handler(), el};
}
test('public gate OFF: no network and no static private preview leak',async()=>{
  let calls=0;
  const ui=launch({publicMode:true,financialFlag:false,preview:{founder:{provider_record_payment_subtotal_usd:999999}},fetchImpl:async()=>{calls++;throw Error('unexpected')}});
  await ui.fire();
  assert.equal(calls,0);
  assert.equal(ui.el('hero-provider-total').textContent,'Unavailable');
});
test('public gate ON but endpoint 404: no static fallback',async()=>{
  let calls=0;
  const ui=launch({publicMode:true,financialFlag:true,preview:{founder:{provider_record_payment_subtotal_usd:999999}},fetchImpl:async(path,opts)=>{calls++;assert.equal(path,'/support/financial-snapshot');assert.equal(opts.cache,'no-store');return {ok:false,status:404}}});
  await ui.fire();
  assert.equal(calls,1);
  assert.equal(ui.el('hero-provider-total').textContent,'Unavailable');
});
test('public gate ON with invalid financial payload: no number displayed',async()=>{
  const ui=launch({publicMode:true,financialFlag:true,preview:{founder:{provider_record_payment_subtotal_usd:999999}},fetchImpl:async()=>({ok:true,json:async()=>({publish_approved:true})})});
  await ui.fire();
  assert.equal(ui.el('hero-provider-total').textContent,'Unavailable');
});
