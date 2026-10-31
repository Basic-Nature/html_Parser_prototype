'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const api=require(path.resolve(__dirname,'..','public_canonical_explore.js'));

// Synthetic contract fixtures ONLY. No invented votes appear in public previews.
const id='12345678-1234-4abc-8def-1234567890ab';
const row=()=>({id,state:'Texas',year:2024,election_date:null,date_precision:null,
  contest:'Synthetic contest',jurisdiction_name:'Sample reporting unit',
  jurisdiction_type:'reporting_unit',aggregation_scope:'jurisdiction',precinct:null,
  candidate:'Synthetic candidate',party:null,is_write_in:false,total_votes:null,
  vote_components:[]});
const payload=(item=row())=>({contract:'canonical_record_instance_v1',
  authority:'canonical_production',data_source:'canonical',
  identity_semantics:'current_record_instance_only',cross_republication_permalink:false,item});

test('only valid current-instance UUIDs produce same-origin GET paths',()=>{
  assert.equal(api.recordInstanceUrl(id),'/api/ballotlens-database/records/'+id);
  for(const bad of [null,'','synthetic-record-1','../../admin','TX&role=admin',
    '12345678-1234-1234-1234-1234567890ab?role=admin']) {
    assert.equal(api.recordInstanceUrl(bad),null);
  }
});
test('valid current record response preserves NULL and recorded jurisdiction type',()=>{
  const result=api.parseRecordInstance(payload(),row());
  assert.equal(result.total_votes,null);
  assert.equal(result.jurisdiction_type,'reporting_unit');
  assert.equal(result.vote_components.length,0);
});
test('numeric zero remains zero, missing methods are not added',()=>{
  const item={...row(),total_votes:0,vote_components:[{vote_method:'Election Day',votes:0}]};
  const result=api.parseRecordInstance(payload(item),item);
  assert.equal(result.total_votes,0);
  assert.deepEqual(result.vote_components,[{vote_method:'Election Day',votes:0}]);
});
test('current-only ID semantics cannot be promoted to permanent permalink',()=>{
  assert.throws(()=>api.parseRecordInstance({...payload(),cross_republication_permalink:true},row()),/contract/);
  assert.throws(()=>api.parseRecordInstance({...payload(),identity_semantics:'permalink'},row()),/contract/);
});
test('record ID, state, jurisdiction, contest and vote total changes fail closed',()=>{
  for(const change of [
    {id:'aaaaaaaa-1234-4abc-8def-1234567890ab'},
    {state:'Arizona'},{jurisdiction_name:'Another unit'},
    {contest:'Another contest'},{total_votes:0},{jurisdiction_type:'county'}
  ]){
    assert.throws(()=>api.parseRecordInstance(payload({...row(),...change}),row()),/identity|allowlist/);
  }
});
test('private keys and methods with hidden columns are rejected',()=>{
  for(const extra of [
    {source_url:'https://unapproved.invalid/secret'},
    {provenance:{internal:true}},
    {reviewer_principal:'private@example.invalid'},
    {vote_components:[{vote_method:'Election Day',votes:0,source_column:'secret'}]}
  ]){
    assert.throws(()=>api.parseRecordInstance(payload({...row(),...extra}),row()),/identity|allowlist/);
  }
});
test('untrusted authority, contract, malformed vote amounts and missing keys rejected',()=>{
  assert.throws(()=>api.parseRecordInstance({...payload(),authority:'warehouse'},row()),/contract/);
  for(const bad of [2.5,Number.MAX_SAFE_INTEGER+1,'1.5']){
    assert.throws(()=>api.parseRecordInstance(payload({...row(),total_votes:bad}),row()),/identity|allowlist/);
  }
  const x=row();delete x.precinct;
  assert.throws(()=>api.parseRecordInstance(payload(x),row()),/identity|allowlist/);
});
