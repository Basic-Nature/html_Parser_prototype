'use strict';
// W24UI13C: permanent no-network regressions for public geographic identity.
// Synthetic data only; never evidence that ElectionPulse has published a race.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const geo = require('../public_geo_coverage.js');
const source = fs.readFileSync(path.join(__dirname, '..', 'public_geo_coverage.js'),'utf8');
function facets(jurisdictions) {
  return {contract:'canonical_facets_v1',authority:'canonical_production',data_source:'canonical',
    active_filters:{state:'New York'},semantic_contract:{facet_mode:'self_excluding',lineage:'not_inferred'},jurisdictions};
}
function coverage(items=[], total=0) {
  return {contract:'canonical_coverage_v1',authority:'canonical_production',data_source:'canonical',
    active_filters:{state:'New York'},semantic_contract:{precinct_null:'preserved',lineage:'not_inferred'},
    pagination:{limit:10,offset:0,total_races:total,has_more:items.length<total},items};
}
test('canonical state codes are display context, API uses full labels',()=>{
  assert.equal(geo.normalizeState('ny'),'NY');
  assert.equal(geo.canonicalStateLabel('NY'),'New York');
  assert.match(geo.facetUrl('NY'),/state=New\+York/);
});
test('publishing scope labels never coerce reporting_unit to county',()=>{
  const x=geo.parseJurisdictionFacets(facets([{name:'Albany',type:'reporting_unit'}]),'NY');
  assert.equal(x.items[0].type,'reporting_unit');
});
test('same-name different-type entries remain distinct',()=>{
  const x=geo.parseJurisdictionFacets(facets([{name:'Albany',type:'reporting_unit'},{name:'Albany',type:'district'}]),'NY');
  assert.equal(x.items.length,2);
});
test('case variants remain visible but are not a durable ID',()=>{
  const x=geo.parseJurisdictionFacets(facets([{name:'ALBANY',type:'reporting_unit'},{name:'Albany',type:'reporting_unit'}]),'NY');
  assert.equal(x.caseVariantEntries,1);
});
test('byte-identical facet entries are rejected',()=>{
  assert.throws(()=>geo.parseJurisdictionFacets(facets([{name:'Albany',type:'district'},{name:'Albany',type:'district'}]),'NY'),/Duplicate/);
});
test('malformed official-source assertions are not silently trusted',()=>{
  const b=coverage([{state:'New York',contest:'Example only',year:2024,result_count:0,jurisdiction_count:0,jurisdiction_scope_rows:0,precinct_scope_rows:0,official_source_relationship:'verified'}],1);
  assert.throws(()=>geo.parseCoverage(b,'NY'),/Unsafe/);
});
test('zero-count published summary is not an inferred precinct total',()=>{
  const b=coverage([{state:'New York',contest:'Example only',year:2024,result_count:0,jurisdiction_count:0,jurisdiction_scope_rows:0,precinct_scope_rows:0,official_source_relationship:'not_established'}],1);
  const result=geo.parseCoverage(b,'NY');
  assert.equal(result.items[0].resultCount,0);
});
test('coverage parser retains exact page-size contract 10',()=>{
  const b=coverage(); b.pagination.limit=20;
  assert.throws(()=>geo.parseCoverage(b,'NY'),/contract/);
});
test('jurisdiction URL is name-only and cannot assert type or GEOID',()=>{
  const url=geo.dataFrameworkJurisdictionUrl('NY','Albany');
  assert.match(url,/state=New\+York/);assert.match(url,/jurisdiction=Albany/);
  assert.doesNotMatch(url,/GEOID|geography_id|jurisdiction_type/);
});
test('paging controls are created with semantic navigation and live announcements',()=>{
  assert.match(source,/Published jurisdiction results pages/);
  assert.match(source,/Previous reporting units/);
  assert.match(source,/Next reporting units/);
  assert.match(source,/pageStatus\.setAttribute\('aria-live', 'polite'\)/);
});
test('pagination advances through the filtered matched set using full-array indices',()=>{
  assert.match(source,/matches\.slice\(start, stop\)/);
  assert.match(source,/option\.value = String\(index\)/);
  assert.match(source,/page = 0; renderOptions\(\)/);
});
test('same-name ambiguity withholds the name-only deep link',()=>{
  assert.match(source,/const ambiguity = entry && items\.some/);
  assert.match(source,/link\.hidden = !href/);
  assert.match(source,/The name-only Data Framework link is withheld/);
});
test('all state/navigation links keep existing API names',()=>{
  assert.equal(geo.worklistUrl('NY'),'/worklist?state=NY');
  assert.equal(geo.ballotLensUrl('NY'),'/ballot_lens?geo_state=NY');
});
