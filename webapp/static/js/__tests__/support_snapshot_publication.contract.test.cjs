/* W24DON2D permanent candidate: node --test, no external services. */
'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const contract=require('../support_snapshot_contract.js');
function good() {
  return {
    schema:contract.SCHEMA, snapshot_date:'2026-10-27',source_revision:'snapshot-rev-001',
    publish_approved:true,approval_evidence:{record_id:'approved-001'},
    founder:{provider_record_payment_subtotal_usd:9000,invoice_paid_status_azure_usd:6000,
      applied_payment_github_usd:2000,received_payment_godaddy_usd:1000},
    contributions:{gross_captured_usd:1000,net_settled_usd:900,spent_on_project_usd:300,
      committed_unspent_usd:200,unallocated_available_usd:400},
    monthly:[{period:'2026-10',azure_paid_invoice_usd:6000,github_applied_payments_usd:2000,
      godaddy_paid_receipt_usd:1000,provider_payment_sum_usd:9000}]
  };
}
test('unapproved content never qualifies for publication',()=>{const s=good();s.publish_approved=false;assert.equal(contract.validateForPublication(s).ok,false);});
test('approved synthetic ledger with provenance reconciles',()=>assert.equal(contract.validateForPublication(good()).ok,true));
test('missing evidence rejects',()=>{const s=good();s.approval_evidence=null;assert.equal(contract.validateForPublication(s).ok,false);});
test('unknown funds are not zero',()=>{const s=good();s.contributions.net_settled_usd=null;assert.equal(contract.validateForPublication(s).ok,false);});
test('gross and net must reconcile',()=>{const s=good();s.contributions.gross_captured_usd=700;assert.equal(contract.validateForPublication(s).ok,false);});
test('spent committed and unallocated cannot double count',()=>{const s=good();s.contributions.committed_unspent_usd=300;assert.equal(contract.validateForPublication(s).ok,false);});
test('founder provider subtotal must reconcile',()=>{const s=good();s.founder.provider_record_payment_subtotal_usd=10000;assert.equal(contract.validateForPublication(s).ok,false);});
test('monthly category subtotal must reconcile',()=>{const s=good();s.monthly[0].provider_payment_sum_usd=10000;assert.equal(contract.validateForPublication(s).ok,false);});
test('duplicate month not accepted',()=>{const s=good();s.monthly.push({...s.monthly[0]});assert.equal(contract.validateForPublication(s).ok,false);});
test('partial cents ledger not misrepresented as complete',()=>{const s=good();s.monthly[0].github_applied_payments_usd=null;assert.equal(contract.validateForPublication(s).ok,false);});
test('wrong revision not accepted',()=>{const s=good();s.schema='REV1';assert.equal(contract.validateForPublication(s).ok,false);});
