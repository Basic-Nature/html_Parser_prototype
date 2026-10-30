/* W24DON2D pure snapshot contract. Browser + Node; no network, payments, or I/O. */
(function(root, factory) {
  const value = factory();
  if (typeof module === 'object' && module.exports) module.exports = value;
  if (root) root.ElectionPulseSupportSnapshotContract = value;
})(typeof window !== 'undefined' ? window : null, function() {
  'use strict';
  const SCHEMA = 'ELECTIONPULSE_SUPPORT_PUBLIC_SNAPSHOT_REV2';
  const funding = ['gross_captured_usd','net_settled_usd','spent_on_project_usd','committed_unspent_usd','unallocated_available_usd'];
  const founder = ['provider_record_payment_subtotal_usd','invoice_paid_status_azure_usd','applied_payment_github_usd','received_payment_godaddy_usd'];
  const month = ['azure_paid_invoice_usd','github_applied_payments_usd','godaddy_paid_receipt_usd','provider_payment_sum_usd'];
  // USD values must be nonnegative, finite, and accurately representable to cents.
  // Reconciliation is performed in integer cents to avoid float drift.
  const cents = v => typeof v === 'number' && Number.isFinite(v) && v >= 0 &&
    Number.isSafeInteger(Math.round(v * 100)) && Math.abs(v * 100 - Math.round(v * 100)) < 0.000001;
  const toCents = v => Math.round(v * 100);
  function validateForPublication(snapshot) {
    const reasons=[];
    if (!snapshot || typeof snapshot !== 'object') return {ok:false,reasons:['SNAPSHOT_MISSING']};
    if (snapshot.schema !== SCHEMA) reasons.push('WRONG_SCHEMA');
    if (snapshot.publish_approved !== true) reasons.push('NOT_APPROVED');
    if (!/^\d{4}-\d{2}-\d{2}$/.test(String(snapshot.snapshot_date||''))) reasons.push('INVALID_DATE');
    if (!/^[A-Za-z0-9_.-]{6,128}$/.test(String(snapshot.source_revision||''))) reasons.push('SOURCE_REVISION_REQUIRED');
    if (!snapshot.approval_evidence || typeof snapshot.approval_evidence !== 'object' ||
        !/^[A-Za-z0-9_.-]{6,128}$/.test(String(snapshot.approval_evidence.record_id||''))) reasons.push('APPROVAL_EVIDENCE_REQUIRED');
    if (!snapshot.founder || !snapshot.contributions || !Array.isArray(snapshot.monthly) || !snapshot.monthly.length) reasons.push('TABLES_MISSING');
    // This is an integrity check, not an authentication or authorization boundary.
    // True publication authority must be enforced by the server/operator release gate.
    if (snapshot.contributions && typeof snapshot.contributions==='object') {
      const values=funding.map(k=>snapshot.contributions[k]);
      if(values.some(v=>!cents(v))) reasons.push('FUNDING_CENTS_INCOMPLETE');
      else {
        const [gross,net,spent,committed,available]=values;
        if (toCents(gross)<toCents(net) || toCents(net)!==toCents(spent)+toCents(committed)+toCents(available)) reasons.push('FUNDING_RECONCILIATION_FAILED');
      }
    }
    if (snapshot.founder && typeof snapshot.founder==='object') {
      if(founder.some(k=>!cents(snapshot.founder[k]))) reasons.push('FOUNDER_CENTS_INCOMPLETE');
      else if (toCents(snapshot.founder.provider_record_payment_subtotal_usd)!==
        toCents(snapshot.founder.invoice_paid_status_azure_usd)+toCents(snapshot.founder.applied_payment_github_usd)+toCents(snapshot.founder.received_payment_godaddy_usd)) reasons.push('FOUNDER_SUBTOTAL_MISMATCH');
    }
    if (Array.isArray(snapshot.monthly)) {
      const periods=new Set();
      for(const row of snapshot.monthly) {
        if(!row || !/^\d{4}-\d{2}$/.test(String(row.period||'')) || periods.has(row.period)) {reasons.push('INVALID_OR_REUSED_PERIOD');break;}
        periods.add(row.period);
        if(month.some(k=>!cents(row[k]))) {reasons.push('MONTHLY_CENTS_INCOMPLETE');break;}
        if(toCents(row.provider_payment_sum_usd)!==toCents(row.azure_paid_invoice_usd)+toCents(row.github_applied_payments_usd)+toCents(row.godaddy_paid_receipt_usd)) {reasons.push('MONTHLY_RECONCILIATION_FAILED');break;}
      }
    }
    if (snapshot.founder && Array.isArray(snapshot.monthly) && snapshot.monthly.length) {
      const sum=(key)=>snapshot.monthly.every(row=>row && cents(row[key]))
        ? snapshot.monthly.reduce((s,row)=>s+toCents(row[key]),0) : null;
      for (const [totalKey,monthlyKey] of [
        ['invoice_paid_status_azure_usd','azure_paid_invoice_usd'],
        ['applied_payment_github_usd','github_applied_payments_usd'],
        ['received_payment_godaddy_usd','godaddy_paid_receipt_usd'],
      ]) {
        if (cents(snapshot.founder[totalKey]) && sum(monthlyKey)!==null &&
            toCents(snapshot.founder[totalKey])!==sum(monthlyKey)) reasons.push('PROVIDER_MONTHLY_TOTAL_MISMATCH');
      }
    }
    return {ok:reasons.length===0,reasons};
  }
  return Object.freeze({SCHEMA,validateForPublication});
});
