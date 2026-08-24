'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..', '..', '..');
const modulePath = path.join(
  repo,
  'webapp',
  'static',
  'js',
  'pipeline_inspection_panel.js'
);

const panel = require(modulePath);

function fixture() {
  return {
    contract: 'pipeline_inspection_socket_v1',
    authority: {
      canonical: false,
      transport: 'same_run_socket',
    },
    session_id: 'session-27',
    inspection: {
      authority: {
        canonical: false,
        inspection: 'noncanonical_parser_evidence',
        write_kind: 'none',
      },
      automatic_timestamp: false,
      completeness: {
        expected_count: null,
        is_complete: null,
        missing_count: null,
        notes: [],
        null_value_count: null,
        observed_count: null,
        state: 'unknown',
      },
      contract: 'pipeline_inspection_v1',
      headers_included: false,
      rows_included: false,
      source_provenance: {
        artifact_id: null,
        evidence_ref: 'fixture://c2g27',
        location: null,
        source_metadata_included: false,
        source_sha256: 'a'.repeat(64),
        source_type: 'csv',
        source_uri_included: false,
      },
      stage: 'interpreted',
      summary: {
        header_count: 1,
        row_count: 3,
        transformation_count: 1,
        warning_count: 0,
      },
      transformations: [
        {
          confidence: null,
          details: {
            after_header: 'Election Day',
            before_header: 'election day',
            vote_value_mutation: false,
          },
          evidence_refs: [],
          from_stage: 'interpreted',
          operation: 'vote_method_header_canonicalization',
          rule_source: (
            'Context_Integration.Context_Library.constants.'
            + 'BALLOT_NAME_CANON_MAP'
          ),
          sequence: 0,
          to_stage: 'interpreted',
        },
      ],
      warnings: [],
    },
  };
}

const view = panel.buildViewModel(fixture());

assert.strictEqual(panel.CUSTOM_EVENT, 'pipeline:inspection');
assert.strictEqual(view.authorityLabel, 'NONE');
assert.strictEqual(view.stage, 'interpreted');
assert.strictEqual(view.sourceType, 'csv');
assert.strictEqual(view.rowCount, 3);
assert.strictEqual(view.headerCount, 1);
assert.strictEqual(view.completeness.state, 'unknown');
assert.strictEqual(view.completeness.expectedCount, null);
assert.strictEqual(view.transformations.length, 1);

const transformation = view.transformations[0];
assert.strictEqual(transformation.beforeHeader, 'election day');
assert.strictEqual(transformation.afterHeader, 'Election Day');
assert.strictEqual(transformation.confidence, null);
assert.strictEqual(transformation.voteValueMutation, false);
assert.strictEqual(
  transformation.ruleSource,
  'Context_Integration.Context_Library.constants.BALLOT_NAME_CANON_MAP'
);

const canonical = fixture();
canonical.authority.canonical = true;
assert.throws(
  () => panel.buildViewModel(canonical),
  /noncanonical/
);

const rawRows = fixture();
rawRows.inspection.rows_included = true;
assert.throws(
  () => panel.buildViewModel(rawRows),
  /raw rows or headers/
);

const sourceUri = fixture();
sourceUri.inspection.source_provenance.source_uri_included = true;
assert.throws(
  () => panel.buildViewModel(sourceUri),
  /source URI/
);

const source = fs.readFileSync(modulePath, 'utf8');
assert.strictEqual(source.includes('innerHTML'), false);
assert.strictEqual(source.includes('localStorage'), false);
assert.strictEqual(source.includes('sessionStorage'), false);
assert.strictEqual(source.includes('fetch('), false);
assert.strictEqual(source.includes('.emit('), false);
assert.strictEqual(source.includes('verify-and-promote'), false);
assert.strictEqual(source.includes('JSON.stringify'), false);

console.log('C2G27_NODE_WHY_PANEL_CONTRACT=PASS');