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
  'pipeline_inspection_consumer.js'
);

const consumer = require(modulePath);

class FakeSocket {
  constructor() {
    this.handlers = new Map();
  }

  on(event, handler) {
    this.handlers.set(event, handler);
  }

  off(event, handler) {
    if (this.handlers.get(event) === handler) {
      this.handlers.delete(event);
    }
  }

  trigger(event, payload) {
    const handler = this.handlers.get(event);
    if (!handler) {
      throw new Error(`No handler for ${event}`);
    }
    return handler(payload);
  }
}

function validEnvelope(sessionId, signedValue) {
  return {
    contract: 'pipeline_inspection_socket_v1',
    authority: {
      canonical: false,
      transport: 'same_run_socket',
    },
    session_id: sessionId,
    inspection: {
      contract: 'pipeline_inspection_v1',
      authority: {
        canonical: false,
      },
      stage: 'interpreted',
      rows_included: false,
      headers_included: false,
      transformations: [
        {
          operation: 'vote_method_header_canonicalization',
          confidence: null,
          details: {
            unknown_example: null,
            confirmed_zero_example: 0,
            signed_example: signedValue,
          },
        },
      ],
    },
  };
}

assert.strictEqual(consumer.SOCKET_EVENT, 'pipeline_inspection');

const valid = consumer.validateEnvelope(
  validEnvelope('s-1', -4),
  's-1'
);
assert.strictEqual(valid.ok, true);
assert.strictEqual(
  valid.value.inspection.transformations[0].details.unknown_example,
  null
);
assert.strictEqual(
  valid.value.inspection.transformations[0].details.confirmed_zero_example,
  0
);
assert.strictEqual(
  valid.value.inspection.transformations[0].details.signed_example,
  -4
);

assert.strictEqual(
  consumer.validateEnvelope(validEnvelope('s-1', -4), 's-2').ok,
  false
);

const canonical = validEnvelope('s-1', -4);
canonical.authority.canonical = true;
assert.strictEqual(
  consumer.validateEnvelope(canonical, 's-1').ok,
  false
);

const rawRows = validEnvelope('s-1', -4);
rawRows.inspection.rows = [{ Precinct: 'P-1' }];
assert.strictEqual(
  consumer.validateEnvelope(rawRows, 's-1').ok,
  false
);

const socket = new FakeSocket();
const accepted = [];
const rejected = [];

const runtime = consumer.attach(
  socket,
  () => 's-1',
  {
    maxPerSession: 2,
    maxSessions: 2,
    onAccepted: (envelope) => accepted.push(envelope),
    onRejected: (reason) => rejected.push(reason),
  }
);

socket.trigger('pipeline_inspection', validEnvelope('s-1', -4));
socket.trigger('pipeline_inspection', validEnvelope('s-1', -3));
socket.trigger('pipeline_inspection', validEnvelope('s-1', -2));

assert.strictEqual(accepted.length, 3);
assert.strictEqual(runtime.getSession('s-1').length, 2);
assert.strictEqual(
  runtime.getLatest('s-1').inspection.transformations[0].details.signed_example,
  -2
);

const copy = runtime.getLatest('s-1');
copy.inspection.transformations[0].details.signed_example = 999;
assert.strictEqual(
  runtime.getLatest('s-1').inspection.transformations[0].details.signed_example,
  -2
);

socket.trigger('pipeline_inspection', validEnvelope('wrong-session', -1));
assert.strictEqual(rejected.length, 1);
assert.strictEqual(rejected[0].reason, 'session_mismatch');

runtime.detach();
assert.strictEqual(socket.handlers.has('pipeline_inspection'), false);

const source = fs.readFileSync(modulePath, 'utf8');
assert.strictEqual(source.includes('localStorage'), false);
assert.strictEqual(source.includes('sessionStorage'), false);
assert.strictEqual(source.includes('fetch('), false);
assert.strictEqual(source.includes('.emit('), false);

console.log('C2G26_NODE_CONTRACT=PASS');