import { describe, expect, it } from 'vitest';
import { parsePublicRegistryPayload } from '../contracts/registry';
import { canSubmitApprovedRegistrySource } from '../state/selectors';
const ut = 'blsrc_v1_267d575a499b910deacc0c4ee660e696aeeab9c162126fe17625239be15f615c';
const md = 'blsrc_v1_fd0d8be9659f0c22e4c03cca55b52742aa809ae8650729e63bc186235cb64837';
const projects = [ut,md].map((registry_source_id,i)=>({registry_source_id, registry_category:'curated',state:i?'Maryland':'Utah',year:'2024',contest:'President',format:'CSV',scope:'statewide'}));
const payload=(ids:string[])=>({contract:'ballot_lens_public_registry_v2',count:2,execution_enabled:ids.length>0,execution_source_ids:ids,sources:projects});
describe('SOURCE2 public authority v2',()=>{
 it('selects only primary by default and permits both when explicitly projected',()=>{
  const a=parsePublicRegistryPayload(payload([ut])); const b=parsePublicRegistryPayload(payload([ut,md]));
  expect(canSubmitApprovedRegistrySource(a,a.sources[0])).toBe(true);
  expect(canSubmitApprovedRegistrySource(a,a.sources[1])).toBe(false);
  expect(canSubmitApprovedRegistrySource(b,b.sources[1])).toBe(true);
  expect(b.execution_source_id).toBeNull();
 });
 it('rejects duplicate, unknown or excessive execution IDs',()=>{
  for(const ids of [[ut,ut],[ut,'blsrc_v1_'+'a'.repeat(64)],[ut,md,ut]]) expect(()=>parsePublicRegistryPayload(payload(ids))).toThrow();
 });
 it('retains fail-closed v1 compatibility',()=>{
  const a=parsePublicRegistryPayload({contract:'ballot_lens_public_registry_v1',count:2,execution_enabled:true,execution_source_id:ut,sources:projects});
  expect(canSubmitApprovedRegistrySource(a,a.sources[0])).toBe(true);
  expect(canSubmitApprovedRegistrySource(a,a.sources[1])).toBe(false);
 });
});
