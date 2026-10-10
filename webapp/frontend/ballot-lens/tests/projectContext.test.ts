import {describe,it,expect,vi} from 'vitest';
import {PROJECT_UUID_RX,resolveProjectReturn} from '../services/projectContext';
const id='123e4567-e89b-42d3-a456-426614174000';
describe('Project selector is never itself authority',()=>{
  it('rejects malformed selector',async()=>{
    expect(PROJECT_UUID_RX.test('../external')).toBe(false);
    expect(await resolveProjectReturn('../external')).toBeNull();
  });
  it('requires matching authorized server response',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({id:'deadbeef'})}));
    expect(await resolveProjectReturn(id)).toBeNull();
    vi.unstubAllGlobals();
  });
  it('rejects denied responses',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:false}));
    expect(await resolveProjectReturn(id)).toBeNull();
    vi.unstubAllGlobals();
  });
});
