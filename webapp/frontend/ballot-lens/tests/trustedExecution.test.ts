import {describe,expect,it} from 'vitest';
import {buildTrustedSubmitPayload} from '../services/trustedExecution';
describe('F2-H trusted execution projection',()=>{
 it('uses existing trusted payload shapes',()=>{expect(buildTrustedSubmitPayload({runMode:'trusted_url',displayLabel:'x',url:'https://example.gov/results'})).toEqual({trusted_run_mode:'trusted_url',direct_urls:['https://example.gov/results']});expect(buildTrustedSubmitPayload({runMode:'manual_upload',displayLabel:'x.csv',uploadPath:'x.csv',uploadName:'x.csv'})).toEqual({trusted_run_mode:'manual_upload',file_source:'uploads',manual_upload_path:'x.csv',manual_upload_name:'x.csv'})});
});
