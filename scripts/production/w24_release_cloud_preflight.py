"""Read-only Azure pre-deployment checkpoint, executed in GitHub Actions.

No local Azure CLI, no database credentials, and no Redis. Refuses to bless a
rollback tag that the existing ACR cannot resolve at inspection time.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path

APP='BallotLens'
GROUP='BallotLens_group'


def az_json(*args):
    try:
        r=subprocess.run(['az',*args,'--output','json','--only-show-errors'],
                         capture_output=True,text=True,timeout=65)
        if r.returncode:
            raise RuntimeError('AZURE_COMMAND_FAILED')
        return json.loads(r.stdout)
    except Exception:
        raise RuntimeError('AZURE_READ_FAILED') from None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release-commit',required=True)
    args=p.parse_args()
    out=Path('output/reports/w24_release_cloud_preflight.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'result':'HOLD','checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'new_commit':args.release_commit,'mutation':False,'redis':False}
    try:
        app=az_json('webapp','show','-n',APP,'-g',GROUP)
        cfg=az_json('webapp','config','show','-n',APP,'-g',GROUP)
        settings=az_json('webapp','config','appsettings','list','-n',APP,'-g',GROUP)
        mapping={str(x.get('name')):str(x.get('value') or '') for x in settings
                 if isinstance(x,dict) and x.get('name')}
        instance_list=az_json('webapp','list-instances','-n',APP,'-g',GROUP)
        if (app.get('name')!=APP or app.get('state')!='Running'
            or app.get('httpsOnly') is not True):
            raise RuntimeError('APP_IDENTITY_OR_HTTPS_INVALID')
        if not isinstance(instance_list,list) or len(instance_list)!=1:
            raise RuntimeError('APP_SERVICE_SINGLE_INSTANCE_NOT_CONFIRMED')
        if mapping.get('GUNICORN_WORKERS','1')!='1':
            raise RuntimeError('GUNICORN_WORKER_COUNT_NOT_ONE')
        if mapping.get('ELECTIONPULSE_KEYCLOAK_OIDC_ENABLED','false').lower()!='false':
            raise RuntimeError('KEYCLOAK_MUST_BE_OFF_DURING_DEPLOY')
        ac_server=os.environ.get('ACR_LOGIN_SERVER','').lower().rstrip('/')
        registry=os.environ.get('ACR_NAME','')
        if not ac_server or not registry:
            raise RuntimeError('ACR_IDENTITY_NOT_CONFIGURED')
        linux=str(cfg.get('linuxFxVersion') or '')
        match=re.fullmatch(r'DOCKER\|([^/]+)/ballotlens:([A-Za-z0-9_.-]{1,128})',linux)
        if not match or match.group(1).lower()!=ac_server:
            raise RuntimeError('ROLLBACK_IMAGE_REGISTRY_OR_FORMAT_MISMATCH')
        prior_tag=match.group(2)
        if not re.fullmatch('[0-9a-f]{40}', prior_tag) or prior_tag==args.release_commit:
            raise RuntimeError('PREVIOUS_IMAGE_MUST_HAVE_DISTINCT_COMMIT_SHA_TAG')
        tags=az_json('acr','repository','show-tags','--name',registry,'--repository','ballotlens')
        if not isinstance(tags,list) or prior_tag not in tags:
            raise RuntimeError('PRIOR_IMAGE_TAG_NOT_FOUND_IN_ACR')
        report.update({'result':'PASS','previous_image':linux,
                       'previous_tag_present_at_preflight':True,
                       'single_instance':True,'gunicorn_workers':1,
                       'app_running_https':True,
                       'retention_after_inspection':'NOT_GUARANTEED_NO_DELETE_ALLOWED'})
    except Exception as exc:
        report['hold_reason']=str(exc)[:140]
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print('W24_RELEASE_CLOUD_PREFLIGHT='+report['result'])
    if report['result']=='HOLD':
        print('W24_RELEASE_CLOUD_HOLD='+report.get('hold_reason','UNKNOWN'))
        return 3
    return 0


if __name__=='__main__':
    raise SystemExit(main())
