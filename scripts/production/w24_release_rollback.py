"""Manual, narrowly scoped App Service image rollback to a verified ACR tag.

This never downgrades PostgreSQL schema. Keycloak is disabled before reverting
an application image, keeping existing session data intact for DBA review.
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


def az(*args):
    try:
        r=subprocess.run(['az',*args,'--only-show-errors'],stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL,text=True,timeout=90)
    except Exception:raise RuntimeError('AZURE_UNAVAILABLE') from None
    if r.returncode:raise RuntimeError('AZURE_OPERATION_FAILED')
    return r.stdout


def az_json(*args):
    try:return json.loads(az(*args,'--output','json'))
    except (TypeError,ValueError):raise RuntimeError('AZURE_JSON_INVALID') from None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--deployed-commit',required=True)
    p.add_argument('--previous-tag',required=True)
    p.add_argument('--confirmation',required=True)
    a=p.parse_args()
    out=Path('output/reports/w24_release_rollback.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'result':'HOLD','utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'original_deployed_commit':a.deployed_commit,
            'previous_tag':a.previous_tag,'redis':False,'db_downgrade':False}
    try:
        if not re.fullmatch('[0-9a-f]{40}',a.deployed_commit):
            raise RuntimeError('COMMIT_SHA_INVALID')
        if not re.fullmatch('[0-9a-f]{40}',a.previous_tag):
            raise RuntimeError('PREVIOUS_TAG_INVALID')
        if a.confirmation != 'RESTORE:'+a.deployed_commit+':'+a.previous_tag:
            raise RuntimeError('RESTORE_CONFIRMATION_MISMATCH')
        registry=os.environ.get('ACR_NAME') or ''
        ac_server=(os.environ.get('ACR_LOGIN_SERVER') or '').rstrip('/').lower()
        if not registry or not ac_server:raise RuntimeError('ACR_IDENTITY_MISSING')
        app=az_json('webapp','show','-n',APP,'-g',GROUP)
        cfg=az_json('webapp','config','show','-n',APP,'-g',GROUP)
        if app.get('name')!=APP or app.get('httpsOnly') is not True:
            raise RuntimeError('APP_IDENTITY_DRIFT')
        current=str(cfg.get('linuxFxVersion') or '')
        if a.deployed_commit not in current:
            raise RuntimeError('CURRENT_IMAGE_SHA_DOES_NOT_MATCH_DEPLOYMENT')
        tags=az_json('acr','repository','show-tags','--name',registry,'--repository','ballotlens')
        if not isinstance(tags,list) or a.previous_tag not in tags:
            raise RuntimeError('PREVIOUS_TAG_NOT_RESOLVABLE')
        image=ac_server+'/ballotlens:'+a.previous_tag
        if 'DOCKER|'+image==current:raise RuntimeError('ALREADY_ON_PREVIOUS_IMAGE')
        az('webapp','config','appsettings','set','-n',APP,'-g',GROUP,
           '--settings','ELECTIONPULSE_KEYCLOAK_OIDC_ENABLED=false',
           'ELECTIONPULSE_OIDC_POSTGRES_ENABLED=false','--output','none')
        report['keycloak_disable_requested']=True
        # If the following operation fails, reconciliation must inspect the
        # live image before attempting anything else.
        az('webapp','config','container','set','-n',APP,'-g',GROUP,
           '--container-image-name',image,
           '--container-registry-url','https://'+ac_server,'--output','none')
        report['result']='ROLLBACK_IMAGE_REQUESTED_POSTDEPLOY_VERIFICATION_REQUIRED'
    except Exception as exc:
        report['error_code']=str(exc)[:160]
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print('W24_ROLLBACK_RESULT='+report['result'])
    if report['result']=='HOLD':
        print('W24_ROLLBACK_HOLD='+report.get('error_code','UNKNOWN'))
        return 3
    return 0


if __name__=='__main__':
    raise SystemExit(main())
