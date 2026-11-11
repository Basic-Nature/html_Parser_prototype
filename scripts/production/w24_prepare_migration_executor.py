"""Point the EXISTING migration-only App Service at a reviewed release image.

The governed schema controller requires its executor container to carry the
exact GitHub workflow commit. No DB SQL, PostgreSQL credentials, or role
changes occur here. The executor's prior image reference is retained for a
manual migration-host rollback if needed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path

APP='BallotLens-Migration'
PRIMARY='BallotLens'
GROUP='BallotLens_group'
DB_HOST='ballotlens-server.postgres.database.azure.com'
DB_NAME='ballotlens-database'
DB_USER='electionpulse_migration'


def az(*args):
    try:
        r=subprocess.run(['az',*args,'--only-show-errors'],
                         stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                         text=True,timeout=90)
    except Exception:raise RuntimeError('AZURE_COMMAND_UNAVAILABLE') from None
    if r.returncode:raise RuntimeError('AZURE_COMMAND_FAILED')
    return r.stdout


def az_json(*args):
    try:return json.loads(az(*args,'--output','json'))
    except (TypeError,ValueError):raise RuntimeError('AZURE_JSON_INVALID') from None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release-commit',required=True)
    parser.add_argument('--confirmation',required=True)
    args=parser.parse_args()
    report={'timestamp':dt.datetime.now(dt.timezone.utc).isoformat(),
            'result':'HOLD','release_commit':args.release_commit,
            'database_mutation':False,'redis':False}
    out=Path('output/reports/w24_migration_executor_prepare.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not re.fullmatch('[0-9a-f]{40}',args.release_commit):
            raise RuntimeError('INVALID_COMMIT')
        if args.confirmation!='PREPARE_MIGRATION:'+args.release_commit:
            raise RuntimeError('CONFIRMATION_MISMATCH')
        registry=os.environ.get('ACR_NAME') or ''
        server=(os.environ.get('ACR_LOGIN_SERVER') or '').lower().rstrip('/')
        if not registry or not server:raise RuntimeError('ACR_CONFIGURATION_MISSING')
        new_image=server+'/ballotlens:'+args.release_commit
        tags=az_json('acr','repository','show-tags','--name',registry,'--repository','ballotlens')
        if not isinstance(tags,list) or args.release_commit not in tags:
            raise RuntimeError('EXACT_RELEASE_IMAGE_NOT_RESOLVABLE_IN_ACR')
        main=az_json('webapp','config','show','-n',PRIMARY,'-g',GROUP)
        if args.release_commit not in str(main.get('linuxFxVersion') or ''):
            raise RuntimeError('PRIMARY_APP_NOT_YET_AT_RELEASE_IMAGE')
        site=az_json('webapp','show','-n',APP,'-g',GROUP)
        config=az_json('webapp','config','show','-n',APP,'-g',GROUP)
        vals=az_json('webapp','config','appsettings','list','-n',APP,'-g',GROUP)
        settings={str(item.get('name')):str(item.get('value') or '')
                  for item in vals if isinstance(item,dict) and item.get('name')}
        if site.get('name')!=APP or site.get('state')!='Running' or site.get('httpsOnly') is not True:
            raise RuntimeError('MIGRATION_EXECUTOR_IDENTITY_OR_HTTPS_DRIFT')
        if settings.get('POSTGRES_HOST','').lower()!=DB_HOST or settings.get('POSTGRES_DB')!=DB_NAME or settings.get('POSTGRES_USER')!=DB_USER:
            raise RuntimeError('MIGRATION_EXECUTOR_DATABASE_IDENTITY_DRIFT')
        if not settings.get('POSTGRES_PASSWORD','').lower().startswith('@microsoft.keyvault('):
            raise RuntimeError('MIGRATION_EXECUTOR_PASSWORD_NOT_KEYVAULT_REF')
        old=str(config.get('linuxFxVersion') or '')
        prior_pull_policy=config.get('acrUseManagedIdentityCreds')
        report['managed_identity_pull_policy_before']=prior_pull_policy
        if not old.startswith('DOCKER|'+server+'/ballotlens:'):
            raise RuntimeError('MIGRATION_EXECUTOR_OLD_IMAGE_IDENTITY_DRIFT')
        report['previous_executor_image']=old
        if old=='DOCKER|'+new_image:
            report['result']='PASS_ALREADY_AT_EXACT_IMAGE'
        else:
            az('webapp','config','container','set','--name',APP,'--resource-group',GROUP,
               '--container-image-name',new_image,
               '--container-registry-url','https://'+server,'--output','none')
            report['image_change_attempted']=True
            updated=az_json('webapp','config','show','-n',APP,'-g',GROUP)
            if args.release_commit not in str(updated.get('linuxFxVersion') or ''):
                raise RuntimeError('MIGRATION_EXECUTOR_IMAGE_UPDATE_UNCONFIRMED')
            if updated.get('acrUseManagedIdentityCreds')!=prior_pull_policy:
                raise RuntimeError('MIGRATION_EXECUTOR_PULL_AUTH_POLICY_CHANGED')
            report['result']='PASS_EXECUTOR_IMAGE_UPDATED'
    except Exception as exc:
        report['hold_code']=str(exc)[:150]
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('W24_MIGRATION_EXECUTOR_RESULT='+report['result'])
    if report['result']=='HOLD':
        print('W24_MIGRATION_EXECUTOR_HOLD='+report.get('hold_code','UNKNOWN'))
        return 3
    return 0


if __name__=='__main__':raise SystemExit(main())
