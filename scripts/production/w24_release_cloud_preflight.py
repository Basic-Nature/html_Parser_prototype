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


def classify_azure_cli_error(stderr):
    # NEVER print raw stderr: Azure errors can contain resource identifiers,
    # credentials, connection strings, or other sensitive configuration.
    s = (stderr or '').casefold()
    patterns = (
        ('authorizationpermissionmismatch', 'ACR_REGISTRY_PERMISSION'),
        ('authorizationfailed', 'RBAC_AUTHORIZATION_FAILED'),
        ('authenticationfailed', 'AUTHENTICATION_FAILED'),
        ('authenticationrequired', 'AUTHENTICATION_REQUIRED'),
        ('not logged in', 'NOT_LOGGED_IN'),
        ('please run \'az login\'', 'NOT_LOGGED_IN'),
        ('forbidden', 'FORBIDDEN'),
        ('unauthorized', 'UNAUTHORIZED'),
        ('resourcegroupnotfound', 'RESOURCE_GROUP_NOT_FOUND'),
        ('resourcenotfound', 'RESOURCE_NOT_FOUND'),
        ('subscriptionnotfound', 'SUBSCRIPTION_NOT_FOUND'),
        ('registry not found', 'REGISTRY_NOT_FOUND'),
        ('unrecognized arguments', 'CLI_ARGUMENTS_NOT_SUPPORTED'),
        ('not in the \'az\' command group', 'CLI_COMMAND_NOT_SUPPORTED'),
        ('not a valid command', 'CLI_COMMAND_NOT_SUPPORTED'),
        ('connection refused', 'CONNECTION_REFUSED'),
        ('timed out', 'NETWORK_TIMEOUT'),
        ('certificate verify failed', 'TLS_CERTIFICATE_FAILURE'),
        ('ssl:', 'TLS_FAILURE'),
        ('403', 'HTTP_403'),
        ('401', 'HTTP_401'),
        ('404', 'HTTP_404'),
    )
    return next((label for pattern, label in patterns if pattern in s),
                'UNCLASSIFIED_REQUIRES_RESTRICTED_LOG_REVIEW')


def az_json(checks, operation, *args):
    # 'operation' is a fixed label chosen by this script, never user input.
    try:
        r = subprocess.run(
            ['az', *args, '--output', 'json', '--only-show-errors'],
            capture_output=True, text=True, timeout=65, check=False,
        )
    except FileNotFoundError:
        checks.append({'operation': operation, 'result': 'HOLD',
                       'reason': 'AZURE_CLI_NOT_FOUND'})
        raise RuntimeError('AZURE_READ_FAILED:' + operation + ':AZURE_CLI_NOT_FOUND') from None
    except subprocess.TimeoutExpired:
        checks.append({'operation': operation, 'result': 'HOLD',
                       'reason': 'AZURE_CLI_TIMEOUT'})
        raise RuntimeError('AZURE_READ_FAILED:' + operation + ':AZURE_CLI_TIMEOUT') from None
    except OSError:
        checks.append({'operation': operation, 'result': 'HOLD',
                       'reason': 'AZURE_CLI_SPAWN_ERROR'})
        raise RuntimeError('AZURE_READ_FAILED:' + operation + ':AZURE_CLI_SPAWN_ERROR') from None
    if r.returncode != 0:
        category = classify_azure_cli_error(r.stderr)
        reason = 'EXIT_' + str(r.returncode) + ':' + category
        checks.append({'operation': operation, 'result': 'HOLD',
                       'reason': reason})
        raise RuntimeError('AZURE_READ_FAILED:' + operation + ':' + reason) from None
    try:
        data = json.loads(r.stdout)
    except (TypeError, ValueError):
        checks.append({'operation': operation, 'result': 'HOLD',
                       'reason': 'CLI_JSON_INVALID'})
        raise RuntimeError('AZURE_READ_FAILED:' + operation + ':CLI_JSON_INVALID') from None
    checks.append({'operation': operation, 'result': 'PASS'})
    print('W24_AZURE_READ_PASS=' + operation)
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release-commit',required=True)
    args=p.parse_args()
    out=Path('output/reports/w24_release_cloud_preflight.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'result':'HOLD','checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'new_commit':args.release_commit,'mutation':False,'redis':False}
    checks=[]
    report['azure_read_checks']=checks
    try:
        app=az_json(checks,'WEBAPP_SHOW','webapp','show','-n',APP,'-g',GROUP)
        cfg=az_json(checks,'WEBAPP_CONFIG_SHOW','webapp','config','show','-n',APP,'-g',GROUP)
        settings=az_json(checks,'WEBAPP_APPSETTINGS_LIST','webapp','config','appsettings','list','-n',APP,'-g',GROUP)
        mapping={str(x.get('name')):str(x.get('value') or '') for x in settings
                 if isinstance(x,dict) and x.get('name')}
        instance_list=az_json(checks,'WEBAPP_LIST_INSTANCES','webapp','list-instances','-n',APP,'-g',GROUP)
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
        tags=az_json(checks,'ACR_REPOSITORY_SHOW_TAGS','acr','repository','show-tags','--name',registry,'--repository','ballotlens')
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
