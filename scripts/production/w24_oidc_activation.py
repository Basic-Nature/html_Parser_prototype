"""Manual, Azure-hosted ElectionPulse Keycloak activation. No Redis or DB SQL.

Only the existing App Service's non-secret feature flags are written here.
The operator must provision Key Vault references and the dedicated DB role
through approved Azure/DBA channels before requesting activation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

APP = 'BallotLens'
GROUP = 'BallotLens_group'
ISSUER = 'https://auth.smartelections.cloud/realms/smartelections'
CALLBACK = 'https://www.electionpulse.org/auth/oidc/callback'
CLIENT_ID = 'electionpulse-web'
SECRETS = ('ELECTIONPULSE_KEYCLOAK_CLIENT_SECRET',
           'ELECTIONPULSE_OIDC_SUBJECT_HMAC_KEY',
           'ELECTIONPULSE_OIDC_PG_DSN')
FLAGS = ('ELECTIONPULSE_KEYCLOAK_OIDC_ENABLED',
         'ELECTIONPULSE_OIDC_POSTGRES_ENABLED',
         'ELECTIONPULSE_OIDC_SESSION_BACKEND',
         'ELECTION_PROJECTS_ENABLED')


def az(*args: str, timeout: int = 65) -> str:
    try:
        r = subprocess.run(['az', *args, '--only-show-errors'],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, timeout=timeout, check=False)
    except Exception:
        raise RuntimeError('AZURE_COMMAND_UNAVAILABLE') from None
    if r.returncode != 0:
        raise RuntimeError('AZURE_COMMAND_FAILED')
    return r.stdout


def json_az(*args: str) -> dict | list:
    try:
        return json.loads(az(*args, '--output', 'json'))
    except (ValueError, TypeError):
        raise RuntimeError('AZURE_JSON_INVALID') from None


def get_app() -> dict:
    result = json_az('webapp', 'show', '-n', APP, '-g', GROUP)
    if not isinstance(result, dict) or result.get('name') != APP or result.get('state') != 'Running' or result.get('httpsOnly') is not True:
        raise RuntimeError('APP_SERVICE_ID_OR_HTTPS_POSTURE_DRIFT')
    return result


def settings() -> dict[str, str]:
    values = json_az('webapp', 'config', 'appsettings', 'list', '-n', APP, '-g', GROUP)
    if not isinstance(values, list):
        raise RuntimeError('APP_SETTINGS_UNAVAILABLE')
    return {str(v.get('name')): str(v.get('value') or '') for v in values
            if isinstance(v, dict) and v.get('name')}


def verify_ref(app_id: str, key: str) -> None:
    if not app_id.startswith('/subscriptions/'):
        raise RuntimeError('APP_RESOURCE_ID_INVALID')
    url = 'https://management.azure.com' + app_id + '/config/configreferences/appsettings/' + key + '?api-version=2026-03-15'
    payload = json_az('rest', '--method', 'get', '--url', url)
    props = payload.get('properties') if isinstance(payload, dict) else None
    if not isinstance(props, dict) or props.get('status') != 'Resolved':
        raise RuntimeError('OIDC_SECRET_REF_UNRESOLVED:' + key)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('inspect', 'enable', 'disable'), required=True)
    p.add_argument('--release-commit', required=True)
    p.add_argument('--confirmation', default='')
    p.add_argument('--operator-verified-keycloak-client', default='false')
    a = p.parse_args()
    report = {'mode': a.mode, 'release_commit': a.release_commit,
              'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'result': 'HOLD', 'database_mutation': False, 'redis': False,
              'secrets_recorded': False, 'live_keycloak_login_tested': False}
    out = Path('output/reports/w24_oidc_activation.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        app = get_app()
        config = json_az('webapp', 'config', 'show', '-n', APP, '-g', GROUP)
        image = str(config.get('linuxFxVersion') or '') if isinstance(config, dict) else ''
        if a.release_commit not in image or len(a.release_commit) != 40:
            raise RuntimeError('DEPLOYED_CONTAINER_SHA_MISMATCH')
        vals = settings()
        report['safe_flags_before'] = {key: vals.get(key, '')[:100] for key in FLAGS}
        report['secret_reference_present'] = {key: vals.get(key, '').lower().startswith('@microsoft.keyvault(')
                                               for key in SECRETS}
        if a.mode == 'disable':
            if a.confirmation != 'DISABLE:' + a.release_commit:
                raise RuntimeError('EXPLICIT_DISABLE_CONFIRMATION_REQUIRED')
            az('webapp', 'config', 'appsettings', 'set', '-n', APP, '-g', GROUP,
               '--settings', 'ELECTIONPULSE_KEYCLOAK_OIDC_ENABLED=false',
               'ELECTIONPULSE_OIDC_POSTGRES_ENABLED=false', '--output', 'none')
            report['result'] = 'DISABLE_REQUESTED'
        else:
            if vals.get('GUNICORN_WORKERS', '1') != '1':
                raise RuntimeError('SINGLE_GUNICORN_WORKER_REQUIRED')
            instances = json_az('webapp', 'list-instances', '-n', APP, '-g', GROUP)
            if not isinstance(instances, list) or len(instances) != 1:
                raise RuntimeError('SINGLE_APP_SERVICE_INSTANCE_REQUIRED')
            report['instance_count'] = len(instances)
            if vals.get('ELECTIONPULSE_KEYCLOAK_CLIENT_ID') != CLIENT_ID:
                raise RuntimeError('KEYCLOAK_CLIENT_ID_MISMATCH')
            if vals.get('ELECTIONPULSE_KEYCLOAK_REDIRECT_URI') != CALLBACK:
                raise RuntimeError('OIDC_CALLBACK_MISMATCH')
            for key in SECRETS:
                if not report['secret_reference_present'][key]:
                    raise RuntimeError('OIDC_SECRET_MUST_USE_EXISTING_KEYVAULT:' + key)
                verify_ref(str(app.get('id') or ''), key)
            if a.mode == 'enable':
                if a.confirmation != 'ENABLE:' + a.release_commit:
                    raise RuntimeError('EXPLICIT_ENABLE_CONFIRMATION_REQUIRED')
                if a.operator_verified_keycloak_client.lower() != 'true':
                    raise RuntimeError('KEYCLOAK_CLIENT_REDIRECT_AND_ROLE_REVIEW_REQUIRED')
                az('webapp', 'config', 'appsettings', 'set', '-n', APP, '-g', GROUP,
                   '--settings', 'ELECTIONPULSE_OIDC_SESSION_BACKEND=postgresql',
                   'ELECTIONPULSE_OIDC_POSTGRES_ENABLED=true',
                   'ELECTIONPULSE_KEYCLOAK_OIDC_ENABLED=true',
                   'ELECTION_PROJECTS_ENABLED=true', '--output', 'none')
                report['result'] = 'ENABLE_REQUESTED_LIVE_LOGIN_MUST_BE_VALIDATED'
            else:
                report['result'] = 'PASS_READ_ONLY'
    except Exception as exc:
        report['error_code'] = str(exc)[:150]
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('W24_OIDC_ACTIVATION_RESULT=' + report['result'])
    if report['result'] == 'HOLD':
        print('W24_OIDC_HOLD_CODE=' + report.get('error_code', 'UNKNOWN'))
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
