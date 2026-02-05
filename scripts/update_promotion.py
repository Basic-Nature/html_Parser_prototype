#!/usr/bin/env python
"""Update dataset_promotion.py with verification gating."""

import os
from pathlib import Path

from webapp.parser.config import PROJECT_ROOT

root_override = os.environ.get("PROJECT_ROOT_OVERRIDE")
if root_override:
    repo_root = Path(root_override).expanduser().resolve()
else:
    repo_root = Path(PROJECT_ROOT).resolve()

dataset_promotion_path = repo_root / "webapp" / "parser" / "health" / "dataset_promotion.py"

with open(dataset_promotion_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the batch insertion loop
old_insertion = '''    batch = create_batch_metadata(
        source=f"dataset_promotion:{dataset_dir.name}",
        status=StatusEnum.PENDING,
    )
    inserted = 0
    try:
        for payload in payloads:
            create_warehouse_election_result(batch_id=batch.batch_id, **payload)
            inserted += 1'''

new_insertion = '''    batch = create_batch_metadata(
        source=f"dataset_promotion:{dataset_dir.name}",
        status=StatusEnum.PENDING,
    )
    inserted = 0
    duplicates_skipped = 0
    blocked_urls_skipped = 0
    
    # Get URL from metadata for verification tier
    source_url = metadata.get('source_url')
    url_tier = get_url_verification_tier(source_url) if source_url else 'pending'
    
    try:
        from webapp.parser.utils.db_utils import get_session
        session = get_session()
        
        for payload in payloads:
            # Set verification status based on URL tier
            if url_tier == 'blocked':
                blocked_urls_skipped += 1
                logger.warning(f"[PROMOTE] Skipping blocked URL: {source_url}")
                continue
            elif url_tier == 'trusted':
                payload['verification_status'] = 'verified'
                payload['verified_at'] = datetime.now(timezone.utc)
            else:  # pending
                payload['verification_status'] = 'pending'
            
            payload['source_url'] = source_url
            payload['source_principal'] = metadata.get('source_principal')
            
            # Check for exact duplicate
            if check_exact_duplicate(
                session,
                state=payload.get('state'),
                county=payload.get('county'),
                contest=payload.get('contest'),
                candidate=payload.get('candidate'),
                party=payload.get('party'),
                votes=payload.get('votes'),
                precinct=payload.get('precinct'),
                election_date=payload.get('election_date'),
            ):
                duplicates_skipped += 1
                continue
            
            create_warehouse_election_result(batch_id=batch.batch_id, **payload)
            inserted += 1'''

content = content.replace(old_insertion, new_insertion)

with open(dataset_promotion_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('promote_dataset function updated with verification gating')
