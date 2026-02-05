#!/usr/bin/env python
"""Update metadata tracking in dataset_promotion.py."""

with open(r'c:\Users\olivi\html_Parser_prototype\webapp\parser\health\dataset_promotion.py', 'r') as f:
    content = f.read()

# Update the metastats dictionary
old_metastats = '''    update_batch_metadata(
        batch.batch_id,
        status=StatusEnum.COMPLETED,
        metastats={
            "dataset_dir": str(dataset_dir),
            "records_inserted": inserted,
            "skipped_rows": skipped,
            "contest": metadata.get("contest"),
            "state": metadata.get("state"),
            "county": metadata.get("county"),
        },
    )
    summary["batch_id"] = str(batch.batch_id)
    summary["inserted_records"] = inserted
    print(f"[PROMOTE] Inserted {inserted} rows into warehouse_election_results (batch={batch.batch_id}).")'''

new_metastats = '''    update_batch_metadata(
        batch.batch_id,
        status=StatusEnum.COMPLETED,
        metastats={
            "dataset_dir": str(dataset_dir),
            "records_inserted": inserted,
            "duplicates_skipped": duplicates_skipped,
            "blocked_urls_skipped": blocked_urls_skipped,
            "skipped_rows": skipped,
            "url_tier": url_tier,
            "contest": metadata.get("contest"),
            "state": metadata.get("state"),
            "county": metadata.get("county"),
        },
    )
    summary["batch_id"] = str(batch.batch_id)
    summary["inserted_records"] = inserted
    summary["duplicates_skipped"] = duplicates_skipped
    summary["blocked_urls_skipped"] = blocked_urls_skipped
    summary["url_tier"] = url_tier
    print(f"[PROMOTE] Inserted {inserted} rows (duplicates_skipped={duplicates_skipped}, blocked={blocked_urls_skipped}) from {dataset_dir} (batch={batch.batch_id}).")'''

content = content.replace(old_metastats, new_metastats)

with open(r'c:\Users\olivi\html_Parser_prototype\webapp\parser\health\dataset_promotion.py', 'w') as f:
    f.write(content)

print('Metadata tracking updated')
