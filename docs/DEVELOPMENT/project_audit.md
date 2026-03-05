---
layout: default
title: "Project Audit"
---

Audit scope: `webapp/parser/` modules.

Modules scanned: 250 | ~88154 non-empty LOC

## Pipeline map (Mermaid)

```mermaid
graph LR
  subgraph Entry["Entry"]
    Smart_Elections_Parser_Webapp["Smart_Elections_Parser_Webapp"]
  end
  subgraph Pipeline["Pipeline"]
    web_pipeline["web_pipeline"]
  end
  subgraph Routing["Routing"]
    state_router["state_router"]
  end
  subgraph Services["Services"]
    election_data_services["election_data_services"]
  end
  subgraph Utils["Utils"]
    browser_utils["browser_utils"]
    contest_selector["contest_selector"]
    detect["detect"]
    dynamic_table_extractor["dynamic_table_extractor"]
    html_scanner["html_scanner"]
    pivot["pivot"]
    shared_logic["shared_logic"]
    table_builder["table_builder"]
    user_prompt["user_prompt"]
  end
  subgraph Context_Integration["Context_Integration"]
    context_coordinator["context_coordinator"]
    librarian["librarian"]
    loader["loader"]
    vocab_loader["vocab_loader"]
    Integrity_check["Integrity_check"]
    constants["constants"]
    context_organizer["context_organizer"]
  end
  subgraph Health["Health"]
    manual_correction_bot["manual_correction_bot"]
    create_test_dataset["create_test_dataset"]
    dataset_promotion["dataset_promotion"]
    fine_tune_bert_ner["fine_tune_bert_ner"]
    health_router["health_router"]
    integrity_check_runner["integrity_check_runner"]
    log_cache_cleaner_bot["log_cache_cleaner_bot"]
    promotion_helpers["promotion_helpers"]
  end
  table_builder -->|37| dynamic_table_extractor
  manual_correction_bot -->|36| librarian
  detect -->|18| browser_utils
  file_io_blueprint -->|16| data_framework_blueprint
  loader -->|13| vocab_loader
  pivot -->|12| contest_selector
  election_data_blueprint -->|11| data_framework_blueprint
  pivot -->|11| html_election_parser
  utility_admin_blueprint -->|10| data_framework_blueprint
  dynamic_table_extractor -->|10| context_coordinator
  Smart_Elections_Parser_Webapp -->|9| google_sheets_client
  url_library_blueprint -->|9| data_framework_blueprint
  html_scanner -->|9| librarian
  user_prompt -->|9| shared_logic
  ui_navigation_blueprint -->|8| data_framework_blueprint
```

## Connection highlights

Key module-to-module and cluster relationships to watch during refactors.

### Top module edges

- `table_builder` → `dynamic_table_extractor` (37 refs, Utils → Utils)
- `manual_correction_bot` → `librarian` (36 refs, Health → Context_Integration)
- `detect` → `browser_utils` (18 refs, Utils → Utils)
- `file_io_blueprint` → `data_framework_blueprint` (16 refs, Other → Other)
- `loader` → `vocab_loader` (13 refs, Context_Integration → Context_Integration)
- `pivot` → `contest_selector` (12 refs, Utils → Utils)
- `election_data_blueprint` → `data_framework_blueprint` (11 refs, Other →
  Other)
- `pivot` → `html_election_parser` (11 refs, Utils → Other)
- `utility_admin_blueprint` → `data_framework_blueprint` (10 refs, Other →
  Other)
- `dynamic_table_extractor` → `context_coordinator` (10 refs, Utils →
  Context_Integration)

### Cluster flow summary

- Utils → Utils: 1126 edges (intra-cluster)
- Other → Other: 699 edges (intra-cluster)
- Context_Integration → Context_Integration: 386 edges (intra-cluster)
- Entry → Entry: 316 edges (intra-cluster)
- Format Handlers → Format Handlers: 234 edges (intra-cluster)
- Health → Health: 192 edges (intra-cluster)
- State Handlers → State Handlers: 69 edges (intra-cluster)
- Health → Context_Integration: 39 edges (cross-cluster)
- Utils → Context_Integration: 38 edges (cross-cluster)
- Routing → Routing: 27 edges (intra-cluster)

## Pipeline focus (compact)

```mermaid
graph LR
  subgraph Entry["Entry"]
    Smart_Elections_Parser_Webapp["Smart_Elections_Parser_Webapp"]
  end
  subgraph Pipeline["Pipeline"]
    web_pipeline["web_pipeline"]
  end
  subgraph Routing["Routing"]
    state_router["state_router"]
  end
  subgraph Services["Services"]
    election_data_services["election_data_services"]
  end
  subgraph Utils["Utils"]
    browser_utils["browser_utils"]
    contest_selector["contest_selector"]
    database_comparison["database_comparison"]
    db_utils["db_utils"]
    detect["detect"]
    download_utils["download_utils"]
    dynamic_table_extractor["dynamic_table_extractor"]
    embedding_cache["embedding_cache"]
    format_router["format_router"]
    pattern_extractor["pattern_extractor"]
    pivot["pivot"]
    privilege_tiers["privilege_tiers"]
    shared_logic["shared_logic"]
    table_builder["table_builder"]
    user_prompt["user_prompt"]
  end
  subgraph Context_Integration["Context_Integration"]
    Integrity_check["Integrity_check"]
    constants["constants"]
    context_coordinator["context_coordinator"]
    context_organizer["context_organizer"]
    librarian["librarian"]
    loader["loader"]
    vocab_loader["vocab_loader"]
  end
  subgraph Health["Health"]
    create_test_dataset["create_test_dataset"]
    dataset_promotion["dataset_promotion"]
    fine_tune_bert_ner["fine_tune_bert_ner"]
    health_router["health_router"]
    integrity_check_runner["integrity_check_runner"]
    log_cache_cleaner_bot["log_cache_cleaner_bot"]
    manual_correction_bot["manual_correction_bot"]
    promotion_helpers["promotion_helpers"]
  end
  table_builder -->|37| dynamic_table_extractor
  detect -->|18| browser_utils
  pivot -->|12| contest_selector
  user_prompt -->|9| shared_logic
  pattern_extractor -->|7| browser_utils
  embedding_cache -->|4| Smart_Elections_Parser_Webapp
  table_builder -->|4| pivot
  Smart_Elections_Parser_Webapp -->|3| database_comparison
  format_router -->|3| download_utils
  shared_logic -->|3| format_router
  Smart_Elections_Parser_Webapp -->|2| privilege_tiers
  pdf_handler -->|2| contest_selector
  example_county -->|2| example_state
  browser_utils -->|2| shared_logic
  database_comparison -->|2| db_utils
```

## Cross-module hotspots

- webapp.parser.models.election_data:String ← 136 refs (election_data.py)
- webapp.parser.routes.data_framework_blueprint:_call_handler ← 86 refs
  (data_framework_blueprint.py)
- webapp.parser.models.election_data:Integer ← 77 refs (election_data.py)
- webapp.parser.utils.dynamic_table_extractor:_emit ← 62 refs
  (dynamic_table_extractor.py)
- webapp.parser.Context_Integration.Context_Library.constants:load_vocab_list ←
  58 refs (constants.py)
- webapp.parser.Context_Integration.librarian:safe_path ← 50 refs (librarian.py)
- webapp.parser.utils.table_builder:_norm_header ← 50 refs (table_builder.py)
- webapp.parser.models.election_data:DateTime ← 43 refs (election_data.py)
- webapp.parser.models.election_data:Text ← 41 refs (election_data.py)
- webapp.parser.Context_Integration.context_coordinator:ContextCoordinator ← 34
  refs (context_coordinator.py)
- webapp.Smart_Elections_Parser_Weba:get_request_principal ← 33 refs
  (Smart_Elections_Parser_Webapp.py)
- webapp.parser.html_election_parser:_safe_int ← 23 refs
  (html_election_parser.py)
- webapp.parser.html_election_parser:mark_url_processed ← 23 refs
  (html_election_parser.py)
- webapp.parser.utils.pdf_table_utils:_record_recon_event ← 23 refs
  (pdf_table_utils.py)
- webapp.parser.Context_Integration.Context_Library.constants:load_vocab_mapping
  ← 20 refs (constants.py)

## Leaf modules (candidates for review)

- `location_inference.py`
- `loader.py`
- `_ocr_helpers.py`
- `ocr_tuning.py`
- `fec_handler.py`
- `download_finder.py`
- `html_dynamic_fallback.py`
- `state_handler_base.py`
- `state_scaffold.py`
- `clarity_base_handler.py`
- `dominion_base_handler.py`
- `voteworks_base_handler.py`
- `arizona.py`
- `california.py`
- `example_county.py`
- `westchester.py`
- `new_york.py`
- `texas.py`
- `vendor_state_map.py`
- `health_config.py`
- `promotion_helpers.py`
- `risk_gates_integration_examples.py`
- `risk_gates_spec.py`
- `session_manager.py`
- `navigation_recipes.py`
- `election_data_blueprint.py`
- `fec_data_assurance_blueprint.py`
- `file_io_blueprint.py`
- `health_blueprint.py`
- `observability_blueprint.py`
- `prometheus_metrics_blueprint.py`
- `public_pages_blueprint.py`
- `session_orchestration_blueprint.py`
- `ui_navigation_blueprint.py`
- `url_library_blueprint.py`
- `utility_admin_blueprint.py`
- `coordinator_protocol.py`
- `date_utils.py`
- `logger_singleton.py`
- `merge_utils.py`
- `session_state.py`
- `strategy_concurrency.py`
- `structure_cache.py`
- `url_ingestion.py`

## Pipeline clusters (Mermaid)

```mermaid
graph LR
  subgraph Entry["Entry"]
    Smart_Elections_Parser_Webapp["Smart_Elections_Parser_Webapp"]
  end
  subgraph Pipeline["Pipeline"]
    web_pipeline["web_pipeline"]
  end
  subgraph Routing["Routing"]
    state_router["state_router"]
  end
  subgraph Handlers["Handlers"]
    csv_handler["csv_handler"]
    example_county["example_county"]
    example_state["example_state"]
    html_handler["html_handler"]
    json_handler["json_handler"]
    pdf_handler["pdf_handler"]
    rockland["rockland"]
    state_handler_base["state_handler_base"]
  end
  subgraph Services["Services"]
    election_data_services["election_data_services"]
  end
  subgraph Utils["Utils"]
    browser_utils["browser_utils"]
    contest_selector["contest_selector"]
    detect["detect"]
    dynamic_table_extractor["dynamic_table_extractor"]
    html_scanner["html_scanner"]
    pivot["pivot"]
    shared_logic["shared_logic"]
    table_builder["table_builder"]
    user_prompt["user_prompt"]
  end
  subgraph Context_Integration["Context_Integration"]
    context_coordinator["context_coordinator"]
    librarian["librarian"]
    loader["loader"]
    vocab_loader["vocab_loader"]
    Integrity_check["Integrity_check"]
    constants["constants"]
    context_organizer["context_organizer"]
  end
  subgraph Health["Health"]
    manual_correction_bot["manual_correction_bot"]
    create_test_dataset["create_test_dataset"]
    dataset_promotion["dataset_promotion"]
    fine_tune_bert_ner["fine_tune_bert_ner"]
    health_router["health_router"]
    integrity_check_runner["integrity_check_runner"]
    log_cache_cleaner_bot["log_cache_cleaner_bot"]
    promotion_helpers["promotion_helpers"]
  end
  table_builder -->|37| dynamic_table_extractor
  manual_correction_bot -->|36| librarian
  detect -->|18| browser_utils
  file_io_blueprint -->|16| data_framework_blueprint
  loader -->|13| vocab_loader
  pivot -->|12| contest_selector
  election_data_blueprint -->|11| data_framework_blueprint
  pivot -->|11| html_election_parser
  utility_admin_blueprint -->|10| data_framework_blueprint
  dynamic_table_extractor -->|10| context_coordinator
  Smart_Elections_Parser_Webapp -->|9| google_sheets_client
  url_library_blueprint -->|9| data_framework_blueprint
  html_scanner -->|9| librarian
  user_prompt -->|9| shared_logic
  ui_navigation_blueprint -->|8| data_framework_blueprint
```

## Modules

### webapp/Smart\_Elections\_Parser\_Webapp.py {#webapp-smart-elections-parser-webapp-py}

- Definitions:
  - function: `\_int\_env` (line 219)
  - function: `\_clone\_payload` (line 231)
  - function: `\_get\_ttl\_cache\_payload` (line 240)
  - function: `\_set\_ttl\_cache\_payload` (line 251)
  - function: `\_log\_endpoint\_latency` (line 260)
  - function: `\_emit\_download\_ready` (line 279)
  - function: `\_flagged\_url\_log\_dir` (line 319)
  - function: `\_rotate\_flagged\_url\_path` (line 327)
  - function: `\_prune\_flagged\_url\_logs` (line 345)
  - function: `\_is\_local\_request` (line 361)
  - function: `\_guarded\_ingestion\_allowed` (line 377)
  - function: `\_request\_wants\_json` (line 394)
  - function: `\_cert\_required\_response` (line 409)
  - function: `\_require\_client\_cert` (line 420)
  - function: `\_require\_cert\_for\_socket\_action` (line 439)
  - function: `\_ingestion\_audit\_context` (line 488)
  - function: `log\_flagged\_url` (line 504)
  - function: `\_require\_health\_auth` (line 682)
  - function: `\_health\_auth\_response` (line 705)
  - function: `\_public\_health\_task\_definitions` (line 713)
  - function: `\_get\_health\_tasks` (line 725)
  - function: `\_get\_health\_task` (line 732)
  - function: `\_append\_health\_task\_log` (line 738)
  - function: `\_trim\_health\_task\_history` (line 754)
  - function: `\_finalize\_health\_task` (line 766)
  - function: `\_launch\_health\_task` (line 777)
  - function: `\_run\_health\_task` (line 801)
  - function: `ensure\_utf8` (line 851)
  - function: `\_is\_request\_secure` (line 865)
  - class: `EnsureWsSecurityHeaders` (line 873)
  - function: `\_register\_legacy\_endpoint\_aliases` (line 1138)
  - function: `\_socket\_payload\_too\_large` (line 1267)
  - function: `\_rate\_limit\_socket\_action` (line 1278)
  - function: `\_rate\_limit` (line 1292)
  - function: `\_generate\_upload\_filename` (line 1297)
  - function: `\_enforce\_request\_size` (line 1303)
  - function: `\_validate\_uploaded\_file` (line 1312)
  - function: `\_save\_uploaded\_file` (line 1355)
  - function: `\_log\_download\_access` (line 1378)
  - function: `\_resolve\_output\_metadata\_path` (line 1388)
  - function: `\_quick\_copy\_session\_dir` (line 1400)
  - function: `\_ensure\_quick\_copy\_dir` (line 1408)
  - function: `\_cleanup\_quick\_copy\_dir` (line 1424)
  - function: `\_unique\_quick\_copy\_name` (line 1438)
  - function: `\_is\_output\_download\_allowed` (line 1452)
  - function: `is\_owner` (line 1471)
  - function: `create\_session\_metadata` (line 1475)
  - function: `\_recover\_stale\_session` (line 1478)
  - function: `cleanup\_sessions` (line 1506)
  - function: `transition\_session` (line 1535)
  - function: `cleanup\_old\_log\_files` (line 1574)
  - function: `client\_fingerprint` (line 1595)
  - function: `get\_request\_principal` (line 1605)
  - function: `\_is\_local\_host` (line 1620)
  - function: `\_is\_azure\_environment` (line 1631)
  - function: `\_get\_dev\_isolation\_bypass\_ips` (line 1641)
  - function: `\_is\_dev\_isolation\_bypass\_request` (line 1647)
  - function: `\_resolve\_cert\_session\_id` (line 1662)
  - function: `\_derive\_auth\_context` (line 1671)
  - function: `\_apply\_auth\_context` (line 1681)
  - function: `\_session\_has\_principal` (line 1691)
  - function: `resolve\_session\_id` (line 1695)
  - function: `emit\_contest\_options` (line 1847)
  - function: `\_promote\_inner` (line 1885)
  - function: `ensure\_db\_tables` (line 1907)
  - function: `normalize\_log\_obj` (line 1936)
  - function: `store\_log` (line 2071)
  - function: `\_heartbeat\_loop` (line 2083)
  - function: `socketio\_emit\_func` (line 2098)
  - function: `get\_prompt\_queue` (line 2266)
  - function: `broadcast\_sessions` (line 2269)
  - function: `lock\_session` (line 2286)
  - function: `unlock\_session` (line 2297)
  - function: `safe\_is\_alive` (line 2308)
  - function: `is\_output\_bypassed` (line 2328)
  - function: `get\_manual\_source` (line 2331)
  - function: `get\_manual\_source\_origin` (line 2334)
  - function: `get\_all\_file\_lists` (line 2337)
  - function: `get\_session\_enums` (line 2344)
  - function: `redirect\_to\_https\_www` (line 2361)
  - function: `\_csp\_nonce` (line 2433)
  - function: `build\_csp` (line 2442)
  - function: `add\_headers` (line 2518)
  - function: `\_handle\_global\_exception` (line 2595)
  - function: `add\_url` (line 2642)
  - function: `allowed\_file` (line 2760)
  - function: `get\_url\_list` (line 2770)
  - function: `list\_urls` (line 2786)
  - function: `log\_run\_event` (line 2812)
  - function: `\_validate\_filter\_value` (line 2836)
  - function: `log\_db\_monitor\_event` (line 2853)
  - function: `index` (line 2863)
  - function: `api\_urls` (line 2867)
  - function: `api\_urls\_parse` (line 3012)
  - function: `api\_urls\_training\_data` (line 3087)
  - function: `api\_urls\_parse\_all` (line 3167)
  - function: `api\_filename\_parse` (line 3235)
  - function: `\_load\_output\_metadata` (line 3311)
  - function: `\_build\_output\_lookup\_match` (line 3322)
  - function: `api\_outputs\_lookup` (line 3361)
  - function: `\_get\_warehouse\_columns` (line 3419)
  - function: `\_collect\_url\_reference\_hint` (line 3428)
  - function: `api\_warehouse\_match` (line 3510)
  - function: `api\_warehouse\_export` (line 3597)
  - function: `api\_warehouse\_coverage` (line 3650)
  - function: `data\_framework` (line 3750)
  - function: `\_collect\_data\_framework\_scaffold` (line 3754)
  - function: `\_extract\_year\_from\_text` (line 3808)
  - function: `\_collect\_data\_framework\_curated` (line 3817)
  - function: `\_resolve\_preview\_filters` (line 3872)
  - function: `\_select\_preview\_context` (line 3892)
  - function: `\_fetch\_preview\_rows` (line 3923)
  - function: `api\_data\_framework\_preview` (line 3962)
  - function: `api\_data\_framework\_scaffold` (line 4081)
  - function: `api\_data\_framework\_scaffold\_csv` (line 4094)
  - function: `api\_data\_framework\_curated` (line 4118)
  - function: `api\_data\_framework\_warehouse\_status` (line 4131)
  - function: `api\_data\_framework\_exports` (line 4480)
  - function: `health\_dashboard` (line 4528)
  - function: `api\_list\_health\_tasks` (line 4572)
  - function: `api\_start\_health\_task` (line 4579)
  - function: `api\_health\_task\_detail` (line 4596)
  - function: `api\_health\_socket\_test` (line 4606)
  - function: `test\_ui\_prompt` (line 4665)
  - function: `api\_fs\_list` (line 4721)
  - function: `api\_list\_dir\_compat` (line 4766)
  - function: `api\_fs\_mkdir` (line 4769)
  - function: `api\_fs\_delete` (line 4801)
  - function: `api\_quick\_copy` (line 4840)
  - function: `api\_quick\_copy\_clear` (line 4902)
  - function: `download\_fs` (line 4913)
  - function: `view\_csv` (line 4988)
  - function: `\_build\_or\_load\_csv\_index` (line 5178)
  - function: `csv\_locate` (line 5225)
  - function: `favicon` (line 5266)
  - function: `robots\_txt` (line 5325)
  - function: `serve\_well\_known\_appspecific` (line 5330)
  - function: `\_normalize\_party\_bucket` (line 5339)
  - function: `\_compute\_dropoff\_items` (line 5350)
  - function: `api\_warehouse\_election\_results` (line 5389)
  - function: `delete\_input\_file` (line 5759)
  - function: `delete\_output\_file` (line 5771)
  - function: `delete\_upload\_file` (line 5783)
  - function: `download\_input\_file` (line 5795)
  - function: `download\_output\_file` (line 5798)
  - function: `download\_upload\_file` (line 5879)
  - function: `ballot\_lens` (line 5882)
  - function: `ballot\_lens\_modern` (line 5926)
  - function: `worklist` (line 5931)
  - function: `api\_validate\_urls` (line 5949)
  - function: `api\_url\_status` (line 6022)
  - function: `site\_webmanifest` (line 6228)
  - function: `quality\_dashboard` (line 6265)
  - function: `\_load\_integrity\_trends` (line 6270)
  - function: `api\_integrity\_trends` (line 6322)
  - function: `api\_integrity\_signal` (line 6336)
  - function: `api\_integrity\_export` (line 6381)
  - function: `url\_status\_dashboard` (line 6418)
  - function: `quick\_reference\_page` (line 6422)
  - function: `api\_quality\_metrics` (line 6437)
  - function: `api\_auth\_certificate\_info` (line 6512)
  - function: `api\_route\_wrapper\_monitor\_snapshot` (line 6553)
  - function: `auth\_welcome` (line 6626)
  - function: `upload\_to\_input` (line 6679)
  - function: `upload\_to\_output` (line 6753)
  - function: `upload\_to\_uploads` (line 6825)
  - function: `health` (line 6926)
  - function: `heartbeat` (line 6932)
  - function: `clear\_history` (line 6935)
  - function: `history` (line 6944)
  - function: `rerun\_prior` (line 6986)
  - function: `api\_election\_data\_worklist` (line 7043)
  - function: `api\_election\_data\_worklist\_overview` (line 7249)
  - function: `api\_election\_data\_db\_lite\_finalized` (line 7298)
  - function: `api\_election\_data\_db\_lite\_down\_ballot` (line 7373)
  - function: `api\_election\_data\_google\_sheets\_health` (line 7421)
  - function: `api\_election\_data\_states\_counties` (line 7499)
  - function: `api\_assign\_dl\_owner` (line 7619)
  - function: `api\_preqc\_check` (line 7687)
  - function: `api\_qc1\_submit` (line 7807)
  - function: `api\_election\_data\_stats` (line 7902)
  - function: `handle\_contest\_selected` (line 8015)
  - function: `handle\_get\_session\_history` (line 8056)
  - function: `handle\_clone\_session` (line 8108)
  - function: `on\_join` (line 8153)
  - function: `handle\_get\_sessions` (line 8193)
  - function: `handle\_connect` (line 8200)
  - function: `handle\_disconnect` (line 8409)
  - function: `handle\_ack\_cert\_reauth` (line 8440)
  - function: `handle\_set\_output\_mode` (line 8476)
  - function: `handle\_parser\_prompt` (line 8505)
  - function: `handle\_prompt\_cancel` (line 8562)
  - function: `handle\_cancel\_parser` (line 8614)
  - function: `handle\_toggle\_output\_bypass` (line 8679)
  - function: `handle\_set\_manual\_source` (line 8708)
  - function: `handle\_delete\_session` (line 8754)
  - function: `handle\_ballot\_lens` (line 8776)
  - function: `\_read\_jsonl` (line 8841)
  - function: `fec\_mappings\_review` (line 8861)
  - function: `api\_fec\_problem\_rows` (line 8903)
  - function: `api\_fec\_save\_mapping` (line 8926)
  - function: `api\_data\_assurance\_classify` (line 9019)
  - function: `api\_data\_assurance\_promote` (line 9148)
  - function: `api\_data\_assurance\_pending\_reviews` (line 9237)
- Imports:
  - **Standard Library** (23):
    - `import os as os` (line 4)
    - `import socket as socket` (line 5)
    - `from typing import Any` (line 6)
    - `from typing import Callable` (line 6)
    - `from typing import Tuple` (line 6)
    - `import asyncio as asyncio` (line 53)
    - `import csv as csv` (line 54)
    - `import io as io` (line 56)
    - `import json as json` (line 57)
    - `import re as re` (line 58)
    - `import shutil as shutil` (line 60)
    - `import subprocess as subprocess` (line 61)
    - `import sys as sys` (line 62)
    - `import threading as threading` (line 63)
    - `import time as time` (line 64)
    - `from datetime import datetime` (line 65)
    - `from datetime import timedelta` (line 65)
    - `from datetime import timezone` (line 65)
    - `from pathlib import Path` (line 66)
    - `from threading import Event` (line 67)
    - `from threading import Thread` (line 67)
    - `from urllib.parse import urlparse` (line 68)
    - `from urllib.parse import urlunparse` (line 68)
  - **Third-party** (91):
    - `import orjson as orjson` (line 70)
    - `import psycopg2 as psycopg2` (line 71)
    - `from flask import Flask` (line 72)
    - `from flask import Response` (line 72)
    - `from flask import flash` (line 72)
    - `from flask import g` (line 72)
    - `from flask import jsonify` (line 72)
    - `from flask import redirect` (line 72)
    - `from flask import render_template` (line 72)
    - `from flask import request` (line 72)
    - `from flask import send_file` (line 72)
    - `from flask import send_from_directory` (line 72)
    - `from flask import session` (line 72)
    - `from flask import url_for` (line 72)
    - `from psycopg2 import errors as pg_errors` (line 86)
    - `from sqlalchemy import inspect` (line 87)
    - `from sqlalchemy import text` (line 87)
    - `from sqlalchemy.exc import OperationalError` (line 88)
    - `from werkzeug.exceptions import HTTPException` (line 89)
    - `from werkzeug.exceptions import NotFound` (line 89)
    - `from webapp.parser.health.integrity_monitor import get_integrity_monitor`
      (line 119)
    - `from webapp.parser.health.session_manager import SessionManager` (line
      120)
    - `from webapp.parser.utils.logger_singleton import logger` (line 121)
    - `from webapp.parser.utils.logger_singleton import prompt` (line 121)
    - `from webapp.parser.utils.session_state import DEFAULT_PHASE_BY_STATE`
      (line 122)
    - `from webapp.parser.utils.session_state import PipelinePhase` (line 122)
    - `from webapp.parser.utils.session_state import SessionState` (line 122)
    - `from webapp.parser.utils.session_state import export_session_enums` (line
      122)
    - `from webapp.parser.config import ALLOW_GOOGLE_DOCS` (line 137)
    - `from webapp.parser.config import ALLOW_LEGACY_OUTPUT_DOWNLOAD` (line 137)
    - `from webapp.parser.config import DATA_API_URL` (line 137)
    - `from webapp.parser.config import DEPLOY_ENV` (line 137)
    - `from webapp.parser.config import INPUT_DIR` (line 137)
    - `from webapp.parser.config import LOG_DIR` (line 137)
    - `from webapp.parser.config import MAX_CSV_ROWS` (line 137)
    - `from webapp.parser.config import MAX_PDF_PAGES` (line 137)
    - `from webapp.parser.config import MAX_SOCKET_EVENT_BYTES` (line 137)
    - `from webapp.parser.config import MAX_SOCKET_LOG_BYTES` (line 137)
    - `from webapp.parser.config import MAX_UPLOAD_BYTES` (line 137)
    - `from webapp.parser.config import MAX_UPLOAD_SIZE_MB` (line 137)
    - `from webapp.parser.config import MAX_XLSX_BYTES` (line 137)
    - `from webapp.parser.config import OUTPUT_DIR` (line 137)
    - `from webapp.parser.config import POSTGRES_DB` (line 137)
    - `from webapp.parser.config import POSTGRES_HOST` (line 137)
    - `from webapp.parser.config import POSTGRES_PASSWORD_RAW` (line 137)
    - `from webapp.parser.config import POSTGRES_PORT` (line 137)
    - `from webapp.parser.config import POSTGRES_USER_RAW` (line 137)
    - `from webapp.parser.config import PROJECT_ROOT` (line 137)
    - `from webapp.parser.config import QUICK_COPY_DIR` (line 137)
    - `from webapp.parser.config import RUN_HISTORY_FILE` (line 137)
  - **Local/Project** (4):
    - `from __future__ import annotations` (line 1)
    - `import hmac as hmac` (line 3)
    - `import gzip as gzip` (line 55)
    - `import secrets as secrets` (line 59)
- Task markers:
  - L263 **WARNING**: " if elapsed_ms &gt;= API_LATENCY_WARN_MS else "DEBUG",
  - L275 **WARNING**: (payload)
  - L478 **WARNING**: ",
  - L600 **WARNING**: ({
  - L601 **WARNING**: ",
  - L871 **WARNING**: ").upper().split(","))
  - L907 **WARNING**: ({
  - L908 **WARNING**: ",
  - L925 **WARNING**: ({
  - L926 **WARNING**: ",
  - L943 **WARNING**: ({
  - L944 **WARNING**: ",
  - L960 **WARNING**: ({
  - L961 **WARNING**: ",
  - L977 **WARNING**: ({
  - L978 **WARNING**: ",
  - L994 **WARNING**: ({
  - L995 **WARNING**: ",
  - L1011 **WARNING**: ({
  - L1012 **WARNING**: ",
  - L1028 **WARNING**: ({
  - L1029 **WARNING**: ",
  - L1045 **WARNING**: ({
  - L1046 **WARNING**: ",
  - L1062 **WARNING**: ({
  - L1063 **WARNING**: ",
  - L1079 **WARNING**: ({
  - L1080 **WARNING**: ",
  - L1096 **WARNING**: ({
  - L1097 **WARNING**: ",
  - L1113 **WARNING**: ({
  - L1114 **WARNING**: ",
  - L1130 **WARNING**: ({
  - L1131 **WARNING**: ",
  - L1327 **WARNING**: ({
  - L1328 **WARNING**: ",
  - L1344 **WARNING**: ({
  - L1345 **WARNING**: ",
  - L1431 **WARNING**: ({
  - L1432 **WARNING**: ",
  - L1939 **WARNING**: , ERROR, CRITICAL, TRACE
  - L1981 **WARNING**: ", "ERROR", "CRITICAL", "TRACE"}
  - L2017 **WARNING**: " in mlow:
  - L2570 **WARNING**:         # For websocket handshake only: add Cache-Control
    so webhint stops warning
  - L2653 **WARNING**: ({"level": "WARNING", "type": "status", "message": "URL
    too long or invalid.", "session_id": None})
  - L2657 **WARNING**: ({"level": "WARNING", "type": "status", "message": "No
    valid http(s) URL found.", "session_id": None})
  - L2666 **WARNING**: ({"level": "WARNING", "type": "status", "message": "URL
    too long.", "session_id": None})
  - L2676 **WARNING**: ({"level": "WARNING", "type": "status", "message": "URLs
    with credentials are not allowed.", "session_id": None})
  - L2696 **WARNING**: ({"level": "WARNING", "type": "status", "message": f"URL
    blocked: {reason}", "session_id": None})
  - L2729 **WARNING**: ({"level": "WARNING", "type": "status", "message": "Only
    http/https URLs with a host are accepted.", "session_id": None})
- Outgoing cross-module calls (sample):
  - origin.strip (line 36)
  - \_RAW\_SOCKETIO\_ORIGINS.split (line 37)
  - origin.strip (line 38)
  - dotenv.load\_dotenv (line 131)
  - threading.Lock (line 210)
  - threading.Lock (line 212)
  - orjson.loads (line 235)
  - orjson.dumps (line 235)
  - time.time (line 241)
  - \_API\_LATENCY\_CACHE.get (line 243)
  - slot.get (line 246)
  - slot.get (line 248)
  - time.time (line 252)
  - time.perf\_counter (line 261)
  - payload.update (line 272)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 275)
  - webapp.parser.utils.logger\_singleton.logger.debug (line 277)
  - \_DOWNLOAD\_READY\_SESSIONS.add (line 285)
  - socketio.emit (line 287)
  - DB\_MONITOR\_FILE.touch (line 309)
  - webapp.parser.config.LOG\_DIR.mkdir (line 321)
  - datetime.datetime.now (line 329)
  - now.strftime (line 331)
  - prefix.with\_suffix (line 332)
  - candidate.exists (line 333)
  - candidate.stat (line 333)
  - prefix.with\_name (line 337)
  - cand.exists (line 338)
  - cand.stat (line 338)
  - datetime.datetime.now (line 346)
  - datetime.timedelta (line 347)
  - base.glob (line 350)
  - datetime.datetime.fromtimestamp (line 352)
  - entry.stat (line 352)
  - entry.unlink (line 354)
  - remote\_addr.startswith (line 370)
  - hmac.compare\_digest (line 381)
  - auth\_hdr.lower (line 384)
  - auth\_hdr.split (line 386)
  - hmac.compare\_digest (line 387)
  - accept.lower (line 400)
  - flask.jsonify (line 412)
  - flask.url\_for (line 415)
  - flask.redirect (line 417)
  - flask.url\_for (line 417)
  - auth\_hdr.lower (line 427)
  - hmac.compare\_digest (line 429)
  - auth\_hdr.split (line 429)
  - principal.startswith (line 434)
  - auth\_hdr.lower (line 446)
- Inbound references:
  - \_int\_env ← Smart_Elections_Parser_Webapp.py:226
  - \_int\_env ← Smart_Elections_Parser_Webapp.py:227
  - \_int\_env ← Smart_Elections_Parser_Webapp.py:228
  - \_int\_env ← embedding_cache.py:128
  - \_int\_env ← embedding_cache.py:129
  - \_int\_env ← embedding_cache.py:131
  - \_int\_env ← embedding_cache.py:132
  - \_clone\_payload ← Smart_Elections_Parser_Webapp.py:248
  - \_clone\_payload ← Smart_Elections_Parser_Webapp.py:256
  - \_get\_ttl\_cache\_payload ← Smart_Elections_Parser_Webapp.py:3653
  - \_get\_ttl\_cache\_payload ← Smart_Elections_Parser_Webapp.py:7534
  - \_set\_ttl\_cache\_payload ← Smart_Elections_Parser_Webapp.py:3716
  - \_set\_ttl\_cache\_payload ← Smart_Elections_Parser_Webapp.py:7602
  - \_log\_endpoint\_latency ← Smart_Elections_Parser_Webapp.py:3655
  - \_log\_endpoint\_latency ← Smart_Elections_Parser_Webapp.py:3717
  - \_log\_endpoint\_latency ← Smart_Elections_Parser_Webapp.py:7524
  - \_log\_endpoint\_latency ← Smart_Elections_Parser_Webapp.py:7536
  - \_log\_endpoint\_latency ← Smart_Elections_Parser_Webapp.py:7603
  - \_emit\_download\_ready ← Smart_Elections_Parser_Webapp.py:2239
  - \_emit\_download\_ready ← Smart_Elections_Parser_Webapp.py:2252
  - \_flagged\_url\_log\_dir ← Smart_Elections_Parser_Webapp.py:330
  - \_flagged\_url\_log\_dir ← Smart_Elections_Parser_Webapp.py:348
  - \_rotate\_flagged\_url\_path ← Smart_Elections_Parser_Webapp.py:507
  - \_prune\_flagged\_url\_logs ← Smart_Elections_Parser_Webapp.py:514
  - \_is\_local\_request ← Smart_Elections_Parser_Webapp.py:423
  - \_is\_local\_request ← Smart_Elections_Parser_Webapp.py:442
  - \_guarded\_ingestion\_allowed ← Smart_Elections_Parser_Webapp.py:2886
  - \_guarded\_ingestion\_allowed ← Smart_Elections_Parser_Webapp.py:5892
  - \_guarded\_ingestion\_allowed ← Smart_Elections_Parser_Webapp.py:6691
  - \_guarded\_ingestion\_allowed ← Smart_Elections_Parser_Webapp.py:6765
  - \_guarded\_ingestion\_allowed ← Smart_Elections_Parser_Webapp.py:6831
  - \_request\_wants\_json ← Smart_Elections_Parser_Webapp.py:410
  - \_request\_wants\_json ← Smart_Elections_Parser_Webapp.py:6680
  - \_request\_wants\_json ← Smart_Elections_Parser_Webapp.py:6754
  - \_request\_wants\_json ← Smart_Elections_Parser_Webapp.py:6826
  - \_cert\_required\_response ← Smart_Elections_Parser_Webapp.py:436
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:2883
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:4583
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:4771
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:4802
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:4841
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:4903
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:5760
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:5772
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:5784
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:5889
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:6518
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:6554
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:6682
  - \_require\_client\_cert ← Smart_Elections_Parser_Webapp.py:6756

### Context\_Integration/Context\_Library/constants.py {#webapp-parser-context-integration-context-library-constants-py}

- Definitions:
  - function: `\_log\_vocab\_fallback` (line 11)
  - function: `\_load\_state\_to\_county\_map\_from\_vocab` (line 21)
  - function: `\_load\_county\_to\_precincts\_map\_from\_vocab` (line 48)
  - class: `\_LazyMapping` (line 74)
  - function: `load\_vocab\_list` (line 107)
  - function: `load\_vocab\_mapping` (line 122)
  - function: `\_load\_division\_type\_defaults\_from\_vocab` (line 137)
  - function: `\_load\_division\_type\_overrides\_from\_vocab` (line 142)
  - function: `\_load\_canonical\_state\_abbr\_from\_vocab` (line 156)
  - function: `\_load\_party\_code\_map\_from\_vocab` (line 171)
  - function: `\_load\_party\_code\_descriptions\_from\_vocab` (line 177)
  - function: `\_load\_party\_aliases\_from\_vocab` (line 183)
  - function: `\_load\_party\_normalization\_overrides\_from\_vocab` (line 189)
  - function: `\_load\_office\_keywords\_from\_vocab` (line 195)
  - function: `\_load\_html\_taxonomy\_from\_vocab` (line 201)
  - function: `\_load\_html\_taxonomy\_category` (line 222)
  - function: `\_load\_html\_tags\_from\_vocab` (line 236)
  - function: `\_load\_button\_tags\_from\_vocab` (line 247)
  - function: `\_load\_heading\_tags\_from\_vocab` (line 258)
  - function: `\_load\_table\_tags\_from\_vocab` (line 269)
  - function: `\_load\_state\_tags\_from\_vocab` (line 280)
  - function: `\_load\_panel\_tags\_from\_vocab` (line 291)
  - function: `\_load\_container\_extra\_keywords\_from\_vocab` (line 302)
  - function: `\_load\_container\_fallback\_selectors\_from\_vocab` (line 308)
  - function: `\_load\_root\_container\_tags\_from\_vocab` (line 314)
  - function: `\_load\_special\_total\_row\_keywords\_from\_vocab` (line 325)
  - function: `\_load\_table\_anchor\_label\_defaults\_from\_vocab` (line 331)
  - function: `\_load\_location\_synonym\_map\_from\_vocab` (line 337)
  - function: `\_load\_rawjson\_column\_aliases\_from\_vocab` (line 343)
  - function: `\_load\_candidate\_value\_keys\_from\_vocab` (line 349)
  - function: `\_load\_total\_value\_keys\_from\_vocab` (line 355)
  - function: `\_load\_custom\_attr\_patterns\_from\_vocab` (line 361)
  - function: `\_load\_noisy\_label\_patterns\_from\_vocab` (line 367)
  - function: `\_load\_precinct\_header\_patterns\_from\_vocab` (line 373)
  - function: `\_load\_selectors\_from\_vocab` (line 379)
  - function: `\_load\_keyword\_priority\_resolver\_from\_vocab` (line 394)
  - function: `\_load\_pseudo\_party\_labels\_from\_vocab` (line 400)
  - function: `\_load\_pseudo\_party\_raw\_keys\_from\_vocab` (line 406)
  - function: `\_load\_pseudo\_exclude\_tokens\_from\_vocab` (line 412)
  - function: `\_load\_ballot\_group\_canon\_order\_from\_vocab` (line 418)
  - function: `\_load\_ballot\_name\_canon\_overrides\_from\_vocab` (line 424)
  - function: `\_load\_ballot\_display\_overrides\_from\_vocab` (line 430)
  - function: `\_load\_division\_suffixes\_from\_vocab` (line 436)
  - function: `\_load\_division\_heuristic\_terms\_from\_vocab` (line 442)
  - function: `\_load\_camelot\_state\_noise\_overrides\_from\_vocab` (line 449)
  - function: `\_load\_camelot\_county\_noise\_overrides\_from\_vocab` (line
    464)
  - function: `\_load\_allowed\_labels\_from\_vocab` (line 479)
  - function: `\_load\_field\_type\_labels\_from\_vocab` (line 485)
  - function: `\_load\_canonical\_segment\_labels\_from\_vocab` (line 491)
  - function: `\_load\_update\_panel\_keywords\_from\_vocab` (line 497)
  - function: `\_load\_nlp\_skip\_phrases\_from\_vocab` (line 503)
  - function: `\_load\_misaligned\_patterns\_from\_vocab` (line 509)
  - function: `\_load\_view\_by\_phrases\_from\_vocab` (line 515)
  - function: `\_load\_container\_fallback\_selectors\_from\_vocab` (line 521)
  - function: `\_load\_noisy\_label\_patterns\_from\_vocab` (line 527)
  - function: `\_load\_precinct\_header\_patterns\_from\_vocab` (line 533)
  - function: `\_load\_contest\_title\_keywords\_extra\_from\_vocab` (line 539)
  - function: `\_load\_contest\_title\_tags\_from\_vocab` (line 545)
  - function: `\_load\_contest\_title\_skip\_phrases\_from\_vocab` (line 551)
  - function: `\_load\_contest\_header\_keywords\_from\_vocab` (line 557)
  - function: `\_load\_contest\_header\_preference\_from\_vocab` (line 563)
  - function: `\_load\_camelot\_noise\_categories\_from\_vocab` (line 569)
  - function: `\_load\_always\_ignore\_classes\_from\_vocab` (line 579)
  - function: `\_load\_always\_ignore\_ids\_from\_vocab` (line 590)
  - function: `\_load\_icon\_classes\_from\_vocab` (line 596)
  - function: `\_load\_icon\_tags\_from\_vocab` (line 607)
  - function: `\_load\_button\_classes\_from\_vocab` (line 618)
  - function: `\_load\_heading\_classes\_from\_vocab` (line 629)
  - function: `\_load\_panel\_classes\_from\_vocab` (line 640)
  - function: `\_load\_timestamp\_classes\_from\_vocab` (line 651)
  - function: `\_load\_structural\_tags\_from\_vocab` (line 662)
  - function: `\_load\_timestamp\_id\_patterns\_from\_vocab` (line 673)
  - function: `\_load\_timestamp\_attrs\_from\_vocab` (line 679)
  - function: `\_load\_always\_ignore\_tags\_from\_vocab` (line 685)
  - function: `\_load\_likely\_row\_classes\_from\_vocab` (line 696)
  - function: `\_load\_table\_builder\_location\_priority\_from\_vocab` (line
    708)
  - function: `\_load\_table\_builder\_location\_tokens\_from\_vocab` (line 714)
  - function: `\_load\_table\_builder\_candidate\_suffix\_defaults\_from\_vocab`
    (line 720)
  - function: `\_load\_location\_abbreviations\_from\_vocab` (line 726)
  - function: `\_load\_party\_canon\_map\_from\_vocab` (line 737)
  - function: `\_load\_ballot\_inline\_alias\_defaults\_from\_vocab` (line 743)
  - function: `\_load\_ballot\_group\_rename\_variants\_from\_vocab` (line 749)
  - function: `\_load\_election\_entity\_labels\_from\_vocab` (line 755)
  - function: `\_load\_extra\_heading\_tags\_from\_vocab` (line 761)
  - function: `\_load\_contest\_panel\_tags\_from\_vocab` (line 772)
  - function: `\_load\_election\_type\_regex\_map\_from\_vocab` (line 783)
  - function: `build\_state\_to\_division\_type\_map` (line 809)
  - function: `\_build\_state\_module\_map` (line 839)
  - function: `get\_party\_code\_info` (line 919)
  - function: `\_parse\_priority\_rule` (line 1198)
  - function: `\_normalize\_token` (line 1218)
  - function: `\_context\_terms` (line 1222)
  - function: `\_context\_has\_any` (line 1235)
  - function: `\_apply\_condition\_scores` (line 1245)
  - function: `resolve\_keyword\_priority` (line 1281)
  - function: `\_sanitize\_party\_token` (line 1471)
  - function: `normalize\_party\_code` (line 1490)
  - function: `canonical\_ballot\_group` (line 1517)
  - function: `split\_and\_normalize\_ballot\_groups` (line 1544)
  - function: `normalize\_result\_group\_label` (line 1563)
  - function: `normalize\_party\_label` (line 1581)
  - function: `is\_pseudo\_result\_party` (line 1611)
  - function: `\_iter\_strings` (line 1723)
  - function: `\_compile\_union` (line 1734)
  - function: `\_norm\_state\_key` (line 1749)
  - function: `\_norm\_county\_key` (line 1760)
  - function: `\_collect\_layered\_patterns` (line 1769)
  - function: `get\_camelot\_title\_regex` (line 1780)
  - function: `get\_camelot\_row\_regex` (line 1790)
  - function: `build\_camelot\_row\_filter` (line 1803)
- Imports:
  - **Standard Library** (13):
    - `import os as os` (line 1)
    - `import re as re` (line 2)
    - `from collections.abc import Mapping` (line 3)
    - `from functools import lru_cache` (line 4)
    - `from typing import Any` (line 5)
    - `from typing import Callable` (line 5)
    - `from typing import Dict` (line 5)
    - `from typing import Iterable` (line 5)
    - `from typing import List` (line 5)
    - `from typing import Optional` (line 5)
    - `from typing import Pattern` (line 5)
    - `from typing import Set` (line 5)
    - `from typing import Tuple` (line 5)
  - **Local/Project** (2):
    - `from utils.logger_singleton import logger` (line 7)
    - `from vocab.loader import get_vocab_loader` (line 8)
- Outgoing cross-module calls (sample):
  - os.getenv (line 14)
  - utils.logger\_singleton.logger.debug (line 15)
  - vocab.loader.get\_vocab\_loader (line 23)
  - loader.load\_canonical (line 24)
  - part.strip (line 36)
  - line.split (line 36)
  - state\_map.setdefault (line 39)
  - state\_map.items (line 41)
  - functools.lru\_cache (line 20)
  - vocab.loader.get\_vocab\_loader (line 50)
  - loader.load\_canonical (line 51)
  - part.strip (line 63)
  - line.split (line 63)
  - county\_map.setdefault (line 66)
  - county\_map.items (line 68)
  - functools.lru\_cache (line 47)
  - self.\_loader (line 81)
  - self.\_data (line 85)
  - self.\_data (line 88)
  - self.\_data (line 91)
  - self.\_data (line 94)
  - self.\_data (line 97)
  - self.\_data (line 100)
  - self.\_data (line 103)
  - vocab.loader.get\_vocab\_loader (line 110)
  - loader.load\_canonical (line 111)
  - functools.lru\_cache (line 106)
  - vocab.loader.get\_vocab\_loader (line 125)
  - loader.load\_mapping (line 126)
  - functools.lru\_cache (line 121)
  - functools.lru\_cache (line 136)
  - mapping.items (line 145)
  - part.strip (line 148)
  - key.split (line 148)
  - nested.setdefault (line 151)
  - functools.lru\_cache (line 141)
  - mapping.items (line 159)
  - state\_map.setdefault (line 162)
  - state\_map.items (line 164)
  - functools.lru\_cache (line 155)
  - functools.lru\_cache (line 170)
  - functools.lru\_cache (line 176)
  - functools.lru\_cache (line 182)
  - functools.lru\_cache (line 188)
  - functools.lru\_cache (line 194)
  - raw.strip (line 209)
  - line.startswith (line 210)
  - part.strip (line 212)
  - line.split (line 212)
  - c.strip (line 215)
- Inbound references:
  - \_log\_vocab\_fallback ← constants.py:26
  - \_log\_vocab\_fallback ← constants.py:30
  - \_log\_vocab\_fallback ← constants.py:53
  - \_log\_vocab\_fallback ← constants.py:57
  - \_log\_vocab\_fallback ← constants.py:113
  - \_log\_vocab\_fallback ← constants.py:117
  - \_log\_vocab\_fallback ← constants.py:128
  - \_log\_vocab\_fallback ← constants.py:132
  - \_LazyMapping ← constants.py:801
  - \_LazyMapping ← constants.py:803
  - \_LazyMapping ← constants.py:829
  - \_LazyMapping ← constants.py:850
  - \_LazyMapping ← constants.py:863
  - load\_vocab\_list ← constants.py:205
  - load\_vocab\_list ← constants.py:304
  - load\_vocab\_list ← constants.py:310
  - load\_vocab\_list ← constants.py:327
  - load\_vocab\_list ← constants.py:333
  - load\_vocab\_list ← constants.py:345
  - load\_vocab\_list ← constants.py:351
  - load\_vocab\_list ← constants.py:357
  - load\_vocab\_list ← constants.py:363
  - load\_vocab\_list ← constants.py:369
  - load\_vocab\_list ← constants.py:375
  - load\_vocab\_list ← constants.py:402
  - load\_vocab\_list ← constants.py:408
  - load\_vocab\_list ← constants.py:414
  - load\_vocab\_list ← constants.py:420
  - load\_vocab\_list ← constants.py:438
  - load\_vocab\_list ← constants.py:452
  - load\_vocab\_list ← constants.py:467
  - load\_vocab\_list ← constants.py:481
  - load\_vocab\_list ← constants.py:499
  - load\_vocab\_list ← constants.py:505
  - load\_vocab\_list ← constants.py:511
  - load\_vocab\_list ← constants.py:517
  - load\_vocab\_list ← constants.py:523
  - load\_vocab\_list ← constants.py:529
  - load\_vocab\_list ← constants.py:535
  - load\_vocab\_list ← constants.py:541
  - load\_vocab\_list ← constants.py:547
  - load\_vocab\_list ← constants.py:553
  - load\_vocab\_list ← constants.py:559
  - load\_vocab\_list ← constants.py:565
  - load\_vocab\_list ← constants.py:572
  - load\_vocab\_list ← constants.py:573
  - load\_vocab\_list ← constants.py:574
  - load\_vocab\_list ← constants.py:592
  - load\_vocab\_list ← constants.py:675
  - load\_vocab\_list ← constants.py:681

### Context\_Integration/Integrity\_check.py {#webapp-parser-context-integration-integrity-check-py}

- Definitions:
  - function: `\_trim\_monitor\_log` (line 53)
  - function: `\_cap\_log\_value` (line 76)
  - function: `log\_integrity\_monitor` (line 98)
  - function: `\_ensure\_alerts\_table` (line 109)
  - function: `find\_date\_anomalies` (line 116)
  - function: `detect\_anomalies\_with\_ml` (line 124)
  - function: `election\_integrity\_checks` (line 211)
  - function: `advanced\_cross\_field\_validation` (line 232)
  - function: `summarize\_context\_entities` (line 241)
  - function: `analyze\_contests` (line 250)
  - function: `auto\_tune\_contamination` (line 296)
  - function: `print\_issues\_table` (line 317)
  - function: `print\_entity\_summary` (line 337)
  - function: `print\_ml\_anomalies` (line 345)
  - function: `print\_date\_anomalies` (line 375)
  - function: `print\_auto\_tune\_result` (line 393)
  - function: `print\_analyze\_contests` (line 399)
  - function: `monitor\_db\_for\_alerts` (line 411)
  - function: `log\_integrity\_issues` (line 457)
  - function: `detect\_statistical\_outliers` (line 474)
  - function: `print\_integrity\_summary` (line 510)
- Imports:
  - **Standard Library** (9):
    - `import re as re` (line 3)
    - `import threading as threading` (line 4)
    - `import time as time` (line 5)
    - `from collections import Counter` (line 6)
    - `from pathlib import Path` (line 7)
    - `from typing import Any` (line 8)
    - `from typing import Dict` (line 8)
    - `from typing import List` (line 8)
    - `from typing import Tuple` (line 8)
  - **Third-party** (3):
    - `import numpy as np` (line 11)
    - `import orjson as orjson` (line 12)
    - `from sqlalchemy import select` (line 18)
  - **Local/Project** (25):
    - `from __future__ import annotations` (line 1)
    - `import matplotlib as matplotlib` (line 10)
    - `from rich.panel import Panel` (line 13)
    - `from rich.table import Table` (line 14)
    - `from sklearn.cluster import DBSCAN` (line 15)
    - `from sklearn.ensemble import IsolationForest` (line 16)
    - `from sklearn.preprocessing import LabelEncoder` (line 17)
    - `from config import CONTEXT_DB_PATH` (line 20)
    - `from config import CONTEXT_LIBRARY_PATH` (line 20)
    - `from config import LOG_DIR` (line 20)
    - `from Context_Integration.librarian import clean_for_json` (line 21)
    - `from utils import misc_utils` (line 22)
    - `from utils.db_utils import get_session` (line 23)
    - `from utils.logger_singleton import console` (line 24)
    - `from utils.models import Alert` (line 25)
    - `from utils.privilege_tiers import PrivilegeTier` (line 26)
    - `from utils.shared_logic import safe_all` (line 27)
    - `from utils.shared_logic import safe_encode` (line 27)
    - `from utils.shared_logic import safe_execute` (line 27)
    - `from utils.shared_logic import safe_get` (line 27)
    - `from utils.shared_logic import safe_items` (line 27)
    - `from utils.shared_logic import safe_tolist` (line 27)
    - `from utils.spacy_utils import extract_dates` (line 35)
    - `from utils.spacy_utils import extract_entities` (line 35)
    - `from utils.spacy_utils import flag_suspicious_contests` (line 35)
- Outgoing cross-module calls (sample):
  - matplotlib.use (line 38)
  - pathlib.Path (line 41)
  - INTEGRITY\_MONITOR\_LOG.touch (line 49)
  - path.exists (line 54)
  - path.stat (line 57)
  - path.open (line 60)
  - handle.seek (line 63)
  - handle.read (line 64)
  - tail.find (line 68)
  - path.open (line 71)
  - handle.write (line 72)
  - value.items (line 88)
  - payload.setdefault (line 100)
  - time.time (line 100)
  - Context\_Integration.librarian.clean\_for\_json (line 101)
  - INTEGRITY\_MONITOR\_LOG.open (line 103)
  - handle.write (line 104)
  - orjson.dumps (line 104)
  - utils.spacy\_utils.extract\_dates (line 119)
  - utils.shared\_logic.safe\_get (line 119)
  - anomalies.append (line 121)
  - numpy.array (line 134)
  - sklearn.preprocessing.LabelEncoder (line 136)
  - sklearn.preprocessing.LabelEncoder (line 137)
  - sklearn.preprocessing.LabelEncoder (line 138)
  - utils.shared\_logic.safe\_get (line 139)
  - utils.shared\_logic.safe\_get (line 140)
  - utils.shared\_logic.safe\_get (line 141)
  - le\_state.fit (line 142)
  - le\_county.fit (line 143)
  - le\_type.fit (line 144)
  - le\_state.transform (line 148)
  - utils.shared\_logic.safe\_get (line 148)
  - le\_county.transform (line 149)
  - utils.shared\_logic.safe\_get (line 149)
  - le\_type.transform (line 150)
  - utils.shared\_logic.safe\_get (line 150)
  - utils.shared\_logic.safe\_get (line 151)
  - utils.shared\_logic.safe\_get (line 151)
  - utils.shared\_logic.safe\_get (line 152)
  - utils.shared\_logic.safe\_get (line 153)
  - utils.shared\_logic.safe\_get (line 153)
  - utils.shared\_logic.safe\_get (line 154)
  - utils.shared\_logic.safe\_get (line 154)
  - trust\_factors.get (line 161)
  - trust\_factors.get (line 162)
  - trust\_factors.get (line 163)
  - trust\_factors.get (line 163)
  - trust\_factors.get (line 164)
  - trust\_factors.get (line 165)
- Inbound references:
  - \_trim\_monitor\_log ← Integrity_check.py:105
  - \_cap\_log\_value ← Integrity_check.py:91
  - \_cap\_log\_value ← Integrity_check.py:95
  - \_cap\_log\_value ← Integrity_check.py:101
  - \_cap\_log\_value ← Integrity_check.py:471
  - \_cap\_log\_value ← librarian.py:97
  - \_cap\_log\_value ← librarian.py:101
  - \_cap\_log\_value ← librarian.py:770
  - \_cap\_log\_value ← librarian.py:795
  - log\_integrity\_monitor ← Integrity_check.py:276
  - \_ensure\_alerts\_table ← Integrity_check.py:112
  - find\_date\_anomalies ← Integrity_check.py:252
  - detect\_anomalies\_with\_ml ← Integrity_check.py:253
  - election\_integrity\_checks ← Integrity_check.py:251
  - advanced\_cross\_field\_validation ← Integrity_check.py:532
  - summarize\_context\_entities ← Integrity_check.py:528
  - summarize\_context\_entities ← manual_correction_bot.py:1058
  - analyze\_contests ← html_election_parser.py:928
  - analyze\_contests ← Integrity_check.py:518
  - analyze\_contests ← manual_correction_bot.py:1038
  - auto\_tune\_contamination ← Integrity_check.py:537
  - print\_issues\_table ← Integrity_check.py:400
  - print\_issues\_table ← Integrity_check.py:533
  - print\_entity\_summary ← Integrity_check.py:529
  - print\_ml\_anomalies ← Integrity_check.py:402
  - print\_date\_anomalies ← Integrity_check.py:401
  - print\_auto\_tune\_result ← Integrity_check.py:538
  - print\_analyze\_contests ← Integrity_check.py:525
  - print\_integrity\_summary ← html_election_parser.py:986
  - print\_integrity\_summary ← html_election_parser.py:1034

### Context\_Integration/\_\_init\_\_.py {#webapp-parser-context-integration-init-py}

> Context integration module for election results.

### Context\_Integration/context\_coordinator.py {#webapp-parser-context-integration-context-coordinator-py}

> context_coordinator.py

- Definitions:
  - function: `get\_semantic\_score` (line 106)
  - function: `merge\_and\_rank\_candidates` (line 175)
  - function: `dynamic\_state\_county\_detection` (line 265)
  - class: `ContextCoordinator` (line 866)
- Imports:
  - **Standard Library** (17):
    - `import hashlib as hashlib` (line 14)
    - `import os as os` (line 16)
    - `import re as re` (line 17)
    - `import subprocess as subprocess` (line 18)
    - `import threading as threading` (line 19)
    - `from collections import Counter` (line 20)
    - `from collections import defaultdict` (line 20)
    - `from collections.abc import Mapping` (line 21)
    - `from datetime import datetime` (line 22)
    - `from datetime import timezone` (line 22)
    - `from typing import Any` (line 23)
    - `from typing import Callable` (line 23)
    - `from typing import Dict` (line 23)
    - `from typing import List` (line 23)
    - `from typing import Optional` (line 23)
    - `from typing import Tuple` (line 23)
    - `from urllib.parse import urlparse` (line 24)
  - **Third-party** (2):
    - `import numpy as np` (line 26)
    - `import orjson as orjson` (line 27)
  - **Local/Project** (75):
    - `from __future__ import annotations` (line 11)
    - `import difflib as difflib` (line 13)
    - `import numbers as numbers` (line 15)
    - `from rapidfuzz import fuzz` (line 28)
    - `from rapidfuzz import process` (line 28)
    - `from sklearn.preprocessing import LabelEncoder` (line 29)
    - `from config import BATCH_MAX_WORKERS` (line 31)
    - `from config import CONTEXT_LIBRARY_PATH` (line 31)
    - `from config import LOG_DIR` (line 31)
    - `from config import PROJECT_ROOT` (line 31)
    - `from handlers.batch_handler import BatchProcessor` (line 32)
    - `from services.election_data_services import ElectionDataService` (line
      33)
    - `from utils.browser_utils import safe_click` (line 34)
    - `from utils.browser_utils import safe_count` (line 34)
    - `from utils.browser_utils import safe_evaluate` (line 34)
    - `from utils.browser_utils import safe_get_attribute` (line 34)
    - `from utils.browser_utils import safe_inner_text` (line 34)
    - `from utils.browser_utils import safe_is_enabled` (line 34)
    - `from utils.browser_utils import safe_is_visible` (line 34)
    - `from utils.browser_utils import safe_locator` (line 34)
    - `from utils.browser_utils import safe_nth` (line 34)
    - `from utils.browser_utils import safe_wait_for_timeout` (line 34)
    - `from utils.browser_utils import scan_buttons_with_progress` (line 34)
    - `from utils.html_scanner import deduplicate_pattern_kb` (line 47)
    - `from utils.html_scanner import get_segment_embedding` (line 47)
    - `from utils.html_scanner import load_pattern_kb` (line 47)
    - `from utils.logger_singleton import logger` (line 52)
    - `from utils.model_registry import ModelRegistry` (line 53)
    - `from utils.shared_logic import keyphrase_match` (line 54)
    - `from utils.shared_logic import normalize_county_name` (line 54)
    - `from utils.shared_logic import normalize_state_name` (line 54)
    - `from utils.shared_logic import safe_append` (line 54)
    - `from utils.shared_logic import safe_endswith` (line 54)
    - `from utils.shared_logic import safe_filename` (line 54)
    - `from utils.shared_logic import safe_get` (line 54)
    - `from utils.shared_logic import safe_get_first` (line 54)
    - `from utils.shared_logic import safe_isupper` (line 54)
    - `from utils.shared_logic import safe_items` (line 54)
    - `from utils.shared_logic import safe_lower` (line 54)
    - `from utils.shared_logic import safe_model_encode` (line 54)
    - `from utils.shared_logic import safe_replace` (line 54)
    - `from utils.shared_logic import safe_similarity` (line 54)
    - `from utils.shared_logic import safe_startswith` (line 54)
    - `from utils.shared_logic import safe_strip` (line 54)
    - `from utils.shared_logic import safe_tolist` (line 54)
    - `from utils.shared_logic import sync_type_and_election_types` (line 54)
    - `from utils.spacy_utils import extract_dates` (line 74)
    - `from utils.spacy_utils import extract_entities` (line 74)
    - `from utils.spacy_utils import extract_locations` (line 74)
    - `from Context_Library.constants import BALLOT_TYPES` (line 75)
- Task markers:
  - L911 **WARNING**: ("\[ALERT MONITOR\] Thread did not stop cleanly.")
  - L999 **WARNING**: ({
  - L1000 **WARNING**: ",
  - L1754 **WARNING**: (f"\[yellow\]Integrity issues:\[/yellow\]
    {issues\['integrity_issues'\]}")
  - L2017 **WARNING**: (f"\[ContextCoordinator\] No table structure found for
    contest: {contest}")
  - L2322 **WARNING**: (f"\[get_feedback_pattern_kb\] Skipping corrupt line:
    {e}")
  - L2434 **WARNING**: ("\[group_dom_nodes_by_label\] No organized DOM parts.
    (Further warnings suppressed)")
  - L2436 **WARNING**: (f"\[group_dom_nodes_by_label\] No organized DOM parts.
    (Occurred {ContextCoordinator._dom_parts_warning_count} times)")
  - L2441 **WARNING**: ("\[group_dom_nodes_by_label\] No DOM nodes found.")
  - L2459 **WARNING**: ("\[submit_user_feedback\] ContextOrganizer has no
    submit_user_feedback method.")
  - L2487 **WARNING**: (f"\[correct_and_update_contest\] Contest {contest_id}
    missing type/election_types after sync.")
  - L2511 **WARNING**: ("\[print_contest_summary\] No organized contests to
    summarize.")
  - L2524 **WARNING**: ("\[plot_contest_distribution\] No organized contests to
    plot.")
  - L2591 **WARNING**: ("No organized DOM parts.")
  - L2594 **WARNING**: ("No organized DOM parts. (Further warnings suppressed)")
  - L2605 **WARNING**: ("\[get_contest_groups\] No contest groups found.")
  - L2614 **WARNING**: ("\[get_panel_groups\] No panel groups found.")
  - L2623 **WARNING**: ("\[get_button_groups\] No button groups found.")
  - L2632 **WARNING**: ("\[get_table_groups\] No table groups found.")
  - L2641 **WARNING**: ("\[get_relationships\] No organized context.")
  - L2749 **WARNING**: (f"\[fuzzy_score\] One or both inputs are empty:
    a='{a_str}', b='{b_str}'")
  - L2755 **WARNING**: (f"\[fuzzy_score\] One or both inputs are too short:
    a='{a_str}', b='{b_str}'")
  - L3229 **WARNING**: (f"\[extract_field\] Unknown field_type: {field_type}")
  - L3487 **WARNING**: (f"\[get_full_contest\] Contest {contest_id} missing
    type/election_types after sync.")
  - L3572 **WARNING**: (f"\[list_tables\] Table '{tbl}' missing metadata or
    columns.")
  - L3604 **WARNING**: (f"\[get_table_metadata\] Table '{table_name}' missing
    columns.")
  - L3622 **WARNING**: (f"\[check_missing_tables\] Missing tables: {missing}")
  - L3683 **WARNING**: (f"\[save_table_structure\] Failed to save structure for
    contest: {contest}")
  - L3858 **WARNING**: (f"\[get_best_button_advanced\] Contest argument was not
    a dict. Converted to: {contest}")
  - L3862 **WARNING**: (f"\[get_best_button_advanced\] Keywords argument was not
    a list. Converted to: {keywords}")
  - L3866 **WARNING**: (f"\[get_best_button_advanced\] Context argument was not
    a dict. Converted to: {context}")
  - L3873 **WARNING**: ("\[get_best_button_advanced\]_semantic_model is not set
    or is not an object. Using None.")
  - L4018 **WARNING**: (f"\[yellow\]\[Coordinator\] Button '{cand.get('label')}'
    rejected, retrying...\[/yellow\]")
- Outgoing cross-module calls (sample):
  - utils.shared\_logic.safe\_model\_encode (line 143)
  - utils.shared\_logic.safe\_model\_encode (line 144)
  - utils.logger\_singleton.logger.debug (line 146)
  - util.pytorch\_cos\_sim (line 149)
  - cos\_sim.item (line 151)
  - cos\_sim.numpy (line 153)
  - arr.flatten (line 154)
  - utils.logger\_singleton.logger.error (line 159)
  - utils.logger\_singleton.logger.error (line 164)
  - utils.shared\_logic.safe\_lower (line 168)
  - text1.split (line 168)
  - utils.shared\_logic.safe\_lower (line 169)
  - text2.split (line 169)
  - utils.shared\_logic.safe\_get (line 186)
  - utils.shared\_logic.safe\_get (line 188)
  - utils.shared\_logic.safe\_get (line 188)
  - seen.add (line 190)
  - all\_candidates.append (line 191)
  - utils.shared\_logic.safe\_get (line 194)
  - utils.shared\_logic.safe\_get (line 195)
  - utils.shared\_logic.safe\_get (line 198)
  - utils.shared\_logic.safe\_get (line 199)
  - utils.shared\_logic.safe\_get (line 200)
  - utils.shared\_logic.safe\_get (line 201)
  - utils.shared\_logic.safe\_get (line 204)
  - utils.shared\_logic.safe\_get (line 205)
  - utils.shared\_logic.safe\_get (line 210)
  - utils.shared\_logic.safe\_lower (line 212)
  - label.strip (line 212)
  - utils.shared\_logic.safe\_lower (line 212)
  - contest\_title.strip (line 212)
  - utils.shared\_logic.keyphrase\_match (line 216)
  - utils.shared\_logic.keyphrase\_match (line 216)
  - difflib.SequenceMatcher (line 221)
  - utils.shared\_logic.safe\_lower (line 221)
  - utils.shared\_logic.safe\_lower (line 221)
  - utils.shared\_logic.safe\_get (line 227)
  - utils.shared\_logic.safe\_get (line 233)
  - utils.shared\_logic.safe\_get (line 234)
  - utils.shared\_logic.safe\_lower (line 235)
  - utils.shared\_logic.safe\_lower (line 237)
  - all\_candidates.sort (line 255)
  - utils.shared\_logic.safe\_get (line 257)
  - utils.shared\_logic.safe\_get (line 258)
  - utils.shared\_logic.safe\_get (line 259)
  - state\_to\_county.keys (line 293)
  - state\_to\_county.values (line 294)
  - utils.shared\_logic.normalize\_county\_name (line 295)
  - county\_to\_precinct.values (line 296)
  - utils.shared\_logic.normalize\_county\_name (line 297)
- Inbound references:
  - get\_semantic\_score ← context_coordinator.py:225
  - get\_semantic\_score ← context_coordinator.py:230
  - get\_semantic\_score ← context_coordinator.py:3340
  - get\_semantic\_score ← context_coordinator.py:3359
  - get\_semantic\_score ← context_coordinator.py:3368
  - get\_semantic\_score ← context_coordinator.py:3404
  - get\_semantic\_score ← context_coordinator.py:3417
  - get\_semantic\_score ← context_coordinator.py:3544
  - get\_semantic\_score ← context_coordinator.py:3730
  - merge\_and\_rank\_candidates ← context_coordinator.py:2386
  - merge\_and\_rank\_candidates ← context_coordinator.py:3984
  - dynamic\_state\_county\_detection ← state_router.py:404
  - dynamic\_state\_county\_detection ← context_organizer.py:1769
  - dynamic\_state\_county\_detection ← shared_logic.py:2050
  - ContextCoordinator ← html_election_parser.py:1328
  - ContextCoordinator ← html_election_parser.py:1495
  - ContextCoordinator ← html_election_parser.py:1790
  - ContextCoordinator ← state_router.py:400
  - ContextCoordinator ← html_handler.py:103
  - ContextCoordinator ← state_handler_base.py:210
  - ContextCoordinator ← example_state.py:34
  - ContextCoordinator ← example_county.py:27
  - ContextCoordinator ← rockland.py:165
  - ContextCoordinator ← westchester.py:69
  - ContextCoordinator ← dom_snapshot.py:182
  - ContextCoordinator ← contest_selector.py:903
  - ContextCoordinator ← contest_selector.py:1057
  - ContextCoordinator ← dynamic_table_extractor.py:285
  - ContextCoordinator ← dynamic_table_extractor.py:313
  - ContextCoordinator ← dynamic_table_extractor.py:479
  - ContextCoordinator ← dynamic_table_extractor.py:498
  - ContextCoordinator ← dynamic_table_extractor.py:992
  - ContextCoordinator ← dynamic_table_extractor.py:1036
  - ContextCoordinator ← dynamic_table_extractor.py:1066
  - ContextCoordinator ← dynamic_table_extractor.py:1084
  - ContextCoordinator ← dynamic_table_extractor.py:1116
  - ContextCoordinator ← dynamic_table_extractor.py:1146
  - ContextCoordinator ← html_scanner.py:733
  - ContextCoordinator ← html_scanner.py:1220
  - ContextCoordinator ← html_scanner.py:2610
  - ContextCoordinator ← html_scanner.py:2859
  - ContextCoordinator ← html_scanner.py:2965
  - ContextCoordinator ← html_scanner.py:3244
  - ContextCoordinator ← output_utils.py:99
  - ContextCoordinator ← table_builder.py:771
  - ContextCoordinator ← table_builder.py:1156
  - ContextCoordinator ← table_builder.py:1489
  - ContextCoordinator ← table_builder.py:1552

### Context\_Integration/context\_organizer.py {#webapp-parser-context-integration-context-organizer-py}

> context_organizer.py

- Definitions:
  - function: `get\_loading\_indicator` (line 63)
  - function: `ensure\_dict` (line 66)
  - function: `remove\_functions` (line 79)
  - function: `contest\_hash` (line 87)
  - function: `repair\_dom\_segments` (line 99)
  - function: `\_defensive\_dom\_check` (line 161)
  - class: `ContextOrganizer` (line 182)
- Imports:
  - **Standard Library** (9):
    - `import itertools as itertools` (line 10)
    - `import os as os` (line 11)
    - `import re as re` (line 12)
    - `from collections import Counter` (line 14)
    - `from collections import defaultdict` (line 14)
    - `from collections.abc import Hashable` (line 15)
    - `from datetime import datetime` (line 16)
    - `from datetime import timezone` (line 16)
    - `from typing import Any` (line 18)
  - **Third-party** (3):
    - `import numpy as np` (line 21)
    - `import orjson as orjson` (line 22)
    - `from sqlalchemy.exc import SQLAlchemyError` (line 24)
  - **Local/Project** (41):
    - `from __future__ import annotations` (line 8)
    - `import types as types` (line 13)
    - `from difflib import get_close_matches` (line 17)
    - `import matplotlib.pyplot as plt` (line 20)
    - `from rich.table import Table` (line 23)
    - `from config import CONTEXT_DB_PATH` (line 26)
    - `from config import CONTEXT_LIBRARY_PATH` (line 26)
    - `from config import LOG_DIR` (line 26)
    - `from services.election_data_services import ElectionDataService` (line
      27)
    - `from utils.html_scanner import load_context_cache_from_disk` (line 28)
    - `from utils.logger_singleton import console` (line 29)
    - `from utils.logger_singleton import logger` (line 29)
    - `from utils.misc_utils import load_output_cache` (line 30)
    - `from utils.misc_utils import load_processed_urls` (line 30)
    - `from utils.model_registry import ModelRegistry` (line 31)
    - `from utils.shared_logic import flatten_raw_field` (line 32)
    - `from utils.shared_logic import infer_contest_fields` (line 32)
    - `from utils.shared_logic import normalize_label` (line 32)
    - `from utils.shared_logic import safe_add` (line 32)
    - `from utils.shared_logic import safe_db_call` (line 32)
    - `from utils.shared_logic import safe_filename` (line 32)
    - `from utils.shared_logic import safe_get_first` (line 32)
    - `from utils.shared_logic import safe_items` (line 32)
    - `from utils.shared_logic import safe_model_encode` (line 32)
    - `from utils.shared_logic import safe_update` (line 32)
    - `from utils.shared_logic import scan_environment` (line 32)
    - `from utils.shared_logic import sync_type_and_election_types` (line 32)
    - `from Context_Library.constants import BALLOT_TYPES` (line 46)
    - `from Context_Library.constants import CANDIDATE_KEYWORDS` (line 46)
    - `from Context_Library.constants import CONTEST_KEYWORDS` (line 46)
    - `from Context_Library.constants import LOCATION_KEYWORDS` (line 46)
    - `from Context_Library.constants import MISC_FOOTER_KEYWORDS` (line 46)
    - `from Context_Library.constants import PARTY_KEYWORDS` (line 46)
    - `from Context_Library.constants import PERCENT_KEYWORDS` (line 46)
    - `from Context_Library.constants import TOTAL_KEYWORDS` (line 46)
    - `from Integrity_check import detect_anomalies_with_ml` (line 56)
    - `from Integrity_check import election_integrity_checks` (line 56)
    - `from Integrity_check import print_ml_anomalies` (line 56)
    - `from librarian import clean_for_json` (line 57)
    - `from librarian import load_context_library` (line 57)
    - `from librarian import update_context_library` (line 57)
- Task markers:
  - L288 **WARNING**: (
  - L413 **WARNING**: (f"\[CONTEST\] Skipping contest with suspiciously large or
    missing title: {str(title)\[:100\]}...")
  - L501 **WARNING**: (f"\[CONTEST\] Filtered out {len(filtered_out)} contests
    due to missing required fields.")
  - L503 **WARNING**: (f"  \[Filtered\] {reason}: {str(c)\[:100\]}...")
  - L506 **WARNING**: ("\[CONTEST\] No contests with required fields for
    downstream output.")
  - L822 **WARNING**: (f"\[ML\] Anomaly index {idx} out of range for contests
    list of length {len(contests)}")
  - L1665 **WARNING**: (f"  \[yellow\]{title}\[/yellow\]: {fixes}")
  - L1671 **WARNING**: (f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Duplicate
    contest detected.\n  \[dim\]Context:\[/dim\] {contest}")
  - L1673 **WARNING**: (f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Contest
    missing location info.\n  \[dim\]Context:\[/dim\] {contest}")
  - L1675 **WARNING**: (f"\[bold yellow\]\[INTEGRITY\]\[/bold yellow\] Contest
    missing year.\n  \[dim\]Context:\[/dim\] {contest}")
  - L2146 **WARNING**: (f"\[ContextOrganizer\] Could not update context library
    with feedback: {e}")
  - L2223 **WARNING**: (f"\[CONTEXT ORGANIZER\] No table structure found for
    contest: {contest}")
- Outgoing cross-module calls (sample):
  - utils.misc\_utils.load\_processed\_urls (line 59)
  - utils.misc\_utils.load\_output\_cache (line 60)
  - itertools.cycle (line 61)
  - v.get (line 73)
  - v.get (line 73)
  - obj.items (line 81)
  - c\_dict.get (line 90)
  - c\_dict.get (line 92)
  - c\_dict.get (line 93)
  - c\_dict.get (line 94)
  - c\_dict.get (line 95)
  - seg.get (line 108)
  - seg.get (line 115)
  - normalized\_children.append (line 119)
  - normalized\_children.append (line 123)
  - seg.get (line 127)
  - parent\_idx.get (line 129)
  - seg.get (line 132)
  - seg.get (line 140)
  - idx\_map.get (line 141)
  - child.get (line 142)
  - seg.get (line 142)
  - seg.get (line 143)
  - seg.get (line 149)
  - seg.get (line 151)
  - idx\_map.get (line 152)
  - child\_node.get (line 155)
  - valid\_children.append (line 157)
  - dom\_parts\_dict.get (line 175)
  - utils.shared\_logic.safe\_get\_first (line 176)
  - utils.logger\_singleton.logger.error (line 179)
  - librarian.load\_context\_library (line 211)
  - self.\_default\_library (line 211)
  - utils.logger\_singleton.logger.error (line 213)
  - utils.logger\_singleton.logger.debug (line 215)
  - utils.logger\_singleton.logger.error (line 217)
  - utils.misc\_utils.load\_processed\_urls (line 220)
  - utils.misc\_utils.load\_output\_cache (line 221)
  - utils.html\_scanner.load\_context\_cache\_from\_disk (line 224)
  - services.election\_data\_services.ElectionDataService (line 227)
  - self.\_resolve\_embedding\_model (line 229)
  - utils.logger\_singleton.logger.error (line 231)
  - utils.model\_registry.ModelRegistry.get\_sentence\_transformer (line 278)
  - utils.logger\_singleton.logger.info (line 280)
  - utils.logger\_singleton.logger.info (line 285)
  - utils.logger\_singleton.logger.warning (line 288)
  - item.get (line 305)
  - item.get (line 306)
  - deduped.append (line 307)
  - seen.add (line 308)
- Inbound references:
  - get\_loading\_indicator ← context_organizer.py:1873
  - ensure\_dict ← context_organizer.py:674
  - ensure\_dict ← context_organizer.py:682
  - ensure\_dict ← context_organizer.py:690
  - ensure\_dict ← context_organizer.py:698
  - ensure\_dict ← context_organizer.py:706
  - ensure\_dict ← context_organizer.py:714
  - ensure\_dict ← context_organizer.py:722
  - ensure\_dict ← context_organizer.py:730
  - ensure\_dict ← context_organizer.py:738
  - ensure\_dict ← context_organizer.py:746
  - remove\_functions ← context_organizer.py:81
  - remove\_functions ← context_organizer.py:83
  - remove\_functions ← context_organizer.py:2192
  - contest\_hash ← context_organizer.py:995
  - repair\_dom\_segments ← context_organizer.py:339
  - repair\_dom\_segments ← context_organizer.py:1847
  - \_defensive\_dom\_check ← context_organizer.py:346
  - ContextOrganizer ← web_pipeline.py:846
  - ContextOrganizer ← html_scanner.py:1581
  - ContextOrganizer ← html_scanner.py:3305

### Context\_Integration/librarian.py {#webapp-parser-context-integration-librarian-py}

- Top-of-file comments:

```python

# webapp/parser/Context\_Integration/librarian.py

# -----------------------------------------------------------------------------------

# This file contains functions to manage the context library for the HTML parser,

# including loading, saving, and updating the context library, as well as

# It also includes utilities for logging unknown HTML tags and attributes,

# extending context library structures, and handling ML feedback.

#

# SECURITY: All file operations are validated using safe\_path() to prevent path traversal attacks.

# -----------------------------------------------------------------------------------

```

- Definitions:
  - function: `\_cap\_log\_value` (line 82)
  - function: `get\_vocab\_constant` (line 105)
  - function: `safe\_path` (line 119)
  - function: `get\_safe\_log\_path` (line 148)
  - function: `atomic\_write\_json` (line 170)
  - function: `extend\_panel\_tags` (line 233)
  - function: `extend\_heading\_tags` (line 237)
  - function: `extend\_html\_tags` (line 241)
  - function: `\_normalize\_custom\_attr\_pattern` (line 245)
  - function: `\_dedupe\_custom\_attr\_pattern\_strings` (line 261)
  - function: `extend\_custom\_attr\_patterns` (line 277)
  - function: `extend\_location\_keywords` (line 287)
  - function: `extend\_candidate\_keywords` (line 291)
  - function: `extend\_ballot\_types` (line 295)
  - function: `safe\_join` (line 299)
  - function: `clean\_for\_json` (line 315)
  - function: `robust\_orjson\_loads` (line 331)
  - function: `load\_context\_library` (line 339)
  - function: `update\_context\_library` (line 439)
  - function: `backup\_context\_library` (line 455)
  - function: `save\_context\_library` (line 525)
  - function: `merge\_and\_save\_context\_library` (line 579)
  - function: `\_dedupe\_string\_list` (line 588)
  - function: `dedupe\_context\_library\_fields` (line 605)
  - function: `update\_context\_library\_field` (line 640)
  - function: `update\_domain\_selector\_cache` (line 652)
  - function: `get\_domain\_selectors` (line 673)
  - function: `log\_selector\_attempt` (line 678)
  - function: `\_get\_log\_path` (line 701)
  - function: `\_deduplicate\_jsonl\_log` (line 717)
  - function: `log\_unknown\_tag` (line 752)
  - function: `log\_unknown\_attr` (line 775)
  - function: `integrate\_feedback` (line 801)
  - function: `lookup\_state` (line 816)
  - function: `get\_state\_abbr` (line 841)
  - function: `lookup\_county` (line 854)
  - function: `normalize\_segment\_text` (line 880)
  - function: `get\_canonical\_segment\_label` (line 886)
  - function: `cache\_segment\_label` (line 891)
  - function: `get\_cached\_segment\_label` (line 895)
  - function: `self\_heal\_context\_library` (line 900)
  - function: `parse\_filename\_for\_location` (line 941)
- Imports:
  - **Standard Library** (19):
    - `import argparse as argparse` (line 12)
    - `import os as os` (line 13)
    - `import random as random` (line 14)
    - `import re as re` (line 15)
    - `import shutil as shutil` (line 16)
    - `import subprocess as subprocess` (line 17)
    - `import sys as sys` (line 18)
    - `import tempfile as tempfile` (line 19)
    - `import threading as threading` (line 20)
    - `import time as time` (line 21)
    - `from datetime import datetime` (line 22)
    - `from datetime import timezone` (line 22)
    - `from pathlib import Path` (line 23)
    - `from typing import Any` (line 24)
    - `from typing import Dict` (line 24)
    - `from typing import List` (line 24)
    - `from typing import Optional` (line 24)
    - `from typing import Pattern` (line 24)
    - `from typing import Set` (line 24)
  - **Third-party** (2):
    - `import numpy as np` (line 26)
    - `import orjson as orjson` (line 27)
  - **Local/Project** (24):
    - `from __future__ import annotations` (line 10)
    - `from config import BASE_DIR` (line 29)
    - `from config import CONTEXT_LIBRARY_PATH` (line 29)
    - `from config import LOG_DIR` (line 29)
    - `from config import PROJECT_ROOT` (line 29)
    - `from utils.logger_singleton import logger` (line 30)
    - `from utils.misc_utils import file_hash` (line 31)
    - `from utils.shared_logic import safe_append` (line 32)
    - `from utils.shared_logic import safe_filename` (line 32)
    - `from utils.shared_logic import safe_get` (line 32)
    - `from utils.shared_logic import safe_merge_defaults` (line 32)
    - `from utils.shared_logic import safe_setdefault` (line 32)
    - `from Context_Library.constants import BALLOT_TYPES` (line 39)
    - `from Context_Library.constants import CANDIDATE_KEYWORDS` (line 39)
    - `from Context_Library.constants import CANONICAL_SEGMENT_LABELS` (line 39)
    - `from Context_Library.constants import CANONICAL_STATE_ABBR` (line 39)
    - `from Context_Library.constants import CUSTOM_ATTR_PATTERNS` (line 39)
    - `from Context_Library.constants import HEADING_TAGS` (line 39)
    - `from Context_Library.constants import HTML_TAGS` (line 39)
    - `from Context_Library.constants import KNOWN_STATE_TO_COUNTY_MAP` (line
      39)
    - `from Context_Library.constants import LOCATION_KEYWORDS` (line 39)
    - `from Context_Library.constants import PANEL_TAGS` (line 39)
    - `from Context_Library.constants import STATE_ABBR` (line 39)
    - `from vocab.loader import get_vocab_loader` (line 52)
- Task markers:
  - L915 **WARNING**: (f"\n\[LIBRARIAN SELF-HEAL\] Attempt {attempt}...")
  - L925 **WARNING**: ("\[LIBRARIAN SELF-HEAL\] Misalignments found. Launching
    manual_correction...")
  - L928 **WARNING**: (f"\[LIBRARIAN SELF-HEAL\] Sleeping {cooldown}s before
    rescanning...")
- Outgoing cross-module calls (sample):
  - threading.Lock (line 54)
  - pathlib.Path (line 59)
  - pathlib.Path (line 60)
  - pathlib.Path (line 61)
  - pathlib.Path (line 62)
  - value.items (line 94)
  - vocab.loader.get\_vocab\_loader (line 113)
  - loader.load\_mapping (line 115)
  - loader.load\_canonical (line 116)
  - pathlib.Path (line 136)
  - pathlib.Path (line 138)
  - path.relative\_to (line 141)
  - log\_dir.mkdir (line 156)
  - utils.shared\_logic.safe\_filename (line 159)
  - path.with\_suffix (line 181)
  - path.with\_suffix (line 182)
  - tmp\_path.exists (line 189)
  - tmp\_path.unlink (line 191)
  - backup\_path.exists (line 196)
  - backup\_path.unlink (line 198)
  - tf.write (line 204)
  - orjson.dumps (line 204)
  - path.exists (line 207)
  - shutil.copy2 (line 208)
  - shutil.move (line 213)
  - os.remove (line 218)
  - time.sleep (line 221)
  - tmp\_path.exists (line 226)
  - tmp\_path.unlink (line 228)
  - t.lower (line 235)
  - t.lower (line 239)
  - t.lower (line 243)
  - value.strip (line 247)
  - re.compile (line 250)
  - re.compile (line 259)
  - value.strip (line 266)
  - seen.add (line 273)
  - deduped.append (line 274)
  - Context\_Library.constants.CUSTOM\_ATTR\_PATTERNS.append (line 284)
  - existing.add (line 285)
  - k.lower (line 289)
  - k.lower (line 293)
  - Context\_Library.constants.BALLOT\_TYPES.extend (line 297)
  - utils.logger\_singleton.logger.debug (line 310)
  - obj.items (line 317)
  - obj.tolist (line 323)
  - orjson.loads (line 333)
  - orjson.loads (line 335)
  - val.encode (line 335)
  - safe\_path\_obj.exists (line 357)
- Inbound references:
  - safe\_path ← librarian.py:164
  - safe\_path ← librarian.py:179
  - safe\_path ← librarian.py:185
  - safe\_path ← librarian.py:186
  - safe\_path ← librarian.py:308
  - safe\_path ← librarian.py:353
  - safe\_path ← librarian.py:393
  - safe\_path ← librarian.py:465
  - safe\_path ← librarian.py:481
  - safe\_path ← librarian.py:539
  - safe\_path ← librarian.py:560
  - safe\_path ← librarian.py:713
  - safe\_path ← librarian.py:724
  - safe\_path ← librarian.py:909
  - safe\_path ← librarian.py:919
  - safe\_path ← log_cache_cleaner_bot.py:408
  - safe\_path ← manual_correction_bot.py:101
  - safe\_path ← manual_correction_bot.py:120
  - safe\_path ← manual_correction_bot.py:231
  - safe\_path ← manual_correction_bot.py:237
  - safe\_path ← manual_correction_bot.py:238
  - safe\_path ← manual_correction_bot.py:339
  - safe\_path ← manual_correction_bot.py:378
  - safe\_path ← manual_correction_bot.py:392
  - safe\_path ← manual_correction_bot.py:408
  - safe\_path ← manual_correction_bot.py:438
  - safe\_path ← manual_correction_bot.py:471
  - safe\_path ← manual_correction_bot.py:494
  - safe\_path ← manual_correction_bot.py:498
  - safe\_path ← manual_correction_bot.py:541
  - safe\_path ← manual_correction_bot.py:559
  - safe\_path ← manual_correction_bot.py:584
  - safe\_path ← manual_correction_bot.py:602
  - safe\_path ← manual_correction_bot.py:605
  - safe\_path ← manual_correction_bot.py:689
  - safe\_path ← manual_correction_bot.py:756
  - safe\_path ← manual_correction_bot.py:1075
  - safe\_path ← manual_correction_bot.py:1093
  - safe\_path ← manual_correction_bot.py:1100
  - safe\_path ← manual_correction_bot.py:1103
  - safe\_path ← manual_correction_bot.py:1111
  - safe\_path ← manual_correction_bot.py:1112
  - safe\_path ← manual_correction_bot.py:1140
  - safe\_path ← manual_correction_bot.py:1168
  - safe\_path ← manual_correction_bot.py:1180
  - safe\_path ← manual_correction_bot.py:1277
  - safe\_path ← manual_correction_bot.py:1278
  - safe\_path ← manual_correction_bot.py:1287
  - safe\_path ← manual_correction_bot.py:1310
  - safe\_path ← manual_correction_bot.py:1430

### Context\_Integration/library/entity\_confidence\_map.py {#webapp-parser-context-integration-library-entity-confidence-map-py}

> Entity Confidence Mapping: Weighted Signal Catalog for Decision Gates

- Definitions:
  - class: `DecisionCode` (line 21)
  - class: `SignalType` (line 28)
  - class: `AnomalyType` (line 42)
  - class: `OverrideTrigger` (line 54)
  - class: `SignalCoefficient` (line 65)
  - class: `AnomalyCoefficient` (line 75)
  - class: `ConfidenceCautionResult` (line 85)
  - class: `EntityConfidenceMap` (line 287)
  - function: `get\_confidence\_map` (line 466)
- Imports:
  - **Standard Library** (5):
    - `from dataclasses import dataclass` (line 16)
    - `from enum import Enum` (line 17)
    - `from typing import List` (line 18)
    - `from typing import Optional` (line 18)
    - `from typing import Tuple` (line 18)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 14)
- Outgoing cross-module calls (sample):
  - catalog\_map.get (line 315)
  - entity\_type.lower (line 315)
  - catalog.get (line 316)
  - self.get\_signal\_coefficient (line 355)
  - observed\_signals.append (line 360)
  - self.get\_anomaly\_coefficient (line 371)
  - observed\_anomalies.append (line 376)
  - self.\_generate\_reasoning (line 401)
  - entity\_type.upper (line 442)
  - parts.append (line 446)
  - parts.append (line 450)
  - parts.append (line 454)
  - parts.append (line 456)
  - parts.append (line 457)
- Inbound references:
  - SignalCoefficient ← entity_confidence_map.py:107
  - SignalCoefficient ← entity_confidence_map.py:114
  - SignalCoefficient ← entity_confidence_map.py:121
  - SignalCoefficient ← entity_confidence_map.py:128
  - SignalCoefficient ← entity_confidence_map.py:135
  - SignalCoefficient ← entity_confidence_map.py:145
  - SignalCoefficient ← entity_confidence_map.py:152
  - SignalCoefficient ← entity_confidence_map.py:159
  - SignalCoefficient ← entity_confidence_map.py:166
  - SignalCoefficient ← entity_confidence_map.py:176
  - SignalCoefficient ← entity_confidence_map.py:183
  - SignalCoefficient ← entity_confidence_map.py:190
  - SignalCoefficient ← entity_confidence_map.py:197
  - SignalCoefficient ← entity_confidence_map.py:207
  - SignalCoefficient ← entity_confidence_map.py:214
  - SignalCoefficient ← entity_confidence_map.py:221
  - AnomalyCoefficient ← entity_confidence_map.py:235
  - AnomalyCoefficient ← entity_confidence_map.py:242
  - AnomalyCoefficient ← entity_confidence_map.py:249
  - AnomalyCoefficient ← entity_confidence_map.py:256
  - AnomalyCoefficient ← entity_confidence_map.py:263
  - AnomalyCoefficient ← entity_confidence_map.py:270
  - AnomalyCoefficient ← entity_confidence_map.py:277
  - ConfidenceCautionResult ← entity_confidence_map.py:413
  - EntityConfidenceMap ← entity_confidence_map.py:470

### Context\_Integration/location\_inference.py {#webapp-parser-context-integration-location-inference-py}

- Definitions:
  - function: `infer\_county\_from\_lines` (line 11)
- Imports:
  - **Standard Library** (4):
    - `import re as re` (line 3)
    - `from collections import Counter` (line 4)
    - `from typing import Sequence` (line 5)
    - `from typing import Tuple` (line 5)
  - **Local/Project** (4):
    - `from __future__ import annotations` (line 1)
    - `from utils.shared_logic import normalize_county_name` (line 7)
    - `from utils.shared_logic import normalize_state_name` (line 7)
    - `from Context_Library.constants import KNOWN_STATE_TO_COUNTY_MAP` (line 8)
- Outgoing cross-module calls (sample):
  - utils.shared\_logic.normalize\_state\_name (line 31)
  - Context\_Library.constants.KNOWN\_STATE\_TO\_COUNTY\_MAP.get (line 35)
  - utils.shared\_logic.normalize\_county\_name (line 40)
  - normalized\_lookup.items (line 43)
  - collections.Counter (line 47)
  - re.sub (line 54)
  - re.sub (line 55)
  - normalized\_lookup.keys (line 56)
  - hits.most\_common (line 66)

### Context\_Integration/vocab/loader.py {#webapp-parser-context-integration-vocab-loader-py}

> Vocab Loader: Safe, audited vocabulary file management for confidence/caution framework.

- Definitions:
  - class: `VocabLoaderError` (line 31)
  - class: `VocabSecurityError` (line 36)
  - class: `VocabFileNotFound` (line 41)
  - class: `VocabIntegrityError` (line 46)
  - class: `RateLimitError` (line 51)
  - class: `VocabLoader` (line 66)
  - function: `get\_vocab\_loader` (line 358)
- Imports:
  - **Standard Library** (7):
    - `import hashlib as hashlib` (line 22)
    - `import time as time` (line 23)
    - `from pathlib import Path` (line 24)
    - `from typing import Dict` (line 25)
    - `from typing import List` (line 25)
    - `from typing import Optional` (line 25)
    - `from typing import Tuple` (line 25)
  - **Third-party** (1):
    - `from webapp.parser.utils.logger_singleton import logger` (line 27)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 58)
  - pathlib.Path (line 83)
  - self.\_make\_cache\_key (line 120)
  - self.\_load\_from\_disk (line 125)
  - self.\_make\_cache\_key (line 151)
  - self.\_load\_from\_disk (line 156)
  - self.\_parse\_mapping (line 157)
  - self.\_make\_cache\_key (line 182)
  - time.time (line 183)
  - self.\_load\_from\_disk (line 193)
  - self.\_make\_cache\_key (line 203)
  - self.\_make\_cache\_key (line 208)
  - self.\_make\_cache\_key (line 219)
  - filename.lower (line 255)
  - file\_path.relative\_to (line 267)
  - file\_path.exists (line 271)
  - file\_path.is\_file (line 274)
  - file\_path.read\_text (line 279)
  - hashlib.sha256 (line 284)
  - content.encode (line 284)
  - self.\_make\_cache\_key (line 285)
  - content.split (line 291)
  - raw\_line.strip (line 292)
  - line.startswith (line 295)
  - seen.add (line 303)
  - entries.append (line 304)
  - webapp.parser.utils.logger\_singleton.logger.info (line 309)
  - entry.split (line 340)
  - key.strip (line 341)
  - value.strip (line 342)

### Context\_Integration/vocab\_loader.py {#webapp-parser-context-integration-vocab-loader-py}

> VocabLoader: Secure, auditable vocabulary file loader for election integrity.

- Definitions:
  - class: `VocabLoaderError` (line 22)
  - class: `VocabFileNotFound` (line 27)
  - class: `VocabIntegrityError` (line 32)
  - class: `VocabSecurityError` (line 37)
  - class: `RateLimitError` (line 42)
  - class: `VocabLoader` (line 47)
  - function: `get\_vocab\_loader` (line 413)
- Imports:
  - **Standard Library** (10):
    - `import hashlib as hashlib` (line 10)
    - `import os as os` (line 11)
    - `import threading as threading` (line 12)
    - `import time as time` (line 13)
    - `from datetime import datetime` (line 14)
    - `from datetime import timezone` (line 14)
    - `from pathlib import Path` (line 15)
    - `from typing import Dict` (line 16)
    - `from typing import List` (line 16)
    - `from typing import Optional` (line 16)
  - **Third-party** (1):
    - `from webapp.parser.utils.logger_singleton import logger` (line 18)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 8)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 81)
  - threading.RLock (line 91)
  - threading.RLock (line 95)
  - self.\_resolve\_path (line 128)
  - self.\_check\_rate\_limit (line 132)
  - self.\_get\_cached (line 136)
  - self.\_audit (line 138)
  - self.\_load\_and\_hash (line 142)
  - self.\_verify\_integrity (line 145)
  - self.\_set\_cached (line 149)
  - self.\_audit (line 152)
  - self.load (line 178)
  - line.split (line 183)
  - alias.strip (line 184)
  - canonical.strip (line 184)
  - line.strip (line 187)
  - line.strip (line 187)
  - self.\_resolve\_path (line 202)
  - normalized.startswith (line 235)
  - abs\_path.exists (line 245)
  - time.time (line 257)
  - f.read (line 286)
  - hashlib.sha256 (line 291)
  - raw\_content.encode (line 291)
  - raw\_content.splitlines (line 295)
  - line.strip (line 296)
  - stripped.startswith (line 297)
  - lines.append (line 299)
  - abs\_path.stat (line 345)
  - cached.get (line 346)
  - abs\_path.stat (line 373)
  - datetime.datetime.now (line 374)
  - datetime.datetime.now (line 388)
  - webapp.parser.utils.logger\_singleton.logger.info (line 398)
- Inbound references:
  - VocabLoaderError ← vocab_loader.py:288
  - VocabLoaderError ← loader.py:281
  - VocabFileNotFound ← vocab_loader.py:246
  - VocabFileNotFound ← loader.py:92
  - VocabFileNotFound ← loader.py:272
  - VocabIntegrityError ← vocab_loader.py:318
  - VocabIntegrityError ← loader.py:300
  - VocabIntegrityError ← loader.py:336
  - VocabIntegrityError ← loader.py:345
  - VocabSecurityError ← vocab_loader.py:236
  - VocabSecurityError ← vocab_loader.py:242
  - VocabSecurityError ← loader.py:250
  - VocabSecurityError ← loader.py:256
  - VocabSecurityError ← loader.py:259
  - VocabSecurityError ← loader.py:269
  - VocabSecurityError ← loader.py:275
  - RateLimitError ← vocab_loader.py:270
  - RateLimitError ← loader.py:188
  - VocabLoader ← vocab_loader.py:431
  - VocabLoader ← loader.py:362
  - get\_vocab\_loader ← context_coordinator.py:2543

### config.py {#webapp-parser-config-py}

> Central configuration module for the Smart Elections Parser Webapp.

- Definitions:
  - function: `get\_subprocess\_env` (line 335)
  - function: `get\_supported\_formats` (line 344)
  - function: `get\_sqlalchemy\_engine` (line 380)
  - function: `get\_ocr\_config\_dict` (line 669)
  - function: `log\_ocr\_config\_summary` (line 721)
  - function: `build\_extraction\_quality\_metrics` (line 739)
  - function: `log\_extraction\_quality` (line 934)
- Imports:
  - **Standard Library** (4):
    - `import os as os` (line 8)
    - `import threading as threading` (line 9)
    - `import urllib.parse as urllib` (line 10)
    - `from pathlib import Path` (line 11)
  - **Third-party** (4):
    - `import orjson as orjson` (line 13)
    - `import psycopg2 as psycopg2` (line 14)
    - `from azure.identity import DefaultAzureCredential` (line 15)
    - `from sqlalchemy import create_engine` (line 16)
  - **Local/Project** (1):
    - `from utils.logger_singleton import logger` (line 18)
- Task markers:
  - L964 **WARNING**: ({
  - L965 **WARNING**: ",
  - L983 **NOTE**: Both DL1 and DL2 are now stored in
    CONTEXT_LIBRARY_DIR/verification
- Outgoing cross-module calls (sample):
  - dotenv.load\_dotenv (line 22)
  - pathlib.Path (line 31)
  - VOCAB\_DIR.mkdir (line 48)
  - INPUT\_DIR.mkdir (line 52)
  - OUTPUT\_DIR.mkdir (line 54)
  - UPLOADS\_DIR.mkdir (line 56)
  - URL\_LIST\_FILE.exists (line 64)
  - f.write (line 66)
  - URL\_LIST\_FILE.stat (line 67)
  - f.write (line 70)
  - LOG\_DIR.mkdir (line 78)
  - CACHE\_DIR.mkdir (line 79)
  - QUICK\_COPY\_DIR.mkdir (line 83)
  - RUN\_HISTORY\_FILE.exists (line 88)
  - RUN\_HISTORY\_FILE.touch (line 90)
  - OCR\_DEBUG\_DIR.mkdir (line 202)
  - threading.Lock (line 206)
  - s.strip (line 244)
  - s.strip (line 246)
  - s.strip (line 249)
  - s.strip (line 251)
  - URL\_ALLOWLIST\_HOSTS.append (line 262)
  - ext.startswith (line 355)
  - env\_formats.split (line 356)
  - CONTEXT\_LIBRARY\_PATH.exists (line 359)
  - orjson.loads (line 361)
  - f.read (line 361)
  - context\_library.get (line 363)
  - json.loads (line 368)
  - ext.lower (line 374)
  - ext.startswith (line 376)
  - ext.lower (line 376)
  - ext.lower (line 376)
  - utils.logger\_singleton.logger.error (line 392)
  - azure.identity.DefaultAzureCredential (line 396)
  - cred.get\_token (line 397)
  - utils.logger\_singleton.logger.info (line 398)
  - psycopg2.connect (line 399)
  - utils.logger\_singleton.logger.info (line 409)
  - sqlalchemy.create\_engine (line 410)
  - utils.logger\_singleton.logger.error (line 417)
  - utils.logger\_singleton.logger.info (line 421)
  - sqlalchemy.create\_engine (line 422)
  - utils.logger\_singleton.logger.info (line 431)
  - sqlalchemy.create\_engine (line 432)
  - x.strip (line 588)
  - x.strip (line 588)
  - x.strip (line 589)
  - x.strip (line 589)
  - x.strip (line 590)
- Inbound references:
  - get\_supported\_formats ← config.py:374
  - get\_ocr\_config\_dict ← config.py:727
  - get\_ocr\_config\_dict ← _ocr_helpers.py:51
  - get\_ocr\_config\_dict ← pdf_handler.py:4595
  - log\_ocr\_config\_summary ← pdf_handler.py:4581
  - build\_extraction\_quality\_metrics ← config.py:951
  - log\_extraction\_quality ← html_election_parser.py:1478
  - log\_extraction\_quality ← csv_handler.py:320
  - log\_extraction\_quality ← csv_handler.py:409
  - log\_extraction\_quality ← json_handler.py:970
  - log\_extraction\_quality ← json_handler.py:1343
  - log\_extraction\_quality ← json_handler.py:1430
  - log\_extraction\_quality ← pdf_handler.py:4586
  - log\_extraction\_quality ← pdf_handler.py:6136
  - log\_extraction\_quality ← xlsx_handler.py:347
  - log\_extraction\_quality ← xlsx_handler.py:429

### config\_helpers/\_ocr\_helpers.py {#webapp-parser-config-helpers-ocr-helpers-py}

> OCR Configuration Helper Functions

- Definitions:
  - function: `get\_ocr\_config\_dict` (line 8)
  - function: `log\_ocr\_config\_summary` (line 43)
- Outgoing cross-module calls (sample):
  - logger\_instance.info (line 54)
  - summary.items (line 57)

### config\_helpers/ocr\_tuning.py {#webapp-parser-config-helpers-ocr-tuning-py}

> OCR Tuning Parameters — Centralized Configuration

- Definitions:
  - class: `OcrTuningConfig` (line 46)
- Imports:
  - **Standard Library** (2):
    - `import os as os` (line 42)
    - `from typing import List` (line 43)
- Outgoing cross-module calls (sample):
  - x.strip (line 68)
  - x.strip (line 68)
  - x.strip (line 74)
  - x.strip (line 74)
  - x.strip (line 80)
  - x.strip (line 80)
  - cls.get (line 147)
  - cls.get (line 148)
  - cls.get (line 149)
  - cls.get (line 150)
  - cls.get (line 151)
  - cls.get (line 152)
  - cls.get (line 153)
  - cls.get (line 154)
  - cls.get (line 155)
  - cls.get (line 156)
  - cls.get (line 157)
  - cls.get (line 158)
  - cls.get (line 159)
  - cls.get (line 160)
  - cls.get (line 161)
  - cls.get (line 162)
  - cls.get (line 163)
  - cls.get (line 164)
  - cls.get (line 165)
  - cls.to\_dict (line 171)
  - logger.info (line 174)
  - summary.items (line 177)

### data\_manager.py {#webapp-parser-data-manager-py}

- Definitions:
  - function: `\_ensure\_parent` (line 9)
  - function: `\_atomic\_write\_lines` (line 16)
  - function: `load\_urls` (line 25)
  - function: `save\_urls` (line 41)
  - function: `add\_url` (line 60)
  - function: `remove\_url` (line 75)
  - function: `replace\_urls` (line 96)
  - function: `list\_urls\_cli` (line 99)
  - function: `list\_files` (line 109)
  - function: `copy\_file\_to\_folder` (line 133)
  - function: `run\_manager` (line 147)
- Imports:
  - **Standard Library** (2):
    - `import os as os` (line 1)
    - `import re as re` (line 2)
  - **Local/Project** (6):
    - `from config import INPUT_DIR` (line 4)
    - `from config import OUTPUT_DIR` (line 4)
    - `from config import URL_LIST_FILE` (line 4)
    - `from utils.logger_singleton import console` (line 5)
    - `from utils.logger_singleton import logger` (line 5)
    - `from utils.logger_singleton import prompt` (line 5)
- Outgoing cross-module calls (sample):
  - re.compile (line 7)
  - os.makedirs (line 12)
  - os.makedirs (line 14)
  - os.fspath (line 17)
  - f.write (line 22)
  - ln.rstrip (line 22)
  - os.replace (line 23)
  - line.strip (line 32)
  - s.startswith (line 33)
  - URL\_LINE\_RE.match (line 35)
  - urls.append (line 36)
  - m.group (line 36)
  - utils.logger\_singleton.logger.error (line 38)
  - u.strip (line 47)
  - s.lower (line 50)
  - seen.add (line 52)
  - clean.append (line 53)
  - utils.logger\_singleton.logger.info (line 56)
  - utils.logger\_singleton.logger.error (line 58)
  - url.strip (line 63)
  - u.lower (line 67)
  - existing.lower (line 67)
  - utils.logger\_singleton.logger.info (line 68)
  - urls.append (line 70)
  - utils.logger\_singleton.logger.info (line 72)
  - urls.pop (line 82)
  - utils.logger\_singleton.logger.info (line 83)
  - u.lower (line 87)
  - utils.logger\_singleton.logger.info (line 90)
  - utils.logger\_singleton.logger.info (line 102)
  - utils.logger\_singleton.logger.info (line 104)
  - utils.logger\_singleton.logger.info (line 106)
  - os.fspath (line 110)
  - utils.logger\_singleton.logger.info (line 111)
  - os.listdir (line 113)
  - utils.logger\_singleton.logger.info (line 115)
  - utils.logger\_singleton.logger.info (line 118)
  - utils.logger\_singleton.logger.info (line 121)
  - utils.logger\_singleton.prompt.prompt\_input (line 123)
  - choice.isdigit (line 124)
  - os.remove (line 128)
  - utils.logger\_singleton.logger.info (line 129)
  - utils.logger\_singleton.logger.error (line 131)
  - os.fspath (line 134)
  - utils.logger\_singleton.logger.error (line 136)
  - d.write (line 142)
  - s.read (line 142)
  - utils.logger\_singleton.logger.info (line 143)
  - utils.logger\_singleton.logger.error (line 145)
  - utils.logger\_singleton.console.panel (line 148)
- Inbound references:
  - \_ensure\_parent ← data_manager.py:18
  - \_ensure\_parent ← data_manager.py:138
  - \_atomic\_write\_lines ← data_manager.py:55
  - load\_urls ← data_manager.py:66
  - load\_urls ← data_manager.py:76
  - load\_urls ← data_manager.py:100
  - load\_urls ← html_election_parser.py:2671
  - save\_urls ← data_manager.py:71
  - save\_urls ← data_manager.py:93
  - save\_urls ← data_manager.py:97
  - remove\_url ← data_manager.py:176
  - replace\_urls ← data_manager.py:180
  - list\_urls\_cli ← data_manager.py:167
  - list\_urls\_cli ← data_manager.py:172
  - list\_files ← data_manager.py:182
  - list\_files ← data_manager.py:184
  - list\_files ← data_manager.py:192
  - list\_files ← data_manager.py:194
  - copy\_file\_to\_folder ← data_manager.py:187
  - copy\_file\_to\_folder ← data_manager.py:190
  - run\_manager ← data_manager.py:199

### data\_standardization/election\_data\_standardizer.py {#webapp-parser-data-standardization-election-data-standardizer-py}

> Election Data Standardizer

- Definitions:
  - class: `DataQualityFlag` (line 18)
  - class: `StandardizationResult` (line 31)
  - class: `PartyCodeMapper` (line 48)
  - class: `CandidateNameStandardizer` (line 124)
  - class: `VoteTypeStandardizer` (line 192)
  - class: `CountyDistrictStandardizer` (line 276)
  - class: `WriteInFlagStandardizer` (line 299)
  - class: `ElectionDataStandardizer` (line 337)
  - class: `CandidateNameMatcher` (line 465)
  - class: `PreQCResult` (line 547)
  - class: `PreQCComparisonEngine` (line 561)
  - class: `QCAutoFlagger` (line 693)
- Imports:
  - **Standard Library** (8):
    - `from dataclasses import dataclass` (line 7)
    - `from dataclasses import field` (line 7)
    - `from enum import Enum` (line 8)
    - `from typing import Any` (line 9)
    - `from typing import Dict` (line 9)
    - `from typing import List` (line 9)
    - `from typing import Optional` (line 9)
    - `from typing import Tuple` (line 9)
- Outgoing cross-module calls (sample):
  - dataclasses.field (line 34)
  - dataclasses.field (line 35)
  - dataclasses.field (line 36)
  - dataclasses.field (line 37)
  - ballot\_party.strip (line 110)
  - name.strip (line 143)
  - name.split (line 150)
  - name.strip (line 152)
  - name.upper (line 155)
  - name.strip (line 157)
  - name.split (line 162)
  - name.split (line 163)
  - cls.\_format\_name (line 164)
  - cls.\_format\_name (line 166)
  - p.strip (line 171)
  - name.split (line 171)
  - p.strip (line 171)
  - vote\_data.get (line 226)
  - cls.\_parse\_vote\_count (line 227)
  - votes.items (line 230)
  - votes.items (line 238)
  - votes.items (line 246)
  - flags.append (line 247)
  - standardized.update (line 249)
  - value.strip (line 263)
  - location.strip (line 288)
  - result.lower (line 292)
  - term.lower (line 292)
  - is\_write\_in.strip (line 317)
  - flags.append (line 324)
  - candidate\_name.upper (line 329)
  - flags.append (line 331)
  - raw\_record.copy (line 357)
  - raw\_record.get (line 361)
  - raw\_record.get (line 361)
  - raw\_record.get (line 365)
  - result.add\_flag (line 371)
  - raw\_record.get (line 375)
  - raw\_record.get (line 375)
  - raw\_record.get (line 381)
  - result.add\_flag (line 387)
  - raw\_record.get (line 391)
  - raw\_record.get (line 391)
  - result.add\_flag (line 396)
  - raw\_record.get (line 401)
  - raw\_record.get (line 402)
  - raw\_record.get (line 402)
  - raw\_record.get (line 403)
  - raw\_record.get (line 404)
  - raw\_record.get (line 405)
- Inbound references:
  - StandardizationResult ← election_data_standardizer.py:357
  - PartyCodeMapper ← election_data_standardizer.py:341
  - CandidateNameStandardizer ← election_data_standardizer.py:342
  - VoteTypeStandardizer ← election_data_standardizer.py:343
  - CountyDistrictStandardizer ← election_data_standardizer.py:344
  - WriteInFlagStandardizer ← election_data_standardizer.py:345
  - PreQCResult ← election_data_standardizer.py:649

### data\_standardization/google\_sheets\_client.py {#webapp-parser-data-standardization-google-sheets-client-py}

> Google Sheets API Client

- Definitions:
  - function: `\_build\_service\_account\_json\_from\_env` (line 23)
  - function: `\_load\_credentials\_from\_file` (line 67)
  - class: `SheetFetchResult` (line 86)
  - class: `GoogleSheetsElectionClient` (line 101)
  - function: `get\_election\_data\_client` (line 456)
  - function: `get\_worklist\_client` (line 461)
  - function: `fetch\_worklist\_overview` (line 473)
- Imports:
  - **Standard Library** (10):
    - `import json as json` (line 6)
    - `import logging as logging` (line 7)
    - `import os as os` (line 8)
    - `from dataclasses import dataclass` (line 9)
    - `from datetime import datetime` (line 10)
    - `from typing import Any` (line 11)
    - `from typing import Dict` (line 11)
    - `from typing import List` (line 11)
    - `from typing import Optional` (line 11)
    - `from typing import Tuple` (line 11)
- Outgoing cross-module calls (sample):
  - logging.getLogger (line 20)
  - required\_fields.items (line 48)
  - os.getenv (line 49)
  - missing\_fields.append (line 51)
  - logger.debug (line 56)
  - json.load (line 76)
  - logger.info (line 77)
  - logger.debug (line 80)
  - os.getenv (line 173)
  - logger.info (line 187)
  - os.getenv (line 191)
  - logger.debug (line 193)
  - os.getenv (line 197)
  - logger.debug (line 199)
  - logger.debug (line 206)
  - logger.info (line 235)
  - logger.info (line 241)
  - json.loads (line 246)
  - logger.info (line 248)
  - logger.error (line 256)
  - datetime.datetime.utcnow (line 276)
  - sheet\_name.lower (line 285)
  - service.spreadsheets (line 288)
  - sheet\_metadata.get (line 296)
  - datetime.datetime.utcnow (line 308)
  - sheet\_metadata.get (line 309)
  - service.spreadsheets (line 313)
  - result.get (line 318)
  - datetime.datetime.utcnow (line 326)
  - seen\_headers.get (line 346)
  - headers.append (line 350)
  - record.values (line 364)
  - records.append (line 365)
  - datetime.datetime.utcnow (line 367)
  - logger.info (line 369)
  - logger.error (line 383)
  - datetime.datetime.utcnow (line 389)
  - self.fetch\_sheet (line 403)
  - results.values (line 406)
  - results.values (line 407)
  - logger.info (line 409)
  - self.fetch\_sheet (line 415)
  - self.fetch\_sheet (line 419)
  - sheet\_name.lower (line 433)
  - sheet\_name.lower (line 435)
  - issues.append (line 448)
  - issues.append (line 450)
  - os.getenv (line 467)
  - os.getenv (line 484)
  - client.fetch\_sheet (line 503)
- Inbound references:
  - \_build\_service\_account\_json\_from\_env ← google_sheets_client.py:185
  - \_load\_credentials\_from\_file ← google_sheets_client.py:238
  - SheetFetchResult ← google_sheets_client.py:303
  - SheetFetchResult ← google_sheets_client.py:321
  - SheetFetchResult ← google_sheets_client.py:374
  - SheetFetchResult ← google_sheets_client.py:384
  - SheetFetchResult ← google_sheets_client.py:519
  - GoogleSheetsElectionClient ← google_sheets_client.py:458
  - GoogleSheetsElectionClient ← google_sheets_client.py:470
  - GoogleSheetsElectionClient ← database_comparison.py:137
  - get\_election\_data\_client ← Smart_Elections_Parser_Webapp.py:7321
  - get\_election\_data\_client ← Smart_Elections_Parser_Webapp.py:7388
  - get\_election\_data\_client ← Smart_Elections_Parser_Webapp.py:7454
  - get\_election\_data\_client ← Smart_Elections_Parser_Webapp.py:7471
  - get\_election\_data\_client ← Smart_Elections_Parser_Webapp.py:7552
  - get\_worklist\_client ← google_sheets_client.py:483
  - fetch\_worklist\_overview ← Smart_Elections_Parser_Webapp.py:7086
  - fetch\_worklist\_overview ← Smart_Elections_Parser_Webapp.py:7266
  - fetch\_worklist\_overview ← Smart_Elections_Parser_Webapp.py:7438
  - fetch\_worklist\_overview ← Smart_Elections_Parser_Webapp.py:7911

### db\_init.py {#webapp-parser-db-init-py}

> Database Initialization for SMART Elections Workflow

- Top-of-file comments:

```python

#!/usr/bin/env python3

```

- Definitions:
  - function: `get\_connection\_string` (line 31)
  - function: `init\_db` (line 47)
  - function: `test\_connection` (line 123)
- Imports:
  - **Standard Library** (2):
    - `import os as os` (line 20)
    - `import sys as sys` (line 21)
  - **Third-party** (3):
    - `from sqlalchemy import create_engine` (line 27)
    - `from sqlalchemy import inspect` (line 27)
    - `from sqlalchemy.orm import sessionmaker` (line 28)
  - **Local/Project** (1):
    - `from models.election_data import Base` (line 26)
- Outgoing cross-module calls (sample):
  - os.getenv (line 36)
  - db\_url.startswith (line 39)
  - db\_url.replace (line 41)
  - sqlalchemy.create\_engine (line 56)
  - engine.connect (line 59)
  - sqlalchemy.inspect (line 63)
  - inspector.get\_table\_names (line 64)
  - sqlalchemy.inspect (line 75)
  - inspector.get\_table\_names (line 76)
  - inspector.get\_indexes (line 86)
  - schema\_info.items (line 109)
  - traceback.print\_exc (line 120)
  - sqlalchemy.create\_engine (line 131)
  - sqlalchemy.orm.sessionmaker (line 132)
  - session.query (line 137)
  - session.close (line 140)
  - sys.exit (line 157)
  - sys.exit (line 160)
- Inbound references:
  - get\_connection\_string ← db_init.py:50
  - get\_connection\_string ← db_init.py:126
  - init\_db ← db_init.py:152
  - test\_connection ← db_init.py:155

### election\_fixtures.py {#webapp-parser-election-fixtures-py}

> Election results fixture loader with lazy caching (mirrors fec_lookup.py pattern).

- Definitions:
  - function: `\_get\_fixture\_dir` (line 37)
  - function: `load\_election\_results\_index` (line 42)
  - function: `load\_election\_results\_shards` (line 77)
  - function: `get\_results\_by\_state` (line 111)
  - function: `get\_results\_by\_contest` (line 166)
  - function: `find\_candidate\_by\_name` (line 207)
  - function: `get\_cache\_metrics` (line 283)
  - function: `clear\_cache` (line 289)
  - function: `reset\_metrics` (line 299)
- Imports:
  - **Standard Library** (7):
    - `import json as json` (line 11)
    - `import threading as threading` (line 12)
    - `from pathlib import Path` (line 13)
    - `from typing import Any` (line 14)
    - `from typing import Dict` (line 14)
    - `from typing import List` (line 14)
    - `from typing import Optional` (line 14)
- Outgoing cross-module calls (sample):
  - threading.RLock (line 23)
  - pathlib.Path (line 39)
  - index\_path.exists (line 59)
  - json.load (line 66)
  - shard\_dir.exists (line 93)
  - shard\_dir.glob (line 98)
  - json.load (line 101)
  - state.upper (line 127)
  - main\_index.items (line 134)
  - key.startswith (line 135)
  - key.split (line 136)
  - results.append (line 144)
  - key.split (line 151)
  - results.append (line 159)
  - state.upper (line 184)
  - name.strip (line 229)
  - idx\_dict.items (line 232)
  - key.split (line 234)
  - state.upper (line 242)
  - candidate.get (line 250)
  - fuzzy\_fuzz.token\_sort\_ratio (line 253)
  - matches.append (line 260)
  - record.get (line 263)
  - shards.items (line 271)
  - matches.sort (line 275)
- Inbound references:
  - \_get\_fixture\_dir ← election_fixtures.py:56
  - \_get\_fixture\_dir ← election_fixtures.py:90
  - load\_election\_results\_index ← election_fixtures.py:128
  - load\_election\_results\_index ← election_fixtures.py:187
  - load\_election\_results\_index ← election_fixtures.py:225
  - load\_election\_results\_index ← election_fixtures.py:318
  - load\_election\_results\_shards ← election_fixtures.py:129
  - load\_election\_results\_shards ← election_fixtures.py:188
  - load\_election\_results\_shards ← election_fixtures.py:226
  - load\_election\_results\_shards ← election_fixtures.py:321
  - get\_results\_by\_state ← election_fixtures.py:326
  - get\_results\_by\_contest ← election_fixtures.py:330
  - find\_candidate\_by\_name ← election_fixtures.py:334
  - get\_cache\_metrics ← election_fixtures.py:338

### fec\_lookup.py {#webapp-parser-fec-lookup-py}

- Definitions:
  - function: `\_normalize\_name` (line 17)
  - function: `load\_fec\_candidates` (line 38)
  - function: `get\_candidate\_by\_id` (line 56)
  - function: `\_build\_name\_index` (line 63)
  - function: `find\_candidate\_by\_name` (line 79)
- Imports:
  - **Standard Library** (5):
    - `import json as json` (line 3)
    - `import os as os` (line 4)
    - `from typing import Any` (line 5)
    - `from typing import Dict` (line 5)
    - `from typing import Optional` (line 5)
  - **Local/Project** (3):
    - `from __future__ import annotations` (line 1)
    - `from config import FUZZY_SCORER` (line 7)
    - `from config import MIN_FUZZY_SCORE_MANUAL` (line 7)
- Outgoing cross-module calls (sample):
  - s.replace (line 23)
  - s.replace (line 24)
  - s.replace (line 25)
  - p.strip (line 28)
  - s.split (line 28)
  - p.strip (line 28)
  - s.split (line 34)
  - json.load (line 48)
  - data.get (line 60)
  - rec.get (line 71)
  - rec.get (line 71)
  - rec.get (line 71)
  - out.append (line 74)
  - out.append (line 120)
  - process.extractOne (line 134)
  - process.extract (line 139)
  - names\_map.keys (line 152)
  - difflib.get\_close\_matches (line 154)
  - names\_map.get (line 157)
  - difflib.SequenceMatcher (line 158)
  - difflib.SequenceMatcher (line 165)
  - scored.append (line 166)
  - scored.sort (line 167)
  - best.get (line 181)
  - best.get (line 182)
  - best.get (line 183)
- Inbound references:
  - \_normalize\_name ← fec_lookup.py:72
  - \_normalize\_name ← fec_lookup.py:105
  - load\_fec\_candidates ← fec_lookup.py:59
  - load\_fec\_candidates ← fec_lookup.py:68
  - load\_fec\_candidates ← fec_lookup.py:119
  - \_build\_name\_index ← fec_lookup.py:102

### filename\_parser.py {#webapp-parser-filename-parser-py}

> Filename Parser for Smart Elections Parser

- Definitions:
  - class: `FilenameComponents` (line 61)
  - function: `split\_filename\_parts` (line 87)
  - function: `detect\_state\_from\_parts` (line 113)
  - function: `detect\_county\_from\_parts` (line 143)
  - function: `detect\_year\_from\_parts` (line 180)
  - function: `detect\_contest\_type\_from\_parts` (line 199)
  - function: `detect\_scope\_from\_parts` (line 216)
  - function: `detect\_format\_hint\_from\_parts` (line 232)
  - function: `parse\_filename` (line 256)
  - function: `parse\_filename\_simple` (line 297)
- Imports:
  - **Standard Library** (9):
    - `import re as re` (line 14)
    - `from dataclasses import asdict` (line 15)
    - `from dataclasses import dataclass` (line 15)
    - `from datetime import datetime` (line 16)
    - `from datetime import timezone` (line 16)
    - `from pathlib import Path` (line 17)
    - `from typing import Dict` (line 18)
    - `from typing import List` (line 18)
    - `from typing import Optional` (line 18)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 99)
  - name.replace (line 102)
  - re.sub (line 105)
  - p.strip (line 108)
  - name.split (line 108)
  - p.strip (line 108)
  - part.strip (line 123)
  - part.lower (line 130)
  - part.lower (line 152)
  - part\_lower.replace (line 160)
  - county\_name.title (line 162)
  - re.match (line 188)
  - re.search (line 192)
  - year\_match.group (line 194)
  - p.lower (line 205)
  - CONTEST\_KEYWORDS.items (line 208)
  - p.lower (line 220)
  - p.lower (line 246)
  - format\_keywords.items (line 248)
  - pathlib.Path (line 267)
  - datetime.datetime.now (line 293)
  - dataclasses.asdict (line 308)
- Inbound references:
  - FilenameComponents ← filename_parser.py:282
  - split\_filename\_parts ← filename_parser.py:272
  - detect\_state\_from\_parts ← filename_parser.py:275
  - detect\_county\_from\_parts ← filename_parser.py:276
  - detect\_year\_from\_parts ← filename_parser.py:277
  - detect\_contest\_type\_from\_parts ← filename_parser.py:278
  - detect\_scope\_from\_parts ← filename_parser.py:279
  - detect\_format\_hint\_from\_parts ← filename_parser.py:280
  - parse\_filename ← filename_parser.py:307
  - parse\_filename\_simple ← filename_parser.py:335

### handlers/batch\_handler.py {#webapp-parser-handlers-batch-handler-py}

- Definitions:
  - function: `\_normalize\_label` (line 14)
  - class: `BatchProcessor` (line 24)
- Imports:
  - **Standard Library** (10):
    - `import copy as copy` (line 3)
    - `import time as time` (line 4)
    - `import uuid as uuid` (line 5)
    - `from typing import Any` (line 7)
    - `from typing import Callable` (line 7)
    - `from typing import Dict` (line 7)
    - `from typing import List` (line 7)
    - `from typing import Optional` (line 7)
    - `from typing import Sequence` (line 7)
    - `from typing import Tuple` (line 7)
  - **Local/Project** (10):
    - `from __future__ import annotations` (line 1)
    - `from concurrent.futures import Future` (line 6)
    - `from concurrent.futures import ThreadPoolExecutor` (line 6)
    - `from utils.logger_singleton import logger` (line 9)
    - `from utils.logger_singleton import prompt` (line 9)
    - `from utils.shared_logic import safe_get` (line 10)
    - `from utils.shared_logic import safe_lower` (line 10)
    - `from utils.shared_logic import safe_parse` (line 10)
    - `from utils.shared_logic import safe_strip` (line 10)
    - `from utils.user_prompt import PromptCancelled` (line 11)
- Task markers:
  - L134 **WARNING**: ({
  - L135 **WARNING**: ",
  - L426 **WARNING**: ({
  - L427 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - utils.shared\_logic.safe\_strip (line 19)
  - utils.shared\_logic.safe\_lower (line 19)
  - uuid.uuid4 (line 56)
  - self.\_prepare\_races (line 62)
  - copy.deepcopy (line 63)
  - self.\_find\_initial\_match\_index (line 64)
  - concurrent.futures.ThreadPoolExecutor (line 77)
  - copy.deepcopy (line 83)
  - time.time (line 93)
  - utils.logger\_singleton.logger.info (line 104)
  - self.\_race\_label (line 111)
  - self.\_emit\_result (line 118)
  - self.\_process\_single\_race (line 129)
  - utils.logger\_singleton.logger.warning (line 134)
  - utils.logger\_singleton.logger.error (line 145)
  - self.\_await\_postprocessing (line 153)
  - self.\_compute\_status (line 155)
  - time.time (line 156)
  - self.\_mark\_processed (line 157)
  - utils.logger\_singleton.logger.info (line 159)
  - prepared.append (line 177)
  - copy.deepcopy (line 177)
  - prepared.append (line 179)
  - prepared.append (line 181)
  - self.\_race\_label (line 192)
  - self.\_race\_label (line 198)
  - utils.logger\_singleton.logger.info (line 199)
  - self.\_queue\_prompt\_responses (line 206)
  - self.\_build\_context (line 207)
  - utils.shared\_logic.safe\_parse (line 209)
  - metadata.get (line 218)
  - utils.logger\_singleton.logger.error (line 219)
  - metadata.get (line 222)
  - self.\_emit\_result (line 228)
  - copy.deepcopy (line 234)
  - context.setdefault (line 240)
  - race.get (line 246)
  - overrides.update (line 248)
  - context.update (line 250)
  - self.\_race\_label (line 252)
  - race.get (line 264)
  - self.\_format\_prompt\_value (line 269)
  - race.get (line 274)
  - self.\_build\_matcher (line 279)
  - entry.get (line 279)
  - self.\_format\_prompt\_value (line 282)
  - entry.get (line 282)
  - entry.get (line 284)
  - utils.logger\_singleton.logger.error (line 297)
  - metadata\_out.get (line 307)
- Inbound references:
  - \_normalize\_label ← batch_handler.py:188
  - \_normalize\_label ← batch_handler.py:193

### handlers/fec\_handler.py {#webapp-parser-handlers-fec-handler-py}

- Definitions:
  - function: `parse` (line 22)
- Imports:
  - **Standard Library** (7):
    - `import csv as csv` (line 3)
    - `import os as os` (line 4)
    - `from typing import Any` (line 5)
    - `from typing import Dict` (line 5)
    - `from typing import List` (line 5)
    - `from typing import Optional` (line 5)
    - `from typing import Tuple` (line 5)
  - **Third-party** (7):
    - `from webapp.parser.fec_lookup import find_candidate_by_name` (line 7)
    - `from webapp.parser.fec_lookup import get_candidate_by_id` (line 7)
    - `from webapp.parser.utils.fec_utils import canonicalize_headers` (line 8)
    - `from webapp.parser.utils.fec_utils import date_normalize` (line 8)
    - `from webapp.parser.utils.fec_utils import incumbent_normalize` (line 8)
    - `from webapp.parser.utils.fec_utils import money_normalize` (line 8)
    - `from webapp.parser.utils.fec_utils import party_normalize` (line 8)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - pd.read\_excel (line 38)
  - webapp.parser.utils.fec\_utils.canonicalize\_headers (line 40)
  - df.iterrows (line 41)
  - r.get (line 44)
  - mapping.get (line 45)
  - webapp.parser.utils.fec\_utils.money\_normalize (line 47)
  - webapp.parser.utils.fec\_utils.date\_normalize (line 49)
  - webapp.parser.utils.fec\_utils.party\_normalize (line 51)
  - webapp.parser.utils.fec\_utils.incumbent\_normalize (line 53)
  - rows.append (line 56)
  - out.get (line 59)
  - out.get (line 59)
  - out.get (line 59)
  - webapp.parser.fec\_lookup.get\_candidate\_by\_id (line 61)
  - out.get (line 65)
  - cand.get (line 66)
  - cand.get (line 66)
  - cand.get (line 66)
  - webapp.parser.utils.fec\_utils.party\_normalize (line 67)
  - out.get (line 68)
  - cand.get (line 69)
  - cand.get (line 69)
  - out.get (line 73)
  - out.get (line 73)
  - out.get (line 73)
  - out.get (line 74)
  - out.get (line 74)
  - out.get (line 74)
  - webapp.parser.fec\_lookup.find\_candidate\_by\_name (line 76)
  - match.get (line 79)
  - match.get (line 80)
  - out.get (line 83)
  - webapp.parser.utils.fec\_utils.party\_normalize (line 84)
  - rec.get (line 84)
  - rec.get (line 84)
  - out.get (line 85)
  - rec.get (line 86)
  - rec.get (line 86)
  - csv.DictReader (line 91)
  - webapp.parser.utils.fec\_utils.canonicalize\_headers (line 93)
  - r.items (line 96)
  - mapping.get (line 99)
  - orig.strip (line 99)
  - orig.strip (line 99)
  - webapp.parser.utils.fec\_utils.money\_normalize (line 102)
  - webapp.parser.utils.fec\_utils.date\_normalize (line 104)
  - webapp.parser.utils.fec\_utils.party\_normalize (line 106)
  - webapp.parser.utils.fec\_utils.incumbent\_normalize (line 108)
  - val.strip (line 110)
  - rows.append (line 111)

### handlers/formats/csv\_handler.py {#webapp-parser-handlers-formats-csv-handler-py}

- Definitions:
  - function: `parse\_csv\_election\_results` (line 44)
  - function: `parse` (line 327)
- Imports:
  - **Standard Library** (9):
    - `import csv as csv` (line 3)
    - `import os as os` (line 4)
    - `import re as re` (line 5)
    - `from typing import Any` (line 6)
    - `from typing import Dict` (line 6)
    - `from typing import List` (line 6)
    - `from typing import Optional` (line 6)
    - `from typing import Tuple` (line 6)
    - `from typing import cast` (line 6)
  - **Local/Project** (20):
    - `from __future__ import annotations` (line 1)
    - `from config import ENABLE_PARALLEL` (line 8)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_TITLE_SKIP_PHRASES` (line 9)
    - `from Context_Integration.librarian import parse_filename_for_location`
      (line 12)
    - `from utils.contest_detection import CONTEST_PATTERN as _CONTEST_RX` (line
      13)
    - `from utils.contest_detection import detect_contest_titles_from_text`
      (line 16)
    - `from utils.contest_detection import gather_lines_for_contest_detection`
      (line 16)
    - `from utils.contest_selector import select_contest_auto_first` (line 20)
    - `from utils.header_utils import normalize_table_headers` (line 23)
    - `from utils.location_helpers import attach_precinct_column` (line 24)
    - `from utils.location_helpers import collect_location_headers` (line 24)
    - `from utils.logger_singleton import logger` (line 28)
    - `from utils.output_utils import finalize_election_output` (line 29)
    - `from utils.pivot import expand_single_rawjson_row` (line 30)
    - `from utils.shared_logic import derive_candidate_party_metadata` (line 31)
    - `from utils.shared_logic import derive_state_county_from_table` (line 31)
    - `from utils.shared_logic import safe_get` (line 31)
    - `from utils.shared_logic import safe_slug` (line 31)
    - `from utils.table_builder import build_table_noninteractive` (line 37)
    - `from utils.table_core import robust_table_extraction` (line 38)
- Outgoing cross-module calls (sample):
  - csv.DictReader (line 62)
  - row.items (line 68)
  - normalized\_row.values (line 70)
  - data.append (line 71)
  - utils.header\_utils.normalize\_table\_headers (line 73)
  - h.strip (line 74)
  - utils.contest\_detection.gather\_lines\_for\_contest\_detection (line 77)
  - utils.contest\_detection.detect\_contest\_titles\_from\_text (line 78)
  - dict.fromkeys (line 83)
  - contest\_detection\_diag.get (line 84)
  - utils.logger\_singleton.logger.info (line 85)
  - utils.contest\_detection.CONTEST\_PATTERN.search (line 93)
  - possible\_contest\_cols.sort (line 95)
  - row.get (line 101)
  - row.get (line 101)
  - contest\_names.append (line 108)
  - Context\_Integration.librarian.parse\_filename\_for\_location (line 111)
  - parsed\_location.get (line 112)
  - s.lower (line 124)
  - utils.contest\_selector.select\_contest\_auto\_first (line 138)
  - utils.logger\_singleton.logger.error (line 147)
  - utils.shared\_logic.safe\_get (line 154)
  - row.get (line 160)
  - utils.location\_helpers.collect\_location\_headers (line 162)
  - utils.location\_helpers.attach\_precinct\_column (line 163)
  - html\_context.get (line 179)
  - html\_context.get (line 180)
  - utils.shared\_logic.derive\_state\_county\_from\_table (line 182)
  - state\_county\_diag.get (line 190)
  - state\_county\_diag.get (line 191)
  - utils.shared\_logic.derive\_candidate\_party\_metadata (line 193)
  - re.search (line 199)
  - m.group (line 200)
  - html\_context.get (line 203)
  - utils.shared\_logic.safe\_slug (line 210)
  - party\_diag.get (line 234)
  - utils.pivot.expand\_single\_rawjson\_row (line 236)
  - utils.table\_builder.build\_table\_noninteractive (line 238)
  - utils.output\_utils.finalize\_election\_output (line 267)
  - result.get (line 282)
  - result.get (line 289)
  - result.get (line 290)
  - utils.logger\_singleton.logger.info (line 305)
  - result.get (line 308)
  - utils.logger\_singleton.logger.info (line 311)
  - result.get (line 314)
  - html\_context.get (line 341)
  - ctx.update (line 345)
  - utils.table\_core.robust\_table\_extraction (line 349)
  - utils.header\_utils.normalize\_table\_headers (line 350)
- Inbound references:
  - parse\_csv\_election\_results ← csv_handler.py:440
  - parse\_csv\_election\_results ← test_csv_handler.py:23
  - parse\_csv\_election\_results ← test_csv_handler.py:46

### handlers/formats/download\_finder.py {#webapp-parser-handlers-formats-download-finder-py}

- Definitions:
  - function: `find\_download\_links` (line 9)
- Imports:
  - **Standard Library** (3):
    - `from typing import Any` (line 3)
    - `from typing import List` (line 3)
    - `from urllib.parse import urljoin` (line 4)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from utils.logger_singleton import logger` (line 6)
- Task markers:
  - L45 **WARNING**:
    ({"level":"WARNING","type":"download_finder","message":f"Download finder
    failed: {e}","session_id":session_id})
- Outgoing cross-module calls (sample):
  - page.query\_selector\_all (line 20)
  - a.get\_attribute (line 27)
  - href.strip (line 29)
  - href.startswith (line 30)
  - urllib.parse.urljoin (line 31)
  - urls.append (line 32)
  - page.evaluate (line 38)
  - urls.extend (line 40)
  - utils.logger\_singleton.logger.warning (line 45)

### handlers/formats/html\_dynamic\_fallback.py {#webapp-parser-handlers-formats-html-dynamic-fallback-py}

- Definitions:
  - function: `parse` (line 9)
- Imports:
  - **Standard Library** (3):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
    - `from typing import Tuple` (line 3)
  - **Third-party** (2):
    - `from webapp.parser.html_election_parser import
      generate_generic_html_result` (line 5)
    - `from webapp.parser.utils.logger_singleton import logger` (line 6)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - ctx.get (line 16)
  - ctx.get (line 16)
  - webapp.parser.html\_election\_parser.generate\_generic\_html\_result (line
    18)
  - webapp.parser.utils.logger\_singleton.logger.error (line 29)

### handlers/formats/html\_handler.py {#webapp-parser-handlers-formats-html-handler-py}

- Definitions:
  - function: `\_attempt\_generic\_fallback` (line 19)
  - function: `parse` (line 80)
- Imports:
  - **Standard Library** (7):
    - `import os as os` (line 4)
    - `from typing import Any` (line 5)
    - `from typing import Dict` (line 5)
    - `from typing import List` (line 5)
    - `from typing import Optional` (line 5)
    - `from typing import Tuple` (line 5)
    - `from typing import cast` (line 5)
  - **Third-party** (1):
    - `import orjson as orjson` (line 7)
  - **Local/Project** (13):
    - `from __future__ import annotations` (line 1)
    - `import importlib as importlib` (line 3)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_COUNTY_TO_PRECINCTS_MAP` (line 9)
    - `from state_router import fuzzy_match_handler` (line 10)
    - `from state_router import get_handler` (line 10)
    - `from state_router import list_available_handlers` (line 10)
    - `from utils.contest_selector import resolve_selection_context` (line 11)
    - `from utils.logger_singleton import logger as app_logger` (line 14)
    - `from utils.logger_singleton import prompt` (line 15)
    - `from utils.shared_logic import normalize_county_name` (line 16)
    - `from utils.shared_logic import normalize_state_name` (line 16)
    - `from utils.shared_logic import safe_get` (line 16)
    - `from utils.shared_logic import safe_parse` (line 16)
- Task markers:
  - L216 **WARNING**: (f"\[HTML Handler\] County '{county}' not found. Closest
    matches: {matches}")
  - L220 **WARNING**: (f"\[HTML Handler\] Detected county '{county}' is not in
    known counties for state '{suggested_state or state}'.")
  - L241 **WARNING**: (f"\[HTML Handler\] State '{user_state}' not found.
    Closest matches: {matches}")
  - L285 **WARNING**: (f"\[HTML Handler\] County '{user_county}' not found.
    Closest matches: {matches}")
- Outgoing cross-module calls (sample):
  - fallback\_ctx.setdefault (line 42)
  - fallback\_ctx.setdefault (line 44)
  - context.get (line 46)
  - fallback\_ctx.get (line 46)
  - context.get (line 47)
  - context.get (line 48)
  - fallback\_ctx.get (line 48)
  - context.get (line 49)
  - context.get (line 53)
  - context.get (line 53)
  - utils.logger\_singleton.logger.debug (line 98)
  - coordinator.organize\_and\_enrich (line 105)
  - utils.contest\_selector.resolve\_selection\_context (line 109)
  - html\_context.get (line 110)
  - html\_context.get (line 112)
  - html\_context.get (line 114)
  - utils.shared\_logic.normalize\_state\_name (line 120)
  - utils.shared\_logic.normalize\_county\_name (line 122)
  - state\_router.get\_handler (line 125)
  - routing\_trace.append (line 131)
  - html\_context.get (line 131)
  - html\_context.get (line 131)
  - attempts.append (line 148)
  - routing\_trace.append (line 149)
  - attempts.append (line 152)
  - routing\_trace.append (line 153)
  - attempts.append (line 156)
  - routing\_trace.append (line 161)
  - utils.logger\_singleton.logger.info (line 162)
  - utils.logger\_singleton.prompt.prompt\_input (line 164)
  - importlib.import\_module (line 167)
  - attempts.append (line 176)
  - routing\_trace.append (line 180)
  - utils.logger\_singleton.logger.error (line 182)
  - routing\_trace.append (line 183)
  - utils.shared\_logic.normalize\_state\_name (line 185)
  - html\_context.get (line 185)
  - utils.shared\_logic.normalize\_county\_name (line 186)
  - html\_context.get (line 186)
  - html\_context.get (line 187)
  - organized.get (line 188)
  - entities.extend (line 191)
  - utils.shared\_logic.safe\_get (line 191)
  - coordinator.validate\_and\_check\_integrity (line 193)
  - utils.shared\_logic.normalize\_state\_name (line 198)
  - ml\_suggestions.get (line 198)
  - utils.shared\_logic.normalize\_county\_name (line 199)
  - ml\_suggestions.get (line 199)
  - attempts.append (line 200)
  - routing\_trace.append (line 207)
- Inbound references:
  - \_attempt\_generic\_fallback ← html_handler.py:139

### handlers/formats/json\_handler.py {#webapp-parser-handlers-formats-json-handler-py}

- Definitions:
  - function: `\_build\_contest\_regex` (line 53)
  - function: `\_canonical\_contest\_key` (line 86)
  - function: `\_split\_primary\_title\_for\_grouping` (line 91)
  - function: `\_format\_county\_preview` (line 121)
  - function: `\_format\_scope\_label` (line 148)
  - function: `\_collect\_contest\_groups` (line 168)
  - function: `find\_key\_by\_keywords` (line 290)
  - function: `\_is\_dict\_list` (line 308)
  - function: `\_state\_key\_for\_county` (line 313)
  - function: `\_extract\_first\_str` (line 324)
  - function: `\_derive\_location\_metadata` (line 332)
  - function: `\_fastpath\_county\_results` (line 360)
  - function: `parse\_json\_election\_results` (line 977)
  - function: `parse` (line 1350)
- Imports:
  - **Standard Library** (15):
    - `import os as os` (line 3)
    - `import re as re` (line 4)
    - `from collections import Counter` (line 5)
    - `from collections import OrderedDict` (line 5)
    - `from collections import defaultdict` (line 5)
    - `from pathlib import Path` (line 6)
    - `from typing import Any` (line 7)
    - `from typing import DefaultDict` (line 7)
    - `from typing import Dict` (line 7)
    - `from typing import Iterable` (line 7)
    - `from typing import List` (line 7)
    - `from typing import Optional` (line 7)
    - `from typing import Set` (line 7)
    - `from typing import Tuple` (line 7)
    - `from typing import cast` (line 7)
  - **Third-party** (1):
    - `import orjson as orjson` (line 9)
  - **Local/Project** (30):
    - `from __future__ import annotations` (line 1)
    - `from config import ENABLE_PARALLEL` (line 11)
    - `from Context_Integration.Context_Library.constants import BALLOT_TYPES`
      (line 12)
    - `from Context_Integration.Context_Library.constants import
      BALLOT_TYPES_SORT_ORDER` (line 12)
    - `from Context_Integration.Context_Library.constants import
      CANDIDATE_KEYWORDS` (line 12)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_KEYWORDS` (line 12)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_TITLE_SKIP_PHRASES` (line 12)
    - `from Context_Integration.Context_Library.constants import
      DEFAULT_TOTAL_RESULT_DISPLAY` (line 12)
    - `from Context_Integration.Context_Library.constants import
      GROUP_RENAME_MAP` (line 12)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_STATE_TO_COUNTY_MAP` (line 12)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_KEYWORDS` (line 12)
    - `from Context_Integration.Context_Library.constants import PARTY_KEYWORDS`
      (line 12)
    - `from Context_Integration.Context_Library.constants import
      canonical_ballot_group` (line 12)
    - `from Context_Integration.librarian import parse_filename_for_location`
      (line 25)
    - `from utils.contest_selector import select_contest_auto_first` (line 26)
    - `from utils.json_export_loader import _ALL_COUNTIES_LABEL` (line 29)
    - `from utils.json_export_loader import load_json_export` (line 29)
    - `from utils.location_helpers import attach_precinct_column` (line 30)
    - `from utils.location_helpers import collect_location_headers` (line 30)
    - `from utils.logger_singleton import logger` (line 34)
    - `from utils.output_utils import finalize_election_output` (line 35)
    - `from utils.pivot import expand_single_rawjson_row` (line 36)
    - `from utils.salvage import normalize_ballot_column_name` (line 37)
    - `from utils.shared_logic import format_county_label` (line 38)
    - `from utils.shared_logic import format_state_label` (line 38)
    - `from utils.shared_logic import normalize_county_name` (line 38)
    - `from utils.shared_logic import safe_get` (line 38)
    - `from utils.shared_logic import safe_slug` (line 38)
    - `from utils.table_builder import build_table_noninteractive` (line 45)
    - `from utils.table_core import robust_table_extraction` (line 46)
- Task markers:
  - L377 **WARNING**: ({
  - L378 **WARNING**: ",
  - L501 **WARNING**: ({
  - L502 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - phrase.strip (line 62)
  - re.split (line 65)
  - phrase.strip (line 65)
  - re.escape (line 68)
  - t.replace (line 69)
  - t.replace (line 70)
  - xtoks.append (line 71)
  - parts.append (line 76)
  - re.compile (line 79)
  - re.compile (line 80)
  - re.sub (line 87)
  - title.lower (line 87)
  - re.sub (line 88)
  - title.strip (line 93)
  - text.split (line 100)
  - head.strip (line 101)
  - tail.strip (line 101)
  - re.search (line 104)
  - match.start (line 105)
  - match.start (line 106)
  - match.group (line 107)
  - re.search (line 111)
  - match.start (line 112)
  - match.start (line 113)
  - match.group (line 114)
  - county.strip (line 126)
  - cleaned.append (line 128)
  - scope.replace (line 155)
  - seen.add (line 158)
  - cleaned.append (line 160)
  - cleaned.append (line 162)
  - cleaned.append (line 164)
  - scope\_s.title (line 164)
  - groups.setdefault (line 173)
  - groups.items (line 211)
  - collections.Counter (line 212)
  - title\_counts.most\_common (line 213)
  - summary\_parts.append (line 237)
  - summary\_parts.append (line 239)
  - summary\_parts.append (line 241)
  - detail\_segments.append (line 258)
  - scope\_label.lower (line 259)
  - detail\_segments.append (line 260)
  - detail\_segments.append (line 262)
  - scope\_label.lower (line 268)
  - base\_office.lower (line 268)
  - group\_list.append (line 279)
  - group\_list.sort (line 287)
  - obj.keys (line 294)
  - key.lower (line 296)
- Inbound references:
  - \_build\_contest\_regex ← json_handler.py:83
  - \_build\_contest\_regex ← contest_detection.py:40
  - \_canonical\_contest\_key ← json_handler.py:172
  - \_split\_primary\_title\_for\_grouping ← json_handler.py:243
  - \_format\_county\_preview ← json_handler.py:245
  - \_format\_scope\_label ← json_handler.py:244
  - \_collect\_contest\_groups ← json_handler.py:398
  - find\_key\_by\_keywords ← json_handler.py:999
  - find\_key\_by\_keywords ← json_handler.py:1008
  - find\_key\_by\_keywords ← json_handler.py:1020
  - find\_key\_by\_keywords ← json_handler.py:1103
  - find\_key\_by\_keywords ← json_handler.py:1116
  - find\_key\_by\_keywords ← json_handler.py:1130
  - find\_key\_by\_keywords ← json_handler.py:1135
  - find\_key\_by\_keywords ← json_handler.py:1136
  - find\_key\_by\_keywords ← json_handler.py:1137
  - find\_key\_by\_keywords ← json_handler.py:1141
  - find\_key\_by\_keywords ← json_handler.py:1179
  - find\_key\_by\_keywords ← json_handler.py:1188
  - \_is\_dict\_list ← json_handler.py:1017
  - \_is\_dict\_list ← json_handler.py:1101
  - \_is\_dict\_list ← json_handler.py:1122
  - \_is\_dict\_list ← json_handler.py:1127
  - \_is\_dict\_list ← json_handler.py:1134
  - \_state\_key\_for\_county ← json_handler.py:353
  - \_extract\_first\_str ← json_handler.py:338
  - \_extract\_first\_str ← json_handler.py:344
  - \_extract\_first\_str ← json_handler.py:345
  - \_extract\_first\_str ← json_handler.py:348
  - \_extract\_first\_str ← json_handler.py:350
  - \_derive\_location\_metadata ← json_handler.py:372
  - \_derive\_location\_metadata ← json_handler.py:394
  - \_derive\_location\_metadata ← json_handler.py:989
  - \_fastpath\_county\_results ← json_handler.py:991
  - parse\_json\_election\_results ← json_handler.py:1461

### handlers/formats/pdf\_handler.py {#webapp-parser-handlers-formats-pdf-handler-py}

- Definitions:
  - function: `\_env\_truthy` (line 190)
  - class: `PDFParseCancelled` (line 212)
  - function: `\_cleanup\_pdf\_resources` (line 216)
  - function: `\_register\_pdf\_cleanup` (line 261)
  - function: `\_sanitize\_cache\_get` (line 270)
  - function: `\_sanitize\_cache\_set` (line 281)
  - function: `\_normalize\_angle` (line 292)
  - function: `\_quantize\_angle` (line 300)
  - function: `\_collect\_page\_orientation` (line 310)
  - function: `\_get\_page\_orientation\_map` (line 390)
  - function: `\_log\_orientation\_application` (line 454)
  - function: `\_apply\_page\_orientation` (line 467)
  - function: `\_expand\_focus\_windows` (line 498)
  - function: `\_normalize\_contest\_key` (line 524)
  - function: `\_contest\_title\_tokens` (line 531)
  - function: `\_ensure\_not\_cancelled` (line 537)
  - function: `\_cancelled\_result` (line 598)
  - function: `\_estimate\_ocr\_time\_budgets` (line 623)
  - function: `\_refine\_focus\_windows\_for\_contest` (line 634)
  - function: `\_focus\_windows\_from\_line\_records` (line 677)
  - function: `\_merge\_focus\_windows` (line 735)
  - function: `\_autopick\_contest\_from\_probe` (line 762)
  - function: `\_compute\_sample\_page\_indices` (line 811)
  - function: `\_contest\_probe\_scan` (line 843)
  - function: `\_yield\_full\_pass\_batches` (line 943)
  - function: `\_camelot\_signal\_sets` (line 1015)
  - function: `\_split\_ws\_blocks` (line 1058)
  - function: `\_is\_bad\_header\_line` (line 1062)
  - function: `\_prepare\_output\_context` (line 1066)
  - function: `\_table\_looks\_bad` (line 1079)
  - function: `\_find\_header\_line` (line 1083)
  - function: `\_extract\_table\_by\_whitespace` (line 1087)
  - function: `\_record\_table\_stage` (line 1091)
  - function: `\_ensure\_fitz` (line 1108)
  - function: `\_coerce\_version\_tuple` (line 1124)
  - function: `\_check\_pymupdf\_version` (line 1150)
  - function: `\_score\_camelot\_table` (line 1191)
  - function: `\_normalize\_camelot\_headers` (line 1256)
  - function: `\_camelot\_table\_to\_rows` (line 1271)
  - function: `\_merge\_camelot\_tables\_if\_compatible` (line 1310)
  - function: `\_extract\_camelot\_tables` (line 1341)
  - function: `\_hybrid\_fill\_camelot` (line 1384)
  - function: `\_norm\_txt` (line 1451)
  - function: `\_token\_set` (line 1455)
  - function: `\_header\_signature` (line 1459)
  - function: `\_looks\_like\_candidate\_header` (line 1463)
  - function: `\_compute\_header\_richness` (line 1467)
  - function: `\_compute\_numeric\_fill` (line 1471)
  - function: `\_evaluate\_table\_candidate\_quality` (line 1475)
  - function: `\_find\_best\_header\_match` (line 1486)
  - function: `\_normalize\_anchor\_value` (line 1490)
  - function: `\_merge\_camelot\_with\_text` (line 1494)
  - function: `\_best\_title\_match\_idx` (line 1502)
  - function: `\_extract\_contest\_block` (line 1506)
  - function: `\_parse\_candidate\_line` (line 1521)
  - function: `extract\_candidate\_totals\_from\_lines` (line 1525)
  - function: `\_is\_numeric\_like` (line 1530)
  - function: `\_normalize\_numeric\_token` (line 1534)
  - function: `\_matches\_anchor\_header` (line 1538)
  - function: `\_reconstruct\_columnar\_block` (line 1542)
  - function: `\_extract\_party\_lookup\_from\_lines` (line 1546)
  - function: `\_parse\_candidate\_header\_with\_party` (line 1550)
  - function: `\_coerce\_vote\_value\_for\_reconstruction` (line 1554)
  - function: `\_split\_dense\_precinct\_segments` (line 1565)
  - function: `\_expand\_dense\_precinct\_block` (line 1584)
  - function: `\_normalize\_precinct\_inline\_rows` (line 1608)
  - function: `\_prepare\_dense\_precinct\_lines` (line 1648)
  - function: `\_try\_columnar\_reconstruction` (line 1662)
  - function: `\_log\_ocr\_environment` (line 2242)
  - function: `\_detect\_poppler\_path` (line 2264)
  - function: `\_detect\_contest\_positions` (line 2297)
  - function: `\_associate\_tables\_with\_contests` (line 2317)
  - function: `\_is\_mostly\_markup` (line 2341)
  - function: `\_sanitize\_extracted\_text` (line 2377)
  - function: `\_split\_dense\_precinct\_line` (line 2531)
  - function: `\_explode\_dense\_ocr\_lines` (line 2558)
  - function: `\_summarize\_pages\_from\_records` (line 2599)
  - function: `\_assemble\_page\_line\_records` (line 2623)
  - function: `\_pdf\_to\_images` (line 2686)
  - function: `\_prep\_variants` (line 2855)
  - function: `\_dedupe\_contest\_titles` (line 2882)
  - function: `\_ocr\_images` (line 2914)
  - function: `adaptive\_ocr\_pipeline` (line 2976)
  - function: `ocr\_multi\_pass` (line 3303)
  - function: `\_extract\_text\_multi` (line 3349)
  - function: `\_save\_ocr\_debug\_images` (line 3397)
  - function: `\_write\_debug\_text` (line 3433)
  - function: `\_header\_token\_score` (line 3462)
  - function: `\_line\_has\_digits` (line 3467)
  - function: `\_line\_is\_candidate\_only` (line 3474)
  - function: `\_group\_words\_by\_gaps` (line 3482)
  - function: `\_build\_layout\_columns` (line 3498)
  - function: `\_assign\_words\_to\_columns` (line 3528)
  - function: `\_clean\_numeric\_cell` (line 3551)
  - function: `\_split\_key\_value\_line` (line 3568)
  - function: `\_finalize\_layout\_table` (line 3583)
  - function: `\_merge\_layout\_tables` (line 3598)
  - function: `\_extract\_tables\_via\_layout` (line 3623)
  - function: `\_identify\_statement\_location\_columns` (line 3859)
  - function: `\_statement\_value\_has\_payload` (line 3863)
  - function: `\_coerce\_statement\_numeric` (line 3885)
  - function: `\_remap\_statement\_summary\_header` (line 3903)
  - function: `\_is\_statement\_summary\_header` (line 3926)
  - function: `\_parse\_statement\_candidate\_header` (line 3935)
  - function: `\_normalize\_statement\_candidate\_results` (line 3956)
  - function: `\_extract\_statement\_return\_blocks` (line 4045)
  - function: `\_attach\_statement\_precinct` (line 4326)
  - function: `\_should\_prefer\_statement\_blocks` (line 4358)
  - function: `\_finalize\_structured\_table\_output` (line 4375)
  - function: `infer\_headers\_and\_methods` (line 4491)
  - function: `\_should\_force\_ocr` (line 4496)
  - function: `\_should\_auto\_select` (line 4512)
  - function: `\_pick\_representative\_title` (line 4541)
  - function: `\_dedupe\_contest\_titles` (line 4573)
  - function: `parse\_pdf\_election\_results` (line 4576)
  - function: `parse` (line 6049)
- Imports:
  - **Standard Library** (14):
    - `import os as os` (line 5)
    - `import re as re` (line 6)
    - `import csv as csv` (line 7)
    - `import time as time` (line 8)
    - `import math as math` (line 9)
    - `import platform as platform` (line 10)
    - `import shutil as shutil` (line 11)
    - `import hashlib as hashlib` (line 13)
    - `import tempfile as tempfile` (line 16)
    - `from typing import Any` (line 17)
    - `from collections import Counter` (line 18)
    - `from collections import OrderedDict` (line 18)
    - `from collections import defaultdict` (line 18)
    - `import html as html` (line 49)
  - **Third-party** (4):
    - `from PIL import Image` (line 20)
    - `from PIL import ImageOps` (line 20)
    - `from PIL import ImageFilter` (line 20)
    - `from PIL import ImageEnhance` (line 20)
  - **Local/Project** (96):
    - `from __future__ import annotations` (line 1)
    - `import importlib as importlib` (line 12)
    - `import atexit as atexit` (line 14)
    - `import gc as gc` (line 15)
    - `from concurrent.futures import ThreadPoolExecutor` (line 19)
    - `from Context_Integration.location_inference import
      infer_county_from_lines` (line 21)
    - `from config import ENABLE_OCR` (line 22)
    - `from config import ENABLE_PARALLEL` (line 22)
    - `from config import OUTPUT_DIR` (line 22)
    - `from config import OCR_CONFIDENCE_THRESHOLD` (line 22)
    - `from config import OCR_MIN_ALPHA_SIGNAL` (line 22)
    - `from config import OCR_AVG_CONF_ACCEPT` (line 22)
    - `from config import OCR_DPI_MIN` (line 22)
    - `from config import OCR_DPI_MAX` (line 22)
    - `from config import OCR_DPI_STEP` (line 22)
    - `from config import OCR_PSM_LIST` (line 22)
    - `from config import OCR_OEM_LIST` (line 22)
    - `from config import OCR_PREPROCESS_VARIANTS` (line 22)
    - `from config import OCR_SAMPLE_BUDGET` (line 22)
    - `from config import OCR_MAX_RUNS` (line 22)
    - `from config import OCR_ORIENTATION_THRESHOLD` (line 22)
    - `from config import OCR_DENSE_LINE_THRESHOLD` (line 22)
    - `from config import OCR_TABLE_SIGNAL_MIN_COLS` (line 22)
    - `from config import OCR_TABLE_SIGNAL_MIN_ROWS` (line 22)
    - `from config import OCR_MARKUP_HTML_TAG_RATIO` (line 22)
    - `from config import OCR_DEBUG_SAVE_IMAGES` (line 22)
    - `from config import OCR_FAST_MODE_DPI_LIMIT` (line 22)
    - `from config import OCR_FAST_MODE_SAMPLE_LIMIT` (line 22)
    - `from config import PDF_FAST_MODE` (line 22)
    - `from config import PDF_PROBE_MAX_PAGES` (line 22)
    - `from utils.camelot_utils import attempt_camelot_extraction` (line 78)
    - `from utils.camelot_utils import hybrid_fill_camelot` (line 78)
    - `from utils.pdf_table_utils import best_title_match_idx as
      utils_best_title_match_idx` (line 82)
    - `from utils.pdf_table_utils import coerce_vote_value_for_reconstruction as
      utils_coerce_vote_value_for_reconstruction` (line 82)
    - `from utils.pdf_table_utils import compute_header_richness as
      utils_compute_header_richness` (line 82)
    - `from utils.pdf_table_utils import compute_numeric_fill as
      utils_compute_numeric_fill` (line 82)
    - `from utils.pdf_table_utils import evaluate_table_candidate_quality as
      utils_evaluate_table_candidate_quality` (line 82)
    - `from utils.pdf_table_utils import extract_candidate_totals_from_lines as
      utils_extract_candidate_totals_from_lines` (line 82)
    - `from utils.pdf_table_utils import extract_contest_block as
      utils_extract_contest_block` (line 82)
    - `from utils.pdf_table_utils import extract_party_lookup_from_lines as
      utils_extract_party_lookup_from_lines` (line 82)
    - `from utils.pdf_table_utils import extract_table_by_whitespace as
      utils_extract_table_by_whitespace` (line 82)
    - `from utils.pdf_table_utils import find_best_header_match as
      utils_find_best_header_match` (line 82)
    - `from utils.pdf_table_utils import find_header_line as
      utils_find_header_line` (line 82)
    - `from utils.pdf_table_utils import detect_district_heading as
      utils_detect_district_heading` (line 82)
    - `from utils.pdf_table_utils import header_signature as
      utils_header_signature` (line 82)
    - `from utils.pdf_table_utils import is_bad_header_line as
      utils_is_bad_header_line` (line 82)
    - `from utils.pdf_table_utils import is_numeric_like as
      utils_is_numeric_like` (line 82)
    - `from utils.pdf_table_utils import looks_like_candidate_header as
      utils_looks_like_candidate_header` (line 82)
    - `from utils.pdf_table_utils import matches_anchor_header as
      utils_matches_anchor_header` (line 82)
    - `from utils.pdf_table_utils import merge_camelot_with_text as
      utils_merge_camelot_with_text` (line 82)
- Task markers:
  - L1004 **WARNING**: ({
  - L1005 **WARNING**: ",
  - L1007 **WARN**: \] Skipping page {page_index} during OCR batch render:
    {exc}",
  - L1160 **WARNING**: ({
  - L1161 **WARNING**: ",
  - L1164 **WARN**: \] Detected PyMuPDF %s. Upgrade to %s or newer to avoid
    parser instability."
  - L2756 **WARNING**: ({
  - L2757 **WARNING**: ",
  - L2759 **WARN**: \] Poppler binaries not detected; skipping pdf2image and
    using PyMuPDF fallback.",
  - L2799 **WARNING**: ({
  - L2800 **WARNING**: ",
  - L2803 **WARN**: \] pdf2image conversion failed; "
  - L3187 **WARNING**: ({
  - L3188 **WARNING**: ",
  - L3190 **WARN**: \] Skipping full-document OCR pass due to expired sample
    budget.",
  - L3238 **WARNING**: ({
  - L3239 **WARNING**: ",
  - L3241 **WARN**: \] Aborting full-document OCR pass due to timeout budget.",
  - L3268 **WARNING**: ({
  - L3269 **WARNING**: ",
  - L3272 **WARN**: \] Full-document OCR pass truncated due to
    OCR_FULLDOC_MAX_PAGES limit. "
  - L3354 **WARNING**: ({
  - L3355 **WARNING**: ",
  - L3357 **WARN**: \] Multi-mode text extraction failed: {e}",
  - L4711 **WARNING**: ({
  - L4712 **WARNING**: ",
  - L4714 **WARN**: \] fitz text extraction failed: {e}",
  - L4753 **WARNING**: ({
  - L4754 **WARNING**: ",
  - L4756 **WARN**: \] ENABLE_OCR_FORCE is set but Tesseract is unavailable;
    skipping OCR fallback.",
  - L4820 **WARNING**: ({
  - L4821 **WARNING**: ",
  - L4823 **WARN**: \] Low-signal text detected but OCR is unavailable or
    disabled.",
  - L5188 **WARNING**: ({
  - L5189 **WARNING**: ",
  - L5191 **WARN**: \] Auto contest selection failed in batch mode; falling back
    to filename.",
  - L5703 **WARNING**: ({
  - L5704 **WARNING**: ",
  - L5706 **WARN**: \] Selected contest '{contest}' not found in column
    '{contest_column}'. Skipping row filter.",
  - L5805 **WARNING**: ({
  - L5806 **WARNING**: ",
  - L5808 **WARN**: \] No structured rows matched the inferred column count of
    {len(headers)}. Total lines scanned: {unmatched_count}",
  - L5848 **WARNING**: ({
  - L5849 **WARNING**: ",
  - L6041 **WARNING**: ({
  - L6042 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - os.makedirs (line 56)
  - collections.OrderedDict (line 176)
  - value.strip (line 193)
  - os.getenv (line 196)
  - img.close (line 227)
  - \_PDF\_IMAGE\_REFS.clear (line 230)
  - gc.collect (line 233)
  - time.sleep (line 236)
  - shutil.rmtree (line 244)
  - time.sleep (line 248)
  - gc.collect (line 249)
  - shutil.rmtree (line 253)
  - \_PDF\_TEMP\_DIRS.clear (line 258)
  - atexit.register (line 265)
  - \_SANITIZE\_CACHE.get (line 271)
  - \_SANITIZE\_CACHE.move\_to\_end (line 277)
  - \_SANITIZE\_CACHE.move\_to\_end (line 287)
  - \_SANITIZE\_CACHE.popitem (line 289)
  - page.get\_text (line 312)
  - raw.get (line 318)
  - collections.Counter (line 319)
  - collections.Counter (line 320)
  - block.get (line 329)
  - line.get (line 333)
  - span.get (line 339)
  - span.get (line 345)
  - math.degrees (line 354)
  - math.atan2 (line 354)
  - counter.most\_common (line 376)
  - \_PAGE\_ORIENTATION\_CACHE.get (line 393)
  - \_PAGE\_ORIENTATION\_DEFAULT.get (line 395)
  - collections.Counter (line 398)
  - fitz.open (line 400)
  - utils.logger\_singleton.logger.debug (line 402)
  - doc.close (line 428)
  - orientation\_counts.most\_common (line 434)
  - orientation\_counts.values (line 435)
  - orientation\_counts.keys (line 443)
  - utils.logger\_singleton.logger.info (line 444)
  - \_PAGE\_ORIENTATION\_LOGGED.add (line 450)
  - \_PAGE\_ORIENTATION\_APPLIED.add (line 458)
  - utils.logger\_singleton.logger.info (line 459)
  - orientation\_map.get (line 478)
  - image.rotate (line 485)
  - utils.logger\_singleton.logger.debug (line 487)
  - normalized.append (line 512)
  - merged.append (line 518)
  - re.sub (line 527)
  - value.lower (line 527)
  - re.sub (line 528)
- Inbound references:
  - \_env\_truthy ← pdf_handler.py:196
  - \_env\_truthy ← contest_selector.py:72
  - PDFParseCancelled ← pdf_handler.py:595
  - \_register\_pdf\_cleanup ← pdf_handler.py:2766
  - \_register\_pdf\_cleanup ← pdf_handler.py:2813
  - \_sanitize\_cache\_get ← pdf_handler.py:2391
  - \_sanitize\_cache\_set ← pdf_handler.py:2521
  - \_normalize\_angle ← pdf_handler.py:301
  - \_quantize\_angle ← pdf_handler.py:357
  - \_quantize\_angle ← pdf_handler.py:481
  - \_collect\_page\_orientation ← pdf_handler.py:417
  - \_get\_page\_orientation\_map ← pdf_handler.py:878
  - \_get\_page\_orientation\_map ← pdf_handler.py:953
  - \_get\_page\_orientation\_map ← pdf_handler.py:2701
  - \_log\_orientation\_application ← pdf_handler.py:494
  - \_apply\_page\_orientation ← pdf_handler.py:895
  - \_apply\_page\_orientation ← pdf_handler.py:994
  - \_apply\_page\_orientation ← pdf_handler.py:2705
  - \_expand\_focus\_windows ← pdf_handler.py:674
  - \_expand\_focus\_windows ← pdf_handler.py:727
  - \_expand\_focus\_windows ← pdf_handler.py:4647
  - \_normalize\_contest\_key ← pdf_handler.py:649
  - \_normalize\_contest\_key ← pdf_handler.py:667
  - \_normalize\_contest\_key ← pdf_handler.py:695
  - \_normalize\_contest\_key ← pdf_handler.py:711
  - \_contest\_title\_tokens ← pdf_handler.py:647
  - \_contest\_title\_tokens ← pdf_handler.py:660
  - \_contest\_title\_tokens ← pdf_handler.py:694
  - \_contest\_title\_tokens ← pdf_handler.py:710
  - \_ensure\_not\_cancelled ← pdf_handler.py:889
  - \_ensure\_not\_cancelled ← pdf_handler.py:985
  - \_ensure\_not\_cancelled ← pdf_handler.py:989
  - \_ensure\_not\_cancelled ← pdf_handler.py:2731
  - \_ensure\_not\_cancelled ← pdf_handler.py:2822
  - \_ensure\_not\_cancelled ← pdf_handler.py:3011
  - \_ensure\_not\_cancelled ← pdf_handler.py:3016
  - \_ensure\_not\_cancelled ← pdf_handler.py:3094
  - \_ensure\_not\_cancelled ← pdf_handler.py:3103
  - \_ensure\_not\_cancelled ← pdf_handler.py:3151
  - \_ensure\_not\_cancelled ← pdf_handler.py:3246
  - \_ensure\_not\_cancelled ← pdf_handler.py:3399
  - \_ensure\_not\_cancelled ← pdf_handler.py:3685
  - \_ensure\_not\_cancelled ← pdf_handler.py:4168
  - \_ensure\_not\_cancelled ← pdf_handler.py:4606
  - \_ensure\_not\_cancelled ← pdf_handler.py:4626
  - \_ensure\_not\_cancelled ← pdf_handler.py:4657
  - \_ensure\_not\_cancelled ← pdf_handler.py:4699
  - \_ensure\_not\_cancelled ← pdf_handler.py:4966
  - \_cancelled\_result ← pdf_handler.py:6172
  - \_estimate\_ocr\_time\_budgets ← pdf_handler.py:4740

### handlers/formats/txt\_handler.py {#webapp-parser-handlers-formats-txt-handler-py}

- Definitions:
  - function: `\_read\_delimited\_file` (line 44)
  - function: `parse\_txt\_election\_results` (line 75)
  - function: `parse` (line 327)
- Imports:
  - **Standard Library** (9):
    - `import csv as csv` (line 3)
    - `import os as os` (line 4)
    - `import re as re` (line 5)
    - `from typing import Any` (line 6)
    - `from typing import Dict` (line 6)
    - `from typing import List` (line 6)
    - `from typing import Optional` (line 6)
    - `from typing import Tuple` (line 6)
    - `from typing import cast` (line 6)
  - **Local/Project** (18):
    - `from __future__ import annotations` (line 1)
    - `from config import ENABLE_PARALLEL` (line 8)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_TITLE_SKIP_PHRASES` (line 9)
    - `from utils.contest_detection import CONTEST_PATTERN as _CONTEST_RX` (line
      12)
    - `from utils.contest_detection import detect_contest_titles_from_text`
      (line 15)
    - `from utils.contest_detection import gather_lines_for_contest_detection`
      (line 15)
    - `from utils.contest_selector import select_contest_auto_first` (line 19)
    - `from utils.location_helpers import attach_precinct_column` (line 20)
    - `from utils.location_helpers import collect_location_headers` (line 20)
    - `from utils.logger_singleton import logger` (line 24)
    - `from utils.output_utils import finalize_election_output` (line 25)
    - `from utils.pivot import expand_single_rawjson_row` (line 26)
    - `from utils.shared_logic import derive_candidate_party_metadata` (line 27)
    - `from utils.shared_logic import derive_state_county_from_table` (line 27)
    - `from utils.shared_logic import safe_get` (line 27)
    - `from utils.shared_logic import safe_slug` (line 27)
    - `from utils.table_builder import build_table_noninteractive` (line 33)
    - `from utils.table_core import robust_table_extraction` (line 34)
- Outgoing cross-module calls (sample):
  - handle.read (line 49)
  - handle.seek (line 50)
  - csv.Sniffer (line 52)
  - csv.DictReader (line 55)
  - raw.items (line 63)
  - clean.values (line 65)
  - rows.append (line 66)
  - utils.logger\_singleton.logger.error (line 84)
  - utils.contest\_detection.CONTEST\_PATTERN.search (line 93)
  - possible\_contest\_cols.sort (line 95)
  - utils.contest\_detection.gather\_lines\_for\_contest\_detection (line 99)
  - utils.contest\_detection.detect\_contest\_titles\_from\_text (line 100)
  - dict.fromkeys (line 105)
  - contest\_detection\_diag.get (line 106)
  - utils.logger\_singleton.logger.info (line 107)
  - row.get (line 117)
  - row.get (line 117)
  - contest\_names.append (line 124)
  - s.lower (line 133)
  - utils.contest\_selector.select\_contest\_auto\_first (line 147)
  - utils.logger\_singleton.logger.error (line 156)
  - utils.shared\_logic.safe\_get (line 163)
  - row.get (line 168)
  - utils.location\_helpers.collect\_location\_headers (line 170)
  - utils.location\_helpers.attach\_precinct\_column (line 171)
  - html\_context.get (line 187)
  - html\_context.get (line 188)
  - utils.shared\_logic.derive\_state\_county\_from\_table (line 190)
  - state\_county\_diag.get (line 198)
  - state\_county\_diag.get (line 199)
  - utils.shared\_logic.derive\_candidate\_party\_metadata (line 201)
  - utils.shared\_logic.safe\_slug (line 206)
  - re.search (line 207)
  - m.group (line 208)
  - html\_context.get (line 211)
  - party\_diag.get (line 240)
  - utils.pivot.expand\_single\_rawjson\_row (line 242)
  - utils.table\_builder.build\_table\_noninteractive (line 244)
  - utils.output\_utils.finalize\_election\_output (line 273)
  - result.get (line 288)
  - result.get (line 295)
  - result.get (line 296)
  - utils.logger\_singleton.logger.info (line 311)
  - result.get (line 314)
  - utils.logger\_singleton.logger.info (line 317)
  - result.get (line 320)
  - html\_context.get (line 336)
  - ctx.update (line 339)
  - utils.table\_core.robust\_table\_extraction (line 343)
  - html\_context.get (line 345)
- Inbound references:
  - \_read\_delimited\_file ← txt_handler.py:82
  - parse\_txt\_election\_results ← txt_handler.py:426

### handlers/formats/xlsx\_handler.py {#webapp-parser-handlers-formats-xlsx-handler-py}

- Definitions:
  - function: `\_dataframe\_to\_records` (line 48)
  - function: `parse\_xlsx\_election\_results` (line 65)
  - function: `parse` (line 354)
- Imports:
  - **Standard Library** (8):
    - `import os as os` (line 3)
    - `import re as re` (line 4)
    - `from typing import Any` (line 5)
    - `from typing import Dict` (line 5)
    - `from typing import List` (line 5)
    - `from typing import Optional` (line 5)
    - `from typing import Tuple` (line 5)
    - `from typing import cast` (line 5)
  - **Local/Project** (18):
    - `from __future__ import annotations` (line 1)
    - `from config import ENABLE_PARALLEL` (line 7)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_TITLE_SKIP_PHRASES` (line 8)
    - `from utils.contest_detection import CONTEST_PATTERN as _CONTEST_RX` (line
      11)
    - `from utils.contest_detection import detect_contest_titles_from_text`
      (line 14)
    - `from utils.contest_detection import gather_lines_for_contest_detection`
      (line 14)
    - `from utils.contest_selector import select_contest_auto_first` (line 18)
    - `from utils.location_helpers import attach_precinct_column` (line 19)
    - `from utils.location_helpers import collect_location_headers` (line 19)
    - `from utils.logger_singleton import logger` (line 23)
    - `from utils.output_utils import finalize_election_output` (line 24)
    - `from utils.pivot import expand_single_rawjson_row` (line 25)
    - `from utils.shared_logic import derive_candidate_party_metadata` (line 26)
    - `from utils.shared_logic import derive_state_county_from_table` (line 26)
    - `from utils.shared_logic import safe_get` (line 26)
    - `from utils.shared_logic import safe_slug` (line 26)
    - `from utils.table_builder import build_table_noninteractive` (line 32)
    - `from utils.table_core import robust_table_extraction` (line 33)
- Outgoing cross-module calls (sample):
  - typing.cast (line 42)
  - df.to\_dict (line 52)
  - record.items (line 54)
  - pd.notna (line 56)
  - pd.isna (line 59)
  - clean.values (line 60)
  - records.append (line 61)
  - utils.logger\_singleton.logger.error (line 74)
  - pd.read\_excel (line 84)
  - utils.logger\_singleton.logger.error (line 86)
  - df.keys (line 96)
  - utils.logger\_singleton.logger.error (line 101)
  - utils.contest\_detection.CONTEST\_PATTERN.search (line 111)
  - possible\_contest\_cols.sort (line 113)
  - utils.contest\_detection.gather\_lines\_for\_contest\_detection (line 117)
  - utils.contest\_detection.detect\_contest\_titles\_from\_text (line 118)
  - dict.fromkeys (line 123)
  - contest\_detection\_diag.get (line 124)
  - utils.logger\_singleton.logger.info (line 125)
  - row.get (line 135)
  - row.get (line 135)
  - contest\_names.append (line 142)
  - s.lower (line 151)
  - utils.contest\_selector.select\_contest\_auto\_first (line 165)
  - utils.logger\_singleton.logger.error (line 174)
  - utils.shared\_logic.safe\_get (line 181)
  - row.get (line 186)
  - utils.location\_helpers.collect\_location\_headers (line 188)
  - utils.location\_helpers.attach\_precinct\_column (line 189)
  - html\_context.get (line 205)
  - html\_context.get (line 206)
  - utils.shared\_logic.derive\_state\_county\_from\_table (line 208)
  - state\_county\_diag.get (line 216)
  - state\_county\_diag.get (line 217)
  - utils.shared\_logic.derive\_candidate\_party\_metadata (line 219)
  - utils.shared\_logic.safe\_slug (line 224)
  - re.search (line 225)
  - m.group (line 226)
  - html\_context.get (line 229)
  - party\_diag.get (line 259)
  - utils.pivot.expand\_single\_rawjson\_row (line 261)
  - utils.table\_builder.build\_table\_noninteractive (line 263)
  - utils.output\_utils.finalize\_election\_output (line 293)
  - result.get (line 308)
  - result.get (line 316)
  - result.get (line 317)
  - utils.logger\_singleton.logger.info (line 332)
  - result.get (line 335)
  - utils.logger\_singleton.logger.info (line 338)
  - result.get (line 341)
- Inbound references:
  - \_dataframe\_to\_records ← xlsx_handler.py:109
  - parse\_xlsx\_election\_results ← xlsx_handler.py:470

### handlers/registry.py {#webapp-parser-handlers-registry-py}

- Definitions:
  - function: `register\_state\_handler` (line 16)
  - function: `register\_county\_handler` (line 23)
  - function: `apply\_vendor\_overrides` (line 32)
  - function: `\_module\_exists` (line 43)
  - function: `get\_state\_handler\_module\_path` (line 50)
  - function: `get\_county\_handler\_module\_path` (line 67)
- Imports:
  - **Standard Library** (2):
    - `from typing import Dict` (line 4)
    - `from typing import Optional` (line 4)
  - **Third-party** (4):
    - `from webapp.parser.Context_Integration.Context_Library.constants import
      STATE_MODULE_MAP` (line 6)
    - `from webapp.parser.handlers.vendor_state_map import VENDOR_STATE_MAP`
      (line 7)
    - `from webapp.parser.utils.shared_logic import normalize_county_name` (line
      8)
    - `from webapp.parser.utils.shared_logic import normalize_state_name` (line
      8)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `import importlib.util as importlib` (line 3)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.shared\_logic.normalize\_state\_name (line 18)
  - webapp.parser.utils.shared\_logic.normalize\_state\_name (line 25)
  - webapp.parser.utils.shared\_logic.normalize\_county\_name (line 26)
  - \_COUNTY\_HANDLER\_OVERRIDES.setdefault (line 29)
  - entry.get (line 35)
  - entry.get (line 37)
  - webapp.parser.utils.shared\_logic.normalize\_state\_name (line 52)
  - \_STATE\_HANDLER\_OVERRIDES.get (line 56)
  - webapp.parser.Context\_Integration.Context\_Library.constants.STATE\_MODULE\_MAP.get
    (line 60)
  - webapp.parser.utils.shared\_logic.normalize\_state\_name (line 69)
  - webapp.parser.utils.shared\_logic.normalize\_county\_name (line 70)
  - \_COUNTY\_HANDLER\_OVERRIDES.get (line 74)
- Inbound references:
  - register\_state\_handler ← registry.py:40
  - \_module\_exists ← registry.py:57
  - \_module\_exists ← registry.py:61
  - \_module\_exists ← registry.py:75

### handlers/shared/\_\_init\_\_.py {#webapp-parser-handlers-shared-init-py}

> Shared handler helpers.

### handlers/shared/parity\_hooks.py {#webapp-parser-handlers-shared-parity-hooks-py}

- Definitions:
  - function: `safe\_parity\_note` (line 10)
  - function: `attach\_router\_parity\_note` (line 21)
  - function: `extract\_router\_parity\_note` (line 30)
  - function: `attach\_parity\_note\_to\_metadata` (line 36)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Task markers:
  - L10 **NOTE**: str | None) -&gt; str | None:
  - L11 **NOTE**: , str):
  - L13 **NOTE**: .strip()
  - L21 **NOTE**: str | None) -&gt; None:
  - L24 **NOTE**: )
  - L36 **NOTE**: str | None) -&gt; Dict\[str, Any\]:
  - L37 **NOTE**: )
- Outgoing cross-module calls (sample):
  - note.strip (line 13)
  - context.get (line 33)
  - metadata.setdefault (line 42)
- Inbound references:
  - safe\_parity\_note ← parity_hooks.py:24
  - safe\_parity\_note ← parity_hooks.py:33
  - safe\_parity\_note ← parity_hooks.py:37

### handlers/shared/state\_handler\_base.py {#webapp-parser-handlers-shared-state-handler-base-py}

> State Handler Base Class

- Definitions:
  - class: `StateHandlerBase` (line 35)
  - class: `SimpleTableHandler` (line 450)
- Imports:
  - **Standard Library** (7):
    - `from abc import ABC` (line 20)
    - `from abc import abstractmethod` (line 20)
    - `from typing import Any` (line 21)
    - `from typing import Dict` (line 21)
    - `from typing import List` (line 21)
    - `from typing import Optional` (line 21)
    - `from typing import Tuple` (line 21)
  - **Third-party** (8):
    - `from webapp.parser.Context_Integration.librarian import clean_for_json`
      (line 23)
    - `from webapp.parser.handlers.shared.parity_hooks import
      attach_parity_note_to_metadata` (line 24)
    - `from webapp.parser.handlers.shared.parity_hooks import
      extract_router_parity_note` (line 24)
    - `from webapp.parser.utils.contest_selector import
      select_contest_auto_first` (line 28)
    - `from webapp.parser.utils.html_scanner import scan_html_for_context` (line
      29)
    - `from webapp.parser.utils.logger_singleton import logger` (line 30)
    - `from webapp.parser.utils.retry_utils import retry_with_snapshot` (line
      31)
    - `from webapp.parser.utils.table_core import robust_table_extraction` (line
      32)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 18)
- Task markers:
  - L157 **WARNING**: (f"\[{self.STATE_NAME}\] No contest selected")
  - L178 **NOTE**:             # Attach parity note
  - L252 **WARNING**: (f"\[{self.STATE_NAME}\] No contests detected in HTML")
- Outgoing cross-module calls (sample):
  - self.\_parse\_internal (line 97)
  - webapp.parser.utils.retry\_utils.retry\_with\_snapshot (line 91)
  - self.\_parse\_internal (line 102)
  - html\_context.get (line 129)
  - webapp.parser.utils.logger\_singleton.logger.info (line 131)
  - webapp.parser.handlers.shared.parity\_hooks.extract\_router\_parity\_note
    (line 134)
  - self.pre\_scan\_hook (line 137)
  - self.should\_use\_fallback (line 140)
  - webapp.parser.utils.logger\_singleton.logger.info (line 141)
  - self.\_fallback\_parse (line 142)
  - self.scan\_for\_contests (line 145)
  - self.\_ensure\_location\_fields (line 148)
  - self.post\_scan\_hook (line 151)
  - self.select\_contest (line 154)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 157)
  - self.pre\_extraction\_hook (line 161)
  - self.extract\_tables (line 164)
  - self.post\_extraction\_hook (line 169)
  - self.build\_metadata (line 174)
  - webapp.parser.handlers.shared.parity\_hooks.attach\_parity\_note\_to\_metadata
    (line 179)
  - self.log\_extraction\_attempt (line 182)
  - selected\_contest.get (line 184)
  - webapp.parser.utils.logger\_singleton.logger.info (line 185)
  - webapp.parser.utils.logger\_singleton.logger.error (line 190)
  - self.log\_extraction\_attempt (line 191)
  - webapp.parser.utils.html\_scanner.scan\_html\_for\_context (line 212)
  - html\_context.get (line 213)
  - html\_context.get (line 219)
  - webapp.parser.Context\_Integration.librarian.clean\_for\_json (line 224)
  - coordinator.organize\_and\_enrich (line 225)
  - coordinator.predict\_missing\_fields (line 229)
  - context\_result.update (line 233)
  - context\_result.get (line 249)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 252)
  - context\_result.get (line 256)
  - context\_result.get (line 257)
  - context\_result.get (line 258)
  - webapp.parser.utils.contest\_selector.select\_contest\_auto\_first (line
    263)
  - html\_context.get (line 268)
  - contest.get (line 339)
  - contest.get (line 340)
  - contest.get (line 341)
  - contest.get (line 342)
  - html\_context.get (line 343)
  - context\_result.get (line 343)
  - contest.get (line 367)
  - metadata.get (line 370)
  - webapp.parser.utils.logger\_singleton.logger.debug (line 377)
  - webapp.parser.utils.logger\_singleton.logger.error (line 379)
  - context\_result.get (line 411)

### handlers/shared/state\_scaffold.py {#webapp-parser-handlers-shared-state-scaffold-py}

- Definitions:
  - function: `parse` (line 12)
- Imports:
  - **Standard Library** (3):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
    - `from typing import Tuple` (line 3)
  - **Third-party** (3):
    - `from webapp.parser.handlers.formats.html_dynamic_fallback import parse as
      dynamic_parse` (line 5)
    - `from webapp.parser.handlers.shared.parity_hooks import
      attach_parity_note_to_metadata` (line 6)
    - `from webapp.parser.handlers.shared.parity_hooks import
      extract_router_parity_note` (line 6)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - webapp.parser.handlers.shared.parity\_hooks.extract\_router\_parity\_note
    (line 22)
  - webapp.parser.handlers.formats.html\_dynamic\_fallback.parse (line 23)
  - webapp.parser.handlers.shared.parity\_hooks.attach\_parity\_note\_to\_metadata
    (line 27)

### handlers/shared/vendor\_dispatch.py {#webapp-parser-handlers-shared-vendor-dispatch-py}

> Vendor dispatch handler.

- Definitions:
  - function: `\_display\_state\_name` (line 27)
  - function: `\_get\_canonical\_state` (line 31)
  - function: `\_get\_handler` (line 44)
  - function: `parse` (line 71)
- Imports:
  - **Standard Library** (4):
    - `from typing import Any` (line 8)
    - `from typing import Dict` (line 8)
    - `from typing import List` (line 8)
    - `from typing import Tuple` (line 8)
  - **Third-party** (8):
    - `from webapp.parser.Context_Integration.librarian import get_state_abbr`
      (line 10)
    - `from webapp.parser.Context_Integration.librarian import lookup_state`
      (line 10)
    - `from webapp.parser.handlers.shared.state_scaffold import parse as
      scaffold_parse` (line 11)
    - `from webapp.parser.handlers.shared.vendors.clarity_base_handler import
      ClarityBaseHandler` (line 12)
    - `from webapp.parser.handlers.shared.vendors.dominion_base_handler import
      DominionBaseHandler` (line 13)
    - `from webapp.parser.handlers.shared.vendors.voteworks_base_handler import
      VoteWorksBaseHandler` (line 14)
    - `from webapp.parser.handlers.vendor_state_map import get_vendor_for_state`
      (line 15)
    - `from webapp.parser.utils.logger_singleton import logger` (line 16)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 6)
- Outgoing cross-module calls (sample):
  - canonical\_state.replace (line 28)
  - ctx.get (line 34)
  - ctx.get (line 35)
  - ctx.get (line 36)
  - ctx.get (line 37)
  - webapp.parser.Context\_Integration.librarian.lookup\_state (line 41)
  - webapp.parser.handlers.vendor\_state\_map.get\_vendor\_for\_state (line 45)
  - \_VENDOR\_CLASS\_MAP.get (line 49)
  - webapp.parser.Context\_Integration.librarian.get\_state\_abbr (line 53)
  - \_HANDLER\_CACHE.get (line 58)
  - webapp.parser.handlers.shared.state\_scaffold.parse (line 82)
  - webapp.parser.utils.logger\_singleton.logger.debug (line 86)
  - webapp.parser.handlers.shared.state\_scaffold.parse (line 87)
  - handler.parse (line 89)
- Inbound references:
  - \_display\_state\_name ← vendor_dispatch.py:63
  - \_get\_canonical\_state ← vendor_dispatch.py:80
  - \_get\_handler ← vendor_dispatch.py:84

### handlers/shared/vendors/\_\_init\_\_.py {#webapp-parser-handlers-shared-vendors-init-py}

- Imports:
  - **Third-party** (3):
    - `from webapp.parser.handlers.shared.vendors.clarity_base_handler import
      ClarityBaseHandler` (line 1)
    - `from webapp.parser.handlers.shared.vendors.dominion_base_handler import
      DominionBaseHandler` (line 2)
    - `from webapp.parser.handlers.shared.vendors.voteworks_base_handler import
      VoteWorksBaseHandler` (line 3)

### handlers/shared/vendors/clarity\_base\_handler.py {#webapp-parser-handlers-shared-vendors-clarity-base-handler-py}

> Clarity Elections base handler.

- Definitions:
  - class: `ClarityBaseHandler` (line 16)
- Imports:
  - **Standard Library** (5):
    - `import re as re` (line 8)
    - `from typing import Any` (line 9)
    - `from typing import Dict` (line 9)
    - `from typing import List` (line 9)
    - `from typing import Tuple` (line 9)
  - **Third-party** (3):
    - `from webapp.parser.handlers.shared.state_handler_base import
      StateHandlerBase` (line 11)
    - `from webapp.parser.utils.logger_singleton import logger` (line 12)
    - `from webapp.parser.utils.table_core import robust_table_extraction` (line
      13)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 6)
- Outgoing cross-module calls (sample):
  - re.search (line 31)
  - webapp.parser.utils.logger\_singleton.logger.info (line 34)
  - webapp.parser.utils.table\_core.robust\_table\_extraction (line 49)
  - result.get (line 57)
  - result.get (line 58)

### handlers/shared/vendors/dominion\_base\_handler.py {#webapp-parser-handlers-shared-vendors-dominion-base-handler-py}

> Dominion base handler.

- Definitions:
  - class: `DominionBaseHandler` (line 16)
- Imports:
  - **Standard Library** (5):
    - `import re as re` (line 8)
    - `from typing import Any` (line 9)
    - `from typing import Dict` (line 9)
    - `from typing import List` (line 9)
    - `from typing import Tuple` (line 9)
  - **Third-party** (3):
    - `from webapp.parser.handlers.shared.state_handler_base import
      StateHandlerBase` (line 11)
    - `from webapp.parser.utils.logger_singleton import logger` (line 12)
    - `from webapp.parser.utils.table_core import robust_table_extraction` (line
      13)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 6)
- Outgoing cross-module calls (sample):
  - re.search (line 31)
  - webapp.parser.utils.logger\_singleton.logger.info (line 34)
  - webapp.parser.utils.table\_core.robust\_table\_extraction (line 49)
  - result.get (line 57)
  - result.get (line 58)

### handlers/shared/vendors/voteworks\_base\_handler.py {#webapp-parser-handlers-shared-vendors-voteworks-base-handler-py}

> VoteWorks base handler.

- Definitions:
  - class: `VoteWorksBaseHandler` (line 16)
- Imports:
  - **Standard Library** (5):
    - `import re as re` (line 8)
    - `from typing import Any` (line 9)
    - `from typing import Dict` (line 9)
    - `from typing import List` (line 9)
    - `from typing import Tuple` (line 9)
  - **Third-party** (3):
    - `from webapp.parser.handlers.shared.state_handler_base import
      StateHandlerBase` (line 11)
    - `from webapp.parser.utils.logger_singleton import logger` (line 12)
    - `from webapp.parser.utils.table_core import robust_table_extraction` (line
      13)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 6)
- Outgoing cross-module calls (sample):
  - re.search (line 31)
  - webapp.parser.utils.logger\_singleton.logger.info (line 34)
  - webapp.parser.utils.table\_core.robust\_table\_extraction (line 49)
  - result.get (line 57)
  - result.get (line 58)

### handlers/states/alabama/alabama.py {#webapp-parser-handlers-states-alabama-alabama-py}

- Definitions:
  - class: `AlabamaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - AlabamaHandler ← alabama.py:15

### handlers/states/alaska/alaska.py {#webapp-parser-handlers-states-alaska-alaska-py}

- Definitions:
  - class: `AlaskaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - AlaskaHandler ← alaska.py:15

### handlers/states/american\_samoa/american\_samoa.py {#webapp-parser-handlers-states-american-samoa-american-samoa-py}

- Definitions:
  - class: `AmericanSamoaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - AmericanSamoaHandler ← american_samoa.py:15

### handlers/states/arizona/\_\_init\_\_.py {#webapp-parser-handlers-states-arizona-init-py}

- Imports:
  - **Local/Project** (1):
    - `from arizona import parse as parse` (line 1)

### handlers/states/arizona/arizona.py {#webapp-parser-handlers-states-arizona-arizona-py}

- Top-of-file comments:

```python

# handlers/arizona.py

# ==============================================================

# Handler for Arizona election result sites with expandable cards

# and toggles between 'Vote Type' and 'By County' views.

# ==============================================================

```

- Definitions:
  - function: `parse` (line 33)
- Imports:
  - **Standard Library** (1):
    - `import os as os` (line 6)
  - **Third-party** (1):
    - `import orjson as orjson` (line 8)
  - **Local/Project** (4):
    - `from config import CONTEXT_LIBRARY_PATH` (line 10)
    - `from Context_Integration.context_organizer import ContextOrganizer` (line
      11)
    - `from utils.logger_singleton import logger` (line 12)
    - `from utils.output_utils import finalize_election_output` (line 13)
- Task markers:
  - L25 **WARNING**: ("\[WARN\] context_library.json not found. Using fallback
    config for Arizona handler.")
  - L51 **WARNING**: (f"\[WARN\] Could not expand card {i+1}: {e}")
  - L64 **WARNING**: (f"\[WARN\] Vote Type toggle failed: {e}")
  - L77 **WARNING**: (f"\[WARN\] County toggle failed: {e}")
  - L164 **WARNING**: ("\[FALLBACK\] No tables were parsed. Either no results
    are published yet or the structure has changed.")
  - L165 **WARNING**: ("\[FALLBACK\] Please verify that the site has posted
    election data.")
- Outgoing cross-module calls (sample):
  - orjson.loads (line 21)
  - f.read (line 21)
  - CONTEXT\_LIBRARY.get (line 22)
  - STATE\_CONFIGS.get (line 23)
  - utils.logger\_singleton.logger.warning (line 25)
  - config.setdefault (line 29)
  - config.setdefault (line 30)
  - config.setdefault (line 31)
  - utils.logger\_singleton.logger.info (line 37)
  - config.get (line 40)
  - page.locator (line 42)
  - buttons.count (line 43)
  - buttons.nth (line 45)
  - btn.scroll\_into\_view\_if\_needed (line 46)
  - btn.click (line 47)
  - page.wait\_for\_timeout (line 48)
  - utils.logger\_singleton.logger.info (line 49)
  - buttons.count (line 49)
  - utils.logger\_singleton.logger.warning (line 51)
  - config.get (line 54)
  - page.locator (line 57)
  - vote\_toggle.count (line 58)
  - utils.logger\_singleton.logger.info (line 61)
  - page.wait\_for\_timeout (line 62)
  - utils.logger\_singleton.logger.warning (line 64)
  - config.get (line 67)
  - page.locator (line 70)
  - county\_toggle.count (line 71)
  - utils.logger\_singleton.logger.info (line 74)
  - page.wait\_for\_timeout (line 75)
  - utils.logger\_singleton.logger.warning (line 77)
  - utils.logger\_singleton.logger.info (line 80)
  - utils.logger\_singleton.logger.info (line 81)
  - page.query\_selector\_all (line 85)
  - el.evaluate (line 89)
  - el.inner\_text (line 90)
  - utils.logger\_singleton.logger.debug (line 91)
  - el.inner\_text (line 94)
  - utils.logger\_singleton.logger.debug (line 97)
  - el.query\_selector\_all (line 99)
  - el.query\_selector\_all (line 100)
  - h.inner\_text (line 103)
  - utils.logger\_singleton.logger.progress\_bar (line 106)
  - cell.inner\_text (line 107)
  - row.query\_selector\_all (line 107)
  - full\_name.split (line 111)
  - full\_name.strip (line 117)
  - all\_candidates.add (line 120)
  - row\_blocks.append (line 121)
  - precinct\_data.append (line 135)

### handlers/states/arkansas/arkansas.py {#webapp-parser-handlers-states-arkansas-arkansas-py}

- Definitions:
  - class: `ArkansasHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - ArkansasHandler ← arkansas.py:15

### handlers/states/california.py {#webapp-parser-handlers-states-california-py}

> California State Handler

- Definitions:
  - class: `CaliforniaHandler` (line 19)
  - function: `parse` (line 44)
- Imports:
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 16)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 46)
- Inbound references:
  - CaliforniaHandler ← california.py:42
  - CaliforniaHandler ← california.py:15

### handlers/states/california/california.py {#webapp-parser-handlers-states-california-california-py}

- Definitions:
  - class: `CaliforniaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)

### handlers/states/colorado/colorado.py {#webapp-parser-handlers-states-colorado-colorado-py}

- Definitions:
  - class: `ColoradoHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - ColoradoHandler ← colorado.py:15

### handlers/states/connecticut/connecticut.py {#webapp-parser-handlers-states-connecticut-connecticut-py}

- Definitions:
  - class: `ConnecticutHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - ConnecticutHandler ← connecticut.py:15

### handlers/states/delaware/delaware.py {#webapp-parser-handlers-states-delaware-delaware-py}

- Definitions:
  - class: `DelawareHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - DelawareHandler ← delaware.py:15

### handlers/states/district\_of\_columbia/district\_of\_columbia.py {#webapp-parser-handlers-states-district-of-columbia-district-of-columbia-py}

- Definitions:
  - class: `DistrictofColumbiaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - DistrictofColumbiaHandler ← district_of_columbia.py:15

### handlers/states/example state/example\_county/example\_county.py {#webapp-parser-handlers-states-example-state-example-county-example-county-py}

- Definitions:
  - function: `parse` (line 16)
  - function: `parse\_single\_contest\_dynamic` (line 75)
- Imports:
  - **Standard Library** (1):
    - `from typing import TYPE_CHECKING` (line 1)
  - **Third-party** (1):
    - `from playwright.sync_api import Page` (line 3)
  - **Local/Project** (6):
    - `from utils.contest_selector import select_contest` (line 5)
    - `from utils.html_scanner import scan_html_for_context` (line 6)
    - `from utils.logger_singleton import logger` (line 7)
    - `from utils.output_utils import finalize_election_output` (line 8)
    - `from utils.table_builder import build_dynamic_table` (line 9)
    - `from utils.table_core import robust_table_extraction` (line 10)
- Task markers:
  - L123 **WARNING**: ("\[yellow\]\[WARNING\] No ballot items found by div
    selectors. Trying table-based extraction...\[/yellow\]")
- Outgoing cross-module calls (sample):
  - utils.logger\_singleton.logger.info (line 28)
  - utils.html\_scanner.scan\_html\_for\_context (line 31)
  - html\_context.update (line 42)
  - html\_context.get (line 43)
  - html\_context.get (line 44)
  - coordinator.organize\_and\_enrich (line 47)
  - utils.contest\_selector.select\_contest (line 50)
  - html\_context.get (line 54)
  - utils.logger\_singleton.logger.error (line 57)
  - contest.get (line 64)
  - results.append (line 68)
  - selected.get (line 71)
  - html\_context.get (line 79)
  - utils.logger\_singleton.logger.info (line 80)
  - coordinator.extract\_entities (line 83)
  - ent.lower (line 84)
  - page.locator (line 93)
  - items.count (line 94)
  - items.nth (line 95)
  - item.locator (line 96)
  - cells.nth (line 97)
  - cells.count (line 97)
  - ballot\_items.append (line 99)
  - cell.lower (line 104)
  - headers.append (line 111)
  - headers.append (line 113)
  - headers.append (line 115)
  - headers.append (line 117)
  - headers.append (line 119)
  - utils.logger\_singleton.logger.warning (line 123)
  - utils.table\_core.robust\_table\_extraction (line 124)
  - utils.logger\_singleton.logger.error (line 126)
  - utils.table\_builder.build\_dynamic\_table (line 130)
  - utils.logger\_singleton.logger.error (line 133)
  - row.keys (line 137)
  - utils.output\_utils.finalize\_election\_output (line 145)
  - metadata.update (line 151)

### handlers/states/example state/example\_state.py {#webapp-parser-handlers-states-example-state-example-state-py}

- Definitions:
  - function: `parse` (line 24)
  - function: `parse\_single\_contest\_dynamic` (line 104)
- Imports:
  - **Standard Library** (1):
    - `from typing import TYPE_CHECKING` (line 2)
  - **Third-party** (1):
    - `from playwright.sync_api import Page` (line 4)
  - **Local/Project** (11):
    - `import importlib as importlib` (line 1)
    - `from utils.contest_selector import select_contest` (line 6)
    - `from utils.html_scanner import scan_html_for_context` (line 7)
    - `from utils.logger_singleton import logger` (line 8)
    - `from utils.output_utils import finalize_election_output` (line 9)
    - `from utils.shared_logic import safe_get` (line 10)
    - `from utils.shared_logic import safe_lower` (line 10)
    - `from utils.shared_logic import safe_parse` (line 10)
    - `from utils.shared_logic import safe_strip` (line 10)
    - `from utils.table_builder import build_dynamic_table` (line 16)
    - `from utils.table_core import robust_table_extraction` (line 17)
- Task markers:
  - L51 **WARNING**: (f"\[Example Handler\] No specific parser implemented for
    county: '{county}'. Continuing with state-level logic.")
  - L152 **WARNING**: ("\[yellow\]\[WARNING\] No ballot items found by div
    selectors. Trying table-based extraction...\[/yellow\]")
- Outgoing cross-module calls (sample):
  - utils.shared\_logic.safe\_get (line 36)
  - utils.shared\_logic.safe\_lower (line 37)
  - utils.shared\_logic.safe\_strip (line 37)
  - importlib.import\_module (line 41)
  - utils.logger\_singleton.logger.info (line 42)
  - utils.shared\_logic.safe\_parse (line 43)
  - utils.logger\_singleton.logger.warning (line 51)
  - utils.logger\_singleton.logger.error (line 53)
  - utils.logger\_singleton.logger.info (line 57)
  - utils.html\_scanner.scan\_html\_for\_context (line 60)
  - html\_context.update (line 71)
  - html\_context.get (line 72)
  - html\_context.get (line 73)
  - coordinator.organize\_and\_enrich (line 76)
  - utils.contest\_selector.select\_contest (line 79)
  - html\_context.get (line 83)
  - utils.logger\_singleton.logger.error (line 86)
  - contest.get (line 93)
  - results.append (line 97)
  - selected.get (line 100)
  - html\_context.get (line 108)
  - utils.logger\_singleton.logger.info (line 109)
  - coordinator.extract\_entities (line 112)
  - ent.lower (line 113)
  - page.locator (line 122)
  - items.count (line 123)
  - items.nth (line 124)
  - item.locator (line 125)
  - cells.nth (line 126)
  - cells.count (line 126)
  - ballot\_items.append (line 128)
  - cell.lower (line 133)
  - headers.append (line 140)
  - headers.append (line 142)
  - headers.append (line 144)
  - headers.append (line 146)
  - headers.append (line 148)
  - utils.logger\_singleton.logger.warning (line 152)
  - utils.table\_core.robust\_table\_extraction (line 154)
  - utils.logger\_singleton.logger.error (line 156)
  - utils.table\_builder.build\_dynamic\_table (line 160)
  - utils.logger\_singleton.logger.error (line 163)
  - row.keys (line 167)
  - utils.output\_utils.finalize\_election\_output (line 175)
  - metadata.update (line 181)
- Inbound references:
  - parse\_single\_contest\_dynamic ← example_state.py:96
  - parse\_single\_contest\_dynamic ← example_state.py:102
  - parse\_single\_contest\_dynamic ← example_county.py:67
  - parse\_single\_contest\_dynamic ← example_county.py:73

### handlers/states/florida/florida.py {#webapp-parser-handlers-states-florida-florida-py}

- Definitions:
  - class: `FloridaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - FloridaHandler ← florida.py:15

### handlers/states/georgia/georgia.py {#webapp-parser-handlers-states-georgia-georgia-py}

- Definitions:
  - class: `GeorgiaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - GeorgiaHandler ← georgia.py:15

### handlers/states/guam/guam.py {#webapp-parser-handlers-states-guam-guam-py}

- Definitions:
  - class: `GuamHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - GuamHandler ← guam.py:15

### handlers/states/hawaii/hawaii.py {#webapp-parser-handlers-states-hawaii-hawaii-py}

- Definitions:
  - class: `HawaiiHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - HawaiiHandler ← hawaii.py:15

### handlers/states/idaho/idaho.py {#webapp-parser-handlers-states-idaho-idaho-py}

- Definitions:
  - class: `IdahoHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - IdahoHandler ← idaho.py:15

### handlers/states/illinois/illinois.py {#webapp-parser-handlers-states-illinois-illinois-py}

- Definitions:
  - class: `IllinoisHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - IllinoisHandler ← illinois.py:15

### handlers/states/indiana/indiana.py {#webapp-parser-handlers-states-indiana-indiana-py}

- Definitions:
  - class: `IndianaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - IndianaHandler ← indiana.py:15

### handlers/states/iowa/iowa.py {#webapp-parser-handlers-states-iowa-iowa-py}

- Definitions:
  - class: `IowaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - IowaHandler ← iowa.py:15

### handlers/states/kansas/kansas.py {#webapp-parser-handlers-states-kansas-kansas-py}

- Definitions:
  - class: `KansasHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - KansasHandler ← kansas.py:15

### handlers/states/kentucky/kentucky.py {#webapp-parser-handlers-states-kentucky-kentucky-py}

- Definitions:
  - class: `KentuckyHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - KentuckyHandler ← kentucky.py:15

### handlers/states/louisiana/louisiana.py {#webapp-parser-handlers-states-louisiana-louisiana-py}

- Definitions:
  - class: `LouisianaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - LouisianaHandler ← louisiana.py:15

### handlers/states/maine/maine.py {#webapp-parser-handlers-states-maine-maine-py}

- Definitions:
  - class: `MaineHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - MaineHandler ← maine.py:15

### handlers/states/maryland/maryland.py {#webapp-parser-handlers-states-maryland-maryland-py}

- Definitions:
  - class: `MarylandHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - MarylandHandler ← maryland.py:15

### handlers/states/massachusetts/massachusetts.py {#webapp-parser-handlers-states-massachusetts-massachusetts-py}

- Definitions:
  - class: `MassachusettsHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - MassachusettsHandler ← massachusetts.py:15

### handlers/states/michigan/michigan.py {#webapp-parser-handlers-states-michigan-michigan-py}

- Definitions:
  - class: `MichiganHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - MichiganHandler ← michigan.py:15

### handlers/states/minnesota/minnesota.py {#webapp-parser-handlers-states-minnesota-minnesota-py}

- Definitions:
  - class: `MinnesotaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - MinnesotaHandler ← minnesota.py:15

### handlers/states/mississippi/mississippi.py {#webapp-parser-handlers-states-mississippi-mississippi-py}

- Definitions:
  - class: `MississippiHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - MississippiHandler ← mississippi.py:15

### handlers/states/missouri/missouri.py {#webapp-parser-handlers-states-missouri-missouri-py}

- Definitions:
  - class: `MissouriHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - MissouriHandler ← missouri.py:15

### handlers/states/montana/montana.py {#webapp-parser-handlers-states-montana-montana-py}

- Definitions:
  - class: `MontanaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - MontanaHandler ← montana.py:15

### handlers/states/nebraska/nebraska.py {#webapp-parser-handlers-states-nebraska-nebraska-py}

- Definitions:
  - class: `NebraskaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - NebraskaHandler ← nebraska.py:15

### handlers/states/nevada/nevada.py {#webapp-parser-handlers-states-nevada-nevada-py}

- Definitions:
  - class: `NevadaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - NevadaHandler ← nevada.py:15

### handlers/states/new\_hampshire/new\_hampshire.py {#webapp-parser-handlers-states-new-hampshire-new-hampshire-py}

- Definitions:
  - class: `NewHampshireHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - NewHampshireHandler ← new_hampshire.py:15

### handlers/states/new\_jersey/new\_jersey.py {#webapp-parser-handlers-states-new-jersey-new-jersey-py}

- Definitions:
  - class: `NewJerseyHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - NewJerseyHandler ← new_jersey.py:15

### handlers/states/new\_mexico/new\_mexico.py {#webapp-parser-handlers-states-new-mexico-new-mexico-py}

- Definitions:
  - class: `NewMexicoHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - NewMexicoHandler ← new_mexico.py:15

### handlers/states/new\_york/county/rockland.py {#webapp-parser-handlers-states-new-york-county-rockland-py}

- Definitions:
  - function: `\_write\_debug\_html` (line 80)
  - function: `\_score\_keyword\_match` (line 89)
  - function: `\_extract\_button\_label` (line 99)
  - function: `\_fallback\_button\_search` (line 116)
  - function: `\_score\_keyword\_groups` (line 136)
  - function: `\_flatten\_panel\_text` (line 142)
  - function: `parse` (line 153)
- Imports:
  - **Standard Library** (2):
    - `from pathlib import Path` (line 1)
    - `from typing import TYPE_CHECKING` (line 2)
  - **Third-party** (1):
    - `from playwright.sync_api import Page` (line 4)
  - **Local/Project** (13):
    - `from Context_Integration.librarian import clean_for_json` (line 6)
    - `from utils.browser_utils import autoscroll_until_stable` (line 7)
    - `from utils.browser_utils import safe_click` (line 7)
    - `from utils.browser_utils import safe_is_enabled` (line 7)
    - `from utils.browser_utils import safe_is_visible` (line 7)
    - `from utils.contest_selector import select_contest_auto_first` (line 13)
    - `from utils.html_scanner import scan_html_for_context` (line 14)
    - `from utils.logger_singleton import logger` (line 15)
    - `from utils.logger_singleton import prompt` (line 15)
    - `from utils.output_utils import finalize_election_output` (line 16)
    - `from utils.shared_logic import safe_get` (line 17)
    - `from utils.table_builder import build_dynamic_table` (line 18)
    - `from utils.table_core import harmonize_headers_and_data` (line 19)
- Task markers:
  - L198 **WARNING**: ("\[WARNING\] dom_parts missing after
    organize_and_enrich.")
  - L221 **WARNING**: ("\[red\]No contest selected. Skipping.\[/red\]")
  - L267 **WARNING**: (f"\[yellow\]\[WARNING\] Button '{btn1.get('label', '')}'
    is not clickable (visible={safe_is_visible(element, logger)},
    enabled={safe_is_enabled(element, logger)})\[/yellow\]")
  - L271 **WARNING**: (f"\[yellow\]\[WARNING\] No suitable '{toggle_name}'
    button found; continuing without toggle.\[/yellow\]")
  - L306 **WARNING**: (f"\[yellow\]\[WARNING\] Button '{btn2.get('label', '')}'
    is not clickable (visible={safe_is_visible(element, logger)},
    enabled={safe_is_enabled(element, logger)})\[/yellow\]")
  - L310 **WARNING**: (f"\[yellow\]\[WARNING\] No suitable '{toggle_name2}'
    button found; continuing without toggle.\[/yellow\]")
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 28)
  - DEBUG\_OUTPUT\_DIR.mkdir (line 82)
  - out\_path.write\_text (line 85)
  - text.lower (line 92)
  - weights.items (line 94)
  - element.inner\_text (line 101)
  - element.get\_attribute (line 106)
  - element.get\_attribute (line 111)
  - label.strip (line 114)
  - page.locator (line 118)
  - candidates.count (line 121)
  - candidates.nth (line 122)
  - vocab.items (line 138)
  - utils.shared\_logic.safe\_get (line 144)
  - parts.append (line 146)
  - utils.shared\_logic.safe\_get (line 147)
  - utils.shared\_logic.safe\_get (line 148)
  - parts.append (line 150)
  - utils.logger\_singleton.logger.info (line 166)
  - utils.html\_scanner.scan\_html\_for\_context (line 169)
  - context\_result.get (line 180)
  - context\_result.get (line 181)
  - context\_result.get (line 182)
  - utils.shared\_logic.safe\_get (line 183)
  - utils.shared\_logic.safe\_get (line 184)
  - utils.shared\_logic.safe\_get (line 186)
  - utils.shared\_logic.safe\_get (line 188)
  - Context\_Integration.librarian.clean\_for\_json (line 193)
  - coordinator.organize\_and\_enrich (line 194)
  - utils.logger\_singleton.logger.debug (line 196)
  - utils.logger\_singleton.logger.warning (line 198)
  - coordinator.get\_for\_selector (line 199)
  - utils.logger\_singleton.logger.debug (line 200)
  - selector\_data.get (line 200)
  - context\_result.get (line 206)
  - html\_context.items (line 207)
  - utils.contest\_selector.select\_contest\_auto\_first (line 212)
  - utils.logger\_singleton.logger.warning (line 221)
  - user\_selected\_contest.get (line 228)
  - utils.logger\_singleton.logger.info (line 229)
  - user\_selected\_contest.get (line 229)
  - utils.logger\_singleton.logger.debug (line 241)
  - coordinator.get\_best\_button\_advanced (line 242)
  - btn1.get (line 256)
  - utils.browser\_utils.safe\_is\_visible (line 257)
  - utils.browser\_utils.safe\_is\_enabled (line 257)
  - utils.logger\_singleton.logger.debug (line 259)
  - btn1.get (line 259)
  - utils.browser\_utils.safe\_click (line 260)
  - page.wait\_for\_timeout (line 261)
- Inbound references:
  - \_write\_debug\_html ← rockland.py:319
  - \_score\_keyword\_match ← rockland.py:126
  - \_score\_keyword\_match ← rockland.py:139
  - \_extract\_button\_label ← rockland.py:123
  - \_fallback\_button\_search ← rockland.py:252
  - \_fallback\_button\_search ← rockland.py:291
  - \_score\_keyword\_groups ← rockland.py:374
  - \_flatten\_panel\_text ← rockland.py:373

### handlers/states/new\_york/county/westchester.py {#webapp-parser-handlers-states-new-york-county-westchester-py}

> Westchester County Handler (New York)

- Definitions:
  - function: `parse` (line 43)
- Imports:
  - **Standard Library** (5):
    - `from typing import TYPE_CHECKING` (line 21)
    - `from typing import Any` (line 21)
    - `from typing import Dict` (line 21)
    - `from typing import List` (line 21)
    - `from typing import Tuple` (line 21)
  - **Third-party** (8):
    - `from playwright.sync_api import Page` (line 23)
    - `from webapp.parser.Context_Integration.librarian import clean_for_json`
      (line 24)
    - `from webapp.parser.utils.browser_utils import autoscroll_until_stable`
      (line 25)
    - `from webapp.parser.utils.contest_selector import
      select_contest_auto_first` (line 28)
    - `from webapp.parser.utils.html_scanner import scan_html_for_context` (line
      29)
    - `from webapp.parser.utils.logger_singleton import logger` (line 30)
    - `from webapp.parser.utils.shared_logic import safe_get` (line 31)
    - `from webapp.parser.utils.table_core import robust_table_extraction` (line
      32)
- Task markers:
  - L61 **TODO**: Customize this handler for Westchester County's specific UI.
  - L122 **WARNING**: ("\[Westchester County\] No contest selected")
  - L132 **TODO**: Add button toggles, navigation sequences, etc. specific to
    Westchester County
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.logger\_singleton.logger.info (line 71)
  - webapp.parser.utils.html\_scanner.scan\_html\_for\_context (line 74)
  - html\_context.get (line 75)
  - html\_context.get (line 81)
  - context\_result.get (line 86)
  - context\_result.get (line 87)
  - context\_result.get (line 88)
  - webapp.parser.utils.shared\_logic.safe\_get (line 91)
  - contest.setdefault (line 92)
  - contest.setdefault (line 93)
  - contest.setdefault (line 95)
  - webapp.parser.Context\_Integration.librarian.clean\_for\_json (line 100)
  - coordinator.organize\_and\_enrich (line 101)
  - context\_result.get (line 108)
  - html\_context.items (line 110)
  - webapp.parser.utils.contest\_selector.select\_contest\_auto\_first (line
    113)
  - html\_context.get (line 118)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 122)
  - webapp.parser.utils.logger\_singleton.logger.info (line 129)
  - selected.get (line 129)
  - html\_context.get (line 151)
  - webapp.parser.utils.logger\_singleton.logger.info (line 152)
  - webapp.parser.utils.browser\_utils.autoscroll\_until\_stable (line 153)
  - webapp.parser.utils.table\_core.robust\_table\_extraction (line 156)
  - result.get (line 164)
  - result.get (line 165)
  - selected.get (line 172)
  - html\_context.get (line 173)
  - context\_result.get (line 173)
  - state.lower (line 175)
  - selected.get (line 180)
  - webapp.parser.utils.logger\_singleton.logger.info (line 181)

### handlers/states/new\_york/new\_york.py {#webapp-parser-handlers-states-new-york-new-york-py}

- Definitions:
  - function: `parse` (line 15)
- Imports:
  - **Standard Library** (3):
    - `from typing import Any` (line 2)
    - `from typing import Optional` (line 2)
    - `from typing import Tuple` (line 2)
  - **Third-party** (1):
    - `from playwright.sync_api import Page` (line 4)
  - **Local/Project** (6):
    - `import importlib as importlib` (line 1)
    - `from utils.logger_singleton import logger` (line 6)
    - `from utils.shared_logic import safe_get` (line 7)
    - `from utils.shared_logic import safe_lower` (line 7)
    - `from utils.shared_logic import safe_parse` (line 7)
    - `from utils.shared_logic import safe_strip` (line 7)
- Task markers:
  - L27 **WARNING**: ("\[NY Handler\] No county specified in html_context.")
  - L47 **WARNING**: (f"\[NY Handler\] No specific parser implemented for
    county: '{county}'. Please add it under {module_path}.py")
- Outgoing cross-module calls (sample):
  - utils.shared\_logic.safe\_get (line 24)
  - utils.shared\_logic.safe\_lower (line 25)
  - utils.shared\_logic.safe\_strip (line 25)
  - utils.logger\_singleton.logger.warning (line 27)
  - importlib.import\_module (line 33)
  - utils.logger\_singleton.logger.info (line 34)
  - utils.shared\_logic.safe\_parse (line 36)
  - utils.logger\_singleton.logger.warning (line 47)
  - utils.logger\_singleton.logger.error (line 50)

### handlers/states/north\_carolina/north\_carolina.py {#webapp-parser-handlers-states-north-carolina-north-carolina-py}

- Definitions:
  - class: `NorthCarolinaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - NorthCarolinaHandler ← north_carolina.py:15

### handlers/states/north\_dakota/north\_dakota.py {#webapp-parser-handlers-states-north-dakota-north-dakota-py}

- Definitions:
  - class: `NorthDakotaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - NorthDakotaHandler ← north_dakota.py:15

### handlers/states/northern\_mariana\_islands/northern\_mariana\_islands.py {#webapp-parser-handlers-states-northern-mariana-islands-northern-mariana-islands-py}

- Definitions:
  - class: `NorthernMarianaIslandsHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - NorthernMarianaIslandsHandler ← northern_mariana_islands.py:15

### handlers/states/ohio/ohio.py {#webapp-parser-handlers-states-ohio-ohio-py}

- Definitions:
  - class: `OhioHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - OhioHandler ← ohio.py:15

### handlers/states/oklahoma/oklahoma.py {#webapp-parser-handlers-states-oklahoma-oklahoma-py}

- Definitions:
  - class: `OklahomaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - OklahomaHandler ← oklahoma.py:15

### handlers/states/oregon/oregon.py {#webapp-parser-handlers-states-oregon-oregon-py}

- Definitions:
  - class: `OregonHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - OregonHandler ← oregon.py:15

### handlers/states/pennsylvania/\_\_init\_\_.py {#webapp-parser-handlers-states-pennsylvania-init-py}

- Imports:
  - **Local/Project** (1):
    - `from pennsylvania import parse as parse` (line 1)

### handlers/states/pennsylvania/pennsylvania.py {#webapp-parser-handlers-states-pennsylvania-pennsylvania-py}

- Definitions:
  - function: `apply\_navigation\_steps` (line 25)
  - function: `parse` (line 46)
- Imports:
  - **Standard Library** (3):
    - `import csv as csv` (line 1)
    - `import os as os` (line 2)
    - `from pathlib import Path` (line 3)
  - **Local/Project** (12):
    - `from config import BASE_DIR` (line 5)
    - `from utils.browser_utils import safe_click` (line 6)
    - `from utils.browser_utils import safe_inner_text` (line 6)
    - `from utils.browser_utils import safe_query_selector_all` (line 6)
    - `from utils.browser_utils import safe_wait_for_timeout` (line 6)
    - `from utils.logger_singleton import logger` (line 12)
    - `from utils.output_utils import finalize_election_output` (line 13)
    - `from utils.shared_logic import safe_get` (line 14)
    - `from utils.shared_logic import safe_isdigit` (line 14)
    - `from utils.shared_logic import safe_lower` (line 14)
    - `from utils.shared_logic import safe_replace` (line 14)
    - `from utils.shared_logic import safe_strip` (line 14)
- Task markers:
  - L44 **WARNING**: (f"\[NAV\] Step failed: {step} — {e}")
  - L55 **WARNING**: (f"\[bold yellow\]Detected election:\[/bold yellow\]
    {header_text}")
  - L76 **WARNING**: ("\[PA\] Invalid index input for election selection.")
  - L78 **WARNING**: ("\[PA\] Elections dropdown not found.")
  - L80 **WARNING**: (f"\[PA\] Failed to expand Elections menu or load
    selection: {e}")
  - L96 **WARNING**: ("\[PA\] County Breakdown link not found.")
  - L98 **WARNING**: (f"\[PA\] Failed to click County Breakdown link: {e}")
  - L113 **WARNING**: ("\[yellow\]Multiple CSV files found in input. Please
    select one:\[/yellow\]")
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 22)
  - pathlib.Path (line 23)
  - utils.shared\_logic.safe\_get (line 26)
  - utils.shared\_logic.safe\_get (line 29)
  - utils.shared\_logic.safe\_get (line 30)
  - utils.shared\_logic.safe\_get (line 31)
  - utils.shared\_logic.safe\_get (line 32)
  - utils.browser\_utils.safe\_query\_selector\_all (line 34)
  - utils.logger\_singleton.logger.info (line 37)
  - utils.browser\_utils.safe\_click (line 38)
  - utils.browser\_utils.safe\_wait\_for\_timeout (line 39)
  - utils.logger\_singleton.logger.info (line 41)
  - utils.browser\_utils.safe\_wait\_for\_timeout (line 42)
  - utils.logger\_singleton.logger.warning (line 44)
  - utils.shared\_logic.safe\_get (line 48)
  - utils.logger\_singleton.logger.info (line 49)
  - utils.shared\_logic.safe\_get (line 54)
  - utils.logger\_singleton.logger.warning (line 55)
  - utils.shared\_logic.safe\_lower (line 56)
  - utils.shared\_logic.safe\_strip (line 56)
  - utils.logger\_singleton.logger.info (line 58)
  - utils.browser\_utils.safe\_query\_selector\_all (line 61)
  - utils.browser\_utils.safe\_click (line 64)
  - utils.browser\_utils.safe\_wait\_for\_timeout (line 65)
  - utils.browser\_utils.safe\_query\_selector\_all (line 66)
  - utils.shared\_logic.safe\_strip (line 68)
  - utils.browser\_utils.safe\_inner\_text (line 68)
  - utils.logger\_singleton.logger.info (line 69)
  - utils.shared\_logic.safe\_strip (line 70)
  - utils.shared\_logic.safe\_isdigit (line 71)
  - utils.browser\_utils.safe\_click (line 73)
  - utils.browser\_utils.safe\_wait\_for\_timeout (line 74)
  - utils.logger\_singleton.logger.warning (line 76)
  - utils.logger\_singleton.logger.warning (line 78)
  - utils.logger\_singleton.logger.warning (line 80)
  - utils.logger\_singleton.logger.info (line 82)
  - utils.shared\_logic.safe\_get (line 86)
  - utils.logger\_singleton.logger.info (line 88)
  - utils.browser\_utils.safe\_query\_selector\_all (line 89)
  - utils.browser\_utils.safe\_click (line 92)
  - utils.browser\_utils.safe\_wait\_for\_timeout (line 93)
  - utils.logger\_singleton.logger.info (line 94)
  - utils.logger\_singleton.logger.warning (line 96)
  - utils.logger\_singleton.logger.warning (line 98)
  - os.listdir (line 102)
  - utils.shared\_logic.safe\_lower (line 102)
  - utils.logger\_singleton.logger.error (line 104)
  - utils.logger\_singleton.logger.error (line 108)
  - utils.logger\_singleton.logger.warning (line 113)
  - utils.logger\_singleton.logger.info (line 115)
- Inbound references:
  - apply\_navigation\_steps ← pennsylvania.py:52
  - apply\_navigation\_steps ← pennsylvania.py:83

### handlers/states/puerto\_rico/puerto\_rico.py {#webapp-parser-handlers-states-puerto-rico-puerto-rico-py}

- Definitions:
  - class: `PuertoRicoHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - PuertoRicoHandler ← puerto_rico.py:15

### handlers/states/rhode\_island/rhode\_island.py {#webapp-parser-handlers-states-rhode-island-rhode-island-py}

- Definitions:
  - class: `RhodeIslandHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - RhodeIslandHandler ← rhode_island.py:15

### handlers/states/south\_carolina/south\_carolina.py {#webapp-parser-handlers-states-south-carolina-south-carolina-py}

- Definitions:
  - class: `SouthCarolinaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - SouthCarolinaHandler ← south_carolina.py:15

### handlers/states/south\_dakota/south\_dakota.py {#webapp-parser-handlers-states-south-dakota-south-dakota-py}

- Definitions:
  - class: `SouthDakotaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - SouthDakotaHandler ← south_dakota.py:15

### handlers/states/tennessee/tennessee.py {#webapp-parser-handlers-states-tennessee-tennessee-py}

- Definitions:
  - class: `TennesseeHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - TennesseeHandler ← tennessee.py:15

### handlers/states/texas.py {#webapp-parser-handlers-states-texas-py}

> Texas State Handler

- Definitions:
  - class: `TexasHandler` (line 19)
  - function: `parse` (line 44)
- Imports:
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 16)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 46)
- Inbound references:
  - TexasHandler ← texas.py:42
  - TexasHandler ← texas.py:15

### handlers/states/texas/texas.py {#webapp-parser-handlers-states-texas-texas-py}

- Definitions:
  - class: `TexasHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)

### handlers/states/us\_virgin\_islands/us\_virgin\_islands.py {#webapp-parser-handlers-states-us-virgin-islands-us-virgin-islands-py}

- Definitions:
  - class: `USVirginIslandsHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - USVirginIslandsHandler ← us_virgin_islands.py:15

### handlers/states/utah/utah.py {#webapp-parser-handlers-states-utah-utah-py}

- Definitions:
  - class: `UtahHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - UtahHandler ← utah.py:15

### handlers/states/vermont/vermont.py {#webapp-parser-handlers-states-vermont-vermont-py}

- Definitions:
  - class: `VermontHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - VermontHandler ← vermont.py:15

### handlers/states/virginia/virginia.py {#webapp-parser-handlers-states-virginia-virginia-py}

- Definitions:
  - class: `VirginiaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - VirginiaHandler ← virginia.py:15

### handlers/states/washington/washington.py {#webapp-parser-handlers-states-washington-washington-py}

- Definitions:
  - class: `WashingtonHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - WashingtonHandler ← washington.py:15

### handlers/states/west\_virginia/west\_virginia.py {#webapp-parser-handlers-states-west-virginia-west-virginia-py}

- Definitions:
  - class: `WestVirginiaHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - WestVirginiaHandler ← west_virginia.py:15

### handlers/states/wisconsin/wisconsin.py {#webapp-parser-handlers-states-wisconsin-wisconsin-py}

- Definitions:
  - class: `WisconsinHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - WisconsinHandler ← wisconsin.py:15

### handlers/states/wyoming/wyoming.py {#webapp-parser-handlers-states-wyoming-wyoming-py}

- Definitions:
  - class: `WyomingHandler` (line 8)
  - function: `parse` (line 17)
- Imports:
  - **Standard Library** (2):
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
  - **Third-party** (1):
    - `from webapp.parser.handlers.shared.state_handler_base import
      SimpleTableHandler` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - \_handler\_instance.parse (line 19)
- Inbound references:
  - WyomingHandler ← wyoming.py:15

### handlers/vendor\_state\_map.py {#webapp-parser-handlers-vendor-state-map-py}

> State to vendor mapping for vendor-dispatch handlers.

- Definitions:
  - function: `get\_vendor\_for\_state` (line 51)
- Imports:
  - **Standard Library** (2):
    - `from typing import Dict` (line 8)
    - `from typing import List` (line 8)
  - **Third-party** (1):
    - `from webapp.parser.utils.shared_logic import normalize_state_name` (line
      10)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 6)
- Task markers:
  - L38 **TODO**: enhancedvoting.com domain; confirm vendor",
  - L45 **TODO**: enhancedvoting.com domain; confirm vendor",
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.shared\_logic.normalize\_state\_name (line 53)
  - webapp.parser.utils.shared\_logic.normalize\_state\_name (line 57)
  - entry.get (line 57)
  - entry.get (line 58)
  - entry.get (line 59)

### health/context\_migration.py {#webapp-parser-health-context-migration-py}

- Definitions:
  - function: `table\_structure\_exists` (line 30)
  - function: `create\_table\_structure` (line 37)
  - function: `migrate\_table\_structures\_from\_jsonl` (line 50)
  - function: `migrate\_table\_structures\_from\_json` (line 74)
  - function: `load\_migration\_state` (line 109)
  - function: `save\_migration\_state` (line 115)
  - function: `\_normalize\_geo` (line 120)
  - function: `\_coerce\_year` (line 128)
  - function: `\_ensure\_contest\_for\_snapshot` (line 137)
  - function: `migrate\_context\_snapshot\_from\_metadata` (line 191)
  - function: `migrate\_all` (line 281)
  - function: `migrate\_context\_cache\_to\_db` (line 320)
- Imports:
  - **Standard Library** (6):
    - `from datetime import datetime` (line 1)
    - `from datetime import timezone` (line 1)
    - `from pathlib import Path` (line 2)
    - `from typing import Any` (line 3)
    - `from typing import Dict` (line 3)
    - `from typing import List` (line 3)
  - **Third-party** (1):
    - `import orjson as orjson` (line 5)
  - **Local/Project** (22):
    - `from config import CACHE_DIR` (line 7)
    - `from config import CONTEXT_LIBRARY_DIR` (line 7)
    - `from config import LOG_DIR` (line 7)
    - `from config import OUTPUT_DIR` (line 7)
    - `from Context_Integration.librarian import clean_for_json` (line 8)
    - `from utils.db_utils import get_or_create_county` (line 9)
    - `from utils.db_utils import get_or_create_state` (line 9)
    - `from utils.db_utils import get_session` (line 9)
    - `from utils.html_scanner import export_context_cache_for_db` (line 10)
    - `from utils.logger_singleton import console` (line 11)
    - `from utils.models import BallotType` (line 12)
    - `from utils.models import CandidatePanel` (line 12)
    - `from utils.models import Contest` (line 12)
    - `from utils.models import Heading` (line 12)
    - `from utils.models import LocationPanel` (line 12)
    - `from utils.models import Panel` (line 12)
    - `from utils.models import PartyLabel` (line 12)
    - `from utils.models import ResultsTimestamp` (line 12)
    - `from utils.models import TableStructure` (line 12)
    - `from utils.models import VoteMethod` (line 12)
    - `from manual_correction_bot import AUX_FIELDS` (line 24)
    - `from manual_correction_bot import MAIN_FIELDS` (line 24)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 28)
  - session.query (line 31)
  - utils.models.TableStructure (line 41)
  - session.add (line 47)
  - utils.logger\_singleton.console.table (line 48)
  - utils.logger\_singleton.console.panel (line 51)
  - utils.db\_utils.get\_session (line 53)
  - orjson.loads (line 57)
  - utils.logger\_singleton.console.log (line 59)
  - utils.logger\_singleton.console.log (line 62)
  - entry.get (line 64)
  - entry.get (line 64)
  - entry.get (line 65)
  - orjson.dumps (line 66)
  - entry.get (line 66)
  - orjson.dumps (line 67)
  - entry.get (line 67)
  - session.commit (line 71)
  - utils.logger\_singleton.console.log (line 72)
  - utils.logger\_singleton.console.log (line 75)
  - utils.db\_utils.get\_session (line 77)
  - orjson.loads (line 80)
  - f.read (line 80)
  - utils.logger\_singleton.console.log (line 82)
  - data.get (line 87)
  - utils.logger\_singleton.console.log (line 92)
  - entry.get (line 94)
  - orjson.dumps (line 95)
  - entry.get (line 95)
  - orjson.dumps (line 96)
  - entry.get (line 96)
  - entry.get (line 103)
  - session.commit (line 106)
  - utils.logger\_singleton.console.panel (line 107)
  - MIGRATION\_STATE\_FILE.exists (line 110)
  - orjson.loads (line 112)
  - f.read (line 112)
  - f.write (line 118)
  - orjson.dumps (line 118)
  - value.lower (line 124)
  - value.strip (line 132)
  - raw.isdigit (line 133)
  - utils.db\_utils.get\_or\_create\_state (line 147)
  - utils.db\_utils.get\_or\_create\_county (line 148)
  - session.query (line 150)
  - query.filter (line 152)
  - query.filter (line 154)
  - query.filter (line 156)
  - query.filter (line 158)
  - query.filter (line 160)
- Inbound references:
  - table\_structure\_exists ← context_migration.py:68
  - table\_structure\_exists ← context_migration.py:97
  - create\_table\_structure ← context_migration.py:69
  - create\_table\_structure ← context_migration.py:98
  - migrate\_table\_structures\_from\_jsonl ← context_migration.py:311
  - migrate\_table\_structures\_from\_json ← context_migration.py:313
  - load\_migration\_state ← context_migration.py:286
  - save\_migration\_state ← context_migration.py:318
  - \_normalize\_geo ← context_migration.py:214
  - \_normalize\_geo ← context_migration.py:215
  - \_coerce\_year ← context_migration.py:220
  - \_ensure\_contest\_for\_snapshot ← context_migration.py:251
  - migrate\_context\_snapshot\_from\_metadata ← context_migration.py:305
  - migrate\_all ← context_migration.py:420
  - migrate\_context\_cache\_to\_db ← context_migration.py:292

### health/create\_test\_dataset.py {#webapp-parser-health-create-test-dataset-py}

> Test Dataset Split Script for NER Model Evaluation

- Definitions:
  - function: `load\_verified\_ner\_data\_from\_db` (line 31)
  - function: `load\_ner\_data\_from\_jsonl` (line 51)
  - function: `split\_train\_test` (line 78)
  - function: `save\_datasets` (line 119)
  - function: `compute\_entity\_distribution` (line 155)
  - function: `print\_dataset\_statistics` (line 166)
  - function: `main` (line 185)
- Imports:
  - **Standard Library** (8):
    - `import os as os` (line 14)
    - `import random as random` (line 15)
    - `import sys as sys` (line 16)
    - `from pathlib import Path` (line 17)
    - `from typing import Any` (line 18)
    - `from typing import Dict` (line 18)
    - `from typing import List` (line 18)
    - `from typing import Tuple` (line 18)
  - **Third-party** (4):
    - `import orjson as orjson` (line 20)
    - `from webapp.parser.config import LOG_DIR` (line 26)
    - `from webapp.parser.utils.db_utils import SessionLocal` (line 27)
    - `from webapp.parser.utils.logger_singleton import logger` (line 28)
- Task markers:
  - L56 **WARNING**: (f"\[TEST_SPLIT\] No JSONL training data found at
    {jsonl_path}")
  - L71 **WARNING**: (f"\[TEST_SPLIT\] Failed to parse JSONL line: {e}")
  - L97 **WARNING**: (
  - L190 **WARNING**: ("\[TEST_SPLIT\] No verified data in DB, falling back to
    JSONL")
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 23)
  - webapp.parser.utils.db\_utils.SessionLocal (line 33)
  - session.execute (line 34)
  - data.append (line 40)
  - webapp.parser.utils.logger\_singleton.logger.info (line 47)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 56)
  - orjson.loads (line 63)
  - data.append (line 64)
  - example.get (line 65)
  - example.get (line 66)
  - example.get (line 67)
  - example.get (line 68)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 71)
  - webapp.parser.utils.logger\_singleton.logger.info (line 74)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 97)
  - random.seed (line 104)
  - data.copy (line 105)
  - random.shuffle (line 106)
  - webapp.parser.utils.logger\_singleton.logger.info (line 115)
  - os.makedirs (line 125)
  - f.write (line 131)
  - orjson.dumps (line 131)
  - webapp.parser.utils.logger\_singleton.logger.info (line 132)
  - f.write (line 139)
  - orjson.dumps (line 139)
  - webapp.parser.utils.logger\_singleton.logger.info (line 140)
  - orjson.dumps (line 148)
  - f.write (line 151)
  - orjson.dumps (line 151)
  - webapp.parser.utils.logger\_singleton.logger.info (line 152)
  - example.get (line 159)
  - ent.get (line 161)
  - entity\_counts.get (line 162)
  - webapp.parser.utils.logger\_singleton.logger.info (line 171)
  - webapp.parser.utils.logger\_singleton.logger.info (line 172)
  - webapp.parser.utils.logger\_singleton.logger.info (line 173)
  - webapp.parser.utils.logger\_singleton.logger.info (line 174)
  - webapp.parser.utils.logger\_singleton.logger.info (line 178)
  - webapp.parser.utils.logger\_singleton.logger.info (line 182)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 190)
  - webapp.parser.utils.logger\_singleton.logger.error (line 194)
  - sys.exit (line 195)
  - webapp.parser.utils.logger\_singleton.logger.info (line 214)
  - webapp.parser.utils.logger\_singleton.logger.info (line 215)
- Inbound references:
  - load\_verified\_ner\_data\_from\_db ← create_test_dataset.py:188
  - load\_ner\_data\_from\_jsonl ← create_test_dataset.py:191
  - load\_ner\_data\_from\_jsonl ← fine_tune_bert_ner.py:166
  - split\_train\_test ← create_test_dataset.py:201
  - save\_datasets ← create_test_dataset.py:209
  - compute\_entity\_distribution ← create_test_dataset.py:177
  - compute\_entity\_distribution ← create_test_dataset.py:181
  - print\_dataset\_statistics ← create_test_dataset.py:212

### health/dataset\_promotion.py {#webapp-parser-health-dataset-promotion-py}

- Definitions:
  - function: `discover\_dataset\_dirs` (line 67)
  - function: `resolve\_dataset\_path` (line 79)
  - function: `\_load\_metadata` (line 94)
  - function: `\_load\_rows` (line 101)
  - function: `\_has\_value` (line 110)
  - function: `\_match\_field` (line 118)
  - function: `\_coerce\_text` (line 137)
  - function: `\_coerce\_votes` (line 144)
  - function: `\_resolve\_election\_date` (line 168)
  - function: `build\_warehouse\_records` (line 193)
  - function: `promote\_dataset` (line 242)
  - function: `\_build\_arg\_parser` (line 350)
  - function: `main` (line 374)
- Imports:
  - **Standard Library** (8):
    - `import argparse as argparse` (line 3)
    - `import csv as csv` (line 4)
    - `from datetime import datetime` (line 5)
    - `from datetime import timezone` (line 5)
    - `from pathlib import Path` (line 6)
    - `from typing import Any` (line 7)
    - `from typing import Iterable` (line 7)
    - `from typing import Sequence` (line 7)
  - **Third-party** (10):
    - `import orjson as orjson` (line 9)
    - `from webapp.parser.config import OUTPUT_DIR` (line 10)
    - `from webapp.parser.Context_Integration.librarian import clean_for_json`
      (line 11)
    - `from webapp.parser.health.promotion_helpers import check_exact_duplicate`
      (line 12)
    - `from webapp.parser.health.promotion_helpers import
      get_url_verification_tier` (line 12)
    - `from webapp.parser.utils.db_utils import create_batch_metadata` (line 13)
    - `from webapp.parser.utils.db_utils import
      create_warehouse_election_result` (line 13)
    - `from webapp.parser.utils.db_utils import update_batch_metadata` (line 13)
    - `from webapp.parser.utils.logger_singleton import logger` (line 18)
    - `from webapp.parser.utils.models import StatusEnum` (line 19)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Task markers:
  - L294 **WARNING**: (f"\[PROMOTE\] Skipping blocked URL: {source_url}")
- Outgoing cross-module calls (sample):
  - root.exists (line 69)
  - root.iterdir (line 72)
  - entry.is\_dir (line 73)
  - candidates.append (line 74)
  - candidates.sort (line 75)
  - p.stat (line 75)
  - pathlib.Path (line 82)
  - candidate.is\_absolute (line 83)
  - candidate.exists (line 85)
  - candidate.is\_dir (line 85)
  - metadata\_path.exists (line 96)
  - orjson.loads (line 98)
  - metadata\_path.read\_bytes (line 98)
  - csv\_path.exists (line 103)
  - csv\_path.open (line 105)
  - csv.DictReader (line 106)
  - value.strip (line 114)
  - header.lower (line 123)
  - row.keys (line 123)
  - lower\_map.get (line 125)
  - exact.lower (line 125)
  - row.get (line 126)
  - row.get (line 127)
  - row.items (line 128)
  - header.strip (line 131)
  - text.replace (line 156)
  - normalized.lower (line 157)
  - metadata.get (line 169)
  - context.get (line 171)
  - context.get (line 172)
  - context.get (line 173)
  - metadata.get (line 174)
  - candidate.replace (line 180)
  - text.replace (line 184)
  - datetime.datetime.fromisoformat (line 186)
  - parsed.replace (line 189)
  - metadata.get (line 201)
  - metadata.get (line 202)
  - context.get (line 202)
  - context.get (line 202)
  - metadata.get (line 203)
  - context.get (line 203)
  - context.get (line 203)
  - metadata.get (line 204)
  - context.get (line 204)
  - context.get (line 204)
  - metadata.get (line 208)
  - webapp.parser.Context\_Integration.librarian.clean\_for\_json (line 235)
  - records.append (line 238)
  - metadata.get (line 254)
- Inbound references:
  - discover\_dataset\_dirs ← dataset_promotion.py:88
  - resolve\_dataset\_path ← dataset_promotion.py:379
  - \_load\_metadata ← dataset_promotion.py:249
  - \_load\_rows ← dataset_promotion.py:250
  - \_has\_value ← dataset_promotion.py:126
  - \_has\_value ← dataset_promotion.py:129
  - \_has\_value ← location_helpers.py:315
  - \_match\_field ← dataset_promotion.py:215
  - \_match\_field ← dataset_promotion.py:216
  - \_match\_field ← dataset_promotion.py:217
  - \_match\_field ← dataset_promotion.py:218
  - \_coerce\_text ← dataset_promotion.py:202
  - \_coerce\_text ← dataset_promotion.py:202
  - \_coerce\_text ← dataset_promotion.py:202
  - \_coerce\_text ← dataset_promotion.py:203
  - \_coerce\_text ← dataset_promotion.py:203
  - \_coerce\_text ← dataset_promotion.py:203
  - \_coerce\_text ← dataset_promotion.py:204
  - \_coerce\_text ← dataset_promotion.py:204
  - \_coerce\_text ← dataset_promotion.py:204
  - \_coerce\_text ← dataset_promotion.py:227
  - \_coerce\_text ← dataset_promotion.py:228
  - \_coerce\_text ← dataset_promotion.py:230
  - \_coerce\_votes ← dataset_promotion.py:219
  - \_resolve\_election\_date ← dataset_promotion.py:209
  - build\_warehouse\_records ← dataset_promotion.py:251
  - promote\_dataset ← dataset_promotion.py:380
  - \_build\_arg\_parser ← dataset_promotion.py:375
  - \_build\_arg\_parser ← integrity_check_runner.py:69

### health/fine\_tune\_bert\_ner.py {#webapp-parser-health-fine-tune-bert-ner-py}

> BERT/RoBERTa NER Fine-Tuning Module for Election Data Extraction

- Definitions:
  - function: `load\_ner\_data\_from\_db` (line 61)
  - function: `load\_ner\_data\_from\_jsonl` (line 89)
  - function: `tokenize\_and\_align\_labels` (line 124)
  - function: `fine\_tune\_bert\_ner` (line 160)
- Imports:
  - **Standard Library** (6):
    - `import os as os` (line 16)
    - `import sys as sys` (line 17)
    - `from pathlib import Path` (line 18)
    - `from typing import Any` (line 19)
    - `from typing import Dict` (line 19)
    - `from typing import List` (line 19)
  - **Third-party** (10):
    - `import orjson as orjson` (line 21)
    - `from transformers import AutoModelForTokenClassification` (line 23)
    - `from transformers import AutoTokenizer` (line 23)
    - `from transformers import DataCollatorForTokenClassification` (line 23)
    - `from transformers import Trainer` (line 23)
    - `from transformers import TrainingArguments` (line 23)
    - `from webapp.parser.config import LOG_DIR` (line 35)
    - `from webapp.parser.config import MODEL_DIR` (line 35)
    - `from webapp.parser.utils.db_utils import SessionLocal` (line 36)
    - `from webapp.parser.utils.logger_singleton import logger` (line 37)
  - **Local/Project** (1):
    - `from datasets import Dataset` (line 22)
- Task markers:
  - L80 **TODO**: Improve token alignment with actual character offsets (start,
    end)
  - L94 **WARNING**: (f"\[BERT_NER\] No JSONL training data found at
    {jsonl_path}")
  - L112 **TODO**: Improve token alignment (start, end offsets)
  - L117 **WARNING**: (f"\[BERT_NER\] Failed to parse JSONL line: {e}")
  - L165 **WARNING**: ("\[BERT_NER\] No verified data in DB, falling back to
    JSONL")
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 32)
  - LABEL2ID.items (line 58)
  - webapp.parser.utils.db\_utils.SessionLocal (line 63)
  - session.execute (line 64)
  - text.split (line 74)
  - data.append (line 83)
  - webapp.parser.utils.logger\_singleton.logger.info (line 85)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 94)
  - orjson.loads (line 101)
  - example.get (line 102)
  - example.get (line 103)
  - text.split (line 106)
  - ent.get (line 111)
  - data.append (line 115)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 117)
  - webapp.parser.utils.logger\_singleton.logger.info (line 120)
  - tokenized\_inputs.word\_ids (line 136)
  - label\_ids.append (line 142)
  - label\_ids.append (line 144)
  - prev\_label.startswith (line 148)
  - label\_ids.append (line 149)
  - prev\_label.replace (line 149)
  - label\_ids.append (line 151)
  - labels.append (line 154)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 165)
  - webapp.parser.utils.logger\_singleton.logger.error (line 169)
  - datasets.Dataset.from\_dict (line 178)
  - datasets.Dataset.from\_dict (line 182)
  - transformers.AutoTokenizer.from\_pretrained (line 189)
  - transformers.AutoModelForTokenClassification.from\_pretrained (line 190)
  - train\_dataset.map (line 198)
  - val\_dataset.map (line 202)
  - transformers.DataCollatorForTokenClassification (line 208)
  - transformers.TrainingArguments (line 212)
  - transformers.Trainer (line 230)
  - webapp.parser.utils.logger\_singleton.logger.info (line 240)
  - trainer.train (line 241)
  - trainer.save\_model (line 245)
  - webapp.parser.utils.logger\_singleton.logger.info (line 246)
  - trainer.evaluate (line 249)
  - webapp.parser.utils.logger\_singleton.logger.info (line 250)
- Inbound references:
  - load\_ner\_data\_from\_db ← fine_tune_bert_ner.py:163
  - tokenize\_and\_align\_labels ← fine_tune_bert_ner.py:199
  - tokenize\_and\_align\_labels ← fine_tune_bert_ner.py:203
  - fine\_tune\_bert\_ner ← fine_tune_bert_ner.py:256
  - fine\_tune\_bert\_ner ← retrain_table_structure_models.py:973

### health/health\_config.py {#webapp-parser-health-health-config-py}

> health_config.py

- Imports:
  - **Standard Library** (1):
    - `from pathlib import Path` (line 7)
  - **Local/Project** (3):
    - `from config import LOG_DIR` (line 9)
    - `from config import MODEL_DIR` (line 9)
    - `from config import PROJECT_ROOT` (line 9)
- Task markers:
  - L110 **WARN**: 0.45 ≤ suspicion &lt; 0.72  (middle third, 45–72%) →
    confirm/verify
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 28)
  - pathlib.Path (line 29)
  - pathlib.Path (line 74)

### health/health\_router.py {#webapp-parser-health-health-router-py}

- Definitions:
  - class: `LocalLearningEngine` (line 70)
  - function: `get\_learning\_engine` (line 127)
  - function: `register\_orchestration\_plugin` (line 136)
  - function: `run\_orchestration\_plugins` (line 139)
  - function: `preclean\_json\_logs` (line 148)
  - class: `BotPipeline` (line 238)
- Imports:
  - **Standard Library** (8):
    - `import os as os` (line 3)
    - `import re as re` (line 4)
    - `import subprocess as subprocess` (line 5)
    - `import sys as sys` (line 6)
    - `import time as time` (line 7)
    - `from datetime import datetime` (line 8)
    - `from datetime import timezone` (line 8)
    - `from pathlib import Path` (line 9)
  - **Third-party** (2):
    - `import orjson as orjson` (line 11)
    - `from sqlalchemy import inspect` (line 12)
  - **Local/Project** (34):
    - `import errno as errno` (line 1)
    - `import glob as glob` (line 2)
    - `from config import BATCH_MODE` (line 14)
    - `from config import CACHE_DIR` (line 14)
    - `from config import CACHE_EXPIRE_DAYS` (line 14)
    - `from config import CONTEXT_PATH` (line 14)
    - `from config import COOLDOWN` (line 14)
    - `from config import CORRECTION_MODE` (line 14)
    - `from config import DB_PATH` (line 14)
    - `from config import DRY_RUN` (line 14)
    - `from config import ENABLE_ENHANCED` (line 14)
    - `from config import EXPORT_AUDIT_LOG` (line 14)
    - `from config import FAST_MODE` (line 14)
    - `from config import FIELDS` (line 14)
    - `from config import FILTER_CONTEXT_KEY` (line 14)
    - `from config import FILTER_VALUE` (line 14)
    - `from config import FLUSH_CACHE` (line 14)
    - `from config import INTEGRITY_CHECK` (line 14)
    - `from config import LOG_DIR` (line 14)
    - `from config import MAX_RETRIES` (line 14)
    - `from config import MODEL_DIR` (line 14)
    - `from config import NO_COORDINATOR` (line 14)
    - `from config import NO_ORGANIZER` (line 14)
    - `from config import PROJECT_ROOT` (line 14)
    - `from config import REST_API` (line 14)
    - `from config import SELF_HEAL` (line 14)
    - `from config import UPDATE_DB` (line 14)
    - `from Context_Integration.librarian import load_context_library` (line 41)
    - `from utils.db_utils import get_engine` (line 42)
    - `from utils.logger_singleton import console` (line 43)
    - `from utils.logger_singleton import logger` (line 43)
    - `from utils.models import Base` (line 44)
    - `from integrity_monitor import get_integrity_monitor` (line 45)
    - `from navigation_feedback_ingest import ingest_navigation_feedback` (line
      46)
- Task markers:
  - L95 **WARNING**: (f"\[LocalLearning\] Failed to record training signal:
    {e}")
  - L357 **WARNING**: (f"\[health_router\] manual_correction failed (attempt
    {attempt}): {result.stderr}")
  - L441 **WARNING**: ("\[SELF-HEAL\] Misalignments found. Launching
    manual_correction...")
  - L443 **WARNING**: (f"\[SELF-HEAL\] Sleeping {cooldown}s before
    rescanning...")
  - L445 **WARNING**: ("\[SELF-HEAL\] Max retries reached. Some misalignments
    may remain.")
  - L480 **WARNING**: (f"\[PIPELINE\] Could not fix corrupted JSON files: {e}")
  - L497 **WARNING**: ("\[PIPELINE\] Misaligned NER examples found. Self-heal
    loop will be handled by scan_misaligned_ner.")
  - L499 **WARNING**: ("\[PIPELINE\] scan_misaligned_ner failed or file missing.
    Proceeding with caution.")
  - L525 **WARNING**: ("\[PIPELINE\] Model retraining failed.")
- Outgoing cross-module calls (sample):
  - integrity\_monitor.get\_integrity\_monitor (line 74)
  - datetime.datetime.now (line 81)
  - session\_context.get (line 82)
  - session\_context.get (line 83)
  - session\_context.get (line 84)
  - session\_context.get (line 85)
  - f.write (line 93)
  - orjson.dumps (line 93)
  - utils.logger\_singleton.logger.warning (line 95)
  - session\_context.get (line 100)
  - session\_context.get (line 101)
  - Context\_Integration.librarian.load\_context\_library (line 105)
  - library.get (line 106)
  - c.get (line 111)
  - c.get (line 112)
  - m.get (line 116)
  - utils.logger\_singleton.logger.debug (line 120)
  - ORCHESTRATION\_PLUGINS.append (line 137)
  - suggestions.extend (line 143)
  - utils.logger\_singleton.logger.error (line 145)
  - glob.glob (line 157)
  - char.strip (line 168)
  - char.strip (line 169)
  - json.load (line 181)
  - out.write (line 187)
  - line.strip (line 195)
  - corrupt\_lines.append (line 199)
  - json.loads (line 203)
  - valid\_lines.append (line 204)
  - re.sub (line 208)
  - re.sub (line 209)
  - re.sub (line 210)
  - fixed.replace (line 211)
  - json.loads (line 213)
  - valid\_lines.append (line 214)
  - corrupt\_lines.append (line 216)
  - out.write (line 220)
  - out.write (line 226)
  - utils.db\_utils.get\_engine (line 248)
  - sqlalchemy.inspect (line 250)
  - inspector.get\_table\_names (line 251)
  - table.add\_column (line 254)
  - table.add\_row (line 256)
  - utils.logger\_singleton.console.table (line 257)
  - utils.logger\_singleton.logger.info (line 258)
  - utils.logger\_singleton.logger.info (line 259)
  - utils.logger\_singleton.logger.error (line 263)
  - args.append (line 270)
  - args.append (line 272)
  - args.append (line 274)
- Inbound references:
  - LocalLearningEngine ← health_router.py:131
  - get\_learning\_engine ← web_pipeline.py:716
  - get\_learning\_engine ← health_router.py:639
  - run\_orchestration\_plugins ← health_router.py:629
  - preclean\_json\_logs ← health_router.py:461
  - BotPipeline ← health_router.py:687

### health/integrity\_check\_runner.py {#webapp-parser-health-integrity-check-runner-py}

- Definitions:
  - function: `load\_contests` (line 13)
  - function: `run\_integrity\_summary` (line 23)
  - function: `\_build\_arg\_parser` (line 45)
  - function: `main` (line 68)
- Imports:
  - **Standard Library** (3):
    - `import argparse as argparse` (line 3)
    - `from pathlib import Path` (line 4)
    - `from typing import Any` (line 5)
  - **Third-party** (4):
    - `from webapp.parser.config import CONTEXT_LIBRARY_PATH` (line 7)
    - `from webapp.parser.Context_Integration.Integrity_check import
      print_integrity_summary` (line 8)
    - `from webapp.parser.Context_Integration.librarian import
      load_context_library` (line 9)
    - `from webapp.parser.utils.logger_singleton import logger` (line 10)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Task markers:
  - L18 **WARNING**: ("\[INTEGRITY\] Context library at %s is missing contest
    data", context_path)
- Outgoing cross-module calls (sample):
  - webapp.parser.Context\_Integration.librarian.load\_context\_library (line
    15)
  - library.get (line 16)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 18)
  - pathlib.Path (line 29)
  - webapp.parser.utils.logger\_singleton.logger.info (line 34)
  - webapp.parser.Context\_Integration.Integrity\_check.print\_integrity\_summary
    (line 38)
  - argparse.ArgumentParser (line 46)
  - parser.add\_argument (line 49)
  - parser.add\_argument (line 54)
  - parser.add\_argument (line 60)
  - parser.parse\_args (line 70)
  - webapp.parser.utils.logger\_singleton.logger.error (line 78)
- Inbound references:
  - load\_contests ← integrity_check_runner.py:30
  - run\_integrity\_summary ← integrity_check_runner.py:72

### health/integrity\_monitor.py {#webapp-parser-health-integrity-monitor-py}

> integrity_monitor.py

- Definitions:
  - class: `IntegrityNeuralNetwork` (line 59)
  - class: `HuggingFaceNLPAnalyzer` (line 96)
  - class: `IntegrityMonitor` (line 203)
  - function: `get\_integrity\_monitor` (line 542)
- Imports:
  - **Standard Library** (11):
    - `import asyncio as asyncio` (line 14)
    - `import hashlib as hashlib` (line 15)
    - `import time as time` (line 16)
    - `from datetime import datetime` (line 17)
    - `from datetime import timezone` (line 17)
    - `from pathlib import Path` (line 18)
    - `from typing import Any` (line 19)
    - `from typing import Dict` (line 19)
    - `from typing import List` (line 19)
    - `from typing import Optional` (line 19)
    - `from typing import Tuple` (line 19)
  - **Third-party** (1):
    - `import orjson as orjson` (line 21)
  - **Local/Project** (7):
    - `from __future__ import annotations` (line 12)
    - `from config import LOG_DIR` (line 40)
    - `from config import OUTPUT_DIR` (line 40)
    - `from config import PROJECT_ROOT` (line 40)
    - `from Context_Integration.librarian import atomic_write_json` (line 41)
    - `from Context_Integration.librarian import clean_for_json` (line 41)
    - `from utils.logger_singleton import logger` (line 42)
- Task markers:
  - L264 **WARNING**: (f"\[IntegrityMonitor\] Hash mismatch for
    {file_path.name}: expected {expected_hash}, got {file_hash}")
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 45)
  - DOWNLOAD\_CACHE\_DIR.mkdir (line 46)
  - pathlib.Path (line 51)
  - asyncio.Lock (line 56)
  - nn.Linear (line 69)
  - nn.Linear (line 70)
  - nn.Linear (line 71)
  - nn.Dropout (line 72)
  - F.relu (line 75)
  - self.fc1 (line 75)
  - self.dropout (line 76)
  - F.relu (line 77)
  - self.fc2 (line 77)
  - self.dropout (line 78)
  - torch.sigmoid (line 79)
  - self.fc3 (line 79)
  - torch.no\_grad (line 88)
  - self.eval (line 89)
  - self.forward (line 90)
  - AutoTokenizer.from\_pretrained (line 116)
  - AutoModel.from\_pretrained (line 117)
  - utils.logger\_singleton.logger.info (line 127)
  - utils.logger\_singleton.logger.error (line 129)
  - self.\_lazy\_init (line 141)
  - session\_context.get (line 154)
  - session\_context.get (line 155)
  - session\_context.get (line 156)
  - session\_context.get (line 157)
  - context\_text.strip (line 161)
  - self.ner\_pipeline (line 161)
  - context\_text.lower (line 169)
  - risk\_factors.append (line 170)
  - session\_context.get (line 173)
  - risk\_factors.append (line 174)
  - session\_context.get (line 175)
  - risk\_factors.append (line 176)
  - utils.logger\_singleton.logger.error (line 194)
  - pathlib.Path (line 210)
  - utils.logger\_singleton.logger.info (line 218)
  - utils.logger\_singleton.logger.error (line 220)
  - asyncio.get\_event\_loop (line 224)
  - loop.run\_in\_executor (line 225)
  - hashlib.sha256 (line 229)
  - f.read (line 232)
  - hasher.update (line 233)
  - hasher.hexdigest (line 234)
  - utils.logger\_singleton.logger.error (line 236)
  - file\_path.exists (line 255)
  - self.compute\_file\_hash (line 258)
  - file\_path.stat (line 259)
- Inbound references:
  - IntegrityNeuralNetwork ← integrity_monitor.py:217
  - HuggingFaceNLPAnalyzer ← integrity_monitor.py:211
  - IntegrityMonitor ← integrity_monitor.py:546

### health/log\_cache\_cleaner\_bot.py {#webapp-parser-health-log-cache-cleaner-bot-py}

> log_cache_cleaner_bot.py

- Definitions:
  - function: `is\_jsonl\_file` (line 44)
  - function: `is\_json\_file` (line 47)
  - function: `is\_html\_file` (line 50)
  - function: `safe\_path` (line 53)
  - function: `log\_empty\_entry` (line 61)
  - function: `clean\_jsonl` (line 72)
  - function: `clean\_json` (line 175)
  - function: `clean\_html` (line 295)
  - function: `human\_size` (line 388)
  - function: `clean\_dir` (line 395)
  - function: `run\_db\_maintenance` (line 441)
  - function: `run\_log\_cache\_cleaner` (line 486)
  - function: `schedule\_log\_cache\_cleaner` (line 520)
  - function: `main` (line 530)
- Imports:
  - **Standard Library** (5):
    - `import argparse as argparse` (line 24)
    - `import os as os` (line 25)
    - `import threading as threading` (line 26)
    - `import time as time` (line 27)
    - `from pathlib import Path` (line 28)
  - **Third-party** (3):
    - `import orjson as orjson` (line 30)
    - `from sqlalchemy import text` (line 31)
    - `from sqlalchemy.exc import SQLAlchemyError` (line 32)
  - **Local/Project** (6):
    - `from config import CACHE_DIR` (line 34)
    - `from config import CONTEXT_LIBRARY_DIR` (line 34)
    - `from config import LOG_DIR` (line 34)
    - `from utils.db_utils import get_engine` (line 35)
    - `from utils.logger_singleton import logger` (line 36)
    - `from context_migration import migrate_all` (line 37)
- Task markers:
  - L151 **WARNING**: (f"Skipping non-dict entry in spacy_ner_train_data.jsonl:
    {entry}")
  - L460 **WARNING**: ("\[DB\]\[WARNING\] No user tables found in schema
    'public'.")
  - L503 **WARNING**: ("\[CLEAN\]\[WARNING\] The following files are still too
    large after cleaning:")
  - L507 **WARNING**: ("\[MISALIGNED\] Consider cleaning or pattern-excluding
    these from your training data:")
- Outgoing cross-module calls (sample):
  - fname.endswith (line 45)
  - fname.endswith (line 48)
  - fname.endswith (line 51)
  - path.startswith (line 57)
  - f.write (line 70)
  - orjson.dumps (line 70)
  - os.remove (line 104)
  - shutil.copy2 (line 107)
  - line.strip (line 109)
  - orjson.loads (line 115)
  - malformed\_examples.append (line 119)
  - null\_examples.append (line 124)
  - empty\_examples.append (line 129)
  - nondict\_examples.append (line 134)
  - missing\_required\_examples.append (line 140)
  - orjson.dumps (line 142)
  - seen.add (line 144)
  - entries.append (line 145)
  - misaligned.append (line 147)
  - utils.logger\_singleton.logger.warning (line 151)
  - f.write (line 153)
  - orjson.dumps (line 153)
  - error\_parts.append (line 156)
  - error\_parts.append (line 158)
  - error\_parts.append (line 160)
  - error\_parts.append (line 162)
  - error\_parts.append (line 164)
  - f.write (line 197)
  - orjson.dumps (line 197)
  - os.remove (line 203)
  - shutil.copy2 (line 206)
  - orjson.loads (line 209)
  - f.read (line 209)
  - wf.write (line 212)
  - orjson.dumps (line 212)
  - data.items (line 219)
  - empty\_keys.append (line 225)
  - seen.add (line 232)
  - f.write (line 236)
  - orjson.dumps (line 236)
  - utils.logger\_singleton.logger.info (line 238)
  - utils.logger\_singleton.logger.info (line 244)
  - empty\_indices.append (line 259)
  - orjson.dumps (line 265)
  - seen.add (line 267)
  - deduped.append (line 268)
  - f.write (line 274)
  - orjson.dumps (line 274)
  - utils.logger\_singleton.logger.info (line 276)
  - utils.logger\_singleton.logger.info (line 282)
- Inbound references:
  - is\_jsonl\_file ← log_cache_cleaner_bot.py:416
  - is\_json\_file ← log_cache_cleaner_bot.py:418
  - is\_html\_file ← log_cache_cleaner_bot.py:420
  - log\_empty\_entry ← log_cache_cleaner_bot.py:226
  - log\_empty\_entry ← log_cache_cleaner_bot.py:260
  - clean\_jsonl ← log_cache_cleaner_bot.py:417
  - clean\_json ← log_cache_cleaner_bot.py:419
  - clean\_html ← log_cache_cleaner_bot.py:421
  - human\_size ← log_cache_cleaner_bot.py:437
  - clean\_dir ← log_cache_cleaner_bot.py:490
  - clean\_dir ← log_cache_cleaner_bot.py:492
  - clean\_dir ← log_cache_cleaner_bot.py:494
  - run\_db\_maintenance ← log_cache_cleaner_bot.py:515
  - run\_log\_cache\_cleaner ← log_cache_cleaner_bot.py:523
  - run\_log\_cache\_cleaner ← log_cache_cleaner_bot.py:557
  - schedule\_log\_cache\_cleaner ← log_cache_cleaner_bot.py:542

### health/manual\_correction\_bot.py {#webapp-parser-health-manual-correction-bot-py}

> manual_correction.py

- Definitions:
  - function: `safe\_path` (line 70)
  - function: `load\_cache` (line 99)
  - function: `close\_cache` (line 114)
  - function: `write\_audit\_log` (line 118)
  - function: `process\_logs\_with\_cache` (line 133)
  - function: `process\_and\_sync` (line 145)
  - function: `discover\_field\_types\_from\_logs` (line 189)
  - function: `atomic\_write\_json` (line 222)
  - function: `ml\_score\_entry` (line 295)
  - function: `ml\_suggest\_field` (line 318)
  - function: `load\_jsonl` (line 337)
  - function: `check\_and\_fix\_json\_files` (line 353)
  - function: `find\_log\_files` (line 515)
  - function: `load\_jsonl\_incremental` (line 582)
  - function: `save\_jsonl` (line 600)
  - function: `deduplicate\_entries` (line 613)
  - function: `entry\_key` (line 627)
  - function: `aggregate\_successful\_field\_entries` (line 638)
  - function: `feedback\_loop` (line 679)
  - function: `trim\_log\_file` (line 747)
  - function: `update\_context\_with\_new\_entries` (line 754)
  - function: `validate\_context\_schema` (line 771)
  - function: `extract\_year` (line 796)
  - function: `extract\_state` (line 810)
  - function: `extract\_county` (line 829)
  - function: `extract\_type` (line 851)
  - function: `autofix\_contest\_fields` (line 871)
  - function: `suggest\_fields\_with\_models` (line 917)
  - function: `prompt\_for\_missing\_fields` (line 997)
  - function: `highlight\_anomalies` (line 1020)
  - function: `update\_database\_with\_context` (line 1068)
  - function: `export\_correction\_session` (line 1087)
  - function: `import\_correction\_session` (line 1108)
  - function: `field\_matches\_log` (line 1116)
  - function: `ensure\_context\_library` (line 1133)
  - function: `process\_auto\_mode` (line 1161)
  - function: `main` (line 1238)
- Imports:
  - **Standard Library** (12):
    - `import argparse as argparse` (line 16)
    - `import os as os` (line 18)
    - `import re as re` (line 19)
    - `import shutil as shutil` (line 21)
    - `import subprocess as subprocess` (line 22)
    - `import sys as sys` (line 23)
    - `import time as time` (line 24)
    - `from collections import Counter` (line 25)
    - `from collections import defaultdict` (line 25)
    - `from datetime import datetime` (line 26)
    - `from datetime import timedelta` (line 26)
    - `from pathlib import Path` (line 27)
  - **Third-party** (1):
    - `import orjson as orjson` (line 29)
  - **Local/Project** (19):
    - `import importlib as importlib` (line 17)
    - `import shelve as shelve` (line 20)
    - `from config import CACHE_DIR` (line 31)
    - `from config import CONTEXT_LIBRARY_DIR` (line 31)
    - `from config import CONTEXT_LIBRARY_PATH` (line 31)
    - `from config import LOG_DIR` (line 31)
    - `from config import PROJECT_ROOT` (line 31)
    - `from config import USER_NAME` (line 31)
    - `from Context_Integration.context_coordinator import ContextCoordinator`
      (line 39)
    - `from Context_Integration.librarian import DEFAULT_STRUCTURE` (line 40)
    - `from Context_Integration.librarian import SCHEMA_VERSION` (line 40)
    - `from Context_Integration.librarian import get_state_abbr` (line 40)
    - `from Context_Integration.librarian import load_context_library` (line 40)
    - `from Context_Integration.librarian import lookup_county` (line 40)
    - `from Context_Integration.librarian import lookup_state` (line 40)
    - `from Context_Integration.librarian import update_context_library` (line
      40)
    - `from utils.logger_singleton import logger` (line 49)
    - `from utils.misc_utils import file_hash` (line 50)
    - `from utils.model_registry import ModelRegistry` (line 51)
- Task markers:
  - L307 **WARNING**: (f"Coordinator ML scoring failed: {e}")
  - L328 **WARNING**: (f"Coordinator field suggestion failed: {e}")
  - L341 **WARNING**: (f"Log file not found: {path}")
  - L350 **WARNING**: (f"\[CORRUPT\] {path} line {i}: {e}")
  - L380 **WARNING**: (f"\[SECURITY\] Skipping invalid directory: {directory} -
    {e}")
  - L394 **WARNING**: (f"\[SECURITY\] Skipping file outside allowed directories:
    {file} - {e}")
  - L400 **WARNING**: (f"\[SKIP\] File not found: {file}")
  - L404 **WARNING**: (f"\[SKIP\] File too large: {file}")
  - L429 **WARNING**: (f"\[CORRUPT-LINE\] {file} line {i+1}: {line\[:80\]}...
    ({e})")
  - L443 **WARNING**: (f"\[CORRUPT\] {len(corrupt_items)} lines saved to
    {corrupt_path}")
  - L448 **WARNING**: (f"\[FIXED\] All lines invalid, recreated empty .jsonl
    file: {file}")
  - L462 **WARNING**: (f"\[CORRUPT\] {file}: {e}")
  - L476 **WARNING**: (f"\[CORRUPT\] Corrupt JSON saved to {corrupt_path}")
  - L482 **WARNING**: (f"\[FIXED\] All content invalid, recreated minimal valid
    JSON in {file}")
  - L487 **WARNING**: (f"\[CORRUPT\] {file}: {e}")
  - L501 **WARNING**: (f"\[QUARANTINED\] {file} -&gt; {dest_path}")
  - L505 **WARNING**: (f"\[DELETED\] {file}")
  - L508 **WARNING**: (f"\[SKIP-DELETE\] File already missing: {file}")
  - L543 **WARNING**: (f"\[SECURITY\] Skipping invalid directory: {d} - {e}")
  - L561 **WARNING**: (f"\[SECURITY\] Skipping file outside allowed directories:
    {f} - {e}")
  - L570 **WARNING**: (f"\[FIND-LOGS\] Skipped {d}: {e}")
  - L595 **WARNING**: (f"\[CORRUPT\] {path} line {line_num}: {e}")
  - L734 **WARNING**: (f"Invalid JSON, skipping edit: {e}")
  - L777 **WARNING**: (
  - L1031 **WARNING**: (
  - L1137 **WARN**: if schema version mismatches.
  - L1158 **WARNING**: (f"Schema version mismatch: found
    {context_lib.get('schema_version')}, expected {SCHEMA_VERSION}. Consider
    migrating.")
  - L1182 **WARNING**: (f"\[SECURITY\] Skipping invalid log file: {log_file} -
    {e}")
  - L1214 **WARNING**: (f"\[AUTO\] Could not delete log file {log_file}: {e}")
  - L1337 **WARNING**: (f"\[SKIP\] Could not load {log_file}: {e}")
  - L1353 **WARNING**: ("No log files matched any of the specified fields. Will
    attempt to process all log files for all fields.")
  - L1434 **WARNING**: (f"\[SECURITY\] Cannot delete file outside allowed
    directories: {log_file} - {e}")
  - L1436 **WARNING**: (f"Could not delete log file {log_file}: {e}")
  - L1456 **WARNING**: ("\[WARNING\] No entries were processed. Check your log
    file naming, field configuration, or use --dry-run for debugging.")
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 54)
  - pathlib.Path (line 55)
  - pathlib.Path (line 56)
  - pathlib.Path (line 57)
  - pathlib.Path (line 58)
  - config.LOG\_DIR.mkdir (line 64)
  - pathlib.Path (line 87)
  - pathlib.Path (line 89)
  - path.relative\_to (line 92)
  - shelve.open (line 102)
  - datetime.datetime.now (line 104)
  - cache.items (line 106)
  - v.get (line 107)
  - datetime.datetime.fromisoformat (line 108)
  - datetime.timedelta (line 108)
  - expired.append (line 109)
  - cache.close (line 115)
  - datetime.datetime.now (line 122)
  - orjson.dumps (line 124)
  - f.write (line 131)
  - orjson.dumps (line 131)
  - orjson.dumps (line 137)
  - cache.sync (line 143)
  - orjson.dumps (line 150)
  - batch.append (line 154)
  - datetime.datetime.now (line 155)
  - batch.clear (line 160)
  - cache.sync (line 166)
  - orjson.loads (line 206)
  - field\_types.add (line 210)
  - entry.keys (line 212)
  - field\_types.add (line 214)
  - path.with\_suffix (line 233)
  - path.with\_suffix (line 234)
  - tmp\_path.exists (line 241)
  - tmp\_path.unlink (line 243)
  - backup\_path.exists (line 248)
  - backup\_path.unlink (line 250)
  - tf.write (line 256)
  - orjson.dumps (line 256)
  - path.exists (line 259)
  - shutil.copy2 (line 260)
  - shutil.move (line 265)
  - os.remove (line 270)
  - time.sleep (line 273)
  - tmp\_path.exists (line 278)
  - tmp\_path.unlink (line 280)
  - spacy.load (line 287)
  - Context\_Integration.context\_coordinator.ContextCoordinator (line 300)
  - entry.get (line 301)
- Inbound references:
  - load\_cache ← manual_correction_bot.py:1267
  - load\_cache ← manual_correction_bot.py:1273
  - close\_cache ← manual_correction_bot.py:1269
  - write\_audit\_log ← manual_correction_bot.py:1204
  - discover\_field\_types\_from\_logs ← manual_correction_bot.py:1323
  - ml\_score\_entry ← manual_correction_bot.py:721
  - ml\_suggest\_field ← manual_correction_bot.py:722
  - load\_jsonl ← manual_correction_bot.py:135
  - load\_jsonl ← manual_correction_bot.py:148
  - load\_jsonl ← manual_correction_bot.py:642
  - load\_jsonl ← manual_correction_bot.py:749
  - load\_jsonl ← manual_correction_bot.py:1335
  - check\_and\_fix\_json\_files ← manual_correction_bot.py:1329
  - find\_log\_files ← manual_correction_bot.py:1317
  - save\_jsonl ← manual_correction_bot.py:751
  - deduplicate\_entries ← manual_correction_bot.py:644
  - deduplicate\_entries ← manual_correction_bot.py:750
  - entry\_key ← manual_correction_bot.py:713
  - entry\_key ← manual_correction_bot.py:713
  - aggregate\_successful\_field\_entries ← manual_correction_bot.py:1188
  - aggregate\_successful\_field\_entries ← manual_correction_bot.py:1365
  - feedback\_loop ← manual_correction_bot.py:1403
  - update\_context\_with\_new\_entries ← manual_correction_bot.py:157
  - update\_context\_with\_new\_entries ← manual_correction_bot.py:163
  - update\_context\_with\_new\_entries ← manual_correction_bot.py:742
  - update\_context\_with\_new\_entries ← manual_correction_bot.py:1226
  - update\_context\_with\_new\_entries ← manual_correction_bot.py:1232
  - update\_context\_with\_new\_entries ← manual_correction_bot.py:1395
  - validate\_context\_schema ← manual_correction_bot.py:768
  - extract\_year ← manual_correction_bot.py:878
  - extract\_year ← manual_correction_bot.py:879
  - extract\_state ← manual_correction_bot.py:888
  - extract\_state ← manual_correction_bot.py:889
  - extract\_county ← manual_correction_bot.py:898
  - extract\_county ← manual_correction_bot.py:899
  - extract\_type ← manual_correction_bot.py:908
  - extract\_type ← manual_correction_bot.py:909
  - suggest\_fields\_with\_models ← manual_correction_bot.py:1051
  - prompt\_for\_missing\_fields ← manual_correction_bot.py:1054
  - highlight\_anomalies ← manual_correction_bot.py:1419
  - update\_database\_with\_context ← manual_correction_bot.py:159
  - update\_database\_with\_context ← manual_correction_bot.py:165
  - update\_database\_with\_context ← manual_correction_bot.py:1426
  - field\_matches\_log ← manual_correction_bot.py:1341
  - ensure\_context\_library ← manual_correction_bot.py:1311
  - process\_auto\_mode ← manual_correction_bot.py:1386

### health/navigation\_feedback\_ingest.py {#webapp-parser-health-navigation-feedback-ingest-py}

- Definitions:
  - function: `ingest\_navigation\_feedback` (line 24)
  - function: `\_read\_offset` (line 66)
  - function: `\_write\_offset` (line 75)
  - function: `\_format\_entry` (line 82)
- Imports:
  - **Standard Library** (3):
    - `from pathlib import Path` (line 3)
    - `from typing import Any` (line 4)
    - `from typing import Dict` (line 4)
  - **Third-party** (1):
    - `import orjson as orjson` (line 6)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 31)
  - log\_path.exists (line 33)
  - log\_path.stat (line 33)
  - log\_path.stat (line 40)
  - log\_path.open (line 45)
  - output\_path.open (line 45)
  - source.seek (line 46)
  - raw.strip (line 48)
  - orjson.loads (line 52)
  - sink.write (line 58)
  - orjson.dumps (line 58)
  - source.tell (line 60)
  - path.exists (line 67)
  - path.read\_text (line 70)
  - path.write\_text (line 77)
  - entry.get (line 86)
  - entry.get (line 87)
  - entry.get (line 88)
  - entry.get (line 89)
  - context\_after.get (line 91)
  - context\_before.get (line 91)
  - context\_after.get (line 92)
  - context\_before.get (line 92)
  - entry.get (line 93)
  - metadata.get (line 93)
  - entry.get (line 101)
  - metadata.get (line 103)
  - metadata.get (line 103)
  - context\_after.get (line 103)
  - context\_before.get (line 103)
  - metadata.get (line 106)
  - metadata.get (line 106)
  - entry.get (line 106)
  - entry.get (line 114)
- Inbound references:
  - \_read\_offset ← navigation_feedback_ingest.py:39
  - \_write\_offset ← navigation_feedback_ingest.py:62
  - \_format\_entry ← navigation_feedback_ingest.py:55

### health/promotion\_helpers.py {#webapp-parser-health-promotion-helpers-py}

> Helper functions for dataset promotion with verification gating.

- Definitions:
  - function: `check\_exact\_duplicate` (line 8)
  - function: `get\_url\_verification\_tier` (line 33)
- Imports:
  - **Third-party** (1):
    - `from webapp.parser.utils.logger_singleton import logger` (line 5)
- Task markers:
  - L54 **WARNING**: (f"\[URL_TIER\] Failed to compute trust score: {exc}")
- Outgoing cross-module calls (sample):
  - session.query (line 17)
  - query.filter (line 27)
  - query.filter (line 29)
  - query.first (line 30)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 54)

### health/quarantine\_queue.py {#webapp-parser-health-quarantine-queue-py}

> Quarantine Queue: Transparent URL quarantine workflow with audit trails.

- Definitions:
  - class: `QuarantineReason` (line 35)
  - class: `ReviewStatus` (line 75)
  - class: `DataCollectionNotice` (line 87)
  - class: `QuarantineEntry` (line 99)
  - class: `QuarantineQueue` (line 179)
  - function: `get\_quarantine\_queue` (line 442)
- Imports:
  - **Standard Library** (15):
    - `import hashlib as hashlib` (line 20)
    - `import json as json` (line 21)
    - `import threading as threading` (line 22)
    - `import time as time` (line 23)
    - `from dataclasses import asdict` (line 24)
    - `from dataclasses import dataclass` (line 24)
    - `from dataclasses import field` (line 24)
    - `from datetime import datetime` (line 25)
    - `from datetime import timezone` (line 25)
    - `from enum import Enum` (line 26)
    - `from pathlib import Path` (line 27)
    - `from typing import Any` (line 28)
    - `from typing import Dict` (line 28)
    - `from typing import List` (line 28)
    - `from typing import Optional` (line 28)
  - **Local/Project** (3):
    - `from __future__ import annotations` (line 18)
    - `from config import LOG_DIR` (line 30)
    - `from utils.logger_singleton import logger` (line 31)
- Task markers:
  - L293 **WARNING**: ({
  - L294 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - explanations.get (line 58)
  - impacts.get (line 72)
  - dataclasses.asdict (line 95)
  - dataclasses.field (line 117)
  - dataclasses.field (line 121)
  - dataclasses.field (line 125)
  - datetime.datetime.now (line 152)
  - dataclasses.asdict (line 163)
  - json.dumps (line 164)
  - json.loads (line 169)
  - data.get (line 172)
  - pathlib.Path (line 189)
  - threading.RLock (line 194)
  - hashlib.sha256 (line 223)
  - time.time (line 224)
  - datetime.datetime.now (line 245)
  - f.write (line 258)
  - entry.to\_json (line 258)
  - utils.logger\_singleton.logger.info (line 260)
  - line.strip (line 283)
  - QuarantineEntry.from\_json (line 287)
  - entries.append (line 289)
  - utils.logger\_singleton.logger.warning (line 293)
  - utils.logger\_singleton.logger.error (line 299)
  - line.strip (line 314)
  - QuarantineEntry.from\_json (line 318)
  - line.strip (line 336)
  - QuarantineEntry.from\_json (line 340)
  - line.strip (line 381)
  - QuarantineEntry.from\_json (line 384)
  - entry.add\_review (line 386)
  - entries.append (line 393)
  - f.write (line 403)
  - approved\_entry.to\_json (line 403)
  - f.write (line 409)
  - entry.to\_json (line 409)
  - utils.logger\_singleton.logger.info (line 411)
  - self.get\_pending (line 425)
  - pending\_by\_reason.get (line 429)
- Inbound references:
  - QuarantineReason ← quarantine_queue.py:131
  - QuarantineReason ← quarantine_queue.py:139
  - DataCollectionNotice ← html_election_parser.py:1668
  - DataCollectionNotice ← html_election_parser.py:1674
  - DataCollectionNotice ← quarantine_queue.py:171
  - DataCollectionNotice ← quarantine_queue.py:230
  - DataCollectionNotice ← quarantine_queue.py:235
  - QuarantineEntry ← quarantine_queue.py:174
  - QuarantineEntry ← quarantine_queue.py:242
  - QuarantineQueue ← quarantine_queue.py:446
  - get\_quarantine\_queue ← html_election_parser.py:1565
  - get\_quarantine\_queue ← html_election_parser.py:1665

### health/retrain\_table\_structure\_models.py {#webapp-parser-health-retrain-table-structure-models-py}

- Definitions:
  - class: `NERPipeProtocol` (line 86)
  - class: `MakeDocProtocol` (line 89)
  - function: `normalize\_entity` (line 98)
  - function: `normalize\_entity\_list` (line 103)
  - function: `update\_advanced\_entities` (line 107)
  - function: `is\_misaligned\_text` (line 152)
  - function: `clean\_misaligned\_ner\_jsonl` (line 158)
  - function: `append\_training\_data` (line 218)
  - function: `save\_training\_data\_jsonl` (line 244)
  - function: `cluster\_container\_patterns` (line 257)
  - function: `auto\_label\_header` (line 297)
  - function: `extract\_candidates\_from\_context` (line 319)
  - function: `entity\_frequency\_analysis` (line 327)
  - function: `update\_db\_with\_new\_entities` (line 334)
  - function: `load\_spacy\_ner\_examples` (line 350)
  - function: `remove\_overlapping\_entities` (line 366)
  - function: `validate\_training\_data` (line 388)
  - function: `retrain\_spacy\_ner\_advanced` (line 411)
  - function: `get\_all\_confirmed\_structures` (line 648)
  - function: `run\_manual\_correction` (line 670)
  - function: `retrain\_sentence\_transformer` (line 689)
  - function: `segment\_hash` (line 786)
  - function: `load\_cached\_segment\_hashes` (line 796)
  - function: `scan\_in\_memory\_ner\_examples` (line 800)
  - function: `ensure\_table\_structures\_exists` (line 817)
  - function: `main` (line 850)
- Imports:
  - **Standard Library** (18):
    - `import copy as copy` (line 1)
    - `import datetime as datetime` (line 2)
    - `import hashlib as hashlib` (line 5)
    - `import os as os` (line 6)
    - `import random as random` (line 7)
    - `import re as re` (line 8)
    - `import shutil as shutil` (line 9)
    - `import subprocess as subprocess` (line 10)
    - `import sys as sys` (line 11)
    - `from collections import Counter` (line 12)
    - `from typing import Any` (line 15)
    - `from typing import Dict` (line 15)
    - `from typing import List` (line 15)
    - `from typing import Optional` (line 15)
    - `from typing import Protocol` (line 15)
    - `from typing import Set` (line 15)
    - `from typing import Tuple` (line 15)
    - `from typing import runtime_checkable` (line 15)
  - **Third-party** (10):
    - `import numpy as np` (line 17)
    - `import orjson as orjson` (line 18)
    - `import spacy as spacy` (line 19)
    - `from spacy.language import Language` (line 23)
    - `from spacy.lookups import Lookups` (line 24)
    - `from spacy.training import Example` (line 25)
    - `from spacy.training import offsets_to_biluo_tags` (line 25)
    - `from sqlalchemy import inspect` (line 26)
    - `from sqlalchemy import select` (line 26)
    - `from torch.utils.data import DataLoader` (line 27)
  - **Local/Project** (55):
    - `import gc as gc` (line 3)
    - `import glob as glob` (line 4)
    - `from importlib.util import find_spec` (line 13)
    - `from types import ModuleType` (line 14)
    - `from sentence_transformers import InputExample` (line 20)
    - `from sentence_transformers import losses` (line 20)
    - `from sklearn.cluster import KMeans` (line 21)
    - `from sklearn.feature_extraction.text import TfidfVectorizer` (line 22)
    - `from config import CONTEXT_DB_PATH` (line 29)
    - `from config import LOG_DIR` (line 29)
    - `from config import MODEL_DIR` (line 29)
    - `from config import PROJECT_ROOT` (line 29)
    - `from config import REVIEW_WITH_MANUAL_BOT` (line 29)
    - `from config import SBERT_BATCH_SIZE` (line 29)
    - `from config import SBERT_EPOCHS` (line 29)
    - `from config import SPACY_NER_BATCH_SIZE` (line 29)
    - `from config import SPACY_NER_EPOCHS` (line 29)
    - `from config import SPACY_NER_MIN_DELTA` (line 29)
    - `from config import SPACY_NER_PATIENCE` (line 29)
    - `from config import get_sqlalchemy_engine` (line 29)
    - `from config import get_subprocess_env` (line 29)
    - `from Context_Integration.Context_Library.constants import
      ELECTION_ENTITY_LABELS` (line 44)
    - `from Context_Integration.Context_Library.constants import
      ENTITY_PATTERNS` (line 44)
    - `from Context_Integration.Context_Library.constants import
      MISALIGNED_PATTERNS` (line 44)
    - `from Context_Integration.Context_Library.constants import PARTY_KEYWORDS`
      (line 44)
    - `from Context_Integration.librarian import load_context_library` (line 50)
    - `from utils.db_utils import get_session` (line 51)
    - `from utils.logger_singleton import console` (line 52)
    - `from utils.logger_singleton import logger` (line 52)
    - `from utils.misc_utils import safe_db_path` (line 53)
    - `from utils.model_registry import ModelRegistry` (line 54)
    - `from utils.models import Base` (line 55)
    - `from utils.models import Candidate` (line 55)
    - `from utils.models import Contest` (line 55)
    - `from utils.models import County` (line 55)
    - `from utils.models import DeclarativeBaseProtocol` (line 55)
    - `from utils.models import District` (line 55)
    - `from utils.models import Entity` (line 55)
    - `from utils.models import MetaDataProtocol` (line 55)
    - `from utils.models import Office` (line 55)
    - `from utils.models import Party` (line 55)
    - `from utils.models import Result` (line 55)
    - `from utils.models import State` (line 55)
    - `from utils.models import TableStructure` (line 55)
    - `from utils.shared_logic import get_or_create` (line 70)
    - `from utils.shared_logic import safe_add` (line 70)
    - `from utils.shared_logic import safe_commit` (line 70)
    - `from utils.shared_logic import safe_encode` (line 70)
    - `from utils.shared_logic import safe_execute` (line 70)
    - `from utils.shared_logic import safe_get` (line 70)
- Task markers:
  - L178 **WARNING**: (f"\[CLEAN\] File not found: {jsonl_path}")
  - L186 **WARNING**: (f"\[CLEAN\] Could not parse line: {e}")
  - L201 **WARNING**: (f"\[CLEAN\] Alignment check failed for text:
    {text\[:50\]}... ({e})")
  - L274 **WARNING**: (f"Failed to load {path}: {e}")
  - L403 **WARNING**: (f"Skipping misaligned entity in: {text}")
  - L408 **WARNING**: (f"Error validating entity alignment: {e}")
  - L434 **WARNING**: (f"\[spaCy\] Could not check GPU availability: {e}")
  - L450 **WARNING**: (f"\[spaCy\] Could not load lexeme normalization table.
    You may ignore this for English. Error: {e}")
  - L536 **WARNING**: (f"\[NER\] Skipped {misaligned_count} misaligned examples.
    Saved to {misaligned_path}")
  - L550 **WARNING**: ("No NER training examples found. Skipping spaCy NER
    retraining.")
  - L619 **WARNING**: ("\[SUGGESTION\] Consider lowering min_delta or increasing
    patience if you want longer training.")
  - L621 **WARNING**: ("\[SUGGESTION\] Model improved until the last epoch.
    Consider increasing epochs for further improvement.")
  - L622 **WARNING**: (f"\[SUGGESTION\] Next run: patience={patience},
    min_delta={min_delta:.2f}, epochs={epochs}")
  - L708 **WARNING**: ("No training examples found. Aborting retraining.")
  - L727 **WARNING**: (f"\[WARN\] Could not delete old model directory
    {oldest_path}: {e}")
  - L739 **WARNING**: (f"\[WARN\] Failed to load existing model: {e}")
  - L742 **WARNING**: ("Falling back to base model (all-MiniLM-L6-v2).")
  - L782 **WARNING**: (f"\[WARN\] Could not update canonical model directory:
    {e}")
  - L810 **WARNING**: (f"MISALIGNED: {text} {annots\['entities'\]}")
  - L840 **WARNING**: ("\[DB\] Base.metadata.tables is empty. No models
    registered? Did you import all model classes?")
- Outgoing cross-module calls (sample):
  - value.strip (line 101)
  - utils.misc\_utils.safe\_db\_path (line 115)
  - utils.db\_utils.get\_session (line 116)
  - utils.shared\_logic.get\_or\_create (line 119)
  - utils.shared\_logic.safe\_get (line 119)
  - utils.shared\_logic.get\_or\_create (line 120)
  - utils.shared\_logic.safe\_get (line 120)
  - utils.shared\_logic.get\_or\_create (line 121)
  - utils.shared\_logic.safe\_get (line 121)
  - utils.shared\_logic.get\_or\_create (line 122)
  - utils.shared\_logic.safe\_get (line 122)
  - utils.shared\_logic.get\_or\_create (line 123)
  - utils.shared\_logic.safe\_get (line 123)
  - utils.shared\_logic.get\_or\_create (line 124)
  - utils.shared\_logic.safe\_get (line 126)
  - utils.shared\_logic.get\_or\_create (line 129)
  - utils.shared\_logic.safe\_get (line 131)
  - utils.shared\_logic.safe\_get (line 132)
  - utils.shared\_logic.get\_or\_create (line 135)
  - utils.shared\_logic.safe\_get (line 138)
  - utils.shared\_logic.safe\_get (line 139)
  - utils.shared\_logic.safe\_get (line 140)
  - utils.shared\_logic.safe\_get (line 141)
  - utils.shared\_logic.safe\_get (line 142)
  - results.append (line 144)
  - utils.logger\_singleton.console.panel (line 145)
  - utils.logger\_singleton.console.table (line 147)
  - utils.shared\_logic.safe\_commit (line 148)
  - utils.logger\_singleton.console.log (line 149)
  - re.match (line 154)
  - spacy.blank (line 164)
  - Context\_Integration.Context\_Library.constants.MISALIGNED\_PATTERNS.copy
    (line 165)
  - patterns.extend (line 167)
  - re.match (line 171)
  - utils.logger\_singleton.logger.warning (line 178)
  - orjson.loads (line 184)
  - utils.logger\_singleton.logger.warning (line 186)
  - utils.shared\_logic.safe\_get (line 188)
  - utils.shared\_logic.safe\_get (line 189)
  - misaligned.append (line 192)
  - spacy.training.offsets\_to\_biluo\_tags (line 196)
  - nlp.make\_doc (line 196)
  - misaligned.append (line 198)
  - utils.logger\_singleton.logger.warning (line 201)
  - misaligned.append (line 202)
  - cleaned.append (line 204)
  - utils.shared\_logic.safe\_replace (line 207)
  - f.write (line 210)
  - orjson.dumps (line 210)
  - f.write (line 214)
- Inbound references:
  - normalize\_entity ← retrain_table_structure_models.py:104
  - normalize\_entity ← retrain_table_structure_models.py:119
  - normalize\_entity ← retrain_table_structure_models.py:120
  - normalize\_entity ← retrain_table_structure_models.py:121
  - normalize\_entity ← retrain_table_structure_models.py:122
  - normalize\_entity ← retrain_table_structure_models.py:123
  - normalize\_entity ← retrain_table_structure_models.py:126
  - normalize\_entity ← retrain_table_structure_models.py:131
  - normalize\_entity\_list ← retrain_table_structure_models.py:644
  - is\_misaligned\_text ← retrain_table_structure_models.py:509
  - is\_misaligned\_text ← retrain_table_structure_models.py:927
  - clean\_misaligned\_ner\_jsonl ← retrain_table_structure_models.py:857
  - save\_training\_data\_jsonl ← retrain_table_structure_models.py:538
  - cluster\_container\_patterns ← retrain_table_structure_models.py:978
  - auto\_label\_header ← retrain_table_structure_models.py:511
  - auto\_label\_header ← retrain_table_structure_models.py:929
  - extract\_candidates\_from\_context ← retrain_table_structure_models.py:484
  - extract\_candidates\_from\_context ← retrain_table_structure_models.py:915
  - entity\_frequency\_analysis ← retrain_table_structure_models.py:539
  - update\_db\_with\_new\_entities ← retrain_table_structure_models.py:645
  - load\_spacy\_ner\_examples ← retrain_table_structure_models.py:469
  - load\_spacy\_ner\_examples ← retrain_table_structure_models.py:898
  - remove\_overlapping\_entities ← retrain_table_structure_models.py:513
  - remove\_overlapping\_entities ← retrain_table_structure_models.py:546
  - remove\_overlapping\_entities ← retrain_table_structure_models.py:931
  - retrain\_spacy\_ner\_advanced ← retrain_table_structure_models.py:959
  - get\_all\_confirmed\_structures ← retrain_table_structure_models.py:859
  - run\_manual\_correction ← retrain_table_structure_models.py:853
  - run\_manual\_correction ← retrain_table_structure_models.py:949
  - retrain\_sentence\_transformer ← retrain_table_structure_models.py:954
  - segment\_hash ← retrain_table_structure_models.py:885
  - segment\_hash ← html_scanner.py:1608
  - segment\_hash ← html_scanner.py:2967
  - load\_cached\_segment\_hashes ← retrain_table_structure_models.py:882
  - scan\_in\_memory\_ner\_examples ← retrain_table_structure_models.py:936
  - ensure\_table\_structures\_exists ← retrain_table_structure_models.py:851

### health/risk\_gates.py {#webapp-parser-health-risk-gates-py}

> risk_gates.py

- Definitions:
  - class: `RiskGateScores` (line 38)
  - class: `RiskGateConfig` (line 49)
  - class: `RiskGateEvaluator` (line 67)
  - function: `evaluate\_risk` (line 390)
- Imports:
  - **Standard Library** (4):
    - `from dataclasses import dataclass` (line 33)
    - `from typing import List` (line 34)
    - `from typing import Optional` (line 34)
    - `from typing import Tuple` (line 34)
- Task markers:
  - L11 **WARN**: /log tiers (⅓-proportioned boundaries).
  - L24 **WARN**: 0.45 ≤ suspicion &lt; 0.72  (middle third → confirm/verify)
  - L44 **WARN**: ", or "log"
  - L274 **WARN**: 0.45 ≤ suspicion &lt; 0.72  (middle ⅓, 45–72%)
  - L278 **WARN**: tier (~27% width) is middle third
  - L287 **WARN**: " | "log"
  - L306 **WARN**: tier
  - L315 **WARN**: ", confidence)
  - L319 **WARN**: threshold
- Outgoing cross-module calls (sample):
  - self.\_validate\_config (line 73)
  - self.compute\_confidence\_gate (line 356)
  - self.compute\_verification\_gate (line 357)
  - self.compute\_anomaly\_gate (line 362)
  - self.compute\_composite\_suspicion (line 370)
  - self.classify\_risk\_tier (line 373)
  - \_default\_evaluator.evaluate (line 408)
- Inbound references:
  - RiskGateScores ← risk_gates.py:375
  - RiskGateConfig ← html_election_parser.py:174
  - RiskGateConfig ← risk_gates.py:72
  - RiskGateEvaluator ← risk_gates.py:387

### health/risk\_gates\_calculus.py {#webapp-parser-health-risk-gates-calculus-py}

> risk_gates_calculus.py

- Definitions:
  - class: `DerivativeGates` (line 43)
  - class: `SubTierClassification` (line 60)
  - class: `CalculusRiskEvaluator` (line 71)
  - function: `evaluate\_risk\_with\_calculus` (line 365)
  - function: `visualize\_sub\_tier\_classification` (line 437)
- Imports:
  - **Standard Library** (5):
    - `import math as math` (line 35)
    - `from dataclasses import dataclass` (line 36)
    - `from typing import List` (line 37)
    - `from typing import Optional` (line 37)
    - `from typing import Tuple` (line 37)
  - **Third-party** (3):
    - `from webapp.parser.health.risk_gates import RiskGateConfig` (line 39)
    - `from webapp.parser.health.risk_gates import RiskGateEvaluator` (line 39)
    - `from webapp.parser.health.risk_gates import RiskGateScores` (line 39)
- Task markers:
  - L15 **WARN**: boundary (approaching 0.45)
  - L16 **WARN**: →BLOCK boundary (approaching 0.72)
  - L63 **WARN**: ", or "block"
  - L78 **WARN**: /BLOCK
  - L156 **WARN**: boundary (0.45)
  - L246 **WARN**: /BLOCK) from base classifier
  - L270 **WARN**: ":
  - L271 **WARN**: tier has two boundaries; pick nearest
  - L404 **WARN**: WARN→BLOCK
  - L412 **WARN**: TIER (0.45 – 0.72)
- Outgoing cross-module calls (sample):
  - webapp.parser.health.risk\_gates.RiskGateEvaluator (line 101)
  - webapp.parser.health.risk\_gates.RiskGateConfig (line 102)
  - self.\_compute\_boundary\_slope (line 157)
  - self.\_compute\_boundary\_slope (line 164)
  - math.sqrt (line 173)
  - self.compute\_derivative\_gates (line 352)
  - self.classify\_sub\_tier (line 355)
  - \_default\_calculus\_evaluator.evaluate\_with\_derivatives (line 383)
  - emoji\_map.get (line 453)
- Inbound references:
  - DerivativeGates ← risk_gates_calculus.py:176
  - SubTierClassification ← risk_gates_calculus.py:304
  - CalculusRiskEvaluator ← html_election_parser.py:189
  - CalculusRiskEvaluator ← risk_gates_calculus.py:362

### health/risk\_gates\_integration\_examples.py {#webapp-parser-health-risk-gates-integration-examples-py}

> risk_gates_integration_examples.py

- Definitions:
  - function: `evaluate\_parser\_extraction` (line 22)
  - function: `evaluate\_data\_framework\_upload` (line 90)
  - function: `evaluate\_ballot\_lens\_display` (line 156)
  - function: `evaluate\_guarded\_action` (line 228)
  - function: `summarize\_risk\_distribution` (line 299)
- Imports:
  - **Standard Library** (3):
    - `from typing import Any` (line 14)
    - `from typing import Dict` (line 14)
    - `from typing import Optional` (line 14)
  - **Third-party** (1):
    - `from webapp.parser.health.risk_gates import RiskGateEvaluator` (line 16)
- Task markers:
  - L8 **WARN**: /log extraction
  - L45 **WARN**: (0.45 ≤ tier &lt; 0.72): Import with explicit confirmation
    prompt
  - L65 **WARN**: " else action
  - L130 **NOTE**: = "Data quality acceptable; direct upload approved."
  - L131 **WARN**: ":
  - L134 **NOTE**: = "Medium-risk data; requires user confirmation before
    upload."
  - L138 **NOTE**: = "High-risk data; escalated to admin review. Do not expose
    to Data Framework."
  - L149 **NOTE**:         "audit_note": note
  - L173 **WARN**: tier: Yellow badge, normal display with "⚠️ verify" tooltip
  - L191 **WARN**: ": "#ffc107",   # Yellow
  - L197 **WARN**: ": "visible",
  - L213 **WARN**: "
  - L219 **WARN**: "),
  - L282 **WARN**: " if profile\["suspicion"\] &gt;= 0.45 else "log")
  - L308 **WARN**: /BLOCK items)
  - L322 **WARN**: ": 0, "block": 0}
  - L337 **WARN**: ": 0, "block": 0}
  - L347 **WARN**: ": round(100_tier_counts\["warn"\] / total, 1) if total &gt;
    0 else 0,
  - L355 **WARN**: '\]} items awaiting user confirmation (WARN tier)"
  - L406 **WARN**: tier (0=near LOG, 1=near BLOCK)
  - L423 **WARN**: to LOG if other gates favorable
  - L427 **WARN**: tier
  - L429 **WARN**: tier
- Outgoing cross-module calls (sample):
  - webapp.parser.health.risk\_gates.RiskGateEvaluator (line 51)
  - evaluator.evaluate (line 54)
  - webapp.parser.health.risk\_gates.RiskGateEvaluator (line 110)
  - evaluator.evaluate (line 112)
  - webapp.parser.health.risk\_gates.RiskGateEvaluator (line 179)
  - evaluator.evaluate (line 181)
  - risk\_profiles.get (line 265)
  - webapp.parser.health.risk\_gates.RiskGateEvaluator (line 320)
  - evaluator.evaluate (line 326)
  - contest.get (line 327)
  - contest.get (line 328)
  - contest.get (line 328)
  - contest.get (line 329)
  - contest.get (line 330)
  - contest.get (line 330)
  - contest.get (line 335)
  - tier\_counts.values (line 340)

### health/risk\_gates\_spec.py {#webapp-parser-health-risk-gates-spec-py}

> TECHNICAL SPECIFICATION: Three-Dimensional Risk Assessment Model

- Task markers:
  - L24 **WARN**: (0.45 – 0.72): User confirmation required
  - L72 **WARN**:   • TOTAL SUSPICION ≈ 0.60 → TIER: WARN
  - L86 **WARN**: ) is narrower: user confirmation gates
  - L94 **WARN**: tier
  - L130 **WARN**: (center of warn, tier_conf ≈ 0.76)
  - L137 **WARN**: (near BLOCK boundary, tier_conf ≈ 0.99)
  - L173 **WARN**: tier → IMPORT_CONFIRM
  - L194 **WARN**: If tier == WARN:
  - L220 **WARN**:   tier == WARN:
  - L239 **WARN**: If tier &gt;= WARN:
  - L280 **WARN**: tier narrower
  - L287 **WARN**: tier broader (0.55–0.85), BLOCK narrower (0.85–1.0)
  - L289 **WARN**: tier is broader and more forgiving.
  - L327 **WARN**: , BLOCK)
  - L394 **WARN**: "

### health/scan\_misaligned\_ner.py {#webapp-parser-health-scan-misaligned-ner-py}

- Definitions:
  - function: `resolve\_jsonl\_path` (line 15)
  - function: `scan\_misaligned` (line 22)
  - function: `self\_heal\_loop` (line 101)
  - function: `main` (line 125)
- Imports:
  - **Standard Library** (5):
    - `import os as os` (line 1)
    - `import subprocess as subprocess` (line 2)
    - `import sys as sys` (line 3)
    - `import time as time` (line 4)
    - `from pathlib import Path` (line 5)
  - **Third-party** (3):
    - `import orjson as orjson` (line 7)
    - `import spacy as spacy` (line 8)
    - `from spacy.training import offsets_to_biluo_tags` (line 9)
  - **Local/Project** (3):
    - `from config import LOG_DIR` (line 11)
    - `from config import PROJECT_ROOT` (line 11)
    - `from utils.logger_singleton import logger` (line 12)
- Task markers:
  - L62 **WARNING**: (f"\[CORRUPT\] Could not parse line: {e}")
  - L83 **WARNING**: (f"\n\[MISALIGNED\] Top {top_n} most frequent misaligned
    NER texts:")
  - L85 **WARNING**: (f"  {repr(text)}: {count} times")
  - L86 **WARNING**: ("\[MISALIGNED\] Consider cleaning or pattern-excluding
    these from your training data.")
  - L87 **WARNING**: ("Run the manual_correction to review and clean these
    examples before retraining.")
  - L88 **WARNING**: ("If you see spaCy entity alignment warnings, consider
    cleaning your training data or using the provided validation function.")
  - L98 **WARNING**: (f"\[WARN\] Could not remove old misaligned file: {e}")
  - L112 **WARNING**: ("\[SELF-HEAL\] Misalignments found. Launching
    manual_correction for spacy_ner_misaligned...")
  - L119 **WARNING**: (f"\[SELF-HEAL\] manual_correction exited with code
    {result.returncode}")
  - L120 **WARNING**: (f"\[SELF-HEAL\] Sleeping {cooldown}s before
    rescanning...")
  - L122 **WARNING**: ("\[SELF-HEAL\] Max retries reached. Some misalignments
    may remain.")
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 17)
  - p.is\_absolute (line 18)
  - pathlib.Path (line 19)
  - spacy.blank (line 31)
  - utils.logger\_singleton.logger.error (line 39)
  - line.strip (line 44)
  - orjson.loads (line 47)
  - entry.get (line 48)
  - entry.get (line 49)
  - spacy.training.offsets\_to\_biluo\_tags (line 52)
  - nlp.make\_doc (line 52)
  - misaligned.append (line 54)
  - utils.logger\_singleton.logger.info (line 56)
  - misaligned.append (line 58)
  - utils.logger\_singleton.logger.error (line 60)
  - utils.logger\_singleton.logger.warning (line 62)
  - utils.logger\_singleton.logger.error (line 64)
  - utils.logger\_singleton.logger.info (line 67)
  - f.write (line 73)
  - orjson.dumps (line 73)
  - utils.logger\_singleton.logger.info (line 74)
  - entry.get (line 79)
  - utils.logger\_singleton.logger.warning (line 83)
  - counter.most\_common (line 84)
  - utils.logger\_singleton.logger.warning (line 85)
  - utils.logger\_singleton.logger.warning (line 86)
  - utils.logger\_singleton.logger.warning (line 87)
  - utils.logger\_singleton.logger.warning (line 88)
  - utils.logger\_singleton.logger.info (line 91)
  - os.remove (line 95)
  - utils.logger\_singleton.logger.info (line 96)
  - utils.logger\_singleton.logger.warning (line 98)
  - utils.logger\_singleton.logger.info (line 107)
  - utils.logger\_singleton.logger.info (line 110)
  - utils.logger\_singleton.logger.warning (line 112)
  - subprocess.run (line 114)
  - utils.logger\_singleton.logger.warning (line 119)
  - utils.logger\_singleton.logger.warning (line 120)
  - time.sleep (line 121)
  - utils.logger\_singleton.logger.warning (line 122)
  - argparse.ArgumentParser (line 129)
  - parser.add\_argument (line 136)
  - parser.add\_argument (line 137)
  - parser.add\_argument (line 138)
  - parser.add\_argument (line 139)
  - parser.add\_argument (line 140)
  - parser.add\_argument (line 141)
  - parser.parse\_args (line 142)
  - utils.logger\_singleton.logger.info (line 151)
  - subprocess.run (line 152)
- Inbound references:
  - resolve\_jsonl\_path ← scan_misaligned_ner.py:37
  - resolve\_jsonl\_path ← scan_misaligned_ner.py:145
  - scan\_misaligned ← scan_misaligned_ner.py:108
  - scan\_misaligned ← scan_misaligned_ner.py:149
  - self\_heal\_loop ← scan_misaligned_ner.py:147

### health/session\_branching.py {#webapp-parser-health-session-branching-py}

> Session Branching and Multi-Tenant Isolation for Smart Elections Parser

- Definitions:
  - class: `SessionBranch` (line 22)
  - function: `get\_isolated\_branch` (line 164)
  - function: `validate\_url\_access` (line 182)
  - function: `add\_url\_to\_isolation` (line 240)
  - function: `get\_isolation\_summary` (line 274)
  - function: `list\_all\_isolation\_branches` (line 289)
  - function: `cleanup\_principal\_isolation` (line 302)
- Imports:
  - **Standard Library** (4):
    - `import threading as threading` (line 15)
    - `from typing import Any` (line 16)
    - `from typing import Dict` (line 16)
    - `from typing import Set` (line 16)
  - **Local/Project** (4):
    - `from __future__ import annotations` (line 13)
    - `from utils.logger_singleton import logger` (line 18)
    - `from utils.privilege_tiers import PrivilegeTier` (line 19)
    - `from utils.privilege_tiers import get_principal_tier` (line 19)
- Task markers:
  - L230 **WARNING**: ({
  - L231 **WARNING**: ",
  - L295 **WARNING**:     WARNING:
- Outgoing cross-module calls (sample):
  - threading.RLock (line 39)
  - time.time (line 137)
  - threading.RLock (line 161)
  - utils.privilege\_tiers.get\_principal\_tier (line 208)
  - utils.logger\_singleton.logger.info (line 211)
  - branch.record\_access (line 219)
  - branch.can\_access\_url (line 225)
  - branch.record\_access (line 227)
  - utils.logger\_singleton.logger.warning (line 230)
  - branch.add\_quarantined\_url (line 256)
  - branch.add\_rejected\_url (line 258)
  - utils.logger\_singleton.logger.info (line 263)
  - branch.get\_summary (line 286)
  - branch.get\_summary (line 299)
  - \_BRANCH\_ISOLATION\_MAP.values (line 299)
  - \_BRANCH\_ISOLATION\_MAP.pop (line 313)
  - utils.logger\_singleton.logger.info (line 314)
- Inbound references:
  - SessionBranch ← session_branching.py:178
  - get\_isolated\_branch ← web_pipeline.py:266
  - get\_isolated\_branch ← session_branching.py:201
  - get\_isolated\_branch ← session_branching.py:251
  - get\_isolated\_branch ← session_branching.py:283
  - validate\_url\_access ← web_pipeline.py:437
  - validate\_url\_access ← web_pipeline.py:511
  - validate\_url\_access ← session_manager.py:703
  - add\_url\_to\_isolation ← session_manager.py:732
  - get\_isolation\_summary ← session_manager.py:754
  - cleanup\_principal\_isolation ← web_pipeline.py:891
  - cleanup\_principal\_isolation ← session_manager.py:776

### health/session\_manager.py {#webapp-parser-health-session-manager-py}

- Definitions:
  - class: `SessionManager` (line 15)
- Imports:
  - **Standard Library** (11):
    - `import os as os` (line 3)
    - `import time as time` (line 4)
    - `from datetime import datetime` (line 5)
    - `from datetime import timezone` (line 5)
    - `from queue import Queue` (line 6)
    - `from threading import RLock` (line 7)
    - `from threading import Thread` (line 7)
    - `from typing import Any` (line 8)
    - `from typing import Callable` (line 8)
    - `from typing import Dict` (line 8)
    - `from typing import Optional` (line 8)
  - **Third-party** (3):
    - `from webapp.parser.utils.session_state import DEFAULT_PHASE_BY_STATE`
      (line 10)
    - `from webapp.parser.utils.session_state import PipelinePhase` (line 10)
    - `from webapp.parser.utils.session_state import SessionState` (line 10)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - threading.RLock (line 19)
  - metadata.get (line 57)
  - dt.fromisoformat (line 62)
  - expiry\_str.replace (line 62)
  - exp\_dt.timestamp (line 63)
  - metadata.items (line 68)
  - hashlib.sha256 (line 69)
  - meta\_str.encode (line 69)
  - time.time (line 74)
  - self.get\_cached\_cert (line 91)
  - cached.get (line 95)
  - new\_metadata.items (line 100)
  - hashlib.sha256 (line 101)
  - meta\_str.encode (line 101)
  - cached.get (line 102)
  - self.get\_cached\_cert (line 110)
  - cached.get (line 111)
  - time.time (line 113)
  - time.perf\_counter (line 125)
  - self.\_build\_metadata (line 130)
  - meta.get (line 132)
  - time.time (line 134)
  - self.\_record\_profile (line 149)
  - time.perf\_counter (line 149)
  - datetime.datetime.now (line 155)
  - time.time (line 156)
  - meta.update (line 180)
  - self.\_infer\_phase\_from\_state (line 210)
  - datetime.datetime.now (line 211)
  - meta.update (line 213)
  - extras.get (line 214)
  - extras.get (line 215)
  - self.set\_manual\_source (line 217)
  - meta.get (line 219)
  - self.set\_manual\_source (line 220)
  - time.time (line 221)
  - webapp.parser.utils.session\_state.DEFAULT\_PHASE\_BY\_STATE.get (line 225)
  - webapp.parser.utils.session\_state.PipelinePhase (line 228)
  - time.time (line 242)
  - time.perf\_counter (line 266)
  - datetime.datetime.now (line 274)
  - time.time (line 275)
  - self.\_record\_profile (line 301)
  - time.perf\_counter (line 301)
  - logs.append (line 309)
  - queue.Queue (line 329)
  - self.\_select\_latest\_session\_locked (line 412)
  - self.\_remove\_session\_from\_principal\_sets\_locked (line 419)
  - meta.get (line 428)
  - sessions.discard (line 439)

### html\_election\_parser.py {#webapp-parser-html-election-parser-py}

- Definitions:
  - function: `\_normalize\_unit\_interval` (line 93)
  - function: `\_safe\_int` (line 105)
  - function: `\_apply\_risk\_assessment` (line 114)
  - function: `\_close\_browser\_quietly` (line 257)
  - function: `\_captcha\_detection\_key` (line 279)
  - function: `\_register\_cloudflare\_detection` (line 283)
  - function: `\_prompt\_for\_captcha\_assist` (line 299)
  - function: `\_sanitize\_error\_metadata` (line 357)
  - function: `\_log\_session\_exception\_metadata` (line 380)
  - function: `\_count\_dom\_table\_rows` (line 392)
  - function: `load\_urls` (line 423)
  - function: `mark\_url\_processed` (line 483)
  - function: `prompt\_url\_selection` (line 544)
  - function: `process\_format\_override` (line 712)
  - function: `ai\_analyze\_results` (line 908)
  - function: `stream\_results` (line 1008)
  - function: `\_read\_text\_file\_with\_fallback` (line 1055)
  - function: `\_extract\_text\_blocks` (line 1071)
  - function: `generate\_generic\_html\_result` (line 1259)
  - function: `orchestrate\_url` (line 1485)
  - function: `\_orchestrate\_url\_worker` (line 2549)
  - function: `main` (line 2566)
  - function: `\_capture\_selenium\_ner\_training` (line 2886)
- Imports:
  - **Standard Library** (11):
    - `import os as os` (line 6)
    - `import re as re` (line 7)
    - `import threading as threading` (line 8)
    - `import time as time` (line 9)
    - `from collections import Counter` (line 10)
    - `from collections import defaultdict` (line 10)
    - `from datetime import datetime` (line 11)
    - `from multiprocessing import Pool` (line 12)
    - `from typing import Any` (line 13)
    - `from typing import Dict` (line 13)
    - `from typing import List` (line 13)
  - **Third-party** (4):
    - `import orjson as orjson` (line 15)
    - `import psycopg2 as psycopg2` (line 16)
    - `from playwright.sync_api import sync_playwright` (line 17)
    - `from sqlalchemy.exc import OperationalError` (line 18)
  - **Local/Project** (59):
    - `from __future__ import annotations` (line 1)
    - `from config import CACHE_LOCK` (line 20)
    - `from config import CACHE_RESET` (line 20)
    - `from config import DEFAULT_CAPTCHA_TIMEOUT` (line 20)
    - `from config import ENABLE_AI_ANALYSIS` (line 20)
    - `from config import ENABLE_PARALLEL` (line 20)
    - `from config import ENABLE_REALTIME_STREAM` (line 20)
    - `from config import ENABLE_SELENIUM_FALLBACK` (line 20)
    - `from config import INPUT_DIR` (line 20)
    - `from config import MAX_URLS_DISPLAYED` (line 20)
    - `from config import NAV_MAX_ATTEMPTS` (line 20)
    - `from config import NAV_TIMEOUT_PLAYWRIGHT_MS` (line 20)
    - `from config import NAV_TIMEOUT_SELENIUM_MS` (line 20)
    - `from config import OUTPUT_DIR` (line 20)
    - `from config import PROCESSED_URLS_FILE` (line 20)
    - `from config import UPLOADS_DIR` (line 20)
    - `from config import URL_LIST_FILE` (line 20)
    - `from Context_Integration.librarian import get_safe_log_path` (line 38)
    - `from navigator import NavigationInstructionRunner` (line 39)
    - `from navigator import NavigationRecipeStore` (line 39)
    - `from navigator.dom_snapshot import snapshot_mode_pipeline` (line 40)
    - `from state_router import get_handler` (line 41)
    - `from state_router import preload_handler_map` (line 41)
    - `from utils.browser_utils import SCROLL_METRIC_KEYS` (line 42)
    - `from utils.browser_utils import TABLE_DISCOVERY_SELECTOR` (line 42)
    - `from utils.browser_utils import autoscroll_until_stable` (line 42)
    - `from utils.browser_utils import safe_content` (line 42)
    - `from utils.browser_utils import safe_count` (line 42)
    - `from utils.browser_utils import safe_locator` (line 42)
    - `from utils.browser_utils import safe_query_selector_all` (line 42)
    - `from utils.browser_utils import sync_browser_pipeline` (line 42)
    - `from utils.browser_utils import sync_safe_browser_close` (line 42)
    - `from utils.captcha_tools import detect_cloudflare_challenge` (line 53)
    - `from utils.download_utils import ensure_input_directory` (line 54)
    - `from utils.download_utils import ensure_output_directory` (line 54)
    - `from utils.dynamic_table_extractor import dynamic_table_extractor` (line
      55)
    - `from utils.format_router import prompt_and_handle_download` (line 56)
    - `from utils.format_router import route_format_handler` (line 56)
    - `from utils.logger_singleton import logger` (line 57)
    - `from utils.logger_singleton import prompt` (line 57)
    - `from utils.misc_utils import extract_url_and_label` (line 58)
    - `from utils.misc_utils import load_processed_urls` (line 58)
    - `from utils.output_utils import finalize_election_output` (line 59)
    - `from utils.seleniumbase_launcher import SELENIUMBASE_AVAILABLE` (line 60)
    - `from utils.seleniumbase_launcher import close_driver` (line 60)
    - `from utils.seleniumbase_launcher import launch_browser` (line 60)
    - `from utils.seleniumbase_launcher import
      relaunch_browser_fullscreen_if_needed` (line 60)
    - `from utils.shared_logic import infer_state_county_from_url` (line 66)
    - `from utils.shared_logic import safe_is_set` (line 66)
    - `from utils.shared_logic import safe_parse` (line 66)
- Task markers:
  - L84 **WARNING**: ("Deleting .processed_urls cache for fresh start...")
  - L248 **WARNING**: ({
  - L249 **WARNING**: ",
  - L268 **WARNING**: ({
  - L269 **WARNING**: ",
  - L341 **WARNING**: ({
  - L342 **WARNING**: ",
  - L807 **WARNING**: ({
  - L808 **WARNING**: ",
  - L822 **WARNING**: ({
  - L823 **WARNING**: ",
  - L885 **WARNING**: ({
  - L886 **WARNING**: ",
  - L985 **WARNING**: (payload_2)
  - L1313 **WARNING**: ({
  - L1314 **WARNING**: ",
  - L1360 **WARNING**: ({
  - L1361 **WARNING**: ",
  - L1414 **WARNING**: ({
  - L1415 **WARNING**: ",
  - L1590 **WARNING**: ({
  - L1591 **WARNING**: ",
  - L1646 **WARNING**: ({
  - L1647 **WARNING**: ",
  - L1935 **WARNING**: ({
  - L1936 **WARNING**: ",
  - L1955 **WARNING**: ({
  - L1956 **WARNING**: ",
  - L1962 **WARNING**: ({
  - L1963 **WARNING**: ",
  - L2010 **WARNING**: ({
  - L2011 **WARNING**: ",
  - L2039 **WARNING**: ({
  - L2040 **WARNING**: ",
  - L2144 **WARNING**: ({
  - L2145 **WARNING**: ",
  - L2239 **WARNING**: ({
  - L2240 **WARNING**: ",
  - L2305 **WARNING**: ",
  - L2310 **WARNING**: (payload)
  - L2440 **WARNING**: ({
  - L2441 **WARNING**: ",
  - L2458 **WARNING**: ",
  - L2463 **WARNING**: (payload)
  - L2474 **WARNING**: ",
  - L2479 **WARNING**: (payload)
  - L2481 **WARN**: \] No output file path returned from parser and no output
    files found."
  - L2483 **WARNING**: ",
  - L2488 **WARNING**: (payload)
  - L2507 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - config.PROCESSED\_URLS\_FILE.exists (line 83)
  - utils.logger\_singleton.logger.warning (line 84)
  - config.PROCESSED\_URLS\_FILE.unlink (line 85)
  - navigator.NavigationRecipeStore (line 87)
  - navigator.NavigationInstructionRunner (line 88)
  - collections.defaultdict (line 89)
  - metadata.get (line 130)
  - metadata.get (line 130)
  - metadata.get (line 131)
  - metadata.get (line 131)
  - quality.get (line 134)
  - metadata.get (line 134)
  - metadata.get (line 139)
  - metadata.get (line 146)
  - metadata.get (line 150)
  - metadata.get (line 152)
  - metadata.get (line 155)
  - audit.get (line 156)
  - metadata.get (line 159)
  - metadata.get (line 160)
  - audit.get (line 161)
  - metadata.get (line 165)
  - metadata.get (line 165)
  - RISK\_GATES\_CONFIG.get (line 175)
  - RISK\_GATES\_CONFIG.get (line 176)
  - RISK\_GATES\_CONFIG.get (line 177)
  - RISK\_GATES\_CONFIG.get (line 178)
  - RISK\_GATES\_CONFIG.get (line 179)
  - RISK\_GATES\_CONFIG.get (line 180)
  - RISK\_GATES\_CONFIG.get (line 181)
  - RISK\_GATES\_CONFIG.get (line 182)
  - \_RISK\_PREVIOUS\_BY\_SESSION.get (line 187)
  - evaluator.evaluate\_with\_derivatives (line 191)
  - metadata.get (line 197)
  - metadata.get (line 197)
  - utils.logger\_singleton.logger.info (line 240)
  - utils.logger\_singleton.logger.warning (line 248)
  - utils.browser\_utils.sync\_safe\_browser\_close (line 265)
  - utils.logger\_singleton.logger.warning (line 268)
  - \_CAPTCHA\_DETECTION\_COUNTS.get (line 285)
  - utils.logger\_singleton.logger.info (line 287)
  - utils.logger\_singleton.prompt.prompt\_input (line 334)
  - utils.logger\_singleton.logger.warning (line 341)
  - utils.shared\_logic.safe\_strip (line 349)
  - metadata.get (line 373)
  - session\_id.startswith (line 384)
  - Context\_Integration.librarian.get\_safe\_log\_path (line 385)
  - handle.write (line 387)
  - orjson.dumps (line 387)
  - page.query\_selector\_all (line 401)
- Inbound references:
  - \_normalize\_unit\_interval ← html_election_parser.py:133
  - \_normalize\_unit\_interval ← html_election_parser.py:138
  - \_normalize\_unit\_interval ← html_election_parser.py:164
  - \_normalize\_unit\_interval ← html_election_parser.py:169
  - \_safe\_int ← html_election_parser.py:146
  - \_safe\_int ← html_election_parser.py:150
  - \_safe\_int ← html_election_parser.py:152
  - \_safe\_int ← html_election_parser.py:155
  - \_safe\_int ← html_election_parser.py:156
  - \_safe\_int ← html_election_parser.py:159
  - \_safe\_int ← html_election_parser.py:160
  - \_safe\_int ← html_election_parser.py:161
  - \_safe\_int ← json_export_loader.py:266
  - \_safe\_int ← json_export_loader.py:267
  - \_safe\_int ← json_export_loader.py:280
  - \_safe\_int ← json_export_loader.py:316
  - \_safe\_int ← pivot.py:1972
  - \_safe\_int ← pivot.py:2003
  - \_safe\_int ← pivot.py:2007
  - \_safe\_int ← pivot.py:2008
  - \_safe\_int ← pivot.py:2032
  - \_safe\_int ← pivot.py:2034
  - \_safe\_int ← pivot.py:2052
  - \_safe\_int ← pivot.py:2054
  - \_safe\_int ← pivot.py:2139
  - \_safe\_int ← pivot.py:2143
  - \_safe\_int ← pivot.py:2152
  - \_apply\_risk\_assessment ← html_election_parser.py:1819
  - \_apply\_risk\_assessment ← html_election_parser.py:2111
  - \_apply\_risk\_assessment ← html_election_parser.py:2423
  - \_close\_browser\_quietly ← html_election_parser.py:1765
  - \_close\_browser\_quietly ← html_election_parser.py:1775
  - \_close\_browser\_quietly ← html_election_parser.py:1859
  - \_close\_browser\_quietly ← html_election_parser.py:1908
  - \_close\_browser\_quietly ← html_election_parser.py:1917
  - \_close\_browser\_quietly ← html_election_parser.py:1943
  - \_close\_browser\_quietly ← html_election_parser.py:1972
  - \_close\_browser\_quietly ← html_election_parser.py:1998
  - \_close\_browser\_quietly ← html_election_parser.py:2335
  - \_close\_browser\_quietly ← html_election_parser.py:2376
  - \_close\_browser\_quietly ← html_election_parser.py:2388
  - \_close\_browser\_quietly ← html_election_parser.py:2418
  - \_close\_browser\_quietly ← html_election_parser.py:2543
  - \_captcha\_detection\_key ← html_election_parser.py:284
  - \_register\_cloudflare\_detection ← html_election_parser.py:1926
  - \_register\_cloudflare\_detection ← html_election_parser.py:2037
  - \_prompt\_for\_captcha\_assist ← html_election_parser.py:1928
  - \_sanitize\_error\_metadata ← html_election_parser.py:2346
  - \_log\_session\_exception\_metadata ← html_election_parser.py:2357
  - \_count\_dom\_table\_rows ← html_election_parser.py:2205

### models/election\_data.py {#webapp-parser-models-election-data-py}

> Election Data SQLAlchemy Models

- Definitions:
  - function: `Integer` (line 23)
  - function: `String` (line 27)
  - function: `Text` (line 33)
  - function: `Boolean` (line 37)
  - function: `DateTime` (line 41)
  - function: `Float` (line 45)
  - function: `SQLEnum` (line 49)
  - class: `DataQualityTier` (line 53)
  - class: `ManualReviewStatus` (line 60)
  - class: `DataQualityFlagType` (line 69)
  - class: `ElectionResult` (line 81)
  - class: `ValidationRecord` (line 146)
  - class: `StagingRecord` (line 214)
  - class: `VoterDropoff` (line 257)
  - class: `RaceMetadata` (line 288)
  - class: `AuditLog` (line 324)
  - class: `ManualReviewQueue` (line 360)
  - class: `GoogleSheetsSync` (line 406)
  - class: `DownloadRecord` (line 439)
  - class: `ValidationRecord\_DL1` (line 513)
  - class: `ValidationRecord\_DL2` (line 577)
  - class: `PreQCComparison` (line 644)
  - class: `QC1Checkpoint` (line 683)
  - class: `QC2Checkpoint` (line 721)
  - class: `ChainOfCustody` (line 763)
- Imports:
  - **Standard Library** (3):
    - `from datetime import datetime` (line 6)
    - `from enum import Enum as PyEnum` (line 7)
    - `from typing import Any` (line 8)
  - **Third-party** (12):
    - `from sqlalchemy import Boolean as _Boolean` (line 10)
    - `from sqlalchemy import Column` (line 11)
    - `from sqlalchemy import ForeignKey` (line 11)
    - `from sqlalchemy import Index` (line 11)
    - `from sqlalchemy import DateTime as _DateTime` (line 12)
    - `from sqlalchemy import Enum as _SQLEnumType` (line 13)
    - `from sqlalchemy import Float as _Float` (line 14)
    - `from sqlalchemy import Integer as _Integer` (line 15)
    - `from sqlalchemy import String as _String` (line 16)
    - `from sqlalchemy import Text as _Text` (line 17)
    - `from sqlalchemy.orm import declarative_base` (line 18)
    - `from sqlalchemy.orm import relationship` (line 18)
- Outgoing cross-module calls (sample):
  - sqlalchemy.orm.declarative\_base (line 20)
  - sqlalchemy.Column (line 24)
  - sqlalchemy.Column (line 29)
  - sqlalchemy.Column (line 30)
  - sqlalchemy.String (line 30)
  - sqlalchemy.Column (line 34)
  - sqlalchemy.Column (line 38)
  - sqlalchemy.Column (line 42)
  - sqlalchemy.Column (line 46)
  - sqlalchemy.Column (line 50)
  - sqlalchemy.Enum (line 50)
  - sqlalchemy.Index (line 88)
  - sqlalchemy.Index (line 89)
  - sqlalchemy.Index (line 90)
  - sqlalchemy.orm.relationship (line 142)
  - sqlalchemy.orm.relationship (line 143)
  - sqlalchemy.Index (line 153)
  - sqlalchemy.Index (line 154)
  - sqlalchemy.Index (line 155)
  - sqlalchemy.ForeignKey (line 162)
  - sqlalchemy.orm.relationship (line 163)
  - sqlalchemy.Index (line 221)
  - sqlalchemy.Index (line 222)
  - sqlalchemy.Index (line 264)
  - sqlalchemy.Index (line 265)
  - sqlalchemy.Index (line 295)
  - sqlalchemy.Index (line 331)
  - sqlalchemy.Index (line 332)
  - sqlalchemy.Index (line 333)
  - sqlalchemy.ForeignKey (line 339)
  - sqlalchemy.orm.relationship (line 340)
  - sqlalchemy.Index (line 367)
  - sqlalchemy.Index (line 368)
  - sqlalchemy.Index (line 369)
  - sqlalchemy.ForeignKey (line 375)
  - sqlalchemy.Index (line 446)
  - sqlalchemy.Index (line 447)
  - sqlalchemy.Index (line 521)
  - sqlalchemy.Index (line 522)
  - sqlalchemy.ForeignKey (line 526)
  - sqlalchemy.Index (line 585)
  - sqlalchemy.Index (line 586)
  - sqlalchemy.ForeignKey (line 590)
  - sqlalchemy.Index (line 651)
  - sqlalchemy.Index (line 652)
  - sqlalchemy.ForeignKey (line 656)
  - sqlalchemy.ForeignKey (line 660)
  - sqlalchemy.ForeignKey (line 661)
  - sqlalchemy.Index (line 690)
  - sqlalchemy.Index (line 691)
- Inbound references:
  - Integer ← election_data.py:94
  - Integer ← election_data.py:97
  - Integer ← election_data.py:113
  - Integer ← election_data.py:114
  - Integer ← election_data.py:115
  - Integer ← election_data.py:116
  - Integer ← election_data.py:117
  - Integer ← election_data.py:118
  - Integer ← election_data.py:159
  - Integer ← election_data.py:162
  - Integer ← election_data.py:166
  - Integer ← election_data.py:183
  - Integer ← election_data.py:184
  - Integer ← election_data.py:185
  - Integer ← election_data.py:186
  - Integer ← election_data.py:187
  - Integer ← election_data.py:188
  - Integer ← election_data.py:225
  - Integer ← election_data.py:228
  - Integer ← election_data.py:268
  - Integer ← election_data.py:271
  - Integer ← election_data.py:279
  - Integer ← election_data.py:280
  - Integer ← election_data.py:298
  - Integer ← election_data.py:301
  - Integer ← election_data.py:307
  - Integer ← election_data.py:308
  - Integer ← election_data.py:309
  - Integer ← election_data.py:312
  - Integer ← election_data.py:313
  - Integer ← election_data.py:314
  - Integer ← election_data.py:336
  - Integer ← election_data.py:339
  - Integer ← election_data.py:372
  - Integer ← election_data.py:375
  - Integer ← election_data.py:378
  - Integer ← election_data.py:403
  - Integer ← election_data.py:413
  - Integer ← election_data.py:419
  - Integer ← election_data.py:420
  - Integer ← election_data.py:427
  - Integer ← election_data.py:450
  - Integer ← election_data.py:453
  - Integer ← election_data.py:486
  - Integer ← election_data.py:525
  - Integer ← election_data.py:526
  - Integer ← election_data.py:529
  - Integer ← election_data.py:548
  - Integer ← election_data.py:549
  - Integer ← election_data.py:550

### navigator/\_\_init\_\_.py {#webapp-parser-navigator-init-py}

> Dynamic navigation recipes for Smart Elections Parser.

- Imports:
  - **Local/Project** (3):
    - `from navigation_recipes import DEFAULT_RECIPE_PATH` (line 8)
    - `from navigation_recipes import NavigationRecipeStore` (line 8)
    - `from navigation_runner import NavigationInstructionRunner` (line 9)

### navigator/dom\_snapshot.py {#webapp-parser-navigator-dom-snapshot-py}

> DOM Snapshot Mode for Medium-Trust URLs

- Definitions:
  - function: `capture\_dom\_snapshot` (line 31)
  - function: `extract\_tables\_from\_snapshot` (line 123)
  - function: `snapshot\_mode\_pipeline` (line 282)
- Imports:
  - **Standard Library** (5):
    - `import time as time` (line 18)
    - `from typing import Any` (line 19)
    - `from typing import Dict` (line 19)
    - `from typing import List` (line 19)
    - `from typing import Tuple` (line 19)
  - **Local/Project** (3):
    - `from __future__ import annotations` (line 16)
    - `from utils.logger_singleton import logger` (line 27)
    - `from utils.telemetry import emit_telemetry_event` (line 28)
- Task markers:
  - L78 **WARNING**: ({
  - L79 **WARNING**: ",
  - L147 **WARNING**: ({
  - L148 **WARNING**: ",
  - L201 **WARNING**: ({
  - L202 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - time.time (line 61)
  - page.wait\_for\_selector (line 66)
  - utils.logger\_singleton.logger.debug (line 71)
  - utils.logger\_singleton.logger.warning (line 78)
  - page.content (line 87)
  - utils.logger\_singleton.logger.error (line 89)
  - time.time (line 97)
  - utils.logger\_singleton.logger.info (line 100)
  - utils.telemetry.emit\_telemetry\_event (line 111)
  - utils.logger\_singleton.logger.warning (line 147)
  - time.time (line 155)
  - utils.logger\_singleton.logger.error (line 163)
  - utils.logger\_singleton.logger.debug (line 172)
  - time.time (line 185)
  - utils.logger\_singleton.logger.info (line 186)
  - parser.css (line 199)
  - utils.logger\_singleton.logger.warning (line 201)
  - utils.logger\_singleton.logger.debug (line 209)
  - t.css (line 218)
  - largest\_table.css\_first (line 222)
  - largest\_table.css\_first (line 222)
  - header\_row.css (line 224)
  - cell.text (line 225)
  - headers.append (line 227)
  - largest\_table.css (line 231)
  - largest\_table.css\_first (line 231)
  - largest\_table.css (line 231)
  - first\_row.css (line 233)
  - largest\_table.css\_first (line 238)
  - largest\_table.css (line 238)
  - largest\_table.css (line 238)
  - row.css (line 241)
  - cell.text (line 247)
  - data\_rows.append (line 252)
  - time.time (line 254)
  - utils.logger\_singleton.logger.info (line 256)
  - utils.telemetry.emit\_telemetry\_event (line 268)
  - utils.logger\_singleton.logger.info (line 309)
  - context.get (line 314)
  - utils.logger\_singleton.logger.error (line 326)
  - utils.logger\_singleton.logger.error (line 347)
  - context.get (line 361)
  - context.get (line 361)
  - context.get (line 367)
  - context.get (line 368)
  - context.get (line 369)
  - context.get (line 373)
  - context.get (line 374)
  - utils.logger\_singleton.logger.info (line 377)
- Inbound references:
  - capture\_dom\_snapshot ← dom_snapshot.py:319
  - extract\_tables\_from\_snapshot ← dom_snapshot.py:341

### navigator/keyword\_bias.py {#webapp-parser-navigator-keyword-bias-py}

- Definitions:
  - function: `\_iter\_lines` (line 16)
  - function: `load\_keyword\_bias` (line 35)
- Imports:
  - **Standard Library** (5):
    - `import threading as threading` (line 3)
    - `from pathlib import Path` (line 4)
    - `from typing import Dict` (line 5)
    - `from typing import Iterable` (line 5)
    - `from typing import List` (line 5)
  - **Third-party** (1):
    - `import orjson as orjson` (line 7)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 9)
  - threading.RLock (line 13)
  - path.exists (line 17)
  - path.is\_file (line 17)
  - path.open (line 21)
  - line.strip (line 23)
  - orjson.loads (line 26)
  - obj.get (line 42)
  - obj.get (line 43)
  - normalized\_phrases.append (line 49)
  - p.lower (line 49)
  - entries.append (line 50)
  - obj.get (line 54)
  - obj.get (line 55)
  - obj.get (line 56)
  - \_cache.clear (line 59)
  - \_cache.extend (line 60)
- Inbound references:
  - \_iter\_lines ← keyword_bias.py:41

### navigator/navigation\_recipes.py {#webapp-parser-navigator-navigation-recipes-py}

- Definitions:
  - class: `NavigationRecipeStore` (line 16)
- Imports:
  - **Standard Library** (8):
    - `import threading as threading` (line 3)
    - `from pathlib import Path` (line 4)
    - `from typing import Any` (line 5)
    - `from typing import Dict` (line 5)
    - `from typing import Iterable` (line 5)
    - `from typing import List` (line 5)
    - `from typing import Sequence` (line 5)
    - `from urllib.parse import urlparse` (line 6)
  - **Third-party** (1):
    - `import orjson as orjson` (line 8)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from config import LOG_DIR` (line 10)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 12)
  - pathlib.Path (line 13)
  - pathlib.Path (line 30)
  - pathlib.Path (line 32)
  - threading.RLock (line 41)
  - orjson.loads (line 56)
  - self.\_maybe\_reload\_locked (line 68)
  - self.\_maybe\_reload\_learned\_locked (line 75)
  - self.load (line 79)
  - value.strip (line 85)
  - normalized.lower (line 86)
  - normalized.upper (line 86)
  - value.lower (line 93)
  - value.upper (line 93)
  - normalized\_candidate.lower (line 96)
  - normalized\_candidate.upper (line 96)
  - self.\_normalize (line 105)
  - context.get (line 105)
  - self.\_normalize (line 106)
  - context.get (line 106)
  - self.iter\_recipes (line 108)
  - recipe.get (line 109)
  - self.\_match\_list (line 112)
  - match.get (line 112)
  - match.get (line 112)
  - self.\_match\_list (line 113)
  - match.get (line 113)
  - match.get (line 113)
  - selected.append (line 115)
  - self.load\_learned (line 116)
  - recipe.get (line 117)
  - self.\_match\_list (line 120)
  - match.get (line 120)
  - match.get (line 120)
  - self.\_match\_list (line 121)
  - match.get (line 121)
  - match.get (line 121)
  - selected.append (line 123)
  - recipe.get (line 131)
  - self.\_maybe\_reload\_locked (line 133)
  - existing.get (line 137)
  - self.\_write\_locked (line 143)
  - orjson.dumps (line 147)
  - self.\_build\_learned\_recipes (line 162)
  - self.\_parse\_log\_line (line 174)
  - self.\_entry\_to\_recipe (line 177)
  - recipes.append (line 179)
  - line.strip (line 186)
  - orjson.loads (line 189)
  - entry.get (line 195)

### navigator/navigation\_runner.py {#webapp-parser-navigator-navigation-runner-py}

- Definitions:
  - class: `NavigationResult` (line 24)
  - class: `NavigationInstructionRunner` (line 32)
- Imports:
  - **Standard Library** (6):
    - `import threading as threading` (line 3)
    - `from dataclasses import dataclass` (line 5)
    - `from typing import Any` (line 6)
    - `from typing import Dict` (line 6)
    - `from typing import List` (line 6)
    - `from typing import Optional` (line 6)
  - **Local/Project** (13):
    - `from __future__ import annotations` (line 1)
    - `from concurrent.futures import ThreadPoolExecutor` (line 4)
    - `from concurrent.futures import wait` (line 4)
    - `from utils.browser_utils import SCROLL_METRIC_KEYS` (line 8)
    - `from utils.browser_utils import autoscroll_until_stable` (line 8)
    - `from utils.browser_utils import safe_click_with_retry` (line 8)
    - `from utils.browser_utils import safe_get_attribute` (line 8)
    - `from utils.browser_utils import safe_inner_text` (line 8)
    - `from utils.html_scanner import scan_html_for_context` (line 15)
    - `from utils.logger_singleton import logger` (line 16)
    - `from keyword_bias import load_keyword_bias` (line 17)
    - `from navigation_recipes import DEFAULT_RECIPE_PATH` (line 18)
    - `from navigation_recipes import NavigationRecipeStore` (line 18)
- Task markers:
  - L250 **WARNING**: ({
  - L251 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - navigation\_recipes.NavigationRecipeStore (line 43)
  - threading.RLock (line 48)
  - threading.RLock (line 49)
  - context.get (line 60)
  - self.\_apply\_keyword\_bias (line 66)
  - self.\_script\_matches (line 69)
  - self.\_execute\_script (line 71)
  - script.get (line 73)
  - script.get (line 75)
  - script.get (line 79)
  - match.get (line 82)
  - target\_url.lower (line 84)
  - substr.lower (line 85)
  - match.get (line 87)
  - page.content (line 90)
  - html\_source.lower (line 93)
  - marker.lower (line 96)
  - script.get (line 102)
  - self.\_execute\_step (line 103)
  - context\_updates.update (line 105)
  - script.get (line 106)
  - context\_updates.update (line 108)
  - action.lower (line 115)
  - step.get (line 118)
  - step.get (line 119)
  - step.get (line 120)
  - self.\_selector\_candidates (line 125)
  - page.wait\_for\_selector (line 127)
  - self.\_record\_trace (line 134)
  - self.\_should\_soft\_skip\_selector\_failure (line 136)
  - self.\_has\_results\_ready (line 136)
  - self.\_record\_trace (line 137)
  - self.\_record\_trace (line 140)
  - step.get (line 142)
  - step.get (line 143)
  - page.wait\_for\_load\_state (line 145)
  - self.\_record\_trace (line 146)
  - step.get (line 148)
  - step.get (line 149)
  - self.\_selector\_candidates (line 153)
  - utils.browser\_utils.safe\_click\_with\_retry (line 154)
  - step.get (line 158)
  - self.\_click\_by\_text\_discovery (line 168)
  - self.\_record\_trace (line 171)
  - self.\_should\_soft\_skip\_selector\_failure (line 173)
  - self.\_has\_results\_ready (line 173)
  - self.\_record\_trace (line 174)
  - step.get (line 177)
  - page.wait\_for\_timeout (line 180)
  - self.\_record\_trace (line 181)
- Inbound references:
  - NavigationResult ← navigation_runner.py:64
  - NavigationResult ← navigation_runner.py:75
  - NavigationResult ← navigation_runner.py:76

### navigator/training\_data.py {#webapp-parser-navigator-training-data-py}

- Definitions:
  - function: `iter\_navigation\_feedback` (line 14)
  - function: `build\_training\_dataset` (line 32)
  - function: `export\_training\_dataset` (line 56)
  - function: `main` (line 69)
- Imports:
  - **Standard Library** (5):
    - `import argparse as argparse` (line 3)
    - `from pathlib import Path` (line 4)
    - `from typing import Iterable` (line 5)
    - `from typing import List` (line 5)
    - `from typing import Optional` (line 5)
  - **Third-party** (1):
    - `import orjson as orjson` (line 7)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from config import LOG_DIR` (line 9)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 11)
  - pathlib.Path (line 15)
  - path.exists (line 16)
  - path.is\_file (line 16)
  - path.open (line 19)
  - line.strip (line 21)
  - orjson.loads (line 24)
  - entry.get (line 40)
  - entry.get (line 41)
  - entry.get (line 42)
  - entry.get (line 43)
  - entry.get (line 44)
  - entry.get (line 45)
  - samples.append (line 50)
  - pathlib.Path (line 63)
  - out\_path.write\_bytes (line 65)
  - orjson.dumps (line 65)
  - argparse.ArgumentParser (line 70)
  - parser.add\_argument (line 71)
  - parser.add\_argument (line 72)
  - parser.add\_argument (line 73)
  - parser.parse\_args (line 74)
  - orjson.dumps (line 80)
- Inbound references:
  - iter\_navigation\_feedback ← training_data.py:38
  - build\_training\_dataset ← training_data.py:62
  - build\_training\_dataset ← training_data.py:76
  - export\_training\_dataset ← training_data.py:78

### quality\_assurance/\_\_init\_\_.py {#webapp-parser-quality-assurance-init-py}

> Quality Assurance Module: Data Classification & Verification Pipeline

- Imports:
  - **Local/Project** (13):
    - `from data_classifier import ActionType` (line 13)
    - `from data_classifier import ClassificationResult` (line 13)
    - `from data_classifier import DatasetMetadata` (line 13)
    - `from data_classifier import DLStatus` (line 13)
    - `from data_classifier import QAIssue` (line 13)
    - `from data_classifier import QAIssueType` (line 13)
    - `from data_classifier import classify_as_dl1` (line 13)
    - `from data_classifier import get_dataset_lineage` (line 13)
    - `from data_classifier import get_dl2_inventory` (line 13)
    - `from data_classifier import get_pending_dl2_reviews` (line 13)
    - `from data_classifier import get_rejected_count` (line 13)
    - `from data_classifier import promote_to_dl2` (line 13)
    - `from qa_endpoints import qa_bp` (line 27)

### quality\_assurance/data\_classifier.py {#webapp-parser-quality-assurance-data-classifier-py}

> Data Classifier: DL1/DL2 Quality Assurance Pipeline

- Definitions:
  - class: `DLStatus` (line 35)
  - class: `QAIssueType` (line 43)
  - class: `IssureSeverity` (line 55)
  - class: `ActionType` (line 63)
  - class: `QAIssue` (line 77)
  - class: `ClassificationResult` (line 91)
  - class: `DatasetMetadata` (line 102)
  - function: `get\_db\_connection` (line 120)
  - function: `classify\_as\_dl1` (line 142)
  - function: `detect\_quality\_issues` (line 258)
  - function: `promote\_to\_dl2` (line 372)
  - function: `get\_pending\_dl2\_reviews` (line 462)
  - function: `get\_dl2\_inventory` (line 495)
  - function: `get\_rejected\_count` (line 542)
  - function: `get\_dataset\_lineage` (line 572)
- Imports:
  - **Standard Library** (12):
    - `import json as json` (line 14)
    - `from dataclasses import asdict` (line 15)
    - `from dataclasses import dataclass` (line 15)
    - `from dataclasses import field` (line 15)
    - `from datetime import datetime` (line 16)
    - `from datetime import timezone` (line 16)
    - `from enum import Enum` (line 17)
    - `from typing import Any` (line 18)
    - `from typing import Dict` (line 18)
    - `from typing import List` (line 18)
    - `from typing import Optional` (line 18)
    - `from uuid import uuid4` (line 19)
  - **Third-party** (2):
    - `import psycopg2 as psycopg2` (line 21)
    - `from psycopg2.extras import RealDictCursor` (line 22)
  - **Local/Project** (7):
    - `from __future__ import annotations` (line 12)
    - `from config import VERIFIED_DATA_DB_HOST` (line 24)
    - `from config import VERIFIED_DATA_DB_NAME` (line 24)
    - `from config import VERIFIED_DATA_DB_PASSWORD` (line 24)
    - `from config import VERIFIED_DATA_DB_PORT` (line 24)
    - `from config import VERIFIED_DATA_DB_USER` (line 24)
    - `from utils.logger_singleton import logger` (line 31)
- Task markers:
  - L58 **WARNING**: = "WARNING"
  - L290 **WARNING**: .value,
  - L333 **WARNING**: .value,
  - L361 **WARNING**: .value,
- Outgoing cross-module calls (sample):
  - dataclasses.asdict (line 87)
  - dataclasses.field (line 96)
  - dataclasses.field (line 114)
  - dataclasses.field (line 115)
  - psycopg2.connect (line 123)
  - utils.logger\_singleton.logger.error (line 132)
  - uuid.uuid4 (line 152)
  - conn.cursor (line 171)
  - cursor.execute (line 174)
  - cursor.execute (line 195)
  - json.dumps (line 205)
  - cursor.execute (line 209)
  - json.dumps (line 217)
  - conn.commit (line 224)
  - cursor.close (line 225)
  - conn.close (line 226)
  - utils.logger\_singleton.logger.info (line 228)
  - utils.logger\_singleton.logger.error (line 238)
  - issues.append (line 275)
  - row.items (line 286)
  - issues.append (line 288)
  - seen\_rows.add (line 295)
  - issues.append (line 305)
  - vote\_counts.append (line 313)
  - issues.append (line 315)
  - issues.append (line 331)
  - issues.append (line 345)
  - issues.append (line 359)
  - conn.cursor (line 395)
  - cursor.execute (line 398)
  - cursor.fetchone (line 403)
  - cursor.execute (line 408)
  - resolve\_issues.items (line 416)
  - cursor.execute (line 417)
  - cursor.execute (line 425)
  - json.dumps (line 433)
  - datetime.datetime.now (line 433)
  - conn.commit (line 436)
  - cursor.close (line 437)
  - conn.close (line 438)
  - utils.logger\_singleton.logger.info (line 440)
  - utils.logger\_singleton.logger.error (line 451)
  - conn.cursor (line 469)
  - cursor.execute (line 470)
  - cursor.fetchall (line 480)
  - cursor.close (line 481)
  - conn.close (line 482)
  - utils.logger\_singleton.logger.error (line 487)
  - conn.cursor (line 502)
  - params.append (line 514)
- Inbound references:
  - QAIssue ← data_classifier.py:275
  - QAIssue ← data_classifier.py:288
  - QAIssue ← data_classifier.py:305
  - QAIssue ← data_classifier.py:315
  - QAIssue ← data_classifier.py:331
  - QAIssue ← data_classifier.py:345
  - QAIssue ← data_classifier.py:359
  - ClassificationResult ← data_classifier.py:246
  - get\_db\_connection ← data_classifier.py:167
  - get\_db\_connection ← data_classifier.py:391
  - get\_db\_connection ← data_classifier.py:465
  - get\_db\_connection ← data_classifier.py:498
  - get\_db\_connection ← data_classifier.py:545
  - get\_db\_connection ← data_classifier.py:575
  - detect\_quality\_issues ← data_classifier.py:155

### quality\_assurance/qa\_endpoints.py {#webapp-parser-quality-assurance-qa-endpoints-py}

> Data Assurance Endpoints: REST API for DL1/DL2 Classification & Review

- Definitions:
  - function: `\_require\_qa\_enabled` (line 39)
  - function: `\_get\_reviewer\_principal` (line 50)
  - function: `\_get\_reviewer\_identity` (line 56)
  - function: `\_normalize\_required\_tier` (line 62)
  - function: `\_require\_reviewer` (line 74)
  - function: `\_require\_reviewer\_tier` (line 108)
  - function: `parse\_and\_classify` (line 148)
  - function: `get\_pending\_reviews` (line 241)
  - function: `verify\_and\_promote` (line 284)
  - function: `get\_inventory` (line 348)
  - function: `get\_lineage` (line 402)
  - function: `export\_dl2\_data` (line 451)
  - function: `get\_stats` (line 519)
- Imports:
  - **Standard Library** (4):
    - `import csv as csv` (line 15)
    - `import io as io` (line 16)
    - `from functools import wraps` (line 17)
    - `from io import StringIO` (line 18)
  - **Third-party** (4):
    - `from flask import Blueprint` (line 20)
    - `from flask import jsonify` (line 20)
    - `from flask import request` (line 20)
    - `from flask import send_file` (line 20)
  - **Local/Project** (15):
    - `from __future__ import annotations` (line 13)
    - `from config import ENABLE_VERIFICATION_FRAMEWORK` (line 22)
    - `from config import QA_REQUIRE_CERT_AUTH` (line 22)
    - `from utils.cert_utils import extract_client_principal` (line 23)
    - `from utils.privilege_tiers import PrivilegeTier` (line 24)
    - `from utils.privilege_tiers import get_principal_tier` (line 24)
    - `from utils.shared_logic import safe_get` (line 25)
    - `from utils.shared_logic import safe_strip` (line 25)
    - `from data_classifier import DatasetMetadata` (line 26)
    - `from data_classifier import classify_as_dl1` (line 26)
    - `from data_classifier import get_dataset_lineage` (line 26)
    - `from data_classifier import get_dl2_inventory` (line 26)
    - `from data_classifier import get_pending_dl2_reviews` (line 26)
    - `from data_classifier import get_rejected_count` (line 26)
    - `from data_classifier import promote_to_dl2` (line 26)
- Outgoing cross-module calls (sample):
  - flask.Blueprint (line 36)
  - flask.jsonify (line 44)
  - functools.wraps (line 41)
  - utils.cert\_utils.extract\_client\_principal (line 52)
  - utils.cert\_utils.extract\_client\_principal (line 58)
  - tier\_map.get (line 71)
  - flask.jsonify (line 82)
  - utils.privilege\_tiers.get\_principal\_tier (line 101)
  - functools.wraps (line 76)
  - flask.jsonify (line 122)
  - utils.privilege\_tiers.get\_principal\_tier (line 124)
  - flask.jsonify (line 126)
  - functools.wraps (line 113)
  - flask.request.get\_json (line 185)
  - utils.shared\_logic.safe\_strip (line 188)
  - utils.shared\_logic.safe\_get (line 188)
  - utils.shared\_logic.safe\_strip (line 189)
  - utils.shared\_logic.safe\_get (line 189)
  - utils.shared\_logic.safe\_strip (line 190)
  - utils.shared\_logic.safe\_get (line 190)
  - utils.shared\_logic.safe\_get (line 191)
  - utils.shared\_logic.safe\_get (line 192)
  - utils.shared\_logic.safe\_strip (line 193)
  - utils.shared\_logic.safe\_get (line 193)
  - utils.shared\_logic.safe\_get (line 194)
  - utils.shared\_logic.safe\_get (line 195)
  - utils.shared\_logic.safe\_get (line 196)
  - utils.shared\_logic.safe\_get (line 197)
  - utils.shared\_logic.safe\_get (line 198)
  - utils.shared\_logic.safe\_get (line 199)
  - flask.jsonify (line 203)
  - data\_classifier.DatasetMetadata (line 207)
  - data\_classifier.classify\_as\_dl1 (line 223)
  - flask.jsonify (line 225)
  - issue.to\_dict (line 229)
  - flask.jsonify (line 235)
  - qa\_bp.route (line 145)
  - data\_classifier.get\_pending\_dl2\_reviews (line 268)
  - flask.jsonify (line 271)
  - flask.jsonify (line 277)
  - qa\_bp.route (line 238)
  - flask.request.get\_json (line 312)
  - utils.shared\_logic.safe\_strip (line 313)
  - utils.shared\_logic.safe\_get (line 313)
  - utils.shared\_logic.safe\_strip (line 314)
  - utils.shared\_logic.safe\_get (line 314)
  - utils.shared\_logic.safe\_get (line 315)
  - flask.jsonify (line 318)
  - data\_classifier.promote\_to\_dl2 (line 321)
  - flask.jsonify (line 329)
- Inbound references:
  - \_get\_reviewer\_identity ← qa_endpoints.py:78
  - \_get\_reviewer\_identity ← qa_endpoints.py:120
  - \_require\_reviewer\_tier ← qa_endpoints.py:283

### quarantine\_endpoints.py {#webapp-parser-quarantine-endpoints-py}

> Quarantine Review Endpoints: Transparent UI for URL quarantine review.

- Definitions:
  - function: `\_require\_quarantine\_enabled` (line 28)
  - function: `\_get\_reviewer\_principal` (line 38)
  - function: `\_require\_reviewer` (line 44)
  - function: `get\_pending\_quarantines` (line 60)
  - function: `get\_quarantine\_detail` (line 114)
  - function: `submit\_quarantine\_review` (line 164)
  - function: `get\_quarantine\_stats` (line 232)
- Imports:
  - **Standard Library** (1):
    - `from functools import wraps` (line 13)
  - **Third-party** (3):
    - `from flask import Blueprint` (line 15)
    - `from flask import jsonify` (line 15)
    - `from flask import request` (line 15)
  - **Local/Project** (7):
    - `from __future__ import annotations` (line 11)
    - `from config import ENABLE_VERIFICATION_FRAMEWORK` (line 17)
    - `from health.quarantine_queue import ReviewStatus` (line 18)
    - `from health.quarantine_queue import get_quarantine_queue` (line 18)
    - `from utils.cert_utils import extract_client_principal` (line 22)
    - `from utils.shared_logic import safe_get` (line 23)
    - `from utils.shared_logic import safe_strip` (line 23)
- Outgoing cross-module calls (sample):
  - flask.Blueprint (line 25)
  - flask.jsonify (line 33)
  - functools.wraps (line 30)
  - utils.cert\_utils.extract\_client\_principal (line 40)
  - flask.jsonify (line 50)
  - functools.wraps (line 46)
  - health.quarantine\_queue.get\_quarantine\_queue (line 75)
  - queue.get\_pending (line 76)
  - result.append (line 80)
  - flask.jsonify (line 105)
  - quarantine\_bp.route (line 57)
  - health.quarantine\_queue.get\_quarantine\_queue (line 127)
  - queue.get\_pending (line 128)
  - flask.jsonify (line 132)
  - flask.jsonify (line 134)
  - quarantine\_bp.route (line 111)
  - flask.jsonify (line 187)
  - flask.request.get\_json (line 189)
  - utils.shared\_logic.safe\_strip (line 190)
  - utils.shared\_logic.safe\_get (line 190)
  - utils.shared\_logic.safe\_strip (line 191)
  - utils.shared\_logic.safe\_get (line 191)
  - utils.shared\_logic.safe\_strip (line 192)
  - utils.shared\_logic.safe\_get (line 192)
  - flask.jsonify (line 195)
  - health.quarantine\_queue.ReviewStatus (line 198)
  - flask.jsonify (line 200)
  - flask.jsonify (line 205)
  - health.quarantine\_queue.get\_quarantine\_queue (line 207)
  - queue.record\_review (line 208)
  - flask.jsonify (line 217)
  - flask.jsonify (line 219)
  - quarantine\_bp.route (line 161)
  - health.quarantine\_queue.get\_quarantine\_queue (line 241)
  - queue.get\_stats (line 242)
  - flask.jsonify (line 244)
  - stats.get (line 245)
  - stats.get (line 246)
  - stats.get (line 247)
  - quarantine\_bp.route (line 229)
- Inbound references:
  - \_get\_reviewer\_principal ← quarantine_endpoints.py:48
  - \_get\_reviewer\_principal ← quarantine_endpoints.py:185

### routes/\_\_init\_\_.py {#webapp-parser-routes-init-py}

- Imports:
  - **Local/Project** (12):
    - `from data_framework_blueprint import create_data_framework_blueprint`
      (line 1)
    - `from election_data_blueprint import create_election_data_blueprint` (line
      2)
    - `from fec_data_assurance_blueprint import
      create_fec_data_assurance_blueprint` (line 3)
    - `from file_io_blueprint import create_file_io_blueprint` (line 4)
    - `from health_blueprint import create_health_blueprint` (line 5)
    - `from observability_blueprint import create_observability_blueprint` (line
      6)
    - `from prometheus_metrics_blueprint import
      create_prometheus_metrics_blueprint` (line 7)
    - `from public_pages_blueprint import create_public_pages_blueprint` (line
      8)
    - `from session_orchestration_blueprint import
      create_session_orchestration_blueprint` (line 9)
    - `from ui_navigation_blueprint import create_ui_navigation_blueprint` (line
      10)
    - `from url_library_blueprint import create_url_library_blueprint` (line 11)
    - `from utility_admin_blueprint import create_utility_admin_blueprint` (line
      12)

### routes/data\_framework\_blueprint.py {#webapp-parser-routes-data-framework-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_data\_framework\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)
  - bp.route (line 45)
  - bp.route (line 49)
  - bp.route (line 53)
- Inbound references:
  - \_call\_handler ← data_framework_blueprint.py:31
  - \_call\_handler ← data_framework_blueprint.py:35
  - \_call\_handler ← data_framework_blueprint.py:39
  - \_call\_handler ← data_framework_blueprint.py:43
  - \_call\_handler ← data_framework_blueprint.py:47
  - \_call\_handler ← data_framework_blueprint.py:51
  - \_call\_handler ← data_framework_blueprint.py:55
  - \_call\_handler ← election_data_blueprint.py:31
  - \_call\_handler ← election_data_blueprint.py:35
  - \_call\_handler ← election_data_blueprint.py:39
  - \_call\_handler ← election_data_blueprint.py:43
  - \_call\_handler ← election_data_blueprint.py:47
  - \_call\_handler ← election_data_blueprint.py:51
  - \_call\_handler ← election_data_blueprint.py:55
  - \_call\_handler ← election_data_blueprint.py:59
  - \_call\_handler ← election_data_blueprint.py:63
  - \_call\_handler ← election_data_blueprint.py:67
  - \_call\_handler ← election_data_blueprint.py:71
  - \_call\_handler ← fec_data_assurance_blueprint.py:31
  - \_call\_handler ← fec_data_assurance_blueprint.py:35
  - \_call\_handler ← fec_data_assurance_blueprint.py:39
  - \_call\_handler ← fec_data_assurance_blueprint.py:43
  - \_call\_handler ← fec_data_assurance_blueprint.py:47
  - \_call\_handler ← fec_data_assurance_blueprint.py:51
  - \_call\_handler ← file_io_blueprint.py:31
  - \_call\_handler ← file_io_blueprint.py:35
  - \_call\_handler ← file_io_blueprint.py:39
  - \_call\_handler ← file_io_blueprint.py:43
  - \_call\_handler ← file_io_blueprint.py:47
  - \_call\_handler ← file_io_blueprint.py:51
  - \_call\_handler ← file_io_blueprint.py:55
  - \_call\_handler ← file_io_blueprint.py:59
  - \_call\_handler ← file_io_blueprint.py:63
  - \_call\_handler ← file_io_blueprint.py:67
  - \_call\_handler ← file_io_blueprint.py:71
  - \_call\_handler ← file_io_blueprint.py:75
  - \_call\_handler ← file_io_blueprint.py:79
  - \_call\_handler ← file_io_blueprint.py:83
  - \_call\_handler ← file_io_blueprint.py:87
  - \_call\_handler ← file_io_blueprint.py:91
  - \_call\_handler ← health_blueprint.py:31
  - \_call\_handler ← health_blueprint.py:35
  - \_call\_handler ← health_blueprint.py:39
  - \_call\_handler ← health_blueprint.py:43
  - \_call\_handler ← health_blueprint.py:47
  - \_call\_handler ← health_blueprint.py:51
  - \_call\_handler ← observability_blueprint.py:31
  - \_call\_handler ← observability_blueprint.py:35
  - \_call\_handler ← observability_blueprint.py:39
  - \_call\_handler ← observability_blueprint.py:43

### routes/election\_data\_blueprint.py {#webapp-parser-routes-election-data-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_election\_data\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)
  - bp.route (line 45)
  - bp.route (line 49)
  - bp.route (line 53)
  - bp.route (line 57)
  - bp.route (line 61)
  - bp.route (line 65)
  - bp.route (line 69)

### routes/fec\_data\_assurance\_blueprint.py {#webapp-parser-routes-fec-data-assurance-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_fec\_data\_assurance\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)
  - bp.route (line 45)
  - bp.route (line 49)

### routes/file\_io\_blueprint.py {#webapp-parser-routes-file-io-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_file\_io\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)
  - bp.route (line 45)
  - bp.route (line 49)
  - bp.route (line 53)
  - bp.route (line 57)
  - bp.route (line 61)
  - bp.route (line 65)
  - bp.route (line 69)
  - bp.route (line 73)
  - bp.route (line 77)
  - bp.route (line 81)
  - bp.route (line 85)
  - bp.route (line 89)

### routes/health\_blueprint.py {#webapp-parser-routes-health-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_health\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)
  - bp.route (line 45)
  - bp.route (line 49)

### routes/observability\_blueprint.py {#webapp-parser-routes-observability-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_observability\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)

### routes/prometheus\_metrics\_blueprint.py {#webapp-parser-routes-prometheus-metrics-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_prometheus\_metrics\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 34)

### routes/public\_pages\_blueprint.py {#webapp-parser-routes-public-pages-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_public\_pages\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)
  - bp.route (line 45)

### routes/route\_monitor.py {#webapp-parser-routes-route-monitor-py}

- Definitions:
  - function: `\_utc\_now\_iso` (line 11)
  - function: `record\_route\_monitor\_event` (line 15)
- Imports:
  - **Standard Library** (3):
    - `import threading as threading` (line 3)
    - `from datetime import datetime` (line 4)
    - `from datetime import timezone` (line 4)
  - **Third-party** (1):
    - `from flask import current_app` (line 6)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - threading.Lock (line 8)
  - datetime.datetime.now (line 12)
  - flask.current\_app.\_get\_current\_object (line 17)
  - monitor.setdefault (line 29)
  - routes.setdefault (line 30)
  - stats.get (line 39)
  - stats.get (line 41)
  - stats.get (line 43)
- Inbound references:
  - \_utc\_now\_iso ← route_monitor.py:25
  - \_utc\_now\_iso ← route_monitor.py:26
  - \_utc\_now\_iso ← route_monitor.py:45

### routes/session\_orchestration\_blueprint.py {#webapp-parser-routes-session-orchestration-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_session\_orchestration\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.get (line 29)
  - bp.route (line 33)

### routes/ui\_navigation\_blueprint.py {#webapp-parser-routes-ui-navigation-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_ui\_navigation\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)
  - bp.route (line 45)
  - bp.route (line 49)
  - bp.route (line 53)
  - bp.route (line 57)

### routes/url\_library\_blueprint.py {#webapp-parser-routes-url-library-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_url\_library\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)
  - bp.route (line 45)
  - bp.route (line 49)
  - bp.route (line 53)
  - bp.route (line 57)
  - bp.route (line 61)

### routes/utility\_admin\_blueprint.py {#webapp-parser-routes-utility-admin-blueprint-py}

- Definitions:
  - function: `\_call\_handler` (line 8)
  - function: `create\_utility\_admin\_blueprint` (line 26)
- Imports:
  - **Third-party** (3):
    - `from flask import Blueprint` (line 3)
    - `from flask import current_app` (line 3)
    - `from flask import jsonify` (line 3)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from route_monitor import record_route_monitor_event` (line 5)
- Outgoing cross-module calls (sample):
  - route\_monitor.record\_route\_monitor\_event (line 11)
  - flask.jsonify (line 12)
  - handlers.get (line 13)
  - route\_monitor.record\_route\_monitor\_event (line 15)
  - flask.jsonify (line 16)
  - route\_monitor.record\_route\_monitor\_event (line 19)
  - route\_monitor.record\_route\_monitor\_event (line 22)
  - flask.Blueprint (line 27)
  - bp.route (line 29)
  - bp.route (line 33)
  - bp.route (line 37)
  - bp.route (line 41)
  - bp.route (line 45)
  - bp.route (line 49)
  - bp.route (line 53)
  - bp.route (line 57)
  - bp.route (line 61)
  - bp.route (line 65)

### services/context\_service.py {#webapp-parser-services-context-service-py}

- Definitions:
  - class: `ContextBasedPredictor` (line 35)
  - class: `ContextService` (line 215)
- Imports:
  - **Standard Library** (11):
    - `import hashlib as hashlib` (line 1)
    - `import json as json` (line 2)
    - `import os as os` (line 3)
    - `import re as re` (line 4)
    - `from datetime import datetime` (line 5)
    - `from datetime import timezone` (line 5)
    - `from typing import Any` (line 6)
    - `from typing import Callable` (line 6)
    - `from typing import Dict` (line 6)
    - `from typing import List` (line 6)
    - `from typing import Optional` (line 6)
  - **Local/Project** (20):
    - `from Context_Integration.Context_Library.constants import
      CANDIDATE_KEYWORDS` (line 8)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_KEYWORDS` (line 8)
    - `from Context_Integration.Context_Library.constants import ELECTION_TYPES`
      (line 8)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_COUNTY_TO_PRECINCTS_MAP` (line 8)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_STATE_TO_COUNTY_MAP` (line 8)
    - `from Context_Integration.Context_Library.constants import PARTY_KEYWORDS`
      (line 8)
    - `from Context_Integration.Context_Library.constants import STATE_ABBR`
      (line 8)
    - `from Context_Integration.librarian import load_context_library` (line 17)
    - `from services.election_data_services import ElectionDataService` (line
      21)
    - `from utils.logger_singleton import logger` (line 22)
    - `from utils.logger_singleton import prompt` (line 22)
    - `from utils.shared_logic import PredictionResult` (line 23)
    - `from utils.shared_logic import normalize_county_name` (line 23)
    - `from utils.shared_logic import normalize_state_name` (line 23)
    - `from utils.shared_logic import resolve_county_alias` (line 23)
    - `from utils.shared_logic import safe_append` (line 23)
    - `from utils.shared_logic import safe_get` (line 23)
    - `from utils.spacy_utils import extract_dates` (line 31)
    - `from utils.spacy_utils import extract_entities` (line 31)
    - `from utils.spacy_utils import extract_locations` (line 31)
- Outgoing cross-module calls (sample):
  - services.election\_data\_services.ElectionDataService (line 38)
  - ModelRegistry.get\_sentence\_transformer (line 43)
  - text.lower (line 49)
  - utils.spacy\_utils.extract\_entities (line 52)
  - utils.spacy\_utils.extract\_dates (line 53)
  - utils.spacy\_utils.extract\_locations (line 54)
  - utils.shared\_logic.normalize\_state\_name (line 59)
  - Context\_Integration.Context\_Library.constants.STATE\_ABBR.items (line 62)
  - utils.shared\_logic.resolve\_county\_alias (line 65)
  - utils.shared\_logic.normalize\_county\_name (line 67)
  - difflib.get\_close\_matches (line 73)
  - Context\_Integration.Context\_Library.constants.KNOWN\_COUNTY\_TO\_PRECINCTS\_MAP.get
    (line 80)
  - re.search (line 88)
  - year\_match.group (line 90)
  - re.match (line 92)
  - ent.lower (line 101)
  - re.search (line 128)
  - ballot\_type\_match.group (line 130)
  - re.search (line 133)
  - vote\_method\_match.group (line 135)
  - re.search (line 138)
  - timestamp\_match.group (line 140)
  - re.match (line 142)
  - re.search (line 147)
  - url\_match.group (line 149)
  - self.\_get\_confidence\_keys (line 157)
  - result.get (line 158)
  - db\_c.get (line 158)
  - db\_c.get (line 162)
  - emb.tolist (line 169)
  - utils.logger\_singleton.logger.error (line 171)
  - self.\_estimate\_confidence (line 174)
  - asyncio.sleep (line 189)
  - self.predict (line 190)
  - context\_fields.append (line 202)
  - self.\_get\_confidence\_keys (line 206)
  - result.get (line 207)
  - result.get (line 209)
  - Context\_Integration.librarian.load\_context\_library (line 223)
  - self.\_compute\_version (line 224)
  - self.get\_all (line 236)
  - self.get\_all (line 239)
  - self.get\_all (line 242)
  - self.get\_all (line 245)
  - self.get\_all (line 248)
  - utils.shared\_logic.normalize\_state\_name (line 253)
  - utils.shared\_logic.resolve\_county\_alias (line 256)
  - self.get\_all (line 262)
  - f.write (line 266)
  - self.\_log\_audit (line 267)
- Inbound references:
  - ContextService ← context_service.py:37
  - ContextService ← context_service.py:431

### services/election\_data\_services.py {#webapp-parser-services-election-data-services-py}

> ElectionDataService: Service layer for all election DB operations.

- Definitions:
  - class: `DictConvertible` (line 66)
  - function: `get\_decl\_class\_registry` (line 83)
  - function: `iter\_orm\_classes` (line 92)
  - function: `get\_orm\_class\_by\_tablename` (line 100)
  - function: `get\_table\_columns` (line 109)
  - function: `get\_row\_table` (line 119)
  - function: `iter\_row\_columns` (line 125)
  - function: `row\_to\_dict` (line 134)
  - function: `\_get\_contest\_id` (line 146)
  - function: `columns\_to\_names` (line 164)
  - function: `get\_metadata\_tables` (line 170)
  - class: `ElectionDataService` (line 183)
- Imports:
  - **Standard Library** (8):
    - `from typing import Any` (line 9)
    - `from typing import Dict` (line 9)
    - `from typing import Iterator` (line 9)
    - `from typing import List` (line 9)
    - `from typing import Optional` (line 9)
    - `from typing import Protocol` (line 9)
    - `from typing import Type` (line 9)
    - `from typing import Union` (line 9)
  - **Third-party** (6):
    - `from sqlalchemy import inspect` (line 11)
    - `from sqlalchemy.engine import Engine` (line 12)
    - `from sqlalchemy.orm import DeclarativeMeta` (line 13)
    - `from sqlalchemy.orm import Session` (line 13)
    - `from sqlalchemy.sql.schema import Column` (line 14)
    - `from sqlalchemy.sql.schema import Table` (line 14)
  - **Local/Project** (45):
    - `from Context_Integration.librarian import clean_for_json` (line 16)
    - `from utils.db_utils import SessionLocal` (line 17)
    - `from utils.db_utils import check_missing_tables` (line 17)
    - `from utils.db_utils import create_batch_metadata` (line 17)
    - `from utils.db_utils import create_staging_election_result` (line 17)
    - `from utils.db_utils import create_warehouse_election_result` (line 17)
    - `from utils.db_utils import fetch_contest_full` (line 17)
    - `from utils.db_utils import fetch_table_structures` (line 17)
    - `from utils.db_utils import get_batch_metadata` (line 17)
    - `from utils.db_utils import get_engine` (line 17)
    - `from utils.db_utils import get_or_create_county` (line 17)
    - `from utils.db_utils import get_or_create_party` (line 17)
    - `from utils.db_utils import get_or_create_state` (line 17)
    - `from utils.db_utils import get_session` (line 17)
    - `from utils.db_utils import get_staging_results_by_batch` (line 17)
    - `from utils.db_utils import get_table_structure_from_db` (line 17)
    - `from utils.db_utils import get_warehouse_results_by_batch` (line 17)
    - `from utils.db_utils import save_table_structure_to_db` (line 17)
    - `from utils.db_utils import search_table_structures` (line 17)
    - `from utils.db_utils import select_table_structures_by_title` (line 17)
    - `from utils.db_utils import update_batch_metadata` (line 17)
    - `from utils.db_utils import update_table_structure_fields` (line 17)
    - `from utils.db_utils import upsert_contest` (line 17)
    - `from utils.logger_singleton import logger` (line 41)
    - `from utils.models import BallotType` (line 42)
    - `from utils.models import Base` (line 42)
    - `from utils.models import Button` (line 42)
    - `from utils.models import Candidate` (line 42)
    - `from utils.models import CandidatePanel` (line 42)
    - `from utils.models import Contest` (line 42)
    - `from utils.models import County` (line 42)
    - `from utils.models import District` (line 42)
    - `from utils.models import Heading` (line 42)
    - `from utils.models import LocationPanel` (line 42)
    - `from utils.models import Office` (line 42)
    - `from utils.models import Panel` (line 42)
    - `from utils.models import Party` (line 42)
    - `from utils.models import PartyLabel` (line 42)
    - `from utils.models import Result` (line 42)
    - `from utils.models import ResultsTimestamp` (line 42)
    - `from utils.models import State` (line 42)
    - `from utils.models import TableStructure` (line 42)
    - `from utils.models import VoteMethod` (line 42)
    - `from utils.shared_logic import safe_items` (line 63)
    - `from utils.shared_logic import safe_values` (line 63)
- Outgoing cross-module calls (sample):
  - k.startswith (line 79)
  - utils.shared\_logic.safe\_values (line 88)
  - row.as\_dict (line 143)
  - contest.get (line 149)
  - contest.get (line 151)
  - session.query (line 152)
  - filters.items (line 153)
  - q.filter (line 155)
  - q.first (line 156)
  - utils.logger\_singleton.logger.error (line 161)
  - utils.db\_utils.get\_session (line 199)
  - utils.db\_utils.fetch\_contest\_full (line 200)
  - utils.logger\_singleton.logger.error (line 202)
  - utils.db\_utils.get\_session (line 221)
  - session.query (line 222)
  - utils.shared\_logic.safe\_items (line 224)
  - query.filter (line 226)
  - query.limit (line 227)
  - query.with\_entities (line 230)
  - query.all (line 232)
  - query.all (line 238)
  - results.append (line 240)
  - results.append (line 243)
  - utils.logger\_singleton.logger.error (line 246)
  - self.get\_contests\_by\_advanced\_filter (line 253)
  - utils.db\_utils.get\_session (line 254)
  - utils.db\_utils.fetch\_contest\_full (line 255)
  - utils.db\_utils.get\_engine (line 262)
  - sqlalchemy.inspect (line 263)
  - inspector.get\_table\_names (line 264)
  - utils.db\_utils.SessionLocal (line 272)
  - session.query (line 274)
  - result.append (line 278)
  - Context\_Integration.librarian.clean\_for\_json (line 278)
  - utils.logger\_singleton.logger.error (line 281)
  - session.close (line 284)
  - utils.db\_utils.get\_session (line 289)
  - session.query (line 292)
  - utils.logger\_singleton.logger.error (line 304)
  - utils.db\_utils.get\_session (line 310)
  - session.query (line 311)
  - utils.logger\_singleton.logger.error (line 324)
  - utils.db\_utils.get\_session (line 330)
  - session.query (line 332)
  - utils.logger\_singleton.logger.error (line 345)
  - utils.db\_utils.get\_session (line 351)
  - session.query (line 353)
  - utils.logger\_singleton.logger.error (line 366)
  - utils.db\_utils.get\_session (line 372)
  - session.query (line 374)
- Inbound references:
  - get\_decl\_class\_registry ← election_data_services.py:96
  - iter\_orm\_classes ← election_data_services.py:104
  - get\_orm\_class\_by\_tablename ← election_data_services.py:268
  - get\_table\_columns ← election_data_services.py:72
  - get\_table\_columns ← election_data_services.py:235
  - get\_row\_table ← election_data_services.py:129
  - row\_to\_dict ← election_data_services.py:240
  - row\_to\_dict ← election_data_services.py:277
  - \_get\_contest\_id ← election_data_services.py:480
  - \_get\_contest\_id ← election_data_services.py:504
  - columns\_to\_names ← election_data_services.py:74
  - columns\_to\_names ← election_data_services.py:236
  - get\_metadata\_tables ← election_data_services.py:867
  - get\_metadata\_tables ← election_data_services.py:874
  - get\_metadata\_tables ← election_data_services.py:886

### socket\_ballot\_lens\_orchestration.py {#webapp-parser-socket-ballot-lens-orchestration-py}

- Definitions:
  - function: `\_normalize\_payload` (line 6)
  - function: `\_initialize\_session\_and\_auth` (line 10)
  - function: `\_prepare\_run\_inputs` (line 118)
  - function: `\_configure\_logging\_and\_prompt` (line 369)
  - function: `\_snapshot\_output\_artifacts` (line 406)
  - function: `\_detect\_new\_artifacts` (line 427)
  - function: `\_emit\_download\_ready\_for\_rel` (line 442)
  - function: `\_finalize\_worker\_session` (line 470)
  - function: `\_start\_pipeline\_worker` (line 528)
  - function: `run\_ballot\_lens\_socket\_handler` (line 681)
- Imports:
  - **Standard Library** (1):
    - `from typing import Any` (line 3)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Task markers:
  - L27 **WARNING**: ",
  - L102 **WARNING**: ({
  - L103 **WARNING**: ",
  - L169 **WARNING**: ({
  - L170 **WARNING**: ",
  - L197 **WARNING**: ({
  - L198 **WARNING**: ",
  - L213 **WARNING**: ({
  - L214 **WARNING**: ",
  - L224 **WARNING**: ({
  - L225 **WARNING**: ",
  - L233 **WARNING**: ({
  - L234 **WARNING**: ",
  - L262 **WARNING**: ({
  - L263 **WARNING**: ",
  - L269 **WARNING**: ({
  - L270 **WARNING**: ",
  - L280 **WARNING**: ({
  - L281 **WARNING**: ",
  - L341 **WARNING**: ({
  - L342 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - principal.startswith (line 40)
  - cert\_metadata.get (line 57)
  - meta.get (line 80)
  - raw\_manual\_upload\_path.replace (line 142)
  - candidate\_path.startswith (line 144)
  - ext.lstrip (line 154)
  - direct\_urls.append (line 221)
  - item.get (line 302)
  - item.get (line 307)
  - item.get (line 310)
  - line.strip (line 376)
  - lvl.upper (line 380)
  - output\_dir.exists (line 409)
  - output\_dir.rglob (line 412)
  - path.is\_file (line 414)
  - path.relative\_to (line 418)
  - path.stat (line 419)
  - artifacts\_after.keys (line 432)
  - artifacts\_after.items (line 437)
  - abs\_path.stat (line 450)
  - watcher\_stop.is\_set (line 596)
  - download\_ready\_emitted.is\_set (line 596)
  - artifacts\_after.get (line 602)
  - download\_ready\_emitted.set (line 604)
  - watcher\_thread.start (line 607)
  - download\_ready\_emitted.is\_set (line 640)
  - artifacts\_after.get (line 641)
  - download\_ready\_emitted.set (line 643)
  - watcher\_stop.set (line 660)
- Inbound references:
  - \_normalize\_payload ← socket_ballot_lens_orchestration.py:682
  - \_initialize\_session\_and\_auth ← socket_ballot_lens_orchestration.py:683
  - \_prepare\_run\_inputs ← socket_ballot_lens_orchestration.py:692
  - \_configure\_logging\_and\_prompt ← socket_ballot_lens_orchestration.py:693
  - \_snapshot\_output\_artifacts ← socket_ballot_lens_orchestration.py:593
  - \_snapshot\_output\_artifacts ← socket_ballot_lens_orchestration.py:598
  - \_snapshot\_output\_artifacts ← socket_ballot_lens_orchestration.py:638
  - \_detect\_new\_artifacts ← socket_ballot_lens_orchestration.py:599
  - \_detect\_new\_artifacts ← socket_ballot_lens_orchestration.py:639
  - \_emit\_download\_ready\_for\_rel ← socket_ballot_lens_orchestration.py:603
  - \_emit\_download\_ready\_for\_rel ← socket_ballot_lens_orchestration.py:642
  - \_finalize\_worker\_session ← socket_ballot_lens_orchestration.py:661
  - \_start\_pipeline\_worker ← socket_ballot_lens_orchestration.py:694

### state\_router.py {#webapp-parser-state-router-py}

- Top-of-file comments:

```python

# state\_router.py

# ===============================================

# Dynamically routes to the correct state or county-specific handler module

# Uses importlib for auto-resolution from folder structure.

# Now uses librarian.py for state/county mapping.

# Also provides state/county info for format\_router and download\_utils.

# ===============================================

```

- Definitions:
  - function: `\_guard\_context\_for\_db` (line 67)
  - function: `list\_available\_states` (line 84)
  - function: `list\_available\_counties` (line 96)
  - function: `import\_handler` (line 115)
  - function: `prompt\_for\_handler\_fallback` (line 159)
  - function: `preload\_handler\_map` (line 231)
  - function: `reload\_handler\_map` (line 258)
  - function: `scan\_url\_for\_state\_county` (line 265)
  - function: `fuzzy\_match\_handler` (line 302)
  - function: `list\_available\_handlers` (line 316)
  - function: `get\_handler` (line 361)
  - function: `cli` (line 535)
- Imports:
  - **Standard Library** (8):
    - `import os as os` (line 10)
    - `import time as time` (line 11)
    - `import traceback as traceback` (line 12)
    - `from typing import Any` (line 13)
    - `from typing import Dict` (line 13)
    - `from typing import List` (line 13)
    - `from typing import Optional` (line 13)
    - `from typing import Tuple` (line 13)
  - **Third-party** (1):
    - `import orjson as orjson` (line 15)
  - **Local/Project** (19):
    - `import difflib as difflib` (line 8)
    - `import importlib as importlib` (line 9)
    - `from config import BASE_DIR` (line 17)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_COUNTY_TO_PRECINCTS_MAP` (line 18)
    - `from Context_Integration.Context_Library.constants import
      STATE_MODULE_MAP` (line 18)
    - `from handlers.registry import apply_vendor_overrides` (line 22)
    - `from handlers.registry import get_county_handler_module_path` (line 22)
    - `from handlers.registry import get_state_handler_module_path` (line 22)
    - `from handlers.shared.parity_hooks import attach_router_parity_note` (line
      27)
    - `from handlers.shared.parity_hooks import safe_parity_note` (line 27)
    - `from utils.logger_singleton import console` (line 28)
    - `from utils.logger_singleton import logger` (line 28)
    - `from utils.logger_singleton import prompt` (line 28)
    - `from utils.shared_logic import normalize_county_name` (line 29)
    - `from utils.shared_logic import normalize_state_name` (line 29)
    - `from utils.shared_logic import safe_append` (line 29)
    - `from utils.shared_logic import safe_get_first` (line 29)
    - `from utils.shared_logic import safe_lower` (line 29)
    - `from utils.user_prompt import PromptCancelled` (line 36)
- Task markers:
  - L88 **WARNING**: ("\[Router\] handlers/states directory not found.")
  - L105 **WARNING**: (f"\[Router\] counties directory not found for state:
    {state_key}")
  - L176 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] No handler states
    available for manual selection.")
  - L193 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
  - L196 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
  - L199 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] State '{state}'
    not found. Please try again.")
  - L218 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] Aborted by user.")
  - L221 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] County '{county}'
    not found for state '{state}'. Please try again.")
  - L228 **WARNING**: (f"\[Fallback\]\[Session:{session_id}\] Too many failed
    attempts. Exiting fallback.")
  - L244 **WARNING**: (f"\[Router\] Requested state '{state_name}' not found on
    disk. Skipping restrict filter.")
  - L565 **WARNING**: (f"No counties found for state '{state}'. Try --fuzzy for
    fuzzy matching.")
  - L576 **WARNING**: (f"Failed to load context from file: {e}")
  - L586 **WARNING**: ("No suitable handler found.")
  - L593 **WARNING**: ("No handler selected. Exiting.")
  - L600 **WARNING**: ("Still could not import a suitable handler.")
- Outgoing cross-module calls (sample):
  - handlers.registry.apply\_vendor\_overrides (line 51)
  - context.get (line 72)
  - utils.logger\_singleton.logger.warning (line 88)
  - utils.shared\_logic.normalize\_state\_name (line 89)
  - Context\_Integration.Context\_Library.constants.STATE\_MODULE\_MAP.keys
    (line 89)
  - utils.shared\_logic.normalize\_state\_name (line 91)
  - os.listdir (line 92)
  - utils.shared\_logic.normalize\_state\_name (line 101)
  - utils.logger\_singleton.logger.warning (line 105)
  - os.listdir (line 108)
  - fname.endswith (line 109)
  - fname.startswith (line 109)
  - counties.append (line 110)
  - utils.shared\_logic.normalize\_county\_name (line 110)
  - counties.append (line 112)
  - utils.shared\_logic.normalize\_county\_name (line 112)
  - module\_or\_file\_path.endswith (line 127)
  - utils.logger\_singleton.logger.error (line 133)
  - utils.logger\_singleton.logger.info (line 134)
  - rel\_path.replace (line 138)
  - module\_path.endswith (line 139)
  - utils.logger\_singleton.logger.info (line 141)
  - importlib.import\_module (line 146)
  - utils.logger\_singleton.logger.error (line 150)
  - utils.logger\_singleton.logger.debug (line 151)
  - traceback.format\_exc (line 151)
  - utils.logger\_singleton.logger.info (line 152)
  - utils.logger\_singleton.logger.info (line 153)
  - utils.logger\_singleton.logger.error (line 156)
  - traceback.format\_exc (line 156)
  - utils.logger\_singleton.logger.warning (line 176)
  - utils.logger\_singleton.logger.error (line 181)
  - utils.logger\_singleton.prompt.prompt\_choice (line 184)
  - utils.logger\_singleton.logger.warning (line 193)
  - utils.logger\_singleton.logger.warning (line 196)
  - utils.logger\_singleton.logger.warning (line 199)
  - available\_counties\_dict.get (line 204)
  - utils.logger\_singleton.prompt.prompt\_choice (line 209)
  - utils.logger\_singleton.logger.warning (line 218)
  - utils.logger\_singleton.logger.warning (line 221)
  - utils.logger\_singleton.logger.info (line 225)
  - utils.logger\_singleton.logger.warning (line 228)
  - utils.shared\_logic.normalize\_state\_name (line 239)
  - states.append (line 242)
  - utils.logger\_singleton.logger.warning (line 244)
  - utils.shared\_logic.normalize\_state\_name (line 250)
  - utils.shared\_logic.normalize\_state\_name (line 253)
  - time.time (line 255)
  - utils.logger\_singleton.logger.info (line 256)
  - counties\_by\_state.values (line 256)
- Inbound references:
  - \_guard\_context\_for\_db ← state_router.py:461
  - list\_available\_states ← state_router.py:246
  - list\_available\_states ← state_router.py:248
  - list\_available\_states ← state_router.py:332
  - list\_available\_states ← state_router.py:588
  - list\_available\_counties ← state_router.py:250
  - list\_available\_counties ← state_router.py:333
  - list\_available\_counties ← state_router.py:589
  - import\_handler ← state_router.py:472
  - import\_handler ← state_router.py:482
  - import\_handler ← state_router.py:493
  - import\_handler ← state_router.py:522
  - import\_handler ← state_router.py:596
  - prompt\_for\_handler\_fallback ← state_router.py:512
  - prompt\_for\_handler\_fallback ← state_router.py:591
  - preload\_handler\_map ← state_router.py:262
  - preload\_handler\_map ← state_router.py:330
  - preload\_handler\_map ← state_router.py:378
  - reload\_handler\_map ← state_router.py:552
  - scan\_url\_for\_state\_county ← state_router.py:394
  - fuzzy\_match\_handler ← state_router.py:290
  - fuzzy\_match\_handler ← state_router.py:296
  - fuzzy\_match\_handler ← state_router.py:427
  - fuzzy\_match\_handler ← state_router.py:448
  - list\_available\_handlers ← state_router.py:555
  - list\_available\_handlers ← state_router.py:559
  - get\_handler ← state_router.py:580

### tests/test\_extract\_url.py {#webapp-parser-tests-test-extract-url-py}

- Definitions:
  - function: `test\_extract\_url\_and\_label\_cases` (line 23)
  - function: `test\_load\_urls\_integration` (line 28)
- Imports:
  - **Standard Library** (2):
    - `import tempfile as tempfile` (line 1)
    - `from pathlib import Path` (line 2)
  - **Third-party** (2):
    - `import pytest as pytest` (line 6)
    - `from webapp.parser.utils.misc_utils import extract_url_and_label` (line
      8)
  - **Local/Project** (1):
    - `import importlib as importlib` (line 4)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.misc\_utils.extract\_url\_and\_label (line 24)
  - importlib.import\_module (line 30)
  - f.write\_text (line 38)
  - monkeypatch.setattr (line 40)
  - pathlib.Path (line 40)
  - mod.load\_urls (line 41)

### tests/test\_fec\_handler.py {#webapp-parser-tests-test-fec-handler-py}

- Definitions:
  - function: `test\_party\_normalize` (line 7)
  - function: `test\_money\_and\_date\_normalize` (line 13)
  - function: `test\_handler\_parse\_fixture` (line 19)
- Imports:
  - **Standard Library** (1):
    - `import os as os` (line 1)
  - **Third-party** (2):
    - `from webapp.parser.handlers import fec_handler` (line 3)
    - `from webapp.parser.utils import fec_utils` (line 4)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.fec\_utils.party\_normalize (line 8)
  - webapp.parser.utils.fec\_utils.party\_normalize (line 9)
  - webapp.parser.utils.fec\_utils.party\_normalize (line 10)
  - webapp.parser.utils.fec\_utils.party\_normalize (line 10)
  - webapp.parser.utils.fec\_utils.money\_normalize (line 14)
  - webapp.parser.utils.fec\_utils.money\_normalize (line 15)
  - webapp.parser.utils.fec\_utils.date\_normalize (line 16)
  - webapp.parser.handlers.fec\_handler.parse (line 22)
  - first.get (line 28)
  - first.get (line 28)
  - first.get (line 29)
  - first.get (line 30)

### url\_parser.py {#webapp-parser-url-parser-py}

> URL Parser for Smart Elections Parser

- Definitions:
  - class: `UrlComponents` (line 64)
  - function: `extract\_root\_domain` (line 106)
  - function: `extract\_state\_from\_url` (line 136)
  - function: `extract\_county\_from\_url` (line 159)
  - function: `extract\_year\_from\_url` (line 184)
  - function: `detect\_contest\_type` (line 208)
  - function: `detect\_vendor\_hint` (line 219)
  - function: `find\_election\_keywords` (line 238)
  - function: `parse\_url\_components` (line 250)
  - function: `format\_url\_components\_for\_training` (line 319)
  - function: `parse\_url\_simple` (line 346)
- Imports:
  - **Standard Library** (11):
    - `import re as re` (line 14)
    - `from dataclasses import dataclass` (line 15)
    - `from datetime import datetime` (line 16)
    - `from datetime import timezone` (line 16)
    - `from typing import Dict` (line 17)
    - `from typing import List` (line 17)
    - `from typing import Optional` (line 17)
    - `from typing import Tuple` (line 17)
    - `from urllib.parse import parse_qs` (line 18)
    - `from urllib.parse import unquote` (line 18)
    - `from urllib.parse import urlparse` (line 18)
- Outgoing cross-module calls (sample):
  - re.compile (line 53)
  - re.compile (line 54)
  - re.compile (line 55)
  - re.compile (line 56)
  - re.compile (line 57)
  - re.compile (line 58)
  - re.compile (line 59)
  - domain.split (line 115)
  - code.lower (line 140)
  - domain.startswith (line 140)
  - code.lower (line 140)
  - segment.lower (line 145)
  - s.lower (line 146)
  - seg\_clean.upper (line 147)
  - name.replace (line 153)
  - segment.lower (line 163)
  - prev\_seg.lower (line 168)
  - prev\_seg.title (line 169)
  - re.search (line 171)
  - county\_match.group (line 173)
  - re.compile (line 176)
  - county\_pattern.search (line 177)
  - match.group (line 179)
  - re.search (line 191)
  - year\_match.group (line 193)
  - re.match (line 198)
  - re.search (line 201)
  - year\_match.group (line 203)
  - s.lower (line 210)
  - CONTEST\_PATTERNS.items (line 212)
  - pattern.search (line 213)
  - vendor\_patterns.items (line 231)
  - re.search (line 232)
  - s.lower (line 240)
  - found\_keywords.append (line 245)
  - urllib.parse.urlparse (line 261)
  - urllib.parse.unquote (line 271)
  - path.split (line 271)
  - urllib.parse.parse\_qs (line 278)
  - url.lower (line 284)
  - datetime.datetime.now (line 315)
- Inbound references:
  - UrlComponents ← url_parser.py:296
  - extract\_root\_domain ← url_parser.py:265
  - extract\_state\_from\_url ← url_parser.py:287
  - extract\_county\_from\_url ← url_parser.py:288
  - extract\_year\_from\_url ← url_parser.py:289
  - detect\_contest\_type ← url_parser.py:290
  - detect\_vendor\_hint ← url_parser.py:291
  - find\_election\_keywords ← url_parser.py:294
  - parse\_url\_components ← url_parser.py:356
  - format\_url\_components\_for\_training ← url_parser.py:357

### utils/audit\_trail\_router.py {#webapp-parser-utils-audit-trail-router-py}

> Audit Trail Router - Multi-Tier Compliance Logging

- Definitions:
  - function: `\_ensure\_audit\_logs` (line 46)
  - class: `AuditEntry` (line 69)
  - class: `ComplianceMetadata` (line 116)
  - function: `log\_decision\_with\_tier` (line 148)
  - function: `add\_event\_chain\_id` (line 211)
  - function: `summarize\_daily\_compliance` (line 223)
  - function: `write\_compliance\_summary` (line 306)
  - function: `get\_audit\_entries\_for\_chain` (line 347)
  - function: `get\_principal\_decisions` (line 377)
- Imports:
  - **Standard Library** (11):
    - `import json as json` (line 20)
    - `import os as os` (line 21)
    - `import threading as threading` (line 22)
    - `import uuid as uuid` (line 23)
    - `from dataclasses import dataclass` (line 24)
    - `from datetime import datetime` (line 25)
    - `from datetime import timezone` (line 25)
    - `from pathlib import Path` (line 26)
    - `from typing import Any` (line 27)
    - `from typing import Dict` (line 27)
    - `from typing import Optional` (line 27)
  - **Local/Project** (3):
    - `from __future__ import annotations` (line 18)
    - `from config import LOG_DIR` (line 29)
    - `from utils.logger_singleton import logger` (line 30)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 36)
  - pathlib.Path (line 37)
  - pathlib.Path (line 38)
  - pathlib.Path (line 39)
  - pathlib.Path (line 40)
  - threading.Lock (line 43)
  - log\_file.touch (line 56)
  - json.dumps (line 112)
  - json.dumps (line 141)
  - event.to\_json\_line (line 157)
  - logs\_to\_write.append (line 164)
  - logs\_to\_write.append (line 167)
  - logs\_to\_write.append (line 170)
  - logs\_to\_write.append (line 173)
  - logs\_to\_write.append (line 177)
  - f.write (line 183)
  - f.flush (line 184)
  - os.fsync (line 185)
  - f.fileno (line 185)
  - utils.logger\_singleton.logger.info (line 187)
  - utils.logger\_singleton.logger.error (line 201)
  - uuid.uuid4 (line 216)
  - datetime.datetime.now (line 234)
  - log\_file.exists (line 246)
  - line.strip (line 251)
  - json.loads (line 256)
  - entry.get (line 261)
  - ts.startswith (line 262)
  - entry.get (line 266)
  - entry.get (line 277)
  - entry.get (line 281)
  - decision.lower (line 282)
  - decision.lower (line 284)
  - pathlib.Path (line 288)
  - malicious\_log.exists (line 289)
  - utils.logger\_singleton.logger.error (line 297)
  - metadata.to\_json\_line (line 313)
  - f.write (line 317)
  - f.flush (line 318)
  - os.fsync (line 319)
  - f.fileno (line 319)
  - utils.logger\_singleton.logger.info (line 321)
  - utils.logger\_singleton.logger.error (line 334)
  - log\_file.exists (line 357)
  - line.strip (line 362)
  - json.loads (line 367)
  - entry.get (line 371)
  - entries.append (line 372)
  - log\_file.exists (line 390)
  - line.strip (line 395)
- Inbound references:
  - \_ensure\_audit\_logs ← audit_trail_router.py:61
  - ComplianceMetadata ← audit_trail_router.py:236
  - summarize\_daily\_compliance ← audit_trail_router.py:312

### utils/browser\_utils.py {#webapp-parser-utils-browser-utils-py}

- Definitions:
  - class: `Closable` (line 117)
  - function: `get\_random\_user\_agent` (line 122)
  - function: `safe\_url` (line 129)
  - function: `safe\_inner\_text` (line 138)
  - function: `safe\_locator` (line 163)
  - function: `safe\_evaluate` (line 174)
  - function: `safe\_wait\_for\_timeout` (line 208)
  - function: `safe\_content` (line 220)
  - function: `safe\_nth` (line 243)
  - function: `safe\_is\_visible` (line 250)
  - function: `safe\_is\_enabled` (line 261)
  - function: `safe\_click` (line 272)
  - function: `capture\_page\_diagnostics` (line 296)
  - function: `safe\_click\_with\_retry` (line 343)
  - function: `safe\_get\_attribute` (line 494)
  - function: `safe\_attributes` (line 506)
  - function: `safe\_query\_selector\_all` (line 576)
  - function: `safe\_context\_library` (line 587)
  - function: `safe\_count` (line 599)
  - function: `safe\_context\_result` (line 634)
  - function: `safe\_launch` (line 660)
  - async_function: `async\_safe\_launch` (line 680)
  - function: `safe\_new\_context` (line 699)
  - async_function: `async\_safe\_new\_context` (line 710)
  - function: `safe\_new\_page` (line 721)
  - async_function: `async\_safe\_new\_page` (line 732)
  - function: `safe\_goto` (line 743)
  - async_function: `async\_safe\_goto` (line 755)
  - async_function: `async\_safe\_browser\_close` (line 767)
  - async_function: `async\_launch\_browser` (line 781)
  - async_function: `async\_detect\_cloudflare\_captcha` (line 797)
  - async_function: `async\_browser\_pipeline` (line 805)
  - function: `sync\_launch\_browser` (line 815)
  - function: `sync\_detect\_cloudflare\_captcha` (line 852)
  - function: `sync\_safe\_browser\_close` (line 860)
  - function: `sync\_browser\_pipeline` (line 872)
  - function: `autoscroll\_until\_stable` (line 905)
  - function: `scan\_buttons\_with\_progress` (line 1087)
- Imports:
  - **Standard Library** (17):
    - `import asyncio as asyncio` (line 3)
    - `import inspect as inspect` (line 4)
    - `import json as json` (line 12)
    - `import os as os` (line 13)
    - `import random as random` (line 14)
    - `import re as re` (line 15)
    - `import time as time` (line 16)
    - `from datetime import datetime` (line 17)
    - `from typing import TYPE_CHECKING` (line 18)
    - `from typing import Any` (line 18)
    - `from typing import Dict` (line 18)
    - `from typing import Optional` (line 18)
    - `from typing import Protocol` (line 18)
    - `from typing import Sequence` (line 18)
    - `from typing import Tuple` (line 18)
    - `from typing import TypeVar` (line 18)
    - `from typing import Union` (line 18)
  - **Third-party** (14):
    - `from playwright.async_api import Browser as AsyncBrowser` (line 20)
    - `from playwright.async_api import BrowserContext as AsyncBrowserContext`
      (line 21)
    - `from playwright.async_api import BrowserType as AsyncBrowserType` (line
      22)
    - `from playwright.async_api import ElementHandle as AsyncElementHandle`
      (line 23)
    - `from playwright.async_api import Locator as AsyncLocator` (line 24)
    - `from playwright.async_api import Page as AsyncPage` (line 25)
    - `from playwright.async_api import async_playwright` (line 26)
    - `from playwright.sync_api import Browser as SyncBrowser` (line 27)
    - `from playwright.sync_api import BrowserContext as SyncBrowserContext`
      (line 28)
    - `from playwright.sync_api import BrowserType as SyncBrowserType` (line 29)
    - `from playwright.sync_api import ElementHandle as SyncElementHandle` (line
      30)
    - `from playwright.sync_api import Locator as SyncLocator` (line 31)
    - `from playwright.sync_api import Page as SyncPage` (line 32)
    - `from playwright.sync_api import sync_playwright` (line 33)
  - **Local/Project** (10):
    - `from __future__ import annotations` (line 1)
    - `from selectolax.parser import Node as SelectolaxNode` (line 34)
    - `from config import CONTEXT_LIBRARY_PATH` (line 49)
    - `from config import HEADLESS_DEFAULT` (line 49)
    - `from logger_singleton import console` (line 50)
    - `from logger_singleton import logger` (line 50)
    - `from logger_singleton import prompt` (line 50)
    - `from shared_logic import safe_get_first` (line 51)
    - `from shared_logic import safe_is_set` (line 51)
    - `from shared_logic import safe_lower` (line 51)
- Task markers:
  - L105 **WARNING**: (f"\[browser_utils\] Failed to safely parse
    context_library value for key '{key}'")
  - L107 **WARNING**: (f"\[browser_utils\] Skipping unsafe context_library value
    for key '{key}'")
  - L296 **NOTE**: str = "click_failure") -&gt; dict:
  - L308 **NOTE**: }\_\_{ts}.html")
  - L316 **NOTE**: }\_\_{ts}.png")
  - L404 **WARNING**: (f"\[safe_click_with_retry\] Re-query failed: {e} (attempt
    {attempt})")
  - L407 **WARNING**: (f"\[safe_click_with_retry\] No element found for
    selector={selector} (attempt {attempt})")
  - L459 **WARNING**: ({"level": "WARNING", "type": "browser", "message":
    f"Click attempt failed (attempt {attempt}/{max_retries}): {e}",
    "session_id": session_id})
  - L465 **WARNING**: (f"\[safe_click_with_retry\] Element has no click()
    (attempt {attempt})")
  - L471 **WARNING**: ({"level": "WARNING", "type": "browser", "message":
    f"Exception during click helper (attempt {attempt}): {e}", "session_id":
    session_id})
  - L477 **WARNING**: ({
  - L478 **WARNING**: ",
  - L488 **NOTE**: =(selector or 'element_click').replace('/', '_'))
  - L527 **WARNING**: (f"\[safe_attributes\] Playwright JS extraction failed:
    {e}")
  - L541 **WARNING**: (f"\[safe_attributes\] Playwright fallback extraction
    failed: {e}")
  - L627 **WARNING**: (f"\[safe_count\] Object is not countable: {type(obj)}")
  - L673 **WARNING**: (f"\[safe_launch\] browser_type is not a SyncBrowserType:
    {type(browser_type)}")
  - L693 **WARNING**: (f"\[async_safe_launch\] browser_type is not an
    AsyncBrowserType: {type(browser_type)}")
  - L772 **WARNING**: ({
  - L773 **WARNING**: ",
  - L801 **WARNING**: (f"\[CAPTCHA\] Detected Cloudflare CAPTCHA indicator:
    '{indicator}'")
  - L810 **WARNING**: (f"\[CAPTCHA\] CAPTCHA detected in async mode. Manual
    intervention not implemented. (Session: {session_id})")
  - L856 **WARNING**: (f"\[CAPTCHA\] Detected Cloudflare CAPTCHA indicator:
    '{indicator}'")
  - L865 **WARNING**: ({
  - L866 **WARNING**: ",
  - L890 **WARNING**: (f"\[CAPTCHA\] CAPTCHA detected in sync mode. Manual
    intervention not implemented. (Session: {session_id})")
  - L1045 **WARNING**: ("\[SCROLL\] User aborted scrolling.")
  - L1081 **WARNING**: ("\[SCROLL\] Max scroll time/attempts exceeded. Page may
    not be fully loaded.")
- Outgoing cross-module calls (sample):
  - typing.TypeVar (line 61)
  - json.dumps (line 72)
  - orjson.loads (line 87)
  - f.read (line 87)
  - context.get (line 90)
  - re.fullmatch (line 97)
  - ast.literal\_eval (line 100)
  - logger\_singleton.logger.warning (line 105)
  - logger\_singleton.logger.warning (line 107)
  - logger\_singleton.logger.error (line 113)
  - random.choice (line 124)
  - logger\_singleton.logger.error (line 135)
  - obj.count (line 146)
  - obj.inner\_text (line 151)
  - obj.inner\_text (line 153)
  - obj.inner\_text (line 155)
  - logger\_singleton.logger.error (line 156)
  - logger\_singleton.logger.error (line 160)
  - page.locator (line 167)
  - logger\_singleton.logger.error (line 171)
  - logger\_singleton.logger.error (line 189)
  - logger\_singleton.logger.error (line 192)
  - re.fullmatch (line 194)
  - script.strip (line 194)
  - logger\_singleton.logger.error (line 195)
  - obj.evaluate (line 199)
  - logger\_singleton.logger.error (line 201)
  - logger\_singleton.logger.error (line 205)
  - page.wait\_for\_timeout (line 212)
  - logger\_singleton.logger.error (line 217)
  - logger\_singleton.logger.error (line 224)
  - logger\_singleton.logger.error (line 228)
  - inspect.iscoroutinefunction (line 230)
  - asyncio.get\_event\_loop (line 232)
  - asyncio.new\_event\_loop (line 234)
  - asyncio.set\_event\_loop (line 235)
  - loop.run\_until\_complete (line 236)
  - logger\_singleton.logger.error (line 240)
  - element.is\_visible (line 254)
  - logger\_singleton.logger.error (line 258)
  - element.is\_enabled (line 265)
  - logger\_singleton.logger.error (line 269)
  - element.click (line 287)
  - logger\_singleton.logger.error (line 292)
  - datetime.datetime.utcnow (line 304)
  - fh.write (line 310)
  - page.screenshot (line 318)
  - out.get (line 321)
  - page.evaluate (line 328)
  - logger\_singleton.logger.info (line 337)
- Inbound references:
  - get\_random\_user\_agent ← browser_utils.py:782
  - get\_random\_user\_agent ← browser_utils.py:823
  - safe\_url ← browser_utils.py:953
  - safe\_inner\_text ← browser_utils.py:968
  - safe\_inner\_text ← browser_utils.py:970
  - safe\_inner\_text ← browser_utils.py:1093
  - safe\_inner\_text ← detect.py:433
  - safe\_inner\_text ← detect.py:446
  - safe\_inner\_text ← pattern_extractor.py:75
  - safe\_locator ← browser_utils.py:392
  - safe\_locator ← browser_utils.py:966
  - safe\_locator ← browser_utils.py:1000
  - safe\_locator ← browser_utils.py:1009
  - safe\_locator ← detect.py:428
  - safe\_locator ← detect.py:430
  - safe\_locator ← detect.py:431
  - safe\_locator ← detect.py:435
  - safe\_locator ← detect.py:437
  - safe\_locator ← detect.py:440
  - safe\_locator ← pattern_extractor.py:60
  - safe\_locator ← pattern_extractor.py:72
  - safe\_evaluate ← browser_utils.py:928
  - safe\_evaluate ← browser_utils.py:979
  - safe\_evaluate ← browser_utils.py:994
  - safe\_wait\_for\_timeout ← browser_utils.py:929
  - safe\_wait\_for\_timeout ← browser_utils.py:995
  - safe\_content ← browser_utils.py:307
  - safe\_nth ← detect.py:430
  - safe\_nth ← detect.py:433
  - safe\_nth ← detect.py:439
  - safe\_nth ← detect.py:447
  - safe\_nth ← pattern_extractor.py:66
  - safe\_nth ← pattern_extractor.py:75
  - capture\_page\_diagnostics ← browser_utils.py:488
  - safe\_click\_with\_retry ← browser_utils.py:277
  - safe\_count ← browser_utils.py:1001
  - safe\_count ← detect.py:429
  - safe\_count ← detect.py:432
  - safe\_count ← detect.py:436
  - safe\_count ← detect.py:438
  - safe\_count ← detect.py:441
  - safe\_count ← detect.py:444
  - safe\_count ← pattern_extractor.py:61
  - safe\_count ← pattern_extractor.py:74
  - safe\_launch ← browser_utils.py:832
  - async\_safe\_launch ← browser_utils.py:787
  - safe\_new\_context ← browser_utils.py:833
  - async\_safe\_new\_context ← browser_utils.py:788
  - safe\_new\_page ← browser_utils.py:834
  - async\_safe\_new\_page ← browser_utils.py:789

### utils/camelot\_utils.py {#webapp-parser-utils-camelot-utils-py}

- Definitions:
  - function: `\_normalize\_headers` (line 22)
  - function: `\_row\_is\_title\_noise` (line 40)
  - function: `\_table\_to\_rows` (line 44)
  - function: `\_score\_table` (line 67)
  - function: `attempt\_camelot\_extraction` (line 83)
  - function: `hybrid\_fill\_camelot` (line 118)
- Imports:
  - **Standard Library** (5):
    - `import re as re` (line 3)
    - `from typing import Any` (line 4)
    - `from typing import Dict` (line 4)
    - `from typing import List` (line 4)
    - `from typing import Tuple` (line 4)
  - **Local/Project** (4):
    - `from __future__ import annotations` (line 1)
    - `from Context_Integration.Context_Library.constants import
      build_camelot_row_filter` (line 12)
    - `from Context_Integration.Context_Library.constants import
      get_camelot_title_regex` (line 12)
    - `from salvage import normalize_ballot_column_name` (line 16)
- Outgoing cross-module calls (sample):
  - Context\_Integration.Context\_Library.constants.get\_camelot\_title\_regex
    (line 19)
  - Context\_Integration.Context\_Library.constants.build\_camelot\_row\_filter
    (line 20)
  - re.sub (line 29)
  - salvage.normalize\_ballot\_column\_name (line 30)
  - hs.lower (line 33)
  - seen.add (line 36)
  - hs.lower (line 36)
  - out.append (line 37)
  - df.head (line 54)
  - r.tolist (line 57)
  - v.strip (line 58)
  - rows.append (line 64)
  - h.lower (line 73)
  - r.values (line 75)
  - s.replace (line 77)
  - core.isdigit (line 78)
  - camelot.read\_pdf (line 89)
  - results.append (line 104)
  - results.sort (line 115)
- Inbound references:
  - \_normalize\_headers ← camelot_utils.py:52
  - \_row\_is\_title\_noise ← camelot_utils.py:62
  - \_table\_to\_rows ← camelot_utils.py:100
  - \_score\_table ← camelot_utils.py:103

### utils/captcha\_tools.py {#webapp-parser-utils-captcha-tools-py}

- Definitions:
  - class: `HasContent` (line 22)
  - class: `HasPageSource` (line 28)
  - class: `HasBringToFront` (line 35)
  - class: `HasMaximizeWindow` (line 41)
  - function: `detect\_cloudflare\_challenge` (line 57)
  - function: `get\_page\_content` (line 70)
  - function: `bring\_to\_front` (line 80)
  - function: `is\_cloudflare\_captcha\_present` (line 120)
  - function: `wait\_for\_user\_to\_solve\_captcha` (line 131)
  - function: `\_capture\_captcha\_dom\_state` (line 179)
  - function: `\_log\_captcha\_transition` (line 212)
- Imports:
  - **Standard Library** (6):
    - `import os as os` (line 4)
    - `import platform as platform` (line 5)
    - `import time as time` (line 11)
    - `from typing import Any` (line 12)
    - `from typing import Protocol` (line 12)
    - `from typing import runtime_checkable` (line 12)
  - **Third-party** (1):
    - `import orjson as orjson` (line 14)
  - **Local/Project** (7):
    - `from __future__ import annotations` (line 1)
    - `import ctypes as ctypes` (line 3)
    - `from config import CONTEXT_LIBRARY_PATH` (line 16)
    - `from config import DEFAULT_CAPTCHA_TIMEOUT` (line 16)
    - `from logger_singleton import logger` (line 17)
    - `from shared_logic import safe_get` (line 18)
    - `from shared_logic import safe_lower` (line 18)
- Task markers:
  - L118 **WARNING**: (f"\[CAPTCHA\] Foreground window fallback failed: {e}")
  - L175 **WARNING**: ("\[CAPTCHA\] CAPTCHA not resolved within timeout.")
- Outgoing cross-module calls (sample):
  - orjson.loads (line 49)
  - f.read (line 49)
  - shared\_logic.safe\_get (line 50)
  - logger\_singleton.logger.error (line 52)
  - shared\_logic.safe\_lower (line 64)
  - shared\_logic.safe\_lower (line 65)
  - logger\_singleton.logger.error (line 67)
  - page\_or\_driver.content (line 75)
  - platform.system (line 85)
  - page\_or\_driver.bring\_to\_front (line 88)
  - page\_or\_driver.maximize\_window (line 90)
  - logger\_singleton.logger.debug (line 106)
  - os.system (line 109)
  - logger\_singleton.logger.debug (line 111)
  - os.system (line 114)
  - logger\_singleton.logger.debug (line 116)
  - logger\_singleton.logger.warning (line 118)
  - shared\_logic.safe\_lower (line 125)
  - shared\_logic.safe\_lower (line 126)
  - logger\_singleton.logger.error (line 128)
  - logger\_singleton.logger.info (line 138)
  - time.time (line 139)
  - time.time (line 143)
  - logger\_singleton.logger.debug (line 150)
  - logger\_singleton.logger.info (line 153)
  - time.time (line 160)
  - logger\_singleton.logger.debug (line 163)
  - logger\_singleton.logger.debug (line 169)
  - time.sleep (line 170)
  - logger\_singleton.logger.error (line 173)
  - logger\_singleton.logger.warning (line 175)
  - kw.lower (line 197)
  - html\_content.lower (line 197)
  - time.time (line 205)
  - logger\_singleton.logger.debug (line 208)
  - initial\_state.get (line 238)
  - cleared\_state.get (line 239)
  - cleared\_state.get (line 241)
  - initial\_state.get (line 241)
  - initial\_state.get (line 242)
  - cleared\_state.get (line 243)
  - time.time (line 244)
  - os.makedirs (line 248)
  - f.write (line 251)
  - orjson.dumps (line 251)
  - f.write (line 252)
  - logger\_singleton.logger.debug (line 254)
  - initial\_state.get (line 254)
  - cleared\_state.get (line 255)
  - logger\_singleton.logger.debug (line 258)
- Inbound references:
  - get\_page\_content ← captcha_tools.py:64
  - get\_page\_content ← captcha_tools.py:125
  - get\_page\_content ← captcha_tools.py:191
  - bring\_to\_front ← captcha_tools.py:167
  - is\_cloudflare\_captcha\_present ← captcha_tools.py:152
  - \_capture\_captcha\_dom\_state ← captcha_tools.py:148
  - \_capture\_captcha\_dom\_state ← captcha_tools.py:156
  - \_log\_captcha\_transition ← captcha_tools.py:157

### utils/cert\_utils.py {#webapp-parser-utils-cert-utils-py}

- Definitions:
  - function: `\_sha256\_hex` (line 28)
  - function: `\_decode\_base64` (line 34)
  - function: `\_extract\_cert\_metadata` (line 41)
  - function: `extract\_client\_cert\_fingerprint` (line 135)
  - function: `extract\_sso\_principal` (line 163)
  - function: `extract\_client\_principal` (line 189)
- Imports:
  - **Standard Library** (10):
    - `import base64 as base64` (line 3)
    - `import hashlib as hashlib` (line 4)
    - `import json as json` (line 5)
    - `from datetime import datetime` (line 6)
    - `from datetime import timezone` (line 6)
    - `from typing import Any` (line 7)
    - `from typing import Dict` (line 7)
    - `from typing import Mapping` (line 7)
    - `from typing import Optional` (line 7)
    - `from typing import Tuple` (line 7)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - hashlib.sha256 (line 29)
  - h.update (line 30)
  - h.hexdigest (line 31)
  - base64.b64decode (line 36)
  - x509.load\_der\_x509\_certificate (line 68)
  - cert.not\_valid\_after (line 94)
  - expiry\_date.isoformat (line 95)
  - datetime.datetime.now (line 98)
  - cert.public\_key (line 106)
  - headers.get (line 143)
  - headers.get (line 143)
  - header.lower (line 143)
  - raw.strip (line 146)
  - value.encode (line 150)
  - value.encode (line 158)
  - headers.get (line 165)
  - headers.get (line 165)
  - SSO\_OID\_HEADER.lower (line 165)
  - oid.strip (line 167)
  - headers.get (line 168)
  - headers.get (line 168)
  - SSO\_PRINCIPAL\_HEADER.lower (line 168)
  - principal\_blob.strip (line 171)
  - json.loads (line 175)
  - decoded.decode (line 175)
  - payload.get (line 177)
  - claim.get (line 182)
  - claim.get (line 182)
  - claim.get (line 183)
- Inbound references:
  - \_sha256\_hex ← cert_utils.py:150
  - \_sha256\_hex ← cert_utils.py:154
  - \_sha256\_hex ← cert_utils.py:158
  - \_decode\_base64 ← cert_utils.py:152
  - \_decode\_base64 ← cert_utils.py:171
  - \_extract\_cert\_metadata ← cert_utils.py:155
  - extract\_client\_cert\_fingerprint ← cert_utils.py:195
  - extract\_sso\_principal ← cert_utils.py:198
  - extract\_client\_principal ← verification_endpoints.py:63
  - extract\_client\_principal ← verification_endpoints.py:73

### utils/confidence\_scorer.py {#webapp-parser-utils-confidence-scorer-py}

> Confidence Scoring System with Harmonic Ranking & Immediate Flush

- Definitions:
  - class: `ConfidenceLevel` (line 55)
  - class: `RunConfidence` (line 68)
  - class: `HarmonicScore` (line 148)
  - class: `MaliciousActFlag` (line 181)
  - class: `CriticalErrorSnapshot` (line 219)
  - function: `\_ensure\_log\_files` (line 255)
  - function: `flush\_run\_confidence` (line 273)
  - function: `flush\_malicious\_act` (line 309)
  - function: `flush\_critical\_error\_snapshot` (line 347)
  - function: `flush\_harmonic\_score` (line 384)
  - function: `store\_traceback\_in\_memory` (line 414)
  - function: `get\_traceback\_from\_memory` (line 440)
  - function: `clear\_traceback\_memory` (line 448)
  - function: `compute\_extraction\_confidence` (line 458)
  - function: `compute\_factor\_integrity\_confidence` (line 478)
  - function: `detect\_malicious\_act` (line 506)
  - function: `create\_run\_confidence` (line 535)
- Imports:
  - **Standard Library** (13):
    - `import os as os` (line 22)
    - `import threading as threading` (line 23)
    - `import time as time` (line 24)
    - `import uuid as uuid` (line 25)
    - `from dataclasses import dataclass` (line 26)
    - `from dataclasses import field` (line 26)
    - `from datetime import datetime` (line 27)
    - `from datetime import timezone` (line 27)
    - `from pathlib import Path` (line 28)
    - `from typing import Any` (line 29)
    - `from typing import Dict` (line 29)
    - `from typing import List` (line 29)
    - `from typing import Optional` (line 29)
  - **Local/Project** (3):
    - `from __future__ import annotations` (line 20)
    - `from config import LOG_DIR` (line 31)
    - `from utils.logger_singleton import logger` (line 32)
- Task markers:
  - L290 **WARNING**: ({
  - L291 **WARNING**: ",
  - L363 **WARNING**: ({
  - L364 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 38)
  - pathlib.Path (line 39)
  - pathlib.Path (line 40)
  - pathlib.Path (line 41)
  - pathlib.Path (line 42)
  - threading.Lock (line 45)
  - threading.Lock (line 49)
  - dataclasses.field (line 83)
  - dataclasses.field (line 161)
  - dataclasses.field (line 162)
  - reasons.append (line 202)
  - reasons.append (line 204)
  - reasons.append (line 206)
  - reasons.append (line 208)
  - log\_file.touch (line 265)
  - time.time (line 280)
  - confidence.to\_txt\_entry (line 282)
  - f.write (line 284)
  - f.flush (line 285)
  - os.fsync (line 286)
  - f.fileno (line 286)
  - time.time (line 288)
  - utils.logger\_singleton.logger.warning (line 290)
  - utils.logger\_singleton.logger.error (line 299)
  - time.time (line 316)
  - malicious.to\_txt\_entry (line 318)
  - f.write (line 320)
  - f.flush (line 321)
  - os.fsync (line 322)
  - f.fileno (line 322)
  - time.time (line 324)
  - utils.logger\_singleton.logger.critical (line 325)
  - utils.logger\_singleton.logger.error (line 337)
  - time.time (line 354)
  - error.to\_txt\_entry (line 356)
  - f.write (line 358)
  - f.flush (line 359)
  - os.fsync (line 360)
  - f.fileno (line 360)
  - time.time (line 362)
  - utils.logger\_singleton.logger.warning (line 363)
  - utils.logger\_singleton.logger.error (line 374)
  - score.to\_txt\_entry (line 392)
  - f.write (line 394)
  - f.flush (line 395)
  - os.fsync (line 396)
  - f.fileno (line 396)
  - utils.logger\_singleton.logger.error (line 400)
  - datetime.datetime.now (line 425)
  - \_TRACEBACK\_MEMORY.keys (line 435)
- Inbound references:
  - RunConfidence ← confidence_scorer.py:554
  - \_ensure\_log\_files ← confidence_scorer.py:270

### utils/contest\_detection.py {#webapp-parser-utils-contest-detection-py}

- Definitions:
  - function: `\_build\_contest\_regex` (line 19)
  - function: `\_should\_drop\_contest\_title` (line 81)
  - function: `detect\_contest\_titles\_from\_text` (line 99)
  - function: `gather\_lines\_for\_contest\_detection` (line 186)
- Imports:
  - **Standard Library** (5):
    - `import os as os` (line 5)
    - `import re as re` (line 6)
    - `from collections import Counter` (line 7)
    - `from typing import Iterable` (line 8)
    - `from typing import Pattern` (line 8)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_KEYWORDS` (line 10)
- Outgoing cross-module calls (sample):
  - phrase.strip (line 22)
  - re.split (line 24)
  - phrase.strip (line 24)
  - re.escape (line 27)
  - escaped.replace (line 28)
  - escaped.replace (line 29)
  - formatted.append (line 30)
  - parts.append (line 34)
  - re.compile (line 36)
  - re.compile (line 37)
  - re.compile (line 41)
  - ch.isalpha (line 85)
  - re.findall (line 88)
  - clean.lower (line 88)
  - collections.Counter (line 111)
  - match.split (line 116)
  - drop\_samples.append (line 136)
  - titles.append (line 142)
  - kept\_samples.append (line 144)
  - scanned\_lines.append (line 152)
  - CONTEST\_PATTERN.search (line 153)
  - CONTEST\_NAME\_REGEX.findall (line 154)
  - CONTEST\_PATTERN.search (line 162)
  - CONTEST\_NAME\_REGEX.findall (line 164)
  - diag\_bucket.update (line 172)
  - lines.append (line 200)
  - row.keys (line 211)
  - row.get (line 213)
  - tokens.append (line 218)
  - lines.append (line 223)
- Inbound references:
  - \_should\_drop\_contest\_title ← contest_detection.py:130

### utils/contest\_normalization.py {#webapp-parser-utils-contest-normalization-py}

> Utilities for normalizing contest titles (referenda, propositions, etc.).

- Definitions:
  - function: `\_split\_referendum\_title` (line 25)
  - function: `\_normalize\_candidate\_label` (line 57)
  - function: `normalize\_contest\_label` (line 63)
- Imports:
  - **Standard Library** (3):
    - `import re as re` (line 5)
    - `from typing import Optional` (line 6)
    - `from typing import Tuple` (line 6)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 3)
- Outgoing cross-module calls (sample):
  - re.compile (line 21)
  - re.compile (line 22)
  - text.strip (line 28)
  - text.find (line 32)
  - \_SHALL\_TOKEN\_RE.search (line 39)
  - shall\_match.start (line 40)
  - shall\_match.start (line 41)
  - shall\_match.start (line 42)
  - text.split (line 47)
  - first\_clause.strip (line 48)
  - candidate\_label.lower (line 49)
  - \_SEP\_RE.sub (line 60)
  - question.strip (line 86)
  - question.strip (line 94)
- Inbound references:
  - \_split\_referendum\_title ← contest_normalization.py:81
  - \_normalize\_candidate\_label ← contest_normalization.py:83
  - \_normalize\_candidate\_label ← contest_normalization.py:88
  - \_normalize\_candidate\_label ← contest_normalization.py:89
  - \_normalize\_candidate\_label ← contest_normalization.py:90
  - \_normalize\_candidate\_label ← pivot.py:799
  - \_normalize\_candidate\_label ← pivot.py:1106

### utils/contest\_selector.py {#webapp-parser-utils-contest-selector-py}

- Definitions:
  - function: `\_env\_truthy` (line 66)
  - class: `ContestRecord` (line 81)
  - function: `\_bundle\_key` (line 95)
  - function: `\_collect\_bundle\_members` (line 108)
  - function: `\_should\_bundle` (line 188)
  - function: `\_inject\_bundle\_records` (line 224)
  - function: `\_merge\_contest\_metadata` (line 279)
  - function: `\_extract\_first\_int` (line 378)
  - function: `\_contest\_sort\_key` (line 390)
  - function: `\_extract\_display\_details` (line 417)
  - function: `\_extract\_year\_tokens` (line 455)
  - function: `\_strip\_years` (line 458)
  - function: `\_base\_canonical\_key` (line 461)
  - function: `\_expand\_contests\_from\_context` (line 471)
  - function: `\_merge\_expanded\_contests` (line 528)
  - function: `\_cluster\_titles\_by\_base` (line 547)
  - function: `\_pick\_rep\_title` (line 564)
  - function: `\_score\_title` (line 576)
  - function: `\_chunk\_log\_options` (line 587)
  - function: `\_render\_paginated\_contest\_menu` (line 601)
  - function: `\_log` (line 638)
  - function: `\_norm\_key` (line 663)
  - function: `\_tokens` (line 669)
  - function: `\_jaccard` (line 672)
  - function: `\_cluster\_titles` (line 677)
  - function: `\_pick\_rep` (line 693)
  - function: `\_build\_effective\_list` (line 700)
  - function: `is\_markup\_like` (line 720)
  - function: `sanitize\_title` (line 730)
  - function: `\_remove\_boilerplate` (line 744)
  - function: `\_remove\_keywords` (line 763)
  - function: `\_stem\_and\_remove\_stopwords` (line 768)
  - function: `normalize\_contest` (line 775)
  - function: `extract\_year\_from\_title` (line 789)
  - function: `infer\_election\_type` (line 819)
  - function: `ensure\_contest` (line 876)
  - function: `ml\_verify\_contest` (line 893)
  - function: `feedback\_loop\_verify\_contests` (line 1045)
  - function: `resolve\_selection\_context` (line 1093)
  - function: `select\_contest\_auto\_first` (line 1148)
  - function: `select\_contest\_noninteractive` (line 1255)
  - function: `\_emit\_contest\_options\_to\_webapp` (line 1357)
  - function: `select\_contest` (line 1430)
- Imports:
  - **Standard Library** (13):
    - `import json as json` (line 3)
    - `import math as math` (line 4)
    - `import os as os` (line 5)
    - `import re as re` (line 8)
    - `from collections import defaultdict` (line 9)
    - `from dataclasses import asdict` (line 10)
    - `from dataclasses import dataclass` (line 10)
    - `from typing import TYPE_CHECKING` (line 12)
    - `from typing import Any` (line 12)
    - `from typing import Dict` (line 12)
    - `from typing import List` (line 12)
    - `from typing import Optional` (line 12)
    - `from typing import Tuple` (line 12)
  - **Third-party** (1):
    - `import numpy as np` (line 14)
  - **Local/Project** (22):
    - `from __future__ import annotations` (line 1)
    - `from difflib import get_close_matches` (line 11)
    - `from config import CONTEST_AUTO_CONFIDENCE_THRESHOLD` (line 16)
    - `from config import CONTEST_FEEDBACK_MIN_THRESHOLD` (line 16)
    - `from config import CONTEST_FEEDBACK_THRESHOLD` (line 16)
    - `from config import CONTEST_VERIFY_FLOOR_NO_MODEL` (line 16)
    - `from config import CONTEST_VERIFY_THRESHOLD` (line 16)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_KEYWORDS` (line 23)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_TITLE_KEYWORDS` (line 23)
    - `from Context_Integration.Context_Library.constants import
      ELECTION_TYPE_REGEX_MAP` (line 23)
    - `from Context_Integration.Context_Library.constants import ELECTION_TYPES`
      (line 23)
    - `from Context_Integration.Context_Library.constants import
      OFFICE_KEYWORDS` (line 23)
    - `from logger_singleton import logger` (line 30)
    - `from logger_singleton import prompt` (line 30)
    - `from shared_logic import normalize_county_name` (line 31)
    - `from shared_logic import normalize_state_name` (line 31)
    - `from shared_logic import safe_capitalize` (line 31)
    - `from shared_logic import safe_get` (line 31)
    - `from shared_logic import safe_lower` (line 31)
    - `from shared_logic import safe_model_encode` (line 31)
    - `from shared_logic import safe_strip` (line 31)
    - `from user_prompt import PromptCancelled` (line 40)
- Task markers:
  - L652 **WARNING**: ":
  - L653 **WARNING**: (entry)
  - L1060 **WARNING**: ", "selector", f"Feedback loop {loop+1}: verifying
    contests", session_id=session_id,
  - L1730 **WARNING**: ({"level": "WARNING", "type": "selector", "message":
    "Empty search term", "session_id": session_id})
  - L1735 **WARNING**: ({"level": "WARNING", "type": "selector", "message": f"No
    matches for '{term}'", "session_id": session_id})
  - L1807 **WARNING**: ({"level": "WARNING", "type": "selector", "message": "No
    match; try again.", "session_id": session_id})
- Outgoing cross-module calls (sample):
  - stopwords.words (line 54)
  - nltk.download (line 56)
  - stopwords.words (line 57)
  - value.strip (line 69)
  - os.getenv (line 72)
  - meta.get (line 98)
  - meta.get (line 98)
  - meta.get (line 100)
  - shared\_logic.safe\_lower (line 101)
  - meta.get (line 101)
  - meta.get (line 101)
  - shared\_logic.safe\_lower (line 102)
  - meta.get (line 102)
  - text\_s.lower (line 124)
  - existing.lower (line 126)
  - summary\_list.append (line 128)
  - meta.get (line 132)
  - meta.get (line 132)
  - union\_ids.update (line 133)
  - meta.get (line 134)
  - union\_counties.add (line 136)
  - meta.get (line 137)
  - union\_scopes.add (line 139)
  - meta.get (line 140)
  - union\_variants.add (line 142)
  - meta.get (line 143)
  - union\_vote\_for.add (line 145)
  - meta.get (line 146)
  - members\_serialized.append (line 152)
  - dataclasses.asdict (line 152)
  - bundle\_confidences.append (line 155)
  - bundle\_meta.setdefault (line 177)
  - meta.get (line 199)
  - meta.get (line 199)
  - union\_ids.add (line 202)
  - meta.get (line 203)
  - union\_counties.update (line 205)
  - meta.get (line 207)
  - union\_variants.add (line 209)
  - meta.get (line 210)
  - collections.defaultdict (line 226)
  - grouped.items (line 232)
  - output.extend (line 234)
  - primary\_meta.get (line 241)
  - primary\_meta.get (line 241)
  - bundle\_meta.get (line 243)
  - aggregate\_metadata.setdefault (line 248)
  - bundle\_meta.get (line 259)
  - output.append (line 263)
  - member\_meta.setdefault (line 270)
- Inbound references:
  - ContestRecord ← contest_selector.py:250
  - ContestRecord ← contest_selector.py:1310
  - ContestRecord ← contest_selector.py:1491
  - \_bundle\_key ← contest_selector.py:228
  - \_collect\_bundle\_members ← contest_selector.py:238
  - \_should\_bundle ← contest_selector.py:233
  - \_inject\_bundle\_records ← contest_selector.py:1345
  - \_inject\_bundle\_records ← contest_selector.py:1509
  - \_merge\_contest\_metadata ← contest_selector.py:1301
  - \_merge\_contest\_metadata ← contest_selector.py:1482
  - \_extract\_first\_int ← contest_selector.py:395
  - \_extract\_display\_details ← contest_selector.py:1381
  - \_extract\_display\_details ← contest_selector.py:1548
  - \_extract\_display\_details ← contest_selector.py:1635
  - \_extract\_year\_tokens ← contest_selector.py:568
  - \_extract\_year\_tokens ← contest_selector.py:1295
  - \_extract\_year\_tokens ← contest_selector.py:1476
  - \_strip\_years ← contest_selector.py:465
  - \_strip\_years ← contest_selector.py:573
  - \_strip\_years ← contest_selector.py:574
  - \_base\_canonical\_key ← contest_selector.py:99
  - \_base\_canonical\_key ← contest_selector.py:551
  - \_base\_canonical\_key ← contest_selector.py:555
  - \_base\_canonical\_key ← contest_selector.py:1316
  - \_base\_canonical\_key ← contest_selector.py:1495
  - \_expand\_contests\_from\_context ← contest_selector.py:1282
  - \_expand\_contests\_from\_context ← contest_selector.py:1457
  - \_merge\_expanded\_contests ← contest_selector.py:1284
  - \_merge\_expanded\_contests ← contest_selector.py:1459
  - \_cluster\_titles\_by\_base ← contest_selector.py:1291
  - \_cluster\_titles\_by\_base ← contest_selector.py:1472
  - \_pick\_rep\_title ← contest_selector.py:1294
  - \_pick\_rep\_title ← contest_selector.py:1475
  - \_score\_title ← contest_selector.py:1297
  - \_score\_title ← contest_selector.py:1478
  - \_log ← contest_selector.py:907
  - \_log ← contest_selector.py:1060
  - \_log ← contest_selector.py:1064
  - \_log ← contest_selector.py:1087
  - \_norm\_key ← contest_selector.py:466
  - \_norm\_key ← contest_selector.py:483
  - \_norm\_key ← contest_selector.py:521
  - \_norm\_key ← contest_selector.py:536
  - \_norm\_key ← contest_selector.py:538
  - \_norm\_key ← contest_selector.py:711
  - \_norm\_key ← contest_selector.py:1326
  - \_norm\_key ← contest_selector.py:1330
  - \_norm\_key ← pivot.py:1724
  - \_norm\_key ← pivot.py:1831
  - \_norm\_key ← pivot.py:1832

### utils/coordinator\_protocol.py {#webapp-parser-utils-coordinator-protocol-py}

- Definitions:
  - class: `CoordinatorProtocol` (line 7)
- Imports:
  - **Standard Library** (5):
    - `from typing import Any` (line 3)
    - `from typing import Mapping` (line 3)
    - `from typing import Protocol` (line 3)
    - `from typing import Sequence` (line 3)
    - `from typing import runtime_checkable` (line 3)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)

### utils/data\_comparator.py {#webapp-parser-utils-data-comparator-py}

- Definitions:
  - class: `ComparisonDifference` (line 9)
  - class: `ComparisonResult` (line 19)
  - class: `DataComparator` (line 31)
- Imports:
  - **Standard Library** (6):
    - `from dataclasses import asdict` (line 3)
    - `from dataclasses import dataclass` (line 3)
    - `from dataclasses import field` (line 3)
    - `from datetime import datetime` (line 4)
    - `from datetime import timezone` (line 4)
    - `from typing import Any` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - dataclasses.field (line 22)
  - dataclasses.field (line 23)
  - dataclasses.field (line 24)
  - dataclasses.field (line 25)
  - text.split (line 42)
  - payload.get (line 58)
  - payload.get (line 62)
  - payload.get (line 62)
  - row.get (line 73)
  - row.get (line 74)
  - row.get (line 75)
  - row.get (line 76)
  - self.\_canonical\_name (line 78)
  - self.\_to\_float (line 83)
  - row.get (line 83)
  - row.get (line 83)
  - row.get (line 83)
  - self.\_to\_float (line 84)
  - row.get (line 84)
  - row.get (line 84)
  - row.get (line 84)
  - row.get (line 85)
  - row.get (line 85)
  - row.get (line 85)
  - row.get (line 85)
  - row.get (line 85)
  - row.get (line 85)
  - tol.update (line 97)
  - tolerance.items (line 97)
  - self.\_normalize\_candidates (line 99)
  - self.\_normalize\_candidates (line 100)
  - dl1.items (line 109)
  - dl1\_row.get (line 118)
  - dl2\_row.get (line 119)
  - vote\_diffs.append (line 122)
  - candidate\_differences.append (line 124)
  - dl1\_row.get (line 136)
  - dl2\_row.get (line 137)
  - candidate\_differences.append (line 141)
  - dl1\_row.get (line 153)
  - dl2\_row.get (line 154)
  - candidate\_differences.append (line 156)
  - dl1\_row.get (line 160)
  - dl2\_row.get (line 161)
  - dl2.items (line 176)
  - failures.append (line 204)
  - failures.append (line 206)
  - self.evaluate\_regression (line 222)
  - datetime.datetime.now (line 229)
  - dataclasses.asdict (line 244)
- Inbound references:
  - ComparisonDifference ← data_comparator.py:125
  - ComparisonDifference ← data_comparator.py:142
  - ComparisonDifference ← data_comparator.py:157
  - ComparisonResult ← data_comparator.py:102

### utils/database\_comparison.py {#webapp-parser-utils-database-comparison-py}

> Database Comparison Utility

- Definitions:
  - function: `check\_existing\_finalized\_data` (line 17)
  - function: `\_check\_google\_sheets\_finalized\_data` (line 125)
  - function: `\_check\_warehouse\_database` (line 196)
  - function: `\_check\_verified\_datasets` (line 271)
- Imports:
  - **Standard Library** (3):
    - `from typing import Any` (line 12)
    - `from typing import Dict` (line 12)
    - `from typing import Optional` (line 12)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 10)
    - `from logger_singleton import logger` (line 14)
- Task markers:
  - L187 **WARNING**: ({
  - L188 **WARNING**: ",
  - L262 **WARNING**: ({
  - L263 **WARNING**: ",
  - L334 **WARNING**: ({
  - L335 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - logger\_singleton.logger.info (line 50)
  - logger\_singleton.logger.info (line 67)
  - logger\_singleton.logger.info (line 86)
  - logger\_singleton.logger.info (line 105)
  - logger\_singleton.logger.info (line 115)
  - client.fetch\_finalized\_data (line 138)
  - record.get (line 148)
  - record\_url.lower (line 153)
  - url.lower (line 153)
  - record.get (line 155)
  - record.get (line 156)
  - record.get (line 157)
  - record.keys (line 158)
  - k.lower (line 158)
  - record.get (line 175)
  - record.get (line 176)
  - record.get (line 177)
  - record.keys (line 178)
  - k.lower (line 178)
  - logger\_singleton.logger.warning (line 187)
  - inspector.get\_columns (line 215)
  - col.get (line 219)
  - col.get (line 219)
  - select\_cols.append (line 227)
  - aggregates.append (line 231)
  - engine.connect (line 244)
  - conn.execute (line 245)
  - row.get (line 250)
  - row.get (line 251)
  - row.get (line 252)
  - row.get (line 253)
  - row.get (line 254)
  - logger\_singleton.logger.warning (line 262)
  - inspector.get\_columns (line 290)
  - col.get (line 294)
  - col.get (line 294)
  - engine.connect (line 315)
  - conn.execute (line 316)
  - row.get (line 321)
  - row.get (line 322)
  - row.get (line 323)
  - row.get (line 324)
  - row.get (line 325)
  - row.get (line 326)
  - logger\_singleton.logger.warning (line 334)
- Inbound references:
  - check\_existing\_finalized\_data ← Smart_Elections_Parser_Webapp.py:3471
  - check\_existing\_finalized\_data ← Smart_Elections_Parser_Webapp.py:5995
  - check\_existing\_finalized\_data ← Smart_Elections_Parser_Webapp.py:6126
  - check\_existing\_finalized\_data ← html_election_parser.py:2754
  - \_check\_google\_sheets\_finalized\_data ← database_comparison.py:59
  - \_check\_warehouse\_database ← database_comparison.py:78
  - \_check\_verified\_datasets ← database_comparison.py:97

### utils/date\_utils.py {#webapp-parser-utils-date-utils-py}

> date_utils.py

- Definitions:
  - function: `is\_date\_like` (line 13)
- Imports:
  - **Standard Library** (1):
    - `import re as re` (line 7)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 5)
- Outgoing cross-module calls (sample):
  - re.compile (line 9)
  - re.compile (line 10)
  - re.compile (line 11)
  - text.strip (line 16)
  - \_DATE\_RE\_YEAR.search (line 17)
  - \_DATE\_RE\_MDY.search (line 17)
  - \_DATE\_RE\_ISO.search (line 17)

### utils/db\_utils.py {#webapp-parser-utils-db-utils-py}

- Definitions:
  - function: `robust\_orjson\_loads` (line 42)
  - function: `get\_session` (line 53)
  - function: `get\_engine` (line 65)
  - function: `update\_contest\_in\_db` (line 72)
  - function: `fetch\_contests\_by\_filter` (line 97)
  - function: `create\_all\_tables` (line 131)
  - function: `create\_batch\_metadata` (line 135)
  - function: `update\_batch\_metadata` (line 142)
  - function: `get\_batch\_metadata` (line 151)
  - function: `create\_staging\_election\_result` (line 156)
  - function: `get\_staging\_results\_by\_batch` (line 163)
  - function: `create\_warehouse\_election\_result` (line 168)
  - function: `get\_warehouse\_results\_by\_batch` (line 175)
  - function: `create\_table\_structure` (line 179)
  - function: `update\_table\_structure` (line 192)
  - function: `get\_table\_structure\_by\_id` (line 201)
  - function: `fetch\_table\_structures` (line 205)
  - function: `search\_table\_structures` (line 219)
  - function: `update\_table\_structure\_fields` (line 235)
  - function: `select\_table\_structures\_by\_title` (line 250)
  - function: `save\_table\_structure\_to\_db` (line 258)
  - function: `get\_table\_structure\_from\_db` (line 286)
  - function: `upsert\_contest` (line 307)
  - function: `get\_or\_create\_state` (line 367)
  - function: `get\_or\_create\_county` (line 375)
  - function: `get\_or\_create\_party` (line 383)
  - function: `fetch\_contest\_full` (line 391)
  - function: `check\_missing\_tables` (line 419)
- Imports:
  - **Standard Library** (5):
    - `import os as os` (line 3)
    - `from contextlib import contextmanager` (line 4)
    - `from typing import Generator` (line 5)
    - `from typing import List` (line 5)
    - `from typing import Optional` (line 5)
  - **Third-party** (11):
    - `import orjson as orjson` (line 11)
    - `from sqlalchemy import and_` (line 12)
    - `from sqlalchemy import create_engine` (line 12)
    - `from sqlalchemy import desc` (line 12)
    - `from sqlalchemy import inspect` (line 12)
    - `from sqlalchemy import or_` (line 12)
    - `from sqlalchemy import select` (line 12)
    - `from sqlalchemy import update` (line 12)
    - `from sqlalchemy.exc import SQLAlchemyError` (line 13)
    - `from sqlalchemy.orm import Session` (line 14)
    - `from sqlalchemy.orm import sessionmaker` (line 14)
  - **Local/Project** (13):
    - `from __future__ import annotations` (line 1)
    - `from config import get_sqlalchemy_engine` (line 16)
    - `from Context_Integration.librarian import clean_for_json` (line 17)
    - `from logger_singleton import logger` (line 18)
    - `from models import Base` (line 19)
    - `from models import BatchMetadata` (line 19)
    - `from models import Contest` (line 19)
    - `from models import County` (line 19)
    - `from models import Party` (line 19)
    - `from models import StagingElectionResult` (line 19)
    - `from models import State` (line 19)
    - `from models import TableStructure` (line 19)
    - `from models import WarehouseElectionResult` (line 19)
- Outgoing cross-module calls (sample):
  - sqlalchemy.create\_engine (line 36)
  - config.get\_sqlalchemy\_engine (line 38)
  - sqlalchemy.orm.sessionmaker (line 39)
  - orjson.loads (line 45)
  - orjson.loads (line 47)
  - val.encode (line 47)
  - session.commit (line 58)
  - session.rollback (line 60)
  - session.close (line 63)
  - session.get (line 81)
  - contest.get (line 81)
  - contest.get (line 83)
  - contest.get (line 84)
  - contest.get (line 85)
  - contest.get (line 86)
  - contest.get (line 87)
  - session.commit (line 89)
  - session.rollback (line 91)
  - session.close (line 95)
  - session.query (line 106)
  - filters.items (line 108)
  - query.filter (line 109)
  - query.order\_by (line 110)
  - sqlalchemy.desc (line 110)
  - contests.append (line 122)
  - session.close (line 126)
  - models.BatchMetadata (line 137)
  - session.add (line 138)
  - session.flush (line 139)
  - session.get (line 144)
  - kwargs.items (line 146)
  - session.commit (line 148)
  - session.get (line 153)
  - models.StagingElectionResult (line 158)
  - session.add (line 159)
  - session.flush (line 160)
  - session.query (line 165)
  - models.WarehouseElectionResult (line 170)
  - session.add (line 171)
  - session.flush (line 172)
  - session.query (line 177)
  - models.TableStructure (line 181)
  - session.add (line 188)
  - session.flush (line 189)
  - session.get (line 194)
  - kwargs.items (line 196)
  - session.commit (line 198)
  - session.get (line 203)
  - session.query (line 207)
  - filters.items (line 209)
- Inbound references:
  - get\_session ← dataset_promotion.py:288
  - get\_session ← db_utils.py:136
  - get\_session ← db_utils.py:143
  - get\_session ← db_utils.py:152
  - get\_session ← db_utils.py:157
  - get\_session ← db_utils.py:164
  - get\_session ← db_utils.py:169
  - get\_session ← db_utils.py:176
  - get\_session ← db_utils.py:180
  - get\_session ← db_utils.py:193
  - get\_session ← db_utils.py:202
  - get\_session ← db_utils.py:206
  - get\_session ← db_utils.py:224
  - get\_session ← db_utils.py:239
  - get\_session ← db_utils.py:254
  - get\_session ← db_utils.py:263
  - get\_session ← db_utils.py:291
  - get\_engine ← database_comparison.py:210
  - get\_engine ← database_comparison.py:285
  - get\_engine ← db_utils.py:421
  - get\_engine ← models.py:485
  - save\_table\_structure\_to\_db ← context_organizer.py:2208
  - get\_table\_structure\_from\_db ← context_organizer.py:2219
  - get\_or\_create\_state ← db_utils.py:318
  - get\_or\_create\_county ← db_utils.py:319

### utils/detect.py {#webapp-parser-utils-detect-py}

> detect.py

- Definitions:
  - function: `emit\_metric` (line 42)
  - class: `EntityInfo` (line 50)
  - class: `StructureInfo` (line 64)
  - function: `\_norm` (line 74)
  - function: `normalize\_text` (line 80)
  - function: `normalize\_for\_matching` (line 83)
  - function: `extract\_percent\_reported\_from\_heading` (line 89)
  - function: `\_is\_percent\_header` (line 100)
  - function: `\_should\_exclude\_as\_location` (line 106)
  - function: `\_is\_bad\_location\_fallback` (line 109)
  - function: `is\_location\_header` (line 115)
  - function: `dynamic\_detect\_location\_header` (line 124)
  - function: `detect\_candidate\_column` (line 178)
  - function: `nlp\_entity\_annotate\_table` (line 239)
  - function: `harmonize\_headers\_and\_data` (line 280)
  - function: `find\_best\_header` (line 379)
  - function: `is\_likely\_header` (line 393)
  - function: `parse\_numeric` (line 409)
  - function: `extract\_table\_data` (line 424)
  - function: `normalize\_header` (line 466)
  - function: `dedupe\_headers\_with\_suffix` (line 491)
  - function: `is\_total\_column` (line 504)
- Imports:
  - **Standard Library** (9):
    - `import re as re` (line 9)
    - `from dataclasses import dataclass` (line 11)
    - `from dataclasses import field` (line 11)
    - `from functools import lru_cache` (line 12)
    - `from typing import Any` (line 13)
    - `from typing import Dict` (line 13)
    - `from typing import List` (line 13)
    - `from typing import Optional` (line 13)
    - `from typing import Tuple` (line 13)
  - **Local/Project** (19):
    - `from __future__ import annotations` (line 6)
    - `import difflib as difflib` (line 8)
    - `import unicodedata as unicodedata` (line 10)
    - `from Context_Integration.Context_Library.constants import BALLOT_TYPES`
      (line 15)
    - `from Context_Integration.Context_Library.constants import
      BALLOT_TYPES_SORT_ORDER` (line 15)
    - `from Context_Integration.Context_Library.constants import
      CANDIDATE_KEYWORDS` (line 15)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_ABBREVIATIONS` (line 15)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_KEYWORDS` (line 15)
    - `from Context_Integration.Context_Library.constants import
      PERCENT_KEYWORDS` (line 15)
    - `from Context_Integration.Context_Library.constants import TOTAL_KEYWORDS`
      (line 15)
    - `from logger_singleton import logger` (line 24)
    - `from shared_logic import safe_add` (line 25)
    - `from shared_logic import safe_append` (line 25)
    - `from shared_logic import safe_get` (line 25)
    - `from shared_logic import safe_items` (line 25)
    - `from shared_logic import safe_keys` (line 25)
    - `from shared_logic import safe_lower` (line 25)
    - `from shared_logic import safe_strip` (line 25)
    - `from shared_logic import safe_translate` (line 25)
- Outgoing cross-module calls (sample):
  - re.compile (line 37)
  - re.compile (line 38)
  - re.compile (line 39)
  - logger\_singleton.logger.info (line 44)
  - dataclasses.field (line 51)
  - dataclasses.field (line 52)
  - dataclasses.field (line 53)
  - dataclasses.field (line 54)
  - dataclasses.field (line 55)
  - dataclasses.field (line 67)
  - dataclasses.field (line 68)
  - s.strip (line 75)
  - unicodedata.normalize (line 76)
  - re.sub (line 77)
  - functools.lru\_cache (line 73)
  - shared\_logic.safe\_lower (line 84)
  - shared\_logic.safe\_strip (line 84)
  - str.maketrans (line 85)
  - shared\_logic.safe\_translate (line 86)
  - PERCENT\_REPORTED\_RE.search (line 92)
  - m.group (line 94)
  - re.search (line 95)
  - re.fullmatch (line 113)
  - shared\_logic.safe\_get (line 127)
  - shared\_logic.safe\_get (line 128)
  - coordinator.extract\_entities (line 196)
  - shared\_logic.safe\_get (line 208)
  - coordinator.extract\_entities (line 214)
  - shared\_logic.safe\_get (line 228)
  - NAME\_LIKE\_RE.match (line 232)
  - value.strip (line 232)
  - coordinator.extract\_entities (line 245)
  - shared\_logic.safe\_add (line 248)
  - shared\_logic.safe\_add (line 250)
  - shared\_logic.safe\_items (line 255)
  - NUMBER\_LIKE\_RE.match (line 258)
  - v.replace (line 258)
  - shared\_logic.safe\_add (line 259)
  - shared\_logic.safe\_add (line 260)
  - bt.lower (line 261)
  - h.lower (line 261)
  - shared\_logic.safe\_add (line 262)
  - shared\_logic.safe\_add (line 263)
  - coordinator.extract\_entities (line 266)
  - shared\_logic.safe\_add (line 269)
  - shared\_logic.safe\_add (line 270)
  - shared\_logic.safe\_add (line 272)
  - shared\_logic.safe\_add (line 273)
  - shared\_logic.safe\_append (line 276)
  - all\_headers.update (line 290)
- Inbound references:
  - EntityInfo ← detect.py:240
  - \_norm ← detect.py:81
  - \_norm ← detector.py:52
  - \_norm ← detector.py:57
  - normalize\_text ← detect.py:107
  - normalize\_text ← detect.py:133
  - normalize\_text ← detect.py:139
  - normalize\_text ← detect.py:145
  - normalize\_text ← detect.py:151
  - normalize\_text ← detect.py:161
  - normalize\_text ← shared_logic.py:2282
  - normalize\_for\_matching ← detect.py:116
  - \_is\_percent\_header ← detect.py:110
  - \_is\_percent\_header ← detect.py:139
  - \_is\_percent\_header ← detect.py:145
  - \_is\_percent\_header ← detect.py:152
  - \_is\_percent\_header ← detect.py:162
  - \_should\_exclude\_as\_location ← detect.py:153
  - \_should\_exclude\_as\_location ← detect.py:163
  - is\_location\_header ← detect.py:313
  - extract\_table\_data ← extraction_strategies.py:88
  - extract\_table\_data ← extraction_strategies.py:127
  - normalize\_header ← detect.py:101
  - normalize\_header ← detect.py:102
  - normalize\_header ← detect.py:171
  - normalize\_header ← detect.py:171
  - normalize\_header ← detect.py:186
  - normalize\_header ← detect.py:189
  - normalize\_header ← detect.py:494

### utils/detector.py {#webapp-parser-utils-detector-py}

> detector.py

- Definitions:
  - function: `\_norm` (line 28)
  - function: `\_numeric\_like` (line 33)
  - class: `EntityAnnotation` (line 40)
  - class: `Detector` (line 46)
- Imports:
  - **Standard Library** (8):
    - `import re as re` (line 9)
    - `from dataclasses import dataclass` (line 11)
    - `from dataclasses import field` (line 11)
    - `from functools import lru_cache` (line 12)
    - `from typing import Any` (line 13)
    - `from typing import Dict` (line 13)
    - `from typing import List` (line 13)
    - `from typing import Optional` (line 13)
  - **Local/Project** (9):
    - `from __future__ import annotations` (line 6)
    - `import difflib as difflib` (line 8)
    - `import unicodedata as unicodedata` (line 10)
    - `from Context_Integration.Context_Library.constants import BALLOT_TYPES`
      (line 15)
    - `from Context_Integration.Context_Library.constants import
      CANDIDATE_KEYWORDS` (line 15)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_ABBREVIATIONS` (line 15)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_KEYWORDS` (line 15)
    - `from Context_Integration.Context_Library.constants import
      PERCENT_KEYWORDS` (line 15)
    - `from shared_logic import safe_add` (line 22)
- Outgoing cross-module calls (sample):
  - re.compile (line 24)
  - re.compile (line 25)
  - unicodedata.normalize (line 29)
  - re.sub (line 30)
  - functools.lru\_cache (line 27)
  - re.fullmatch (line 37)
  - dataclasses.field (line 41)
  - dataclasses.field (line 42)
  - dataclasses.field (line 43)
  - dataclasses.field (line 44)
  - self.norm (line 56)
  - \_PERCENT\_INLINE\_RE.search (line 64)
  - m.group (line 66)
  - re.search (line 67)
  - self.norm (line 73)
  - self.norm (line 74)
  - self.is\_location\_header (line 82)
  - self.is\_percent\_header (line 82)
  - self.norm (line 84)
  - difflib.SequenceMatcher (line 86)
  - self.norm (line 86)
  - self.is\_percent\_header (line 87)
  - self.is\_percent\_header (line 90)
  - self.is\_percent\_header (line 96)
  - self.norm (line 109)
  - self.norm (line 111)
  - row.get (line 130)
  - row.get (line 147)
  - \_NAME\_LIKE\_RE.match (line 151)
  - value.strip (line 151)
  - header.lower (line 157)
  - self.is\_percent\_header (line 160)
  - self.norm (line 168)
  - self.norm (line 170)
  - ballot\_type\_headers.append (line 171)
  - header.lower (line 175)
  - row.get (line 177)
  - ballot\_type\_headers.append (line 183)
  - bt.lower (line 190)
  - header.lower (line 190)
  - shared\_logic.safe\_add (line 191)
  - row.items (line 194)
  - shared\_logic.safe\_add (line 198)
  - self.is\_location\_header (line 199)
  - shared\_logic.safe\_add (line 200)
  - self.norm (line 206)
  - self.norm (line 206)
  - row.get (line 211)
  - candidates.add (line 213)
  - shared\_logic.safe\_add (line 218)
- Inbound references:
  - \_numeric\_like ← detector.py:181
  - \_numeric\_like ← detector.py:197
  - EntityAnnotation ← detector.py:188

### utils/dom\_extractor.py {#webapp-parser-utils-dom-extractor-py}

> dom_extractor.py

- Definitions:
  - function: `\_row\_score` (line 17)
  - function: `\_extract\_row\_cells` (line 23)
  - function: `\_pick\_header` (line 36)
  - function: `extract\_rows\_and\_headers\_from\_dom` (line 72)
  - function: `guess\_headers\_from\_row` (line 156)
- Imports:
  - **Standard Library** (5):
    - `import statistics as statistics` (line 8)
    - `from typing import Any` (line 9)
    - `from typing import Dict` (line 9)
    - `from typing import List` (line 9)
    - `from typing import Tuple` (line 9)
  - **Local/Project** (8):
    - `from __future__ import annotations` (line 6)
    - `from browser_utils import safe_count` (line 11)
    - `from browser_utils import safe_inner_text` (line 11)
    - `from browser_utils import safe_locator` (line 11)
    - `from browser_utils import safe_nth` (line 11)
    - `from detect import is_likely_header` (line 12)
    - `from detect import normalize_header` (line 12)
    - `from logger_singleton import logger` (line 13)
- Task markers:
  - L153 **WARNING**: (f"\[DOM_EXTRACTOR\] failure: {e}")
- Outgoing cross-module calls (sample):
  - c.strip (line 20)
  - browser\_utils.safe\_locator (line 24)
  - browser\_utils.safe\_count (line 25)
  - browser\_utils.safe\_inner\_text (line 28)
  - browser\_utils.safe\_nth (line 32)
  - out.append (line 33)
  - browser\_utils.safe\_inner\_text (line 33)
  - detect.is\_likely\_header (line 50)
  - detect.normalize\_header (line 51)
  - h.strip (line 62)
  - detect.normalize\_header (line 65)
  - seen.add (line 68)
  - detect.normalize\_header (line 68)
  - headers.append (line 69)
  - browser\_utils.safe\_locator (line 79)
  - browser\_utils.safe\_count (line 81)
  - browser\_utils.safe\_nth (line 83)
  - browser\_utils.safe\_locator (line 86)
  - browser\_utils.safe\_count (line 87)
  - browser\_utils.safe\_nth (line 93)
  - widths.append (line 98)
  - rows\_cells.append (line 99)
  - statistics.median (line 104)
  - row\_dict.values (line 123)
  - dict\_rows.append (line 125)
  - row.get (line 129)
  - best.update (line 133)
  - diagnostics.update (line 142)
  - best.get (line 145)
  - best.get (line 146)
  - best.get (line 147)
  - logger\_singleton.logger.warning (line 153)
  - c.strip (line 161)
  - detect.normalize\_header (line 164)
  - seen.add (line 167)
  - detect.normalize\_header (line 167)
  - headers.append (line 168)
- Inbound references:
  - \_row\_score ← dom_extractor.py:129
  - \_extract\_row\_cells ← dom_extractor.py:94
  - \_extract\_row\_cells ← dom_extractor.py:157
  - \_pick\_header ← dom_extractor.py:112

### utils/download\_utils.py {#webapp-parser-utils-download-utils-py}

- Definitions:
  - function: `ensure\_input\_directory` (line 22)
  - function: `ensure\_output\_directory` (line 26)
  - function: `load\_download\_manifest` (line 30)
  - function: `\_normalize\_download\_url` (line 46)
  - function: `\_retry\_step\_back\_url` (line 57)
  - function: `update\_download\_manifest` (line 87)
  - function: `is\_already\_downloaded` (line 92)
  - function: `download\_file` (line 112)
  - function: `download\_multiple\_files` (line 222)
  - function: `download\_confirmed\_file` (line 253)
  - function: `summarize\_downloads` (line 278)
  - function: `get\_downloaded\_files\_by\_status` (line 289)
- Imports:
  - **Standard Library** (7):
    - `import os as os` (line 7)
    - `import re as re` (line 8)
    - `from datetime import datetime` (line 9)
    - `from urllib.parse import urljoin` (line 10)
    - `from urllib.parse import urlparse` (line 10)
    - `from urllib.parse import urlsplit` (line 10)
    - `from urllib.parse import urlunsplit` (line 10)
  - **Third-party** (2):
    - `import orjson as orjson` (line 12)
    - `import requests as requests` (line 13)
  - **Local/Project** (11):
    - `from __future__ import annotations` (line 1)
    - `from config import DOWNLOAD_MANIFEST` (line 15)
    - `from config import INPUT_DIR` (line 15)
    - `from config import MAX_DOWNLOAD_BYTES` (line 15)
    - `from config import OUTPUT_DIR` (line 15)
    - `from config import URL_MAX_REDIRECTS` (line 15)
    - `from Context_Integration.context_organizer import ContextOrganizer` (line
      16)
    - `from utils.logger_singleton import logger` (line 17)
    - `from utils.misc_utils import file_hash` (line 18)
    - `from utils.shared_logic import safe_get` (line 19)
    - `from utils.shared_logic import safe_validate_external_url` (line 19)
- Task markers:
  - L144 **WARNING**: (f"\[DOWNLOAD\] HTTP {response.status_code} for
    {file_url}, trying fallback URLs...")
- Outgoing cross-module calls (sample):
  - os.makedirs (line 24)
  - os.makedirs (line 28)
  - orjson.loads (line 38)
  - utils.shared\_logic.safe\_get (line 39)
  - utils.shared\_logic.safe\_get (line 39)
  - urllib.parse.urlsplit (line 49)
  - re.sub (line 52)
  - urllib.parse.urlunsplit (line 53)
  - urllib.parse.urlsplit (line 61)
  - path.endswith (line 64)
  - path.rstrip (line 65)
  - candidates.append (line 66)
  - urllib.parse.urlunsplit (line 66)
  - path.split (line 68)
  - candidates.append (line 71)
  - urllib.parse.urlunsplit (line 71)
  - candidates.append (line 75)
  - urllib.parse.urlunsplit (line 75)
  - seen.add (line 83)
  - deduped.append (line 84)
  - f.write (line 90)
  - orjson.dumps (line 90)
  - utils.shared\_logic.safe\_get (line 95)
  - utils.shared\_logic.safe\_get (line 97)
  - utils.misc\_utils.file\_hash (line 98)
  - manifest.values (line 103)
  - utils.shared\_logic.safe\_get (line 104)
  - utils.shared\_logic.safe\_get (line 105)
  - utils.misc\_utils.file\_hash (line 106)
  - urllib.parse.urljoin (line 127)
  - utils.logger\_singleton.logger.info (line 130)
  - utils.logger\_singleton.logger.info (line 132)
  - utils.shared\_logic.safe\_validate\_external\_url (line 136)
  - requests.get (line 140)
  - utils.logger\_singleton.logger.warning (line 144)
  - utils.logger\_singleton.logger.info (line 147)
  - requests.get (line 149)
  - utils.logger\_singleton.logger.info (line 151)
  - utils.logger\_singleton.logger.debug (line 155)
  - response.raise\_for\_status (line 158)
  - urllib.parse.urlparse (line 163)
  - urllib.parse.urlparse (line 168)
  - utils.shared\_logic.safe\_validate\_external\_url (line 171)
  - urllib.parse.urlparse (line 176)
  - response.iter\_content (line 189)
  - f.write (line 195)
  - utils.misc\_utils.file\_hash (line 196)
  - utils.logger\_singleton.logger.info (line 197)
  - datetime.datetime.now (line 201)
  - Context\_Integration.context\_organizer.ContextOrganizer (line 207)
- Inbound references:
  - ensure\_input\_directory ← download_utils.py:124
  - ensure\_input\_directory ← download_utils.py:238
  - load\_download\_manifest ← download_utils.py:94
  - load\_download\_manifest ← download_utils.py:280
  - load\_download\_manifest ← download_utils.py:291
  - \_normalize\_download\_url ← download_utils.py:129
  - \_normalize\_download\_url ← format_router.py:302
  - \_retry\_step\_back\_url ← download_utils.py:145
  - \_retry\_step\_back\_url ← format_router.py:1019
  - update\_download\_manifest ← download_utils.py:205
  - update\_download\_manifest ← download_utils.py:219
  - is\_already\_downloaded ← download_utils.py:131
  - download\_file ← download_utils.py:241
  - download\_file ← download_utils.py:269
  - summarize\_downloads ← format_router.py:920

### utils/dynamic\_table\_extractor.py {#webapp-parser-utils-dynamic-table-extractor-py}

- Definitions:
  - function: `\_emit` (line 86)
  - function: `dynamic\_table\_extractor` (line 109)
  - function: `find\_tabular\_candidates` (line 193)
  - function: `analyze\_candidate\_nlp` (line 278)
  - function: `score\_candidate` (line 304)
  - function: `remove\_low\_signal\_columns` (line 392)
  - function: `infer\_column\_types` (line 407)
  - function: `advanced\_party\_candidate\_detection` (line 473)
  - function: `extract\_candidates\_and\_parties` (line 492)
  - function: `entity\_linking` (line 543)
  - function: `find\_tables\_with\_headings` (line 597)
  - function: `discover\_container\_selectors` (line 714)
  - function: `log\_new\_dom\_pattern` (line 761)
  - function: `review\_dom\_patterns` (line 776)
  - function: `auto\_approve\_dom\_pattern` (line 822)
  - function: `find\_tables\_with\_panel\_headings` (line 840)
  - function: `find\_tables\_with\_section\_headings` (line 910)
  - function: `is\_candidate\_major\_row` (line 986)
  - function: `is\_candidate\_major\_col` (line 1030)
  - function: `is\_precinct\_major` (line 1060)
  - function: `is\_flat\_candidate\_table` (line 1078)
  - function: `is\_single\_row\_summary` (line 1104)
  - function: `is\_candidate\_footer` (line 1110)
  - function: `detect\_wide\_vs\_long` (line 1129)
  - function: `classify\_ambiguous\_tables` (line 1140)
- Imports:
  - **Standard Library** (8):
    - `import os as os` (line 19)
    - `import re as re` (line 20)
    - `from typing import TYPE_CHECKING` (line 21)
    - `from typing import Any` (line 21)
    - `from typing import Dict` (line 21)
    - `from typing import List` (line 21)
    - `from typing import Optional` (line 21)
    - `from typing import Tuple` (line 21)
  - **Third-party** (2):
    - `import numpy as np` (line 24)
    - `import orjson as orjson` (line 25)
  - **Local/Project** (49):
    - `from __future__ import annotations` (line 1)
    - `import difflib as difflib` (line 18)
    - `import dateutil.parser as dateutil` (line 23)
    - `from selectolax.parser import HTMLParser` (line 26)
    - `from config import ENTITY_LINKING_THRESHOLD` (line 28)
    - `from Context_Integration.Context_Library.constants import BALLOT_TYPES`
      (line 29)
    - `from Context_Integration.Context_Library.constants import
      BALLOT_TYPES_SORT_ORDER` (line 29)
    - `from Context_Integration.Context_Library.constants import
      CANDIDATE_KEYWORDS` (line 29)
    - `from Context_Integration.Context_Library.constants import
      CONTAINER_EXTRA_KEYWORDS` (line 29)
    - `from Context_Integration.Context_Library.constants import
      CONTAINER_FALLBACK_SELECTORS` (line 29)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_KEYWORDS` (line 29)
    - `from Context_Integration.Context_Library.constants import
      EXTRA_HEADING_TAGS` (line 29)
    - `from Context_Integration.Context_Library.constants import HEADING_TAGS`
      (line 29)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_ABBREVIATIONS` (line 29)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_KEYWORDS` (line 29)
    - `from Context_Integration.Context_Library.constants import
      MISC_FOOTER_KEYWORDS` (line 29)
    - `from Context_Integration.Context_Library.constants import
      NLP_SKIP_PHRASES` (line 29)
    - `from Context_Integration.Context_Library.constants import PANEL_TAGS`
      (line 29)
    - `from Context_Integration.Context_Library.constants import PARTY_KEYWORDS`
      (line 29)
    - `from Context_Integration.Context_Library.constants import TOTAL_KEYWORDS`
      (line 29)
    - `from Context_Integration.librarian import extend_heading_tags` (line 46)
    - `from Context_Integration.librarian import extend_panel_tags` (line 46)
    - `from Context_Integration.librarian import get_safe_log_path` (line 46)
    - `from Context_Integration.librarian import log_unknown_tag` (line 46)
    - `from browser_utils import safe_count` (line 52)
    - `from browser_utils import safe_evaluate` (line 52)
    - `from browser_utils import safe_get_attribute` (line 52)
    - `from browser_utils import safe_inner_text` (line 52)
    - `from browser_utils import safe_locator` (line 52)
    - `from browser_utils import safe_nth` (line 52)
    - `from date_utils import is_date_like` (line 60)
    - `from detect import extract_table_data` (line 61)
    - `from detect import is_location_header` (line 61)
    - `from detect import normalize_header` (line 61)
    - `from detect import normalize_text` (line 61)
    - `from dom_extractor import extract_rows_and_headers_from_dom` (line 62)
    - `from dom_extractor import guess_headers_from_row` (line 62)
    - `from logger_singleton import logger` (line 63)
    - `from pattern_extractor import extract_with_patterns` (line 64)
    - `from pattern_extractor import load_dom_patterns` (line 64)
    - `from shared_logic import safe_append` (line 65)
    - `from shared_logic import safe_copy` (line 65)
    - `from shared_logic import safe_get` (line 65)
    - `from shared_logic import safe_lower` (line 65)
    - `from shared_logic import safe_replace` (line 65)
    - `from shared_logic import safe_split` (line 65)
    - `from shared_logic import safe_strip` (line 65)
    - `from shared_logic import safe_values` (line 65)
    - `from table_core import robust_table_extraction` (line 75)
- Task markers:
  - L125 **WARNING**: ", "extractor", "\[EXTRACTOR\] No &lt;table&gt; found in
    provided table_html.", session_id)
  - L130 **WARNING**: ", "extractor", "\[EXTRACTOR\] No &lt;tr&gt; rows found in
    table_html.", session_id)
  - L172 **WARNING**: ", "extractor", "\[EXTRACTOR\] Candidate NLP/score step
    failed", session_id, error=str(e))
  - L188 **WARNING**: ", "extractor", "\[EXTRACTOR\] No suitable table
    candidates found.", session_id)
  - L218 **WARNING**: ", "extractor", "\[EXTRACTOR\] Error while scanning
    &lt;table&gt; elements", session_id, error=str(e))
  - L230 **WARNING**: ", "extractor", "\[EXTRACTOR\] DOM extraction failed",
    session_id, error=str(e))
  - L273 **WARNING**: ", "extractor", "\[EXTRACTOR\] Pattern extraction failed",
    session_id, error=str(e))
  - L784 **WARNING**: ", "extractor", "No learned DOM patterns found.")
  - L808 **WARNING**: ", "extractor", "Entry deleted.")
  - L813 **WARNING**: ", "extractor", "Unknown action.")
  - L815 **WARNING**: ", "extractor", "Invalid entry number.")
- Outgoing cross-module calls (sample):
  - level.upper (line 96)
  - fields.items (line 101)
  - level.lower (line 105)
  - shared\_logic.safe\_get (line 115)
  - selectolax.parser.HTMLParser (line 122)
  - soup.css\_first (line 123)
  - table.css (line 128)
  - cell.text (line 134)
  - row.css (line 137)
  - row.css (line 137)
  - data.append (line 142)
  - shared\_logic.safe\_get (line 144)
  - shared\_logic.safe\_get (line 144)
  - enriched\_candidates.append (line 170)
  - enriched\_candidates.sort (line 174)
  - c.get (line 174)
  - shared\_logic.safe\_get (line 178)
  - shared\_logic.safe\_get (line 178)
  - shared\_logic.safe\_get (line 179)
  - shared\_logic.safe\_get (line 179)
  - shared\_logic.safe\_get (line 181)
  - shared\_logic.safe\_get (line 181)
  - shared\_logic.safe\_get (line 182)
  - browser\_utils.safe\_locator (line 202)
  - browser\_utils.safe\_count (line 203)
  - browser\_utils.safe\_nth (line 206)
  - detect.extract\_table\_data (line 210)
  - shared\_logic.safe\_copy (line 214)
  - candidates.append (line 215)
  - dom\_extractor.extract\_rows\_and\_headers\_from\_dom (line 222)
  - shared\_logic.safe\_copy (line 226)
  - candidates.append (line 227)
  - pattern\_extractor.extract\_with\_patterns (line 234)
  - browser\_utils.safe\_locator (line 241)
  - browser\_utils.safe\_count (line 242)
  - dom\_extractor.guess\_headers\_from\_row (line 243)
  - browser\_utils.safe\_locator (line 250)
  - browser\_utils.safe\_count (line 251)
  - browser\_utils.safe\_count (line 254)
  - browser\_utils.safe\_nth (line 255)
  - cell.inner\_text (line 256)
  - data.append (line 265)
  - shared\_logic.safe\_copy (line 269)
  - candidates.append (line 270)
  - shared\_logic.safe\_get (line 286)
  - coordinator.extract\_entities (line 291)
  - header\_entities.append (line 294)
  - coordinator.score\_header (line 296)
  - header\_scores.append (line 299)
  - shared\_logic.safe\_get (line 314)
- Inbound references:
  - \_emit ← dynamic_table_extractor.py:116
  - \_emit ← dynamic_table_extractor.py:125
  - \_emit ← dynamic_table_extractor.py:130
  - \_emit ← dynamic_table_extractor.py:150
  - \_emit ← dynamic_table_extractor.py:154
  - \_emit ← dynamic_table_extractor.py:161
  - \_emit ← dynamic_table_extractor.py:172
  - \_emit ← dynamic_table_extractor.py:177
  - \_emit ← dynamic_table_extractor.py:188
  - \_emit ← dynamic_table_extractor.py:204
  - \_emit ← dynamic_table_extractor.py:216
  - \_emit ← dynamic_table_extractor.py:218
  - \_emit ← dynamic_table_extractor.py:228
  - \_emit ← dynamic_table_extractor.py:230
  - \_emit ← dynamic_table_extractor.py:271
  - \_emit ← dynamic_table_extractor.py:273
  - \_emit ← dynamic_table_extractor.py:275
  - \_emit ← dynamic_table_extractor.py:784
  - \_emit ← dynamic_table_extractor.py:791
  - \_emit ← dynamic_table_extractor.py:793
  - \_emit ← dynamic_table_extractor.py:794
  - \_emit ← dynamic_table_extractor.py:795
  - \_emit ← dynamic_table_extractor.py:808
  - \_emit ← dynamic_table_extractor.py:811
  - \_emit ← dynamic_table_extractor.py:813
  - \_emit ← dynamic_table_extractor.py:815
  - \_emit ← dynamic_table_extractor.py:820
  - \_emit ← table_builder.py:742
  - \_emit ← table_builder.py:807
  - \_emit ← table_builder.py:821
  - \_emit ← table_builder.py:825
  - \_emit ← table_builder.py:828
  - \_emit ← table_builder.py:833
  - \_emit ← table_builder.py:837
  - \_emit ← table_builder.py:841
  - \_emit ← table_builder.py:850
  - \_emit ← table_builder.py:867
  - \_emit ← table_builder.py:872
  - \_emit ← table_builder.py:878
  - \_emit ← table_builder.py:929
  - \_emit ← table_builder.py:955
  - \_emit ← table_builder.py:960
  - \_emit ← table_builder.py:980
  - \_emit ← table_builder.py:1000
  - \_emit ← table_builder.py:1010
  - \_emit ← table_builder.py:1037
  - \_emit ← table_builder.py:1058
  - \_emit ← table_builder.py:1207
  - \_emit ← table_builder.py:1242
  - \_emit ← table_builder.py:1257

### utils/embedding\_cache.py {#webapp-parser-utils-embedding-cache-py}

- Definitions:
  - function: `\_int\_env` (line 53)
  - function: `\_warn\_on\_large\_disk\_cache` (line 149)
  - function: `\_checkpoint\_disk\_cache` (line 160)
  - function: `\_note\_disk\_cache\_mutation` (line 171)
  - function: `get\_embedding\_cache\_status` (line 177)
  - function: `\_log\_cache\_status` (line 197)
  - function: `\_save\_disk\_cache\_on\_exit` (line 221)
  - function: `ensure\_embedding\_cache\_table` (line 228)
  - function: `\_db\_write\_allowed` (line 273)
  - function: `\_seed\_cache\_from\_db` (line 289)
  - function: `compute\_embedding\_for\_hash` (line 326)
  - function: `save\_embedding` (line 340)
  - function: `load\_embedding` (line 366)
  - function: `get\_embedding\_from\_memory` (line 397)
  - function: `save\_embeddings\_batch` (line 416)
  - function: `load\_embeddings\_batch` (line 481)
  - function: `fix\_missing\_embeddings` (line 541)
- Imports:
  - **Standard Library** (5):
    - `import logging as logging` (line 4)
    - `import os as os` (line 10)
    - `import threading as threading` (line 11)
    - `import time as time` (line 12)
    - `from functools import lru_cache` (line 13)
  - **Third-party** (7):
    - `import numpy as np` (line 15)
    - `import orjson as orjson` (line 16)
    - `from sqlalchemy import inspect` (line 17)
    - `from sqlalchemy import select` (line 17)
    - `from sqlalchemy.dialects.postgresql import insert` (line 18)
    - `from sqlalchemy.exc import SQLAlchemyError` (line 19)
    - `from sqlalchemy.orm.exc import DetachedInstanceError` (line 20)
  - **Local/Project** (11):
    - `from __future__ import annotations` (line 1)
    - `import atexit as atexit` (line 3)
    - `from config import DISK_CACHE_PATH` (line 22)
    - `from config import MISSING_LOG_PATH` (line 22)
    - `from db_utils import TEST_SQLITE_URL` (line 23)
    - `from db_utils import engine` (line 23)
    - `from db_utils import get_session` (line 23)
    - `from logger_singleton import console` (line 24)
    - `from logger_singleton import logger` (line 24)
    - `from models import EmbeddingCache` (line 25)
    - `from shared_logger import SQLAlchemyToSharedLoggerHandler` (line 26)
- Task markers:
  - L313 **WARNING**: (f"\[EMBEDDING CACHE\] DB seed skipped due to error: {e}")
- Outgoing cross-module calls (sample):
  - logging.getLogger (line 41)
  - logger\_obj.addHandler (line 42)
  - shared\_logger.SQLAlchemyToSharedLoggerHandler (line 42)
  - threading.Lock (line 46)
  - threading.Lock (line 47)
  - threading.Lock (line 48)
  - time.time (line 50)
  - joblib.load (line 64)
  - logger\_singleton.console.print (line 67)
  - logger\_singleton.console.print (line 70)
  - os.makedirs (line 77)
  - joblib.dump (line 78)
  - time.time (line 79)
  - logger\_singleton.console.print (line 82)
  - logger\_singleton.console.print (line 86)
  - pickle.load (line 92)
  - logger\_singleton.console.print (line 95)
  - logger\_singleton.console.print (line 98)
  - os.makedirs (line 105)
  - pickle.dump (line 107)
  - time.time (line 108)
  - logger\_singleton.console.print (line 111)
  - logger\_singleton.console.print (line 115)
  - logger\_singleton.console.print (line 154)
  - time.time (line 165)
  - logger\_singleton.console.print (line 206)
  - logger\_singleton.console.print (line 215)
  - atexit.register (line 225)
  - logger\_singleton.console.print (line 233)
  - sqlalchemy.inspect (line 245)
  - inspector.has\_table (line 247)
  - logger\_singleton.console.print (line 250)
  - logger\_singleton.console.print (line 257)
  - logger\_singleton.console.print (line 259)
  - logger\_singleton.console.print (line 265)
  - logger\_singleton.console.print (line 280)
  - db\_utils.get\_session (line 296)
  - sqlalchemy.select (line 297)
  - session.execute (line 298)
  - numpy.frombuffer (line 301)
  - logger\_singleton.logger.warning (line 313)
  - logger\_singleton.console.print (line 320)
  - numpy.array (line 343)
  - db\_utils.get\_session (line 346)
  - session.get (line 348)
  - models.EmbeddingCache (line 352)
  - session.add (line 353)
  - session.commit (line 354)
  - session.rollback (line 356)
  - logger\_singleton.console.print (line 357)
- Inbound references:
  - \_warn\_on\_large\_disk\_cache ← embedding_cache.py:163
  - \_warn\_on\_large\_disk\_cache ← embedding_cache.py:168
  - \_checkpoint\_disk\_cache ← embedding_cache.py:174
  - \_checkpoint\_disk\_cache ← embedding_cache.py:222
  - \_note\_disk\_cache\_mutation ← embedding_cache.py:310
  - \_note\_disk\_cache\_mutation ← embedding_cache.py:364
  - \_note\_disk\_cache\_mutation ← embedding_cache.py:389
  - \_note\_disk\_cache\_mutation ← embedding_cache.py:479
  - \_note\_disk\_cache\_mutation ← embedding_cache.py:527
  - get\_embedding\_cache\_status ← embedding_cache.py:214
  - \_log\_cache\_status ← embedding_cache.py:212
  - ensure\_embedding\_cache\_table ← embedding_cache.py:286
  - ensure\_embedding\_cache\_table ← embedding_cache.py:290
  - ensure\_embedding\_cache\_table ← embedding_cache.py:377
  - ensure\_embedding\_cache\_table ← embedding_cache.py:486
  - ensure\_embedding\_cache\_table ← embedding_cache.py:583
  - \_db\_write\_allowed ← embedding_cache.py:342
  - \_db\_write\_allowed ← embedding_cache.py:423
  - \_seed\_cache\_from\_db ← embedding_cache.py:318
  - compute\_embedding\_for\_hash ← embedding_cache.py:567
  - save\_embedding ← embedding_cache.py:569
  - load\_embedding ← embedding_cache.py:399
  - load\_embedding ← embedding_cache.py:562
  - fix\_missing\_embeddings ← embedding_cache.py:584

### utils/extraction\_strategies.py {#webapp-parser-utils-extraction-strategies-py}

> extraction_strategies.py

- Definitions:
  - function: `register\_strategy` (line 36)
  - function: `run\_registered\_strategies` (line 45)
  - function: `strategy\_html\_tables` (line 81)
  - function: `strategy\_dom\_repetition` (line 94)
  - function: `strategy\_pattern\_based` (line 100)
  - function: `strategy\_heading\_associated` (line 104)
  - function: `strategy\_ml\_detection` (line 157)
  - function: `strategy\_selectolax\_fallback` (line 173)
  - function: `strategy\_nlp\_fallback` (line 196)
  - function: `\_normalized\_header\_tuple` (line 236)
  - function: `\_merge\_similar\_tables` (line 239)
- Imports:
  - **Standard Library** (7):
    - `import re as re` (line 9)
    - `import time as time` (line 10)
    - `from typing import Any` (line 11)
    - `from typing import Callable` (line 11)
    - `from typing import Dict` (line 11)
    - `from typing import List` (line 11)
    - `from typing import Tuple` (line 11)
  - **Local/Project** (17):
    - `from __future__ import annotations` (line 7)
    - `from selectolax.parser import HTMLParser` (line 13)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_KEYWORDS` (line 15)
    - `from Context_Integration.Context_Library.constants import
      NLP_SKIP_PHRASES` (line 15)
    - `from browser_utils import safe_content` (line 19)
    - `from browser_utils import safe_count` (line 19)
    - `from browser_utils import safe_inner_text` (line 19)
    - `from browser_utils import safe_locator` (line 19)
    - `from browser_utils import safe_nth` (line 19)
    - `from detect import emit_metric` (line 20)
    - `from detect import extract_percent_reported_from_heading` (line 20)
    - `from detect import find_best_header` (line 20)
    - `from detect import normalize_header` (line 20)
    - `from dom_extractor import extract_rows_and_headers_from_dom` (line 26)
    - `from logger_singleton import logger` (line 27)
    - `from pattern_extractor import extract_with_patterns` (line 28)
    - `from shared_logic import safe_append` (line 29)
- Task markers:
  - L68 **WARNING**: (f"\[STRATEGY\] {name} failed: {e}")
- Outgoing cross-module calls (sample):
  - STRATEGY\_REGISTRY.append (line 37)
  - time.perf\_counter (line 48)
  - meta.get (line 51)
  - time.perf\_counter (line 54)
  - time.perf\_counter (line 57)
  - detect.emit\_metric (line 58)
  - diag.setdefault (line 62)
  - shared\_logic.safe\_append (line 64)
  - detect.emit\_metric (line 65)
  - time.perf\_counter (line 67)
  - logger\_singleton.logger.warning (line 68)
  - detect.emit\_metric (line 69)
  - context.get (line 72)
  - time.perf\_counter (line 75)
  - detect.emit\_metric (line 76)
  - browser\_utils.safe\_locator (line 85)
  - browser\_utils.safe\_count (line 86)
  - browser\_utils.safe\_nth (line 87)
  - results.append (line 91)
  - dom\_extractor.extract\_rows\_and\_headers\_from\_dom (line 95)
  - pattern\_extractor.extract\_with\_patterns (line 101)
  - browser\_utils.safe\_locator (line 111)
  - browser\_utils.safe\_count (line 112)
  - browser\_utils.safe\_inner\_text (line 113)
  - browser\_utils.safe\_nth (line 113)
  - detect.extract\_percent\_reported\_from\_heading (line 114)
  - browser\_utils.safe\_locator (line 118)
  - browser\_utils.safe\_count (line 119)
  - browser\_utils.safe\_nth (line 120)
  - browser\_utils.safe\_locator (line 122)
  - browser\_utils.safe\_count (line 124)
  - browser\_utils.safe\_nth (line 125)
  - browser\_utils.safe\_inner\_text (line 126)
  - detect.find\_best\_header (line 130)
  - r.pop (line 135)
  - r.get (line 135)
  - detect.find\_best\_header (line 141)
  - detect.extract\_percent\_reported\_from\_heading (line 142)
  - h.append (line 145)
  - r.get (line 150)
  - results.append (line 153)
  - browser\_utils.safe\_content (line 160)
  - t.get (line 166)
  - t.get (line 167)
  - res.append (line 169)
  - browser\_utils.safe\_content (line 175)
  - selectolax.parser.HTMLParser (line 178)
  - tree.css (line 180)
  - tbl.css (line 181)
  - c.text (line 185)
- Inbound references:
  - register\_strategy ← extraction_strategies.py:282
  - register\_strategy ← extraction_strategies.py:283
  - register\_strategy ← extraction_strategies.py:284
  - register\_strategy ← extraction_strategies.py:285
  - register\_strategy ← extraction_strategies.py:286
  - register\_strategy ← extraction_strategies.py:287
  - register\_strategy ← extraction_strategies.py:288
  - \_normalized\_header\_tuple ← extraction_strategies.py:249
  - \_merge\_similar\_tables ← extraction_strategies.py:73

### utils/factor\_chain\_tracker.py {#webapp-parser-utils-factor-chain-tracker-py}

> Deterministic Factor Chain Tracker with Breaking-Chain Detection

- Definitions:
  - function: `\_ensure\_anomaly\_log` (line 41)
  - class: `FactorSnapshot` (line 57)
  - class: `FactorChain` (line 71)
  - function: `detect\_breaking\_chains` (line 175)
  - function: `flush\_factor\_chain\_analysis` (line 306)
  - function: `create\_factor\_chain` (line 349)
  - function: `finalize\_factor\_chain` (line 369)
- Imports:
  - **Standard Library** (9):
    - `import uuid as uuid` (line 25)
    - `from dataclasses import dataclass` (line 26)
    - `from dataclasses import field` (line 26)
    - `from datetime import datetime` (line 27)
    - `from datetime import timezone` (line 27)
    - `from pathlib import Path` (line 28)
    - `from typing import Any` (line 29)
    - `from typing import Dict` (line 29)
    - `from typing import List` (line 29)
  - **Local/Project** (3):
    - `from __future__ import annotations` (line 23)
    - `from config import LOG_DIR` (line 31)
    - `from utils.logger_singleton import logger` (line 32)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 38)
  - FACTOR\_CHAIN\_ANOMALIES\_FILE.touch (line 44)
  - dataclasses.field (line 81)
  - dataclasses.field (line 84)
  - datetime.datetime.now (line 98)
  - factors.copy (line 103)
  - datetime.datetime.now (line 114)
  - severity\_order.get (line 121)
  - severity\_order.get (line 122)
  - s.to\_txt\_line (line 128)
  - start\_factors.get (line 196)
  - current\_factors.get (line 197)
  - anomalies.append (line 203)
  - anomalies.append (line 214)
  - start\_factors.get (line 227)
  - current\_factors.get (line 228)
  - anomalies.append (line 233)
  - FACTOR\_BOUNDS.items (line 242)
  - current\_factors.get (line 246)
  - anomalies.append (line 252)
  - FACTOR\_DEPENDENCIES.items (line 261)
  - antecedent.split (line 266)
  - expected\_str.lower (line 267)
  - expected\_str.lower (line 267)
  - current\_factors.get (line 272)
  - consequent.split (line 279)
  - cons\_expected\_str.lower (line 280)
  - cons\_expected\_str.lower (line 280)
  - current\_factors.get (line 285)
  - anomalies.append (line 291)
  - chain.to\_txt\_entry (line 313)
  - f.write (line 315)
  - f.flush (line 316)
  - os.fsync (line 318)
  - f.fileno (line 318)
  - utils.logger\_singleton.logger.error (line 321)
  - utils.logger\_singleton.logger.error (line 335)
  - uuid.uuid4 (line 356)
  - datetime.datetime.now (line 357)
  - a.get (line 380)
- Inbound references:
  - \_ensure\_anomaly\_log ← factor_chain_tracker.py:49
  - FactorSnapshot ← factor_chain_tracker.py:99
  - FactorChain ← factor_chain_tracker.py:359

### utils/fec\_utils.py {#webapp-parser-utils-fec-utils-py}

- Definitions:
  - function: `\_load\_json` (line 16)
  - function: `\_append\_ambiguous\_log` (line 64)
  - function: `canonicalize\_headers` (line 74)
  - function: `money\_normalize` (line 104)
  - function: `date\_normalize` (line 126)
  - function: `party\_normalize` (line 152)
  - function: `incumbent\_normalize` (line 166)
- Imports:
  - **Standard Library** (8):
    - `import json as json` (line 3)
    - `import os as os` (line 4)
    - `import re as re` (line 5)
    - `from datetime import datetime` (line 6)
    - `from typing import Dict` (line 7)
    - `from typing import List` (line 7)
    - `from typing import Optional` (line 7)
    - `from typing import Tuple` (line 7)
  - **Third-party** (1):
    - `from webapp.parser.config import LOG_DIR` (line 9)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - json.load (line 19)
  - os.makedirs (line 66)
  - fh.write (line 69)
  - json.dumps (line 69)
  - h.strip (line 81)
  - \_ALIASES.items (line 84)
  - v.lower (line 86)
  - orig.lower (line 86)
  - canonical\_order.append (line 89)
  - re.sub (line 96)
  - canonical\_order.append (line 99)
  - s.replace (line 111)
  - s.startswith (line 113)
  - s.endswith (line 113)
  - re.search (line 120)
  - m.group (line 121)
  - datetime.datetime.strptime (line 134)
  - dt.date (line 135)
  - re.search (line 139)
  - m.groups (line 141)
  - datetime.datetime (line 144)
  - dt.date (line 145)
  - re.sub (line 159)
- Inbound references:
  - \_load\_json ← fec_utils.py:24
  - \_load\_json ← fec_utils.py:46
  - \_append\_ambiguous\_log ← fec_utils.py:100
  - \_append\_ambiguous\_log ← fec_utils.py:117
  - \_append\_ambiguous\_log ← fec_utils.py:148
  - \_append\_ambiguous\_log ← fec_utils.py:162

### utils/format\_router.py {#webapp-parser-utils-format-router-py}

- Definitions:
  - function: `\_guard\_text` (line 63)
  - function: `\_guard\_download\_links` (line 72)
  - function: `\_guard\_google\_sheet\_meta` (line 85)
  - function: `\_normalize\_text` (line 91)
  - function: `\_infer\_format\_from\_text` (line 95)
  - function: `\_infer\_format\_from\_attr\_value` (line 106)
  - function: `\_extract\_candidate\_urls` (line 117)
  - function: `\_clean\_filename` (line 144)
  - function: `\_guess\_filename\_from\_url` (line 150)
  - function: `\_extract\_filename\_from\_disposition` (line 169)
  - function: `\_extract\_google\_sheet\_metadata` (line 179)
  - function: `\_probe\_remote\_format` (line 224)
  - function: `\_browser\_headers` (line 275)
  - function: `\_build\_download\_url` (line 296)
  - function: `\_normalize\_download\_url` (line 306)
  - function: `\_cookies\_header\_from\_page` (line 317)
  - function: `extract\_contest\_from\_filename` (line 331)
  - function: `summarize\_downloads` (line 370)
  - function: `\_infer\_format\_from\_url` (line 380)
  - function: `\_expose\_download\_interfaces` (line 388)
  - function: `detect\_format\_from\_links` (line 437)
  - function: `route\_format\_handler` (line 488)
  - function: `extract\_download\_links\_from\_html` (line 515)
  - function: `prompt\_and\_handle\_download` (line 535)
- Imports:
  - **Standard Library** (14):
    - `import os as os` (line 1)
    - `import re as re` (line 2)
    - `import tempfile as tempfile` (line 3)
    - `import time as time` (line 4)
    - `from typing import Dict` (line 6)
    - `from typing import List` (line 6)
    - `from typing import Optional` (line 6)
    - `from typing import Tuple` (line 6)
    - `from urllib.parse import parse_qs` (line 7)
    - `from urllib.parse import unquote` (line 7)
    - `from urllib.parse import urljoin` (line 7)
    - `from urllib.parse import urlparse` (line 7)
    - `from urllib.parse import urlsplit` (line 7)
    - `from urllib.parse import urlunsplit` (line 7)
  - **Third-party** (1):
    - `import requests as requests` (line 9)
  - **Local/Project** (29):
    - `from difflib import get_close_matches` (line 5)
    - `from config import ALLOW_GOOGLE_DOCS` (line 11)
    - `from config import DISABLE_HTML_FALLBACK` (line 11)
    - `from config import SUPPORTED_FORMATS` (line 11)
    - `from config import URL_MAX_REDIRECTS` (line 11)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_KEYWORDS` (line 12)
    - `from handlers import fec_handler` (line 13)
    - `from handlers.formats import csv_handler` (line 14)
    - `from handlers.formats import json_handler` (line 14)
    - `from handlers.formats import pdf_handler` (line 14)
    - `from handlers.formats import txt_handler` (line 14)
    - `from handlers.formats import xlsx_handler` (line 14)
    - `from browser_utils import safe_click_with_retry` (line 15)
    - `from browser_utils import safe_content` (line 15)
    - `from browser_utils import safe_context_library` (line 15)
    - `from browser_utils import safe_context_result` (line 15)
    - `from browser_utils import safe_get_attribute` (line 15)
    - `from browser_utils import safe_inner_text` (line 15)
    - `from browser_utils import safe_query_selector_all` (line 15)
    - `from browser_utils import safe_url` (line 15)
    - `from browser_utils import safe_wait_for_timeout` (line 15)
    - `from download_utils import download_file` (line 26)
    - `from download_utils import ensure_input_directory` (line 26)
    - `from html_scanner import append_pattern_kb` (line 27)
    - `from html_scanner import load_pattern_kb` (line 27)
    - `from logger_singleton import logger` (line 28)
    - `from logger_singleton import prompt` (line 28)
    - `from shared_logic import safe_lower` (line 29)
    - `from shared_logic import safe_parse` (line 29)
- Task markers:
  - L473 **WARNING**: ({
  - L474 **WARNING**: ",
  - L476 **WARN**: \] No supported file formats found on the page.",
  - L501 **WARNING**: ({
  - L502 **WARNING**: ",
  - L504 **WARN**: \] Unsupported format requested: {format_str}",
  - L508 **WARNING**: ({
  - L509 **WARNING**: ",
  - L803 **WARNING**: ({
  - L804 **WARNING**: ",
  - L1012 **WARNING**: ({
  - L1013 **WARNING**: ",
  - L1063 **WARNING**: ({
  - L1064 **WARNING**: ",
  - L1170 **WARNING**: ({
  - L1171 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - re.compile (line 52)
  - link.get (line 79)
  - link.get (line 79)
  - cleaned.get (line 80)
  - cleaned.get (line 80)
  - guarded.append (line 81)
  - meta.get (line 88)
  - meta.get (line 88)
  - raw\_value.strip (line 120)
  - urls.extend (line 123)
  - re.findall (line 123)
  - urls.extend (line 124)
  - re.findall (line 124)
  - urls.extend (line 125)
  - re.findall (line 127)
  - url.lower (line 133)
  - deduped.append (line 137)
  - raw\_value.lower (line 139)
  - urllib.parse.unquote (line 145)
  - name.strip (line 146)
  - urllib.parse.urlparse (line 152)
  - segment.split (line 160)
  - val.lower (line 162)
  - url.split (line 164)
  - FILENAME\_FROM\_DISPOSITION.search (line 172)
  - match.group (line 175)
  - match.group (line 175)
  - urllib.parse.urlparse (line 183)
  - path\_parts.index (line 195)
  - urllib.parse.parse\_qs (line 200)
  - qs.get (line 201)
  - urllib.parse.parse\_qs (line 205)
  - qs.get (line 206)
  - urllib.parse.parse\_qs (line 208)
  - frag\_qs.get (line 209)
  - requests.head (line 240)
  - headers\_map.get (line 255)
  - headers\_map.get (line 255)
  - headers\_map.get (line 256)
  - headers\_map.get (line 256)
  - logger\_singleton.logger.debug (line 259)
  - content\_type.split (line 267)
  - CONTENT\_TYPE\_FORMAT\_MAP.get (line 268)
  - page.evaluate (line 277)
  - urllib.parse.urlparse (line 283)
  - re.sub (line 300)
  - urllib.parse.urljoin (line 301)
  - urllib.parse.urlsplit (line 309)
  - re.sub (line 312)
  - urllib.parse.urlunsplit (line 313)
- Inbound references:
  - \_guard\_text ← format_router.py:79
  - \_guard\_text ← format_router.py:88
  - \_guard\_download\_links ← format_router.py:848
  - \_guard\_google\_sheet\_meta ← format_router.py:850
  - \_normalize\_text ← format_router.py:96
  - \_normalize\_text ← format_router.py:107
  - \_normalize\_text ← shared_logic.py:1327
  - \_normalize\_text ← shared_logic.py:1333
  - \_normalize\_text ← shared_logic.py:1341
  - \_infer\_format\_from\_text ← format_router.py:111
  - \_infer\_format\_from\_text ← format_router.py:113
  - \_infer\_format\_from\_text ← format_router.py:270
  - \_infer\_format\_from\_text ← format_router.py:767
  - \_infer\_format\_from\_attr\_value ← format_router.py:773
  - \_infer\_format\_from\_attr\_value ← format_router.py:777
  - \_extract\_candidate\_urls ← format_router.py:768
  - \_clean\_filename ← format_router.py:156
  - \_clean\_filename ← format_router.py:161
  - \_clean\_filename ← format_router.py:164
  - \_clean\_filename ← format_router.py:176
  - \_guess\_filename\_from\_url ← format_router.py:800
  - \_guess\_filename\_from\_url ← format_router.py:902
  - \_extract\_filename\_from\_disposition ← format_router.py:257
  - \_extract\_google\_sheet\_metadata ← format_router.py:676
  - \_extract\_google\_sheet\_metadata ← format_router.py:676
  - \_probe\_remote\_format ← format_router.py:715
  - \_browser\_headers ← format_router.py:226
  - \_browser\_headers ← format_router.py:1001
  - \_build\_download\_url ← format_router.py:769
  - \_build\_download\_url ← format_router.py:997
  - \_cookies\_header\_from\_page ← format_router.py:226
  - \_cookies\_header\_from\_page ← format_router.py:1000
  - extract\_contest\_from\_filename ← format_router.py:903
  - \_infer\_format\_from\_url ← format_router.py:272
  - \_infer\_format\_from\_url ← format_router.py:772
  - \_expose\_download\_interfaces ← format_router.py:666
  - route\_format\_handler ← format_router.py:620
  - route\_format\_handler ← format_router.py:1139
  - extract\_download\_links\_from\_html ← format_router.py:811

### utils/header\_confidence.py {#webapp-parser-utils-header-confidence-py}

> Header mapping confidence scoring and validation.

- Definitions:
  - function: `get\_header\_confidence` (line 35)
  - function: `validate\_row\_headers` (line 88)
  - function: `should\_insert\_row` (line 128)
- Imports:
  - **Standard Library** (4):
    - `import logging as logging` (line 6)
    - `from typing import Dict` (line 7)
    - `from typing import Optional` (line 7)
    - `from typing import Tuple` (line 7)
  - **Local/Project** (2):
    - `from config import HEADER_CONFIDENCE_THRESHOLD` (line 9)
    - `from config import HEADER_INSERT_CONFIDENCE_THRESHOLD` (line 9)
- Outgoing cross-module calls (sample):
  - logging.getLogger (line 11)
  - header.strip (line 56)
  - a.replace (line 63)
  - aliases.get (line 63)
  - aliases.get (line 67)
  - exact\_alias.replace (line 68)
  - aliases.get (line 72)
  - exact\_alias.replace (line 73)
  - aliases.get (line 79)
  - fuzzy\_alias.replace (line 80)
  - flagged.append (line 120)
  - confidence\_scores.get (line 122)
  - confidence\_scores.get (line 140)
  - mapped\_row.get (line 145)
- Inbound references:
  - get\_header\_confidence ← header_confidence.py:112

### utils/header\_utils.py {#webapp-parser-utils-header-utils-py}

- Definitions:
  - function: `build\_candidate\_group\_hierarchical` (line 10)
  - function: `normalize\_headers\_list` (line 37)
  - function: `\_clean\_header\_fragment` (line 46)
  - function: `\_assemble\_header\_label` (line 57)
  - function: `compact\_header\_tokens` (line 84)
  - function: `collapse\_multiline\_header` (line 147)
  - function: `\_register\_header\_mapping` (line 171)
  - function: `normalize\_table\_headers` (line 178)
- Imports:
  - **Standard Library** (6):
    - `import re as re` (line 3)
    - `from typing import Any` (line 4)
    - `from typing import Dict` (line 4)
    - `from typing import Iterable` (line 4)
    - `from typing import List` (line 4)
    - `from typing import Tuple` (line 4)
  - **Local/Project** (4):
    - `from __future__ import annotations` (line 1)
    - `from detect import dedupe_headers_with_suffix` (line 6)
    - `from detect import normalize_header` (line 6)
    - `from salvage import normalize_ballot_column_name` (line 7)
- Outgoing cross-module calls (sample):
  - row1.append (line 23)
  - row2.append (line 24)
  - h.rsplit (line 27)
  - row1.append (line 30)
  - row2.append (line 31)
  - row1.append (line 33)
  - row2.append (line 34)
  - detect.normalize\_header (line 41)
  - salvage.normalize\_ballot\_column\_name (line 42)
  - detect.dedupe\_headers\_with\_suffix (line 43)
  - cleaned.strip (line 50)
  - cleaned.replace (line 51)
  - cleaned.replace (line 52)
  - re.sub (line 53)
  - cleaned.strip (line 54)
  - re.search (line 64)
  - match.group (line 67)
  - match.start (line 68)
  - pieces.append (line 73)
  - top\_fragment.lower (line 74)
  - bottom\_core.lower (line 74)
  - pieces.append (line 75)
  - re.sub (line 79)
  - cleaned\_prior.insert (line 102)
  - cleaned\_tokens.append (line 109)
  - top.append (line 120)
  - token.startswith (line 124)
  - merged\_bottom.append (line 127)
  - merged\_bottom.append (line 130)
  - combined.append (line 136)
  - re.split (line 156)
  - key.strip (line 173)
  - collapsed.lower (line 190)
  - seen.get (line 191)
  - normalized\_headers.append (line 194)
  - row.items (line 205)
  - header\_mapping.get (line 207)
  - header\_mapping.get (line 209)
  - key\_str.strip (line 209)
  - fallback.lower (line 212)
  - seen.get (line 213)
  - key\_str.strip (line 217)
  - normalized\_headers.append (line 219)
  - normalized\_rows.append (line 221)
- Inbound references:
  - normalize\_headers\_list ← header_utils.py:232
  - \_clean\_header\_fragment ← header_utils.py:58
  - \_clean\_header\_fragment ← header_utils.py:59
  - \_clean\_header\_fragment ← header_utils.py:92
  - \_clean\_header\_fragment ← header_utils.py:92
  - \_clean\_header\_fragment ← header_utils.py:93
  - \_clean\_header\_fragment ← header_utils.py:93
  - \_clean\_header\_fragment ← header_utils.py:152
  - \_clean\_header\_fragment ← header_utils.py:155
  - \_clean\_header\_fragment ← header_utils.py:157
  - \_clean\_header\_fragment ← header_utils.py:160
  - \_assemble\_header\_label ← header_utils.py:136
  - \_assemble\_header\_label ← header_utils.py:142
  - \_assemble\_header\_label ← header_utils.py:164
  - \_assemble\_header\_label ← header_utils.py:168
  - collapse\_multiline\_header ← header_utils.py:187
  - collapse\_multiline\_header ← header_utils.py:211
  - \_register\_header\_mapping ← header_utils.py:197
  - \_register\_header\_mapping ← header_utils.py:199
  - \_register\_header\_mapping ← header_utils.py:200
  - normalize\_table\_headers ← html_election_parser.py:1385

### utils/html\_scanner.py {#webapp-parser-utils-html-scanner-py}

- Definitions:
  - function: `robust\_orjson\_loads` (line 129)
  - function: `\_get\_label\_cache\_path` (line 149)
  - function: `\_load\_label\_cache` (line 202)
  - function: `\_save\_label\_cache` (line 222)
  - function: `cache\_segment\_label` (line 233)
  - function: `get\_cached\_segment\_label` (line 242)
  - function: `safe\_cache\_path` (line 270)
  - function: `safe\_log\_path` (line 331)
  - function: `is\_trivial\_segment` (line 396)
  - function: `segment\_identity\_hash` (line 473)
  - function: `embedding\_cache\_hash` (line 499)
  - function: `get\_segment\_embedding` (line 518)
  - function: `batch\_get\_segment\_embeddings` (line 620)
  - function: `deduplicate\_pattern\_kb` (line 692)
  - function: `prune\_embedding\_cache` (line 702)
  - function: `submit\_segment\_correction` (line 714)
  - function: `auto\_label\_segment` (line 723)
  - function: `\_extract\_clean\_text` (line 953)
  - function: `\_label\_in` (line 968)
  - function: `\_extract\_segments\_by\_label` (line 976)
  - function: `extract\_year\_and\_type` (line 1078)
  - function: `is\_update\_panel` (line 1155)
  - function: `split\_possible\_contests` (line 1172)
  - function: `extract\_tagged\_segments\_with\_attrs` (line 1196)
  - function: `get\_page\_hash` (line 1755)
  - function: `load\_context\_cache\_from\_disk` (line 1802)
  - function: `save\_context\_cache\_to\_disk` (line 1838)
  - function: `add\_context\_entry` (line 1874)
  - function: `get\_context\_entry` (line 1886)
  - function: `export\_context\_cache\_for\_db` (line 1893)
  - function: `load\_pattern\_kb` (line 1906)
  - function: `append\_pattern\_kb` (line 1937)
  - function: `append\_feedback\_log` (line 1962)
  - function: `label\_validator` (line 1991)
  - function: `prompt\_for\_segment\_label` (line 1994)
  - function: `segment\_hash` (line 2047)
  - function: `canonicalize\_segment` (line 2051)
  - function: `validate\_dom\_parts` (line 2111)
  - function: `scan\_html\_for\_context` (line 2589)
  - function: `\_load\_context\_resources` (line 2856)
  - function: `\_prepare\_html\_and\_cache` (line 2943)
  - function: `\_fast\_path\_cache\_hit` (line 2962)
  - function: `\_organize\_segments\_and\_sections` (line 2997)
  - function: `\_enrich\_and\_validate\_context` (line 3230)
  - function: `\_extract\_heading\_text` (line 3414)
  - function: `\_build\_model\_signals` (line 3424)
  - function: `\_build\_context\_digest` (line 3499)
  - function: `\_update\_digest\_trends` (line 3565)
  - function: `\_write\_context\_digest` (line 3615)
- Imports:
  - **Standard Library** (15):
    - `import datetime as datetime` (line 4)
    - `import hashlib as hashlib` (line 10)
    - `import os as os` (line 11)
    - `import re as re` (line 12)
    - `import tempfile as tempfile` (line 13)
    - `import threading as threading` (line 14)
    - `import time as time` (line 15)
    - `import traceback as traceback` (line 16)
    - `from collections import Counter` (line 17)
    - `from typing import Any` (line 19)
    - `from typing import Dict` (line 19)
    - `from typing import List` (line 19)
    - `from typing import Optional` (line 19)
    - `from typing import Pattern` (line 19)
    - `from typing import Set` (line 19)
  - **Third-party** (2):
    - `import numpy as np` (line 21)
    - `import orjson as orjson` (line 22)
  - **Local/Project** (89):
    - `from __future__ import annotations` (line 1)
    - `import concurrent.futures as concurrent` (line 3)
    - `from difflib import get_close_matches` (line 18)
    - `from selectolax.parser import HTMLParser` (line 23)
    - `from config import CACHE_DIR` (line 25)
    - `from config import CONTEXT_CACHE_PATH` (line 25)
    - `from config import CONTEXT_LIBRARY_PATH` (line 25)
    - `from config import ENABLE_SEGMENT_LABEL_PROMPT` (line 25)
    - `from config import LOG_DIR` (line 25)
    - `from config import SEGMENT_ML_LABEL_THRESHOLD` (line 25)
    - `from config import SEGMENT_ML_LABEL_THRESHOLD_STRICT` (line 25)
    - `from Context_Integration.Context_Library.constants import ALLOWED_LABELS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import
      ALWAYS_IGNORE_CLASSES` (line 34)
    - `from Context_Integration.Context_Library.constants import
      ALWAYS_IGNORE_IDS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      ALWAYS_IGNORE_TAGS` (line 34)
    - `from Context_Integration.Context_Library.constants import BALLOT_TYPES`
      (line 34)
    - `from Context_Integration.Context_Library.constants import
      BALLOT_TYPES_SORT_ORDER` (line 34)
    - `from Context_Integration.Context_Library.constants import BUTTON_CLASSES`
      (line 34)
    - `from Context_Integration.Context_Library.constants import BUTTON_TAGS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import
      CANDIDATE_KEYWORDS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      CANONICAL_SEGMENT_LABELS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_KEYWORDS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_PANEL_TAGS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      CUSTOM_ATTR_PATTERNS` (line 34)
    - `from Context_Integration.Context_Library.constants import DISTRICT_REGEX`
      (line 34)
    - `from Context_Integration.Context_Library.constants import ELECTION_TYPES`
      (line 34)
    - `from Context_Integration.Context_Library.constants import
      EXTRA_HEADING_TAGS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      HEADING_CLASSES` (line 34)
    - `from Context_Integration.Context_Library.constants import HEADING_TAGS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import HTML_TAGS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import ICON_CLASSES`
      (line 34)
    - `from Context_Integration.Context_Library.constants import ICON_TAGS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_COUNTY_TO_PRECINCTS_MAP` (line 34)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_STATE_TO_COUNTY_MAP` (line 34)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_ABBREVIATIONS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_KEYWORDS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      MISC_FOOTER_KEYWORDS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      NOISY_LABEL_PATTERNS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      OFFICE_KEYWORDS` (line 34)
    - `from Context_Integration.Context_Library.constants import PANEL_CLASSES`
      (line 34)
    - `from Context_Integration.Context_Library.constants import PANEL_TAGS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import PARTY_KEYWORDS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import
      PERCENT_KEYWORDS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      PRECINCT_HEADER_PATTERNS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      ROOT_CONTAINER_TAGS` (line 34)
    - `from Context_Integration.Context_Library.constants import SELECTORS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import STATE_ABBR`
      (line 34)
    - `from Context_Integration.Context_Library.constants import STATE_TAGS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import
      STRUCTURAL_TAGS` (line 34)
    - `from Context_Integration.Context_Library.constants import TABLE_TAGS`
      (line 34)
- Task markers:
  - L166 **WARNING**: ",
  - L170 **WARNING**: (payload)
  - L192 **WARNING**: ",
  - L196 **WARNING**: (payload)
  - L291 **WARNING**: ",
  - L295 **WARNING**: (payload)
  - L318 **WARNING**: ",
  - L322 **WARNING**: (payload)
  - L356 **WARNING**: ",
  - L360 **WARNING**: (payload)
  - L383 **WARNING**: ",
  - L387 **WARNING**: (payload)
  - L582 **WARNING**: ",
  - L586 **WARNING**: (payload)
  - L798 **WARNING**: (f"\[ML SIMILARITY\] No embedding computed for segment:
    {safe_get(segment, 'segment_hash', None)}")
  - L832 **WARNING**: (f"\[ML SIMILARITY\] No embedding computed for segment:
    {safe_get(segment, 'segment_hash', None)}")
  - L1059 **WARNING**: ",
  - L1063 **WARNING**: (payload)
  - L1070 **WARNING**: ",
  - L1074 **WARNING**: (payload)
  - L1401 **WARNING**: ",
  - L1405 **WARNING**: (payload)
  - L1463 **WARNING**: ",
  - L1467 **WARNING**: (payload)
  - L1716 **WARNING**: ({"level": "WARNING", "type": "dom_segments", "message":
    msg_warn})
  - L1772 **WARNING**: ({"level": "WARNING", "type": "page_hash", "message":
    msg})
  - L1779 **WARNING**: ({"level": "WARNING", "type": "page_hash", "message":
    msg})
  - L1791 **WARNING**: ({"level": "WARNING", "type": "page_hash", "message":
    msg})
  - L1814 **WARNING**: ({"level": "WARNING", "type": "cache", "message": msg})
  - L1849 **WARNING**: ({"level": "WARNING", "type": "cache", "message": msg})
  - L2028 **WARNING**: ({"level": "WARNING", "type": "segment_review",
    "message": msg})
  - L2037 **WARNING**: ({
  - L2038 **WARNING**: ",
  - L2154 **WARNING**: ",
  - L2158 **WARNING**: (payload)
  - L2170 **WARNING**: ",
  - L2174 **WARNING**: (payload)
  - L2183 **WARNING**: ",
  - L2187 **WARNING**: (payload)
  - L2202 **WARNING**: ",
  - L2206 **WARNING**: (payload)
  - L2218 **WARNING**: ",
  - L2222 **WARNING**: (payload)
  - L2231 **WARNING**: ",
  - L2235 **WARNING**: (payload)
  - L2244 **WARNING**: ",
  - L2248 **WARNING**: (payload)
  - L2258 **WARNING**: ",
  - L2262 **WARNING**: (payload)
  - L2273 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - threading.Lock (line 121)
  - val.isascii (line 137)
  - val.encode (line 139)
  - orjson.loads (line 143)
  - orjson.loads (line 145)
  - shared\_logic.safe\_encode (line 145)
  - tempfile.gettempdir (line 160)
  - logger\_singleton.console.print (line 163)
  - logger\_singleton.logger.warning (line 170)
  - os.remove (line 175)
  - logger\_singleton.console.print (line 178)
  - logger\_singleton.logger.info (line 185)
  - logger\_singleton.console.print (line 189)
  - logger\_singleton.logger.warning (line 196)
  - \_TEMP\_FILES\_TRACKER.discard (line 197)
  - \_TEMP\_FILES\_TRACKER.add (line 198)
  - os.makedirs (line 208)
  - f.read (line 215)
  - f.write (line 231)
  - orjson.dumps (line 231)
  - time.time (line 239)
  - re.match (line 251)
  - shared\_logic.safe\_get (line 256)
  - cache.get (line 265)
  - shared\_logic.safe\_get (line 267)
  - re.match (line 278)
  - shared\_logic.safe\_filename (line 280)
  - tempfile.gettempdir (line 285)
  - logger\_singleton.console.print (line 288)
  - logger\_singleton.logger.warning (line 295)
  - os.makedirs (line 296)
  - os.remove (line 301)
  - logger\_singleton.console.print (line 304)
  - logger\_singleton.logger.info (line 311)
  - logger\_singleton.console.print (line 315)
  - logger\_singleton.logger.warning (line 322)
  - \_TEMP\_FILES\_TRACKER.discard (line 323)
  - \_TEMP\_FILES\_TRACKER.add (line 324)
  - os.makedirs (line 326)
  - abs\_path.startswith (line 327)
  - re.match (line 341)
  - shared\_logic.safe\_filename (line 343)
  - filename.endswith (line 344)
  - re.sub (line 345)
  - tempfile.gettempdir (line 350)
  - logger\_singleton.console.print (line 353)
  - logger\_singleton.logger.warning (line 360)
  - os.makedirs (line 361)
  - os.remove (line 366)
  - logger\_singleton.console.print (line 369)
- Inbound references:
  - \_get\_label\_cache\_path ← html_scanner.py:211
  - \_get\_label\_cache\_path ← html_scanner.py:229
  - \_get\_label\_cache\_path ← html_scanner.py:258
  - \_load\_label\_cache ← html_scanner.py:226
  - \_load\_label\_cache ← html_scanner.py:237
  - \_load\_label\_cache ← html_scanner.py:246
  - \_load\_label\_cache ← html_scanner.py:264
  - \_save\_label\_cache ← html_scanner.py:240
  - safe\_cache\_path ← html_scanner.py:157
  - safe\_log\_path ← html_scanner.py:1916
  - safe\_log\_path ← html_scanner.py:1958
  - safe\_log\_path ← html_scanner.py:1970
  - is\_trivial\_segment ← html_scanner.py:538
  - is\_trivial\_segment ← html_scanner.py:633
  - segment\_identity\_hash ← html_scanner.py:734
  - segment\_identity\_hash ← html_scanner.py:1479
  - segment\_identity\_hash ← html_scanner.py:1975
  - segment\_identity\_hash ← html_scanner.py:1999
  - embedding\_cache\_hash ← html_scanner.py:535
  - embedding\_cache\_hash ← html_scanner.py:633
  - get\_segment\_embedding ← html_scanner.py:766
  - get\_segment\_embedding ← html_scanner.py:805
  - get\_segment\_embedding ← html_scanner.py:2693
  - deduplicate\_pattern\_kb ← html_scanner.py:2935
  - prune\_embedding\_cache ← html_scanner.py:2723
  - prune\_embedding\_cache ← html_scanner.py:2805
  - auto\_label\_segment ← html_scanner.py:2008
  - \_extract\_clean\_text ← html_scanner.py:844
  - \_extract\_clean\_text ← html_scanner.py:1010
  - \_extract\_clean\_text ← html_scanner.py:1482
  - \_extract\_clean\_text ← html_scanner.py:1699
  - \_extract\_clean\_text ← html_scanner.py:2662
  - \_extract\_clean\_text ← html_scanner.py:3050
  - \_extract\_clean\_text ← html_scanner.py:3062
  - \_extract\_clean\_text ← html_scanner.py:3076
  - \_extract\_clean\_text ← html_scanner.py:3090
  - \_extract\_clean\_text ← html_scanner.py:3105
  - \_extract\_clean\_text ← html_scanner.py:3118
  - \_extract\_clean\_text ← html_scanner.py:3132
  - \_extract\_clean\_text ← html_scanner.py:3144
  - \_extract\_clean\_text ← html_scanner.py:3156
  - \_extract\_clean\_text ← html_scanner.py:3421
  - \_label\_in ← html_scanner.py:1008
  - \_extract\_segments\_by\_label ← html_scanner.py:2661
  - \_extract\_segments\_by\_label ← html_scanner.py:3031
  - \_extract\_segments\_by\_label ← html_scanner.py:3049
  - \_extract\_segments\_by\_label ← html_scanner.py:3061
  - \_extract\_segments\_by\_label ← html_scanner.py:3075
  - \_extract\_segments\_by\_label ← html_scanner.py:3089
  - \_extract\_segments\_by\_label ← html_scanner.py:3104

### utils/json\_export\_loader.py {#webapp-parser-utils-json-export-loader-py}

- Definitions:
  - function: `\_safe\_int` (line 34)
  - function: `\_collapse\_spaces` (line 50)
  - function: `\_strip\_party\_from\_name` (line 54)
  - function: `\_normalize\_candidate` (line 73)
  - class: `NormalizedResultRow` (line 87)
  - class: `ContestCoverage` (line 101)
  - class: `NormalizedExport` (line 120)
  - function: `\_iter\_county\_contests` (line 132)
  - function: `\_normalize\_group\_labels` (line 139)
  - function: `\_derive\_division\_metadata` (line 143)
  - function: `\_build\_context\_snapshot` (line 179)
  - function: `load\_state\_export` (line 223)
  - function: `load\_json\_export` (line 405)
- Imports:
  - **Standard Library** (11):
    - `import json as json` (line 3)
    - `import re as re` (line 4)
    - `from collections import defaultdict` (line 5)
    - `from dataclasses import dataclass` (line 6)
    - `from dataclasses import field` (line 6)
    - `from pathlib import Path` (line 7)
    - `from typing import Dict` (line 8)
    - `from typing import Iterable` (line 8)
    - `from typing import List` (line 8)
    - `from typing import Optional` (line 8)
    - `from typing import Tuple` (line 8)
  - **Local/Project** (7):
    - `from __future__ import annotations` (line 1)
    - `from Context_Integration.Context_Library.constants import
      DEFAULT_TOTAL_RESULT_DISPLAY` (line 10)
    - `from Context_Integration.Context_Library.constants import PARTY_CODE_MAP`
      (line 10)
    - `from Context_Integration.Context_Library.constants import
      normalize_party_label` (line 10)
    - `from Context_Integration.Context_Library.constants import
      normalize_result_group_label` (line 10)
    - `from Context_Integration.librarian import clean_for_json` (line 16)
    - `from contest_normalization import normalize_contest_label` (line 17)
- Outgoing cross-module calls (sample):
  - code.upper (line 29)
  - Context\_Integration.Context\_Library.constants.PARTY\_CODE\_MAP.keys (line
    29)
  - re.compile (line 30)
  - re.compile (line 31)
  - re.compile (line 32)
  - re.compile (line 33)
  - \_EXTRA\_SPACE\_RE.sub (line 51)
  - \_INCUMBENT\_TOKEN\_RE.sub (line 57)
  - \_PARTY\_SUFFIX\_RE.search (line 59)
  - match.group (line 63)
  - match.group (line 66)
  - base.rstrip (line 67)
  - base.strip (line 68)
  - option.get (line 74)
  - option.get (line 76)
  - option.get (line 78)
  - Context\_Integration.Context\_Library.constants.normalize\_party\_label
    (line 80)
  - dataclasses.dataclass (line 86)
  - dataclasses.field (line 112)
  - dataclasses.field (line 113)
  - dataclasses.field (line 128)
  - dataclasses.field (line 129)
  - payload.get (line 133)
  - county.get (line 134)
  - county.get (line 135)
  - Context\_Integration.Context\_Library.constants.normalize\_result\_group\_label
    (line 140)
  - statewide\_contest.get (line 146)
  - name.lower (line 147)
  - statewide\_contest.get (line 160)
  - statewide\_contest.get (line 160)
  - \_DISTRICT\_LABEL\_RE.search (line 162)
  - district\_match.group (line 164)
  - coverage.items (line 195)
  - coverage.items (line 203)
  - payload.get (line 207)
  - payload.get (line 208)
  - payload.get (line 215)
  - Context\_Integration.librarian.clean\_for\_json (line 218)
  - pathlib.Path (line 224)
  - json.loads (line 225)
  - path.read\_text (line 225)
  - payload.get (line 227)
  - contest.get (line 228)
  - collections.defaultdict (line 232)
  - collections.defaultdict (line 232)
  - contest.get (line 237)
  - contest.get (line 238)
  - contest.get (line 239)
  - contest\_normalization.normalize\_contest\_label (line 240)
  - coverage.setdefault (line 245)
- Inbound references:
  - \_collapse\_spaces ← json_export_loader.py:58
  - \_collapse\_spaces ← json_export_loader.py:69
  - \_collapse\_spaces ← json_export_loader.py:82
  - \_strip\_party\_from\_name ← json_export_loader.py:77
  - \_normalize\_candidate ← json_export_loader.py:278
  - NormalizedResultRow ← json_export_loader.py:283
  - NormalizedResultRow ← json_export_loader.py:319
  - NormalizedResultRow ← json_export_loader.py:346
  - ContestCoverage ← json_export_loader.py:247
  - NormalizedExport ← json_export_loader.py:392
  - \_iter\_county\_contests ← json_export_loader.py:236
  - \_normalize\_group\_labels ← json_export_loader.py:281
  - \_normalize\_group\_labels ← json_export_loader.py:317
  - \_derive\_division\_metadata ← json_export_loader.py:383
  - \_build\_context\_snapshot ← json_export_loader.py:390
  - load\_state\_export ← json_export_loader.py:407

### utils/location\_helpers.py {#webapp-parser-utils-location-helpers-py}

- Definitions:
  - function: `\_normalize\_location\_text` (line 75)
  - function: `\_location\_phrases` (line 84)
  - function: `is\_strict\_location\_header` (line 127)
  - function: `collect\_location\_headers` (line 149)
  - function: `format\_location\_fragment` (line 189)
  - function: `attach\_precinct\_column` (line 238)
- Imports:
  - **Standard Library** (8):
    - `import re as re` (line 3)
    - `from functools import lru_cache` (line 4)
    - `from typing import Any` (line 5)
    - `from typing import Dict` (line 5)
    - `from typing import Iterable` (line 5)
    - `from typing import List` (line 5)
    - `from typing import Sequence` (line 5)
    - `from typing import Tuple` (line 5)
  - **Local/Project** (5):
    - `from __future__ import annotations` (line 1)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_ABBREVIATIONS` (line 7)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_KEYWORDS` (line 7)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_SYNONYM_MAP` (line 7)
    - `from detect import is_location_header` (line 12)
- Outgoing cross-module calls (sample):
  - Context\_Integration.Context\_Library.constants.LOCATION\_ABBREVIATIONS.keys
    (line 69)
  - re.sub (line 70)
  - abbr.lower (line 70)
  - \_SHORT\_LOCATION\_TOKENS.add (line 72)
  - value.lower (line 78)
  - re.sub (line 79)
  - re.sub (line 80)
  - phrases.add (line 89)
  - Context\_Integration.Context\_Library.constants.LOCATION\_SYNONYM\_MAP.items
    (line 90)
  - phrases.add (line 93)
  - phrases.add (line 96)
  - phrases.update (line 97)
  - functools.lru\_cache (line 83)
  - normalized.split (line 136)
  - detect.is\_location\_header (line 144)
  - ordered.append (line 160)
  - seen.add (line 161)
  - ordered.append (line 165)
  - seen.add (line 166)
  - extra.strip (line 171)
  - ordered.append (line 173)
  - seen.add (line 174)
  - header.strip (line 179)
  - ordered.append (line 183)
  - seen.add (line 184)
  - header.lower (line 196)
  - header.strip (line 234)
  - value.strip (line 254)
  - part.lower (line 261)
  - seen.add (line 264)
  - ordered.append (line 265)
  - canonical\_label.lower (line 272)
  - header.strip (line 276)
  - working\_headers.insert (line 282)
  - header.strip (line 289)
  - header.strip (line 291)
  - normalized\_seen.add (line 302)
  - ordered\_locations.append (line 303)
  - row.get (line 315)
  - row.get (line 322)
  - fragments.append (line 324)
  - row.setdefault (line 331)
- Inbound references:
  - \_normalize\_location\_text ← location_helpers.py:87
  - \_normalize\_location\_text ← location_helpers.py:91
  - \_normalize\_location\_text ← location_helpers.py:94
  - \_normalize\_location\_text ← location_helpers.py:131
  - \_location\_phrases ← location_helpers.py:140
  - is\_strict\_location\_header ← location_helpers.py:182
  - collect\_location\_headers ← location_helpers.py:294
  - format\_location\_fragment ← location_helpers.py:322

### utils/logger\_singleton.py {#webapp-parser-utils-logger-singleton-py}

- Definitions:
  - function: `set\_log\_level` (line 20)
  - function: `get\_shared\_logger` (line 23)
- Imports:
  - **Standard Library** (1):
    - `import os as os` (line 8)
  - **Local/Project** (3):
    - `from __future__ import annotations` (line 1)
    - `from shared_logger import RichConsoleProxy` (line 10)
    - `from shared_logger import SharedLogger` (line 10)
- Outgoing cross-module calls (sample):
  - shared\_logger.SharedLogger (line 16)
  - shared\_logger.RichConsoleProxy (line 17)
  - logger.set\_level (line 21)

### utils/merge\_utils.py {#webapp-parser-utils-merge-utils-py}

> merge_utils.py

- Definitions:
  - function: `merge\_table\_data` (line 19)
- Imports:
  - **Standard Library** (4):
    - `from typing import Any` (line 7)
    - `from typing import Dict` (line 7)
    - `from typing import List` (line 7)
    - `from typing import Tuple` (line 7)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 5)
    - `from salvage import collapse_ballot_synonym_columns` (line 16)
- Outgoing cross-module calls (sample):
  - ordered.append (line 31)
  - r.get (line 36)
  - d.values (line 41)
  - rows\_out.append (line 42)
  - salvage.collapse\_ballot\_synonym\_columns (line 45)

### utils/metrics\_prom.py {#webapp-parser-utils-metrics-prom-py}

> Prometheus metrics integration (optional).

- Definitions:
  - function: `increment\_test\_counter` (line 53)
  - function: `\_push\_registry\_async` (line 68)
  - function: `increment\_prom\_counter` (line 85)
- Imports:
  - **Standard Library** (3):
    - `import os as os` (line 5)
    - `import threading as threading` (line 6)
    - `from typing import Dict` (line 7)
- Outgoing cross-module calls (sample):
  - \_counters.get (line 57)
  - c.inc (line 60)
  - threading.Thread (line 79)
  - t.start (line 80)
  - \_counters.get (line 89)
  - c.inc (line 92)
- Inbound references:
  - increment\_test\_counter ← Smart_Elections_Parser_Webapp.py:574
  - \_push\_registry\_async ← metrics_prom.py:62
  - \_push\_registry\_async ← metrics_prom.py:95
  - increment\_prom\_counter ← telemetry_agg.py:57

### utils/misc\_utils.py {#webapp-parser-utils-misc-utils-py}

- Definitions:
  - function: `load\_processed\_urls` (line 29)
  - function: `safe\_db\_path` (line 48)
  - function: `load\_output\_cache` (line 51)
  - function: `file\_hash` (line 60)
  - function: `is\_safe\_path` (line 75)
  - function: `extract\_url\_and\_label` (line 92)
- Imports:
  - **Standard Library** (7):
    - `import hashlib as hashlib` (line 7)
    - `import os as os` (line 8)
    - `import re as re` (line 9)
    - `from pathlib import Path` (line 10)
    - `from typing import Any` (line 11)
    - `from typing import Dict` (line 11)
    - `from typing import List` (line 11)
  - **Third-party** (1):
    - `import orjson as orjson` (line 13)
  - **Local/Project** (11):
    - `from __future__ import annotations` (line 1)
    - `from config import CONTEXT_LIBRARY_PATH` (line 15)
    - `from config import OUTPUT_CACHE` (line 15)
    - `from config import PROCESSED_URLS_FILE` (line 15)
    - `from config import URL_ALLOWLIST_HOSTS` (line 15)
    - `from config import URL_ALLOWLIST_SUFFIXES` (line 15)
    - `from config import URL_BLOCK_PRIVATE_IPS` (line 15)
    - `from config import URL_ENFORCE_ALLOWLIST` (line 15)
    - `from logger_singleton import logger` (line 24)
    - `from shared_logic import safe_get` (line 25)
    - `from shared_logic import safe_validate_external_url` (line 25)
- Task markers:
  - L127 **WARNING**: ({
  - L128 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 30)
  - cache\_path.exists (line 31)
  - cache\_path.open (line 33)
  - orjson.loads (line 35)
  - f.read (line 35)
  - shared\_logic.safe\_get (line 42)
  - pathlib.Path (line 49)
  - pathlib.Path (line 54)
  - safe\_path.exists (line 55)
  - orjson.loads (line 58)
  - line.strip (line 58)
  - logger\_singleton.logger.error (line 63)
  - hashlib.new (line 66)
  - f.read (line 68)
  - h.update (line 69)
  - h.hexdigest (line 70)
  - logger\_singleton.logger.error (line 72)
  - pathlib.Path (line 84)
  - pathlib.Path (line 84)
  - line.strip (line 106)
  - s.startswith (line 107)
  - re.search (line 112)
  - m.group (line 117)
  - shared\_logic.safe\_validate\_external\_url (line 118)
  - logger\_singleton.logger.warning (line 127)
  - s.replace (line 135)
  - m.group (line 135)
  - re.sub (line 137)
  - re.sub (line 138)
- Inbound references:
  - safe\_db\_path ← misc_utils.py:54

### utils/ml\_table\_detector.py {#webapp-parser-utils-ml-table-detector-py}

- Definitions:
  - function: `detect\_tables\_ml` (line 47)
  - function: `\_ml\_detect\_tables` (line 115)
  - function: `\_vision\_detect\_tables` (line 134)
  - function: `\_extract\_table\_from\_selectolax` (line 145)
  - function: `\_looks\_like\_table\_selectolax` (line 188)
  - function: `\_extract\_table\_from\_selectolax` (line 213)
  - function: `\_looks\_like\_table\_selectolax` (line 254)
  - function: `\_extract\_table\_like\_structure\_selectolax` (line 284)
  - function: `\_regex\_table\_detection` (line 327)
  - function: `\_normalize\_header` (line 366)
- Imports:
  - **Standard Library** (7):
    - `import re as re` (line 26)
    - `from collections import Counter` (line 27)
    - `from typing import Any` (line 28)
    - `from typing import Dict` (line 28)
    - `from typing import List` (line 28)
    - `from typing import Optional` (line 28)
    - `from typing import Tuple` (line 28)
  - **Local/Project** (7):
    - `from __future__ import annotations` (line 1)
    - `from selectolax.parser import HTMLParser` (line 30)
    - `from config import TABLE_MODEL_PATH` (line 32)
    - `from browser_utils import safe_attributes` (line 35)
    - `from browser_utils import safe_content` (line 35)
    - `from logger_singleton import logger` (line 36)
    - `from model_registry import TableDetectionModel` (line 37)
- Outgoing cross-module calls (sample):
  - re.compile (line 40)
  - re.compile (line 41)
  - options.get (line 53)
  - options.get (line 54)
  - options.get (line 55)
  - options.get (line 56)
  - options.get (line 63)
  - model\_registry.TableDetectionModel.load\_from\_checkpoint (line 65)
  - table\_model.predict\_tables (line 67)
  - tables.extend (line 69)
  - logger\_singleton.logger.error (line 71)
  - selectolax.parser.HTMLParser (line 76)
  - html\_tree.css (line 79)
  - tables.append (line 82)
  - html\_tree.css (line 85)
  - tables.append (line 89)
  - tables.extend (line 95)
  - tables.extend (line 101)
  - t.get (line 107)
  - unique\_tables.append (line 110)
  - seen.add (line 111)
  - browser\_utils.safe\_content (line 150)
  - selectolax.parser.HTMLParser (line 151)
  - html\_tree.css (line 156)
  - cell.text (line 166)
  - row.css (line 172)
  - row.css (line 174)
  - row\_data.values (line 176)
  - data.append (line 177)
  - browser\_utils.safe\_content (line 192)
  - selectolax.parser.HTMLParser (line 193)
  - browser\_utils.safe\_attributes (line 196)
  - attrs.get (line 197)
  - html\_tree.css (line 200)
  - child.css (line 202)
  - html\_tree.css (line 208)
  - table\_node.css (line 221)
  - c.text (line 230)
  - row.css (line 239)
  - row.css (line 239)
  - row\_map.values (line 241)
  - data.append (line 242)
  - tag.lower (line 261)
  - browser\_utils.safe\_attributes (line 264)
  - attrs.get (line 265)
  - node.css (line 268)
  - ch.css (line 272)
  - node.css (line 278)
  - tag.lower (line 293)
  - node.css (line 296)
- Inbound references:
  - detect\_tables\_ml ← extraction_strategies.py:164
  - \_vision\_detect\_tables ← ml_table_detector.py:93
  - \_extract\_table\_from\_selectolax ← ml_table_detector.py:80
  - \_looks\_like\_table\_selectolax ← ml_table_detector.py:86
  - \_extract\_table\_like\_structure\_selectolax ← ml_table_detector.py:87
  - \_regex\_table\_detection ← ml_table_detector.py:99
  - \_normalize\_header ← location_helpers.py:299
  - \_normalize\_header ← location_helpers.py:307
  - \_normalize\_header ← ml_table_detector.py:108

### utils/model\_registry.py {#webapp-parser-utils-model-registry-py}

- Definitions:
  - function: `\_hf\_offline` (line 41)
  - function: `load\_vocab\_from\_file` (line 50)
  - function: `build\_reverse\_vocab` (line 68)
  - function: `advanced\_tokenizer` (line 92)
  - class: `ModelRegistry` (line 255)
- Imports:
  - **Standard Library** (9):
    - `import os as os` (line 12)
    - `import re as re` (line 13)
    - `import subprocess as subprocess` (line 14)
    - `import sys as sys` (line 15)
    - `import threading as threading` (line 16)
    - `from collections import Counter` (line 17)
    - `from typing import Any` (line 18)
    - `from typing import Callable` (line 18)
    - `from typing import Dict` (line 18)
  - **Local/Project** (8):
    - `from __future__ import annotations` (line 1)
    - `from selectolax.parser import HTMLParser` (line 30)
    - `from config import MODEL_DIR` (line 32)
    - `from config import PROJECT_ROOT` (line 32)
    - `from config import TABLE_MODEL_PATH` (line 32)
    - `from config import VOCAB_DIR` (line 32)
    - `from Context_Integration.librarian import load_context_library` (line 33)
    - `from logger_singleton import logger` (line 34)
- Task markers:
  - L425 **WARNING**: (f"Failed loading local override for SentenceTransformer:
    {e}")
  - L445 **WARNING**: ("TRANSFORMERS_OFFLINE/HUGGINGFACE_HUB_OFFLINE set;
    skipping HF download. Embeddings disabled.")
  - L462 **WARNING**: for noisy environments
  - L465 **WARNING**: (f"Failed to load base SentenceTransformer (network/DNS).
    Running without embeddings. Error: {e}")
- Outgoing cross-module calls (sample):
  - threading.Lock (line 39)
  - os.getenv (line 43)
  - os.getenv (line 44)
  - os.getenv (line 45)
  - line.strip (line 59)
  - logger\_singleton.logger.error (line 62)
  - logger\_singleton.logger.error (line 65)
  - vocab.items (line 74)
  - vocab.items (line 75)
  - re.findall (line 97)
  - text.lower (line 97)
  - WORD2IDX.get (line 98)
  - torch.tensor (line 102)
  - nn.Embedding (line 114)
  - nn.LSTM (line 115)
  - nn.Linear (line 116)
  - nn.Linear (line 117)
  - nn.Linear (line 118)
  - nn.Linear (line 119)
  - self.embedding (line 122)
  - self.encoder (line 123)
  - torch.cat (line 124)
  - self.year\_head (line 125)
  - self.state\_head (line 126)
  - self.county\_head (line 127)
  - self.type\_head (line 128)
  - WORD2IDX.values (line 135)
  - YEAR2IDX.values (line 137)
  - STATE2IDX.values (line 138)
  - COUNTY2IDX.values (line 139)
  - TYPE2IDX.values (line 140)
  - model.load\_state\_dict (line 142)
  - torch.load (line 142)
  - model.eval (line 143)
  - torch.no\_grad (line 152)
  - self.forward (line 153)
  - F.softmax (line 154)
  - F.softmax (line 155)
  - F.softmax (line 156)
  - F.softmax (line 157)
  - year\_probs.argmax (line 159)
  - state\_probs.argmax (line 160)
  - county\_probs.argmax (line 161)
  - type\_probs.argmax (line 162)
  - IDX2YEAR.get (line 166)
  - IDX2STATE.get (line 171)
  - IDX2COUNTY.get (line 176)
  - IDX2TYPE.get (line 181)
  - nn.Embedding (line 204)
  - nn.LSTM (line 205)
- Inbound references:
  - \_hf\_offline ← model_registry.py:444
  - load\_vocab\_from\_file ← model_registry.py:79
  - load\_vocab\_from\_file ← model_registry.py:80
  - load\_vocab\_from\_file ← model_registry.py:81
  - load\_vocab\_from\_file ← model_registry.py:82
  - load\_vocab\_from\_file ← model_registry.py:83
  - build\_reverse\_vocab ← model_registry.py:85
  - build\_reverse\_vocab ← model_registry.py:86
  - build\_reverse\_vocab ← model_registry.py:87
  - build\_reverse\_vocab ← model_registry.py:88
  - advanced\_tokenizer ← model_registry.py:151

### utils/models.py {#webapp-parser-utils-models-py}

- Definitions:
  - class: `MetaDataProtocol` (line 36)
  - class: `DeclarativeBaseProtocol` (line 40)
  - class: `ElectionTypeEnum` (line 45)
  - class: `OfficeLevelEnum` (line 51)
  - class: `StatusEnum` (line 57)
  - class: `State` (line 64)
  - class: `County` (line 76)
  - class: `District` (line 89)
  - class: `Office` (line 104)
  - class: `Party` (line 115)
  - class: `Candidate` (line 125)
  - class: `Contest` (line 143)
  - class: `Result` (line 171)
  - class: `Panel` (line 189)
  - class: `Button` (line 204)
  - class: `CandidatePanel` (line 217)
  - class: `LocationPanel` (line 234)
  - class: `Heading` (line 251)
  - class: `BallotType` (line 267)
  - class: `ResultsTimestamp` (line 284)
  - class: `PartyLabel` (line 299)
  - class: `VoteMethod` (line 314)
  - class: `Entity` (line 331)
  - class: `MiscEntity` (line 341)
  - class: `TableStructure` (line 353)
  - class: `BatchMetadata` (line 369)
  - class: `StagingElectionResult` (line 384)
  - class: `WarehouseElectionResult` (line 402)
  - class: `DataFrameworkPreviewCache` (line 429)
  - class: `EmbeddingCache` (line 450)
  - class: `Alert` (line 462)
  - function: `main` (line 476)
- Imports:
  - **Standard Library** (6):
    - `import enum as enum` (line 3)
    - `import uuid as uuid` (line 4)
    - `from datetime import datetime` (line 5)
    - `from datetime import timezone` (line 5)
    - `from typing import Any` (line 6)
    - `from typing import Protocol` (line 6)
  - **Third-party** (19):
    - `from sqlalchemy import JSON` (line 12)
    - `from sqlalchemy import Boolean` (line 12)
    - `from sqlalchemy import Column` (line 12)
    - `from sqlalchemy import DateTime` (line 12)
    - `from sqlalchemy import Enum` (line 12)
    - `from sqlalchemy import Float` (line 12)
    - `from sqlalchemy import ForeignKey` (line 12)
    - `from sqlalchemy import Index` (line 12)
    - `from sqlalchemy import Integer` (line 12)
    - `from sqlalchemy import LargeBinary` (line 12)
    - `from sqlalchemy import String` (line 12)
    - `from sqlalchemy import Text` (line 12)
    - `from sqlalchemy import UniqueConstraint` (line 12)
    - `from sqlalchemy import inspect` (line 12)
    - `from sqlalchemy.dialects.postgresql import UUID` (line 28)
    - `from sqlalchemy.engine.base import Engine` (line 29)
    - `from sqlalchemy.orm import backref` (line 30)
    - `from sqlalchemy.orm import declarative_base` (line 30)
    - `from sqlalchemy.orm import relationship` (line 30)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 1)
    - `from logger_singleton import logger` (line 32)
- Outgoing cross-module calls (sample):
  - sqlalchemy.orm.declarative\_base (line 34)
  - sqlalchemy.Column (line 69)
  - sqlalchemy.Column (line 70)
  - sqlalchemy.Column (line 71)
  - sqlalchemy.orm.relationship (line 72)
  - sqlalchemy.orm.relationship (line 73)
  - sqlalchemy.orm.relationship (line 74)
  - sqlalchemy.Column (line 81)
  - sqlalchemy.Column (line 82)
  - sqlalchemy.Column (line 83)
  - sqlalchemy.ForeignKey (line 83)
  - sqlalchemy.orm.relationship (line 84)
  - sqlalchemy.orm.relationship (line 85)
  - sqlalchemy.orm.relationship (line 86)
  - sqlalchemy.UniqueConstraint (line 87)
  - sqlalchemy.Column (line 94)
  - sqlalchemy.Column (line 95)
  - sqlalchemy.Column (line 96)
  - sqlalchemy.Column (line 97)
  - sqlalchemy.ForeignKey (line 97)
  - sqlalchemy.orm.relationship (line 98)
  - sqlalchemy.Column (line 99)
  - sqlalchemy.ForeignKey (line 99)
  - sqlalchemy.orm.relationship (line 100)
  - sqlalchemy.orm.relationship (line 101)
  - sqlalchemy.orm.relationship (line 102)
  - sqlalchemy.Column (line 109)
  - sqlalchemy.Column (line 110)
  - sqlalchemy.Column (line 111)
  - sqlalchemy.Enum (line 111)
  - sqlalchemy.orm.relationship (line 112)
  - sqlalchemy.orm.relationship (line 113)
  - sqlalchemy.Column (line 120)
  - sqlalchemy.Column (line 121)
  - sqlalchemy.Column (line 122)
  - sqlalchemy.orm.relationship (line 123)
  - sqlalchemy.Column (line 130)
  - sqlalchemy.Column (line 131)
  - sqlalchemy.Column (line 132)
  - sqlalchemy.ForeignKey (line 132)
  - sqlalchemy.orm.relationship (line 133)
  - sqlalchemy.Column (line 134)
  - sqlalchemy.ForeignKey (line 134)
  - sqlalchemy.orm.relationship (line 135)
  - sqlalchemy.Column (line 136)
  - sqlalchemy.ForeignKey (line 136)
  - sqlalchemy.orm.relationship (line 137)
  - sqlalchemy.orm.relationship (line 138)
  - sqlalchemy.Column (line 139)
  - sqlalchemy.Column (line 140)
- Inbound references:
  - OfficeLevelEnum ← election_data_services.py:574
  - OfficeLevelEnum ← election_data_services.py:578
  - Entity ← election_data_services.py:645
  - BatchMetadata ← election_data_services.py:690
  - EmbeddingCache ← election_data_services.py:730
  - Alert ← election_data_services.py:711

### utils/output\_utils.py {#webapp-parser-utils-output-utils-py}

- Definitions:
  - function: `coerce\_percent\_strings` (line 42)
  - function: `get\_project\_root` (line 50)
  - function: `get\_output\_root` (line 54)
  - function: `safe\_join` (line 66)
  - function: `get\_output\_path` (line 89)
  - function: `format\_timestamp` (line 189)
  - function: `update\_output\_cache` (line 192)
  - function: `check\_existing\_output` (line 213)
  - function: `convert\_sets\_to\_lists` (line 255)
  - function: `deep\_merge\_dicts` (line 265)
  - function: `\_slug` (line 282)
  - function: `build\_filename\_triplet` (line 292)
  - function: `\_ensure\_dir` (line 306)
  - function: `\_coerce\_headers` (line 312)
  - function: `apply\_results\_conditional\_formatting` (line 324)
  - function: `export\_dataframe\_with\_format` (line 361)
  - function: `\_compute\_structure\_hash` (line 370)
  - function: `finalize\_election\_output` (line 384)
- Imports:
  - **Standard Library** (11):
    - `import csv as csv` (line 3)
    - `import datetime as dt` (line 4)
    - `import hashlib as hashlib` (line 5)
    - `import os as os` (line 6)
    - `import re as re` (line 12)
    - `from collections import deque` (line 13)
    - `from datetime import datetime` (line 14)
    - `from typing import Any` (line 15)
    - `from typing import Dict` (line 15)
    - `from typing import List` (line 15)
    - `from typing import Optional` (line 15)
  - **Third-party** (2):
    - `import orjson as orjson` (line 17)
    - `import pandas as pd` (line 18)
  - **Local/Project** (17):
    - `from __future__ import annotations` (line 1)
    - `from config import BASE_DIR` (line 20)
    - `from config import LOG_DIR` (line 20)
    - `from config import OUTPUT_CACHE` (line 20)
    - `from config import OUTPUT_DIR` (line 20)
    - `from logger_singleton import logger` (line 21)
    - `from pivot import transform_wide_to_smart_standard` (line 22)
    - `from rawjson_utils import extract_rawjson_enrichment_from_rows` (line 23)
    - `from rawjson_utils import offload_rawjson_to_ndjson as
      _shared_offload_rawjson_to_ndjson` (line 26)
    - `from shared_logic import is_path_safe` (line 29)
    - `from shared_logic import safe_filename` (line 29)
    - `from shared_logic import safe_get` (line 29)
    - `from shared_logic import safe_get_first` (line 29)
    - `from shared_logic import safe_items` (line 29)
    - `from shared_logic import safe_join_path` (line 29)
    - `from shared_logic import safe_lower` (line 29)
    - `from shared_logic import safe_resolve_path` (line 29)
- Task markers:
  - L136 **WARNING**: ("\[yellow\]\[OUTPUT\] Year could not be verified. Using
    'Unknown'.\[/yellow\]")
  - L139 **WARNING**: ("\[yellow\]\[OUTPUT\] contests could not be verified.
    Using 'unknown_contests'.\[/yellow\]")
  - L610 **WARNING**: (f"\[OUTPUT_UTILS\] Enrichment build failed: {e}")
  - L689 **WARNING**: (f"\[OUTPUT_UTILS\] XLSX export failed: {e}")
- Outgoing cross-module calls (sample):
  - re.compile (line 40)
  - row.items (line 43)
  - PERCENT\_COL\_REGEX.search (line 44)
  - v.replace (line 45)
  - sv.replace (line 46)
  - shared\_logic.safe\_resolve\_path (line 61)
  - shared\_logic.safe\_join\_path (line 83)
  - logger\_singleton.logger.error (line 86)
  - shared\_logic.safe\_get (line 101)
  - coordinator.get\_states (line 102)
  - shared\_logic.safe\_get\_first (line 102)
  - shared\_logic.safe\_get (line 104)
  - coordinator.get\_precincts (line 105)
  - shared\_logic.safe\_get\_first (line 105)
  - shared\_logic.safe\_get (line 107)
  - shared\_logic.safe\_get (line 108)
  - shared\_logic.safe\_get (line 109)
  - coordinator.get\_years (line 116)
  - shared\_logic.safe\_get\_first (line 118)
  - shared\_logic.safe\_get (line 120)
  - shared\_logic.safe\_lower (line 121)
  - coordinator.get\_contests (line 123)
  - shared\_logic.safe\_get\_first (line 125)
  - shared\_logic.safe\_get (line 127)
  - shared\_logic.safe\_get (line 131)
  - logger\_singleton.logger.warning (line 136)
  - logger\_singleton.logger.warning (line 139)
  - safe\_components.append (line 148)
  - shared\_logic.safe\_filename (line 148)
  - shared\_logic.safe\_lower (line 148)
  - safe\_components.append (line 150)
  - shared\_logic.safe\_filename (line 150)
  - shared\_logic.safe\_lower (line 150)
  - safe\_components.append (line 152)
  - shared\_logic.safe\_filename (line 152)
  - shared\_logic.safe\_lower (line 152)
  - safe\_components.append (line 154)
  - shared\_logic.safe\_filename (line 154)
  - safe\_components.append (line 156)
  - safe\_components.append (line 158)
  - shared\_logic.safe\_filename (line 158)
  - shared\_logic.safe\_lower (line 158)
  - shared\_logic.safe\_filename (line 158)
  - shared\_logic.safe\_filename (line 161)
  - safe\_components.append (line 162)
  - safe\_components.append (line 164)
  - safe\_components.append (line 166)
  - shared\_logic.safe\_filename (line 166)
  - shared\_logic.safe\_join\_path (line 173)
  - shared\_logic.is\_path\_safe (line 176)
- Inbound references:
  - coerce\_percent\_strings ← output_utils.py:498
  - get\_project\_root ← output_utils.py:56
  - get\_output\_root ← output_utils.py:169
  - format\_timestamp ← output_utils.py:198
  - convert\_sets\_to\_lists ← output_utils.py:257
  - convert\_sets\_to\_lists ← output_utils.py:261
  - deep\_merge\_dicts ← output_utils.py:277
  - \_slug ← output_utils.py:297
  - \_slug ← output_utils.py:298
  - \_slug ← output_utils.py:299
  - build\_filename\_triplet ← output_utils.py:143
  - build\_filename\_triplet ← output_utils.py:413
  - \_ensure\_dir ← output_utils.py:405
  - \_ensure\_dir ← output_utils.py:428
  - \_coerce\_headers ← output_utils.py:465
  - apply\_results\_conditional\_formatting ← output_utils.py:368
  - \_compute\_structure\_hash ← output_utils.py:437
  - finalize\_election\_output ← table_core.py:347
  - finalize\_election\_output ← table_core.py:475

### utils/pattern\_extractor.py {#webapp-parser-utils-pattern-extractor-py}

> pattern_extractor.py

- Definitions:
  - function: `load\_dom\_patterns` (line 17)
  - function: `extract\_with\_patterns` (line 29)
- Imports:
  - **Standard Library** (6):
    - `import json as json` (line 8)
    - `import os as os` (line 9)
    - `from typing import Any` (line 10)
    - `from typing import Dict` (line 10)
    - `from typing import List` (line 10)
    - `from typing import Tuple` (line 10)
  - **Local/Project** (4):
    - `from __future__ import annotations` (line 6)
    - `from detect import normalize_header` (line 12)
    - `from logger_singleton import logger` (line 13)
    - `from shared_logic import safe_get` (line 14)
- Task markers:
  - L26 **WARNING**: (f"\[PATTERN\] load fail {e}")
  - L95 **WARNING**: (f"\[PATTERN\] pattern error {pat.get('name')}: {e}")
- Outgoing cross-module calls (sample):
  - json.load (line 22)
  - logger\_singleton.logger.warning (line 26)
  - shared\_logic.safe\_get (line 46)
  - pat.get (line 56)
  - pat.get (line 57)
  - pat.get (line 62)
  - cdef.get (line 70)
  - cdef.get (line 71)
  - tmp\_rows.append (line 80)
  - pat.get (line 81)
  - cdef.get (line 85)
  - cdef.get (line 85)
  - detect.normalize\_header (line 86)
  - hdrs.append (line 88)
  - seen.add (line 89)
  - logger\_singleton.logger.warning (line 95)
  - pat.get (line 95)
- Inbound references:
  - load\_dom\_patterns ← pattern_extractor.py:47

### utils/pdf\_table\_utils.py {#webapp-parser-utils-pdf-table-utils-py}

- Definitions:
  - function: `\_recon\_debug\_enabled` (line 72)
  - function: `\_record\_recon\_event` (line 80)
  - function: `consume\_reconstruction\_debug\_events` (line 86)
  - function: `detect\_district\_heading` (line 156)
  - function: `build\_contest\_regex` (line 223)
  - function: `normalize\_text\_token` (line 246)
  - function: `token\_set` (line 252)
  - function: `header\_signature` (line 256)
  - function: `looks\_like\_candidate\_header` (line 262)
  - function: `compute\_header\_richness` (line 276)
  - function: `is\_numeric\_like` (line 301)
  - function: `normalize\_numeric\_token` (line 312)
  - function: `compute\_numeric\_fill` (line 321)
  - function: `evaluate\_table\_candidate\_quality` (line 344)
  - function: `find\_best\_header\_match` (line 428)
  - function: `normalize\_anchor\_value` (line 449)
  - function: `merge\_camelot\_with\_text` (line 455)
  - function: `best\_title\_match\_idx` (line 519)
  - function: `extract\_contest\_block` (line 543)
  - function: `parse\_candidate\_line` (line 663)
  - function: `extract\_candidate\_totals\_from\_lines` (line 751)
  - function: `\_split\_crammed\_numeric\_row` (line 789)
  - function: `split\_ws\_blocks` (line 831)
  - function: `is\_bad\_header\_line` (line 849)
  - function: `table\_looks\_bad` (line 887)
  - function: `find\_header\_line` (line 903)
  - function: `extract\_table\_by\_whitespace` (line 926)
  - function: `matches\_anchor\_header` (line 950)
  - function: `\_extract\_anchor\_tokens` (line 976)
  - function: `\_looks\_like\_vertical\_stub` (line 1006)
  - function: `\_merge\_token\_fragments` (line 1022)
  - function: `\_clean\_candidate\_stub` (line 1035)
  - function: `\_compose\_vertical\_headers` (line 1045)
  - function: `\_gather\_vertical\_rows` (line 1103)
  - function: `\_reconstruct\_vertical\_table` (line 1153)
  - function: `\_combine\_header\_rows` (line 1230)
  - function: `reconstruct\_columnar\_block` (line 1252)
  - function: `extract\_party\_lookup\_from\_lines` (line 1836)
  - function: `parse\_candidate\_header\_with\_party` (line 1856)
  - function: `coerce\_vote\_value\_for\_reconstruction` (line 1901)
- Imports:
  - **Standard Library** (6):
    - `import os as os` (line 11)
    - `import re as re` (line 12)
    - `from collections import Counter` (line 13)
    - `from typing import Any` (line 14)
    - `from typing import Iterable` (line 14)
    - `from typing import Sequence` (line 14)
  - **Local/Project** (8):
    - `from __future__ import annotations` (line 1)
    - `from Context_Integration.Context_Library.constants import
      BALLOT_INLINE_TOKEN_ALIASES` (line 16)
    - `from Context_Integration.Context_Library.constants import BALLOT_TYPES`
      (line 16)
    - `from Context_Integration.Context_Library.constants import
      CONTEST_KEYWORDS` (line 16)
    - `from Context_Integration.Context_Library.constants import PARTY_KEYWORDS`
      (line 16)
    - `from Context_Integration.Context_Library.constants import
      TABLE_ANCHOR_LABELS` (line 16)
    - `from Context_Integration.Context_Library.constants import
      normalize_party_label` (line 16)
    - `from header_utils import collapse_multiline_header` (line 24)
- Outgoing cross-module calls (sample):
  - label.lower (line 49)
  - flag.strip (line 76)
  - \_RECON\_DEBUG\_EVENTS.append (line 83)
  - \_RECON\_DEBUG\_EVENTS.clear (line 90)
  - re.compile (line 94)
  - re.compile (line 95)
  - re.compile (line 147)
  - re.compile (line 148)
  - re.compile (line 149)
  - re.compile (line 151)
  - re.sub (line 160)
  - text.strip (line 160)
  - cleaned.lower (line 163)
  - tok.strip (line 171)
  - cleaned.replace (line 171)
  - token.strip (line 176)
  - \_DISTRICT\_NUM\_RE.match (line 177)
  - match.group (line 179)
  - token.lower (line 184)
  - normalized.startswith (line 185)
  - district\_positions.append (line 186)
  - district\_number.lstrip (line 217)
  - cleaned.strip (line 219)
  - phrase.strip (line 226)
  - re.split (line 228)
  - phrase.strip (line 228)
  - re.escape (line 231)
  - token.replace (line 232)
  - token.replace (line 233)
  - escaped.append (line 234)
  - parts.append (line 237)
  - re.compile (line 239)
  - re.compile (line 240)
  - re.sub (line 248)
  - re.sub (line 249)
  - re.findall (line 253)
  - header\_utils.collapse\_multiline\_header (line 257)
  - re.findall (line 258)
  - collapsed.lower (line 258)
  - label.strip (line 263)
  - ch.isalpha (line 270)
  - h.lower (line 289)
  - token.strip (line 304)
  - ch.isalpha (line 307)
  - \_NUMERIC\_TOKEN\_RE.match (line 309)
  - text.endswith (line 316)
  - text.replace (line 317)
  - text.replace (line 318)
  - row.get (line 330)
  - candidate\_headers.append (line 353)
- Inbound references:
  - \_recon\_debug\_enabled ← pdf_table_utils.py:81
  - \_record\_recon\_event ← pdf_table_utils.py:1159
  - \_record\_recon\_event ← pdf_table_utils.py:1208
  - \_record\_recon\_event ← pdf_table_utils.py:1215
  - \_record\_recon\_event ← pdf_table_utils.py:1221
  - \_record\_recon\_event ← pdf_table_utils.py:1258
  - \_record\_recon\_event ← pdf_table_utils.py:1404
  - \_record\_recon\_event ← pdf_table_utils.py:1438
  - \_record\_recon\_event ← pdf_table_utils.py:1489
  - \_record\_recon\_event ← pdf_table_utils.py:1504
  - \_record\_recon\_event ← pdf_table_utils.py:1513
  - \_record\_recon\_event ← pdf_table_utils.py:1545
  - \_record\_recon\_event ← pdf_table_utils.py:1580
  - \_record\_recon\_event ← pdf_table_utils.py:1643
  - \_record\_recon\_event ← pdf_table_utils.py:1653
  - \_record\_recon\_event ← pdf_table_utils.py:1664
  - \_record\_recon\_event ← pdf_table_utils.py:1672
  - \_record\_recon\_event ← pdf_table_utils.py:1722
  - \_record\_recon\_event ← pdf_table_utils.py:1732
  - \_record\_recon\_event ← pdf_table_utils.py:1743
  - \_record\_recon\_event ← pdf_table_utils.py:1767
  - \_record\_recon\_event ← pdf_table_utils.py:1783
  - \_record\_recon\_event ← pdf_table_utils.py:1808
  - \_record\_recon\_event ← pdf_table_utils.py:1821
  - detect\_district\_heading ← pdf_table_utils.py:616
  - detect\_district\_heading ← pdf_table_utils.py:1564
  - build\_contest\_regex ← pdf_table_utils.py:243
  - normalize\_text\_token ← pdf_table_utils.py:710
  - normalize\_text\_token ← pdf_table_utils.py:712
  - token\_set ← pdf_table_utils.py:396
  - token\_set ← pdf_table_utils.py:526
  - token\_set ← pdf_table_utils.py:530
  - token\_set ← pdf_table_utils.py:1881
  - header\_signature ← pdf_table_utils.py:265
  - header\_signature ← pdf_table_utils.py:288
  - header\_signature ← pdf_table_utils.py:399
  - header\_signature ← pdf_table_utils.py:404
  - header\_signature ← pdf_table_utils.py:429
  - header\_signature ← pdf_table_utils.py:436
  - looks\_like\_candidate\_header ← pdf_table_utils.py:352
  - compute\_header\_richness ← pdf_table_utils.py:391
  - is\_numeric\_like ← pdf_table_utils.py:339
  - is\_numeric\_like ← pdf_table_utils.py:797
  - is\_numeric\_like ← pdf_table_utils.py:1123
  - is\_numeric\_like ← pdf_table_utils.py:1127
  - is\_numeric\_like ← pdf_table_utils.py:1135
  - is\_numeric\_like ← pdf_table_utils.py:1197
  - is\_numeric\_like ← pdf_table_utils.py:1199
  - is\_numeric\_like ← pdf_table_utils.py:1289
  - is\_numeric\_like ← pdf_table_utils.py:1421

### utils/pivot.py {#webapp-parser-utils-pivot-py}

> pivot.py

- Definitions:
  - function: `\_token\_to\_pattern` (line 149)
  - function: `\_build\_division\_token\_patterns` (line 157)
  - function: `\_parse\_numeric\_token` (line 235)
  - function: `\_coerce\_int` (line 269)
  - function: `\_normalized\_header\_cache` (line 285)
  - function: `\_natural\_key` (line 288)
  - function: `\_sort\_precincts` (line 299)
  - function: `\_infer\_division\_type\_by\_suffix` (line 318)
  - function: `\_extract\_municipality` (line 327)
  - function: `\_numeric\_ratio` (line 352)
  - function: `\_is\_numeric\_column` (line 363)
  - function: `\_fast\_path\_already\_wide` (line 367)
  - function: `debug\_dump\_pivot\_state` (line 416)
  - function: `\_strip\_party\_fragment` (line 420)
  - function: `\_normalize\_candidate\_label` (line 469)
  - function: `\_collect\_ballot\_types` (line 494)
  - function: `\_derive\_party\_map` (line 528)
  - function: `\_normalize\_division\_name` (line 542)
  - function: `\_division\_type\_for` (line 550)
  - function: `\_s` (line 566)
  - function: `\_safe\_col\_name` (line 573)
  - function: `\_norm\_text` (line 581)
  - function: `\_normalize\_state\_key` (line 585)
  - function: `\_detect\_division\_type\_for\_precinct` (line 597)
  - function: `\_detect\_division\_name\_for\_precinct` (line 656)
  - function: `\_normalize\_party\_value` (line 709)
  - function: `\_extract\_party\_from\_label` (line 726)
  - function: `\_candidate\_display\_and\_key` (line 798)
  - function: `\_normalize\_ballot\_suffix` (line 818)
  - function: `\_is\_misc\_candidate\_label` (line 822)
  - function: `\_map\_ballot\_suffix` (line 828)
  - function: `\_pluralize\_division` (line 850)
  - function: `\_choose\_division\_header` (line 860)
  - function: `pivot\_to\_wide` (line 898)
  - function: `transform\_wide\_to\_smart\_standard` (line 1357)
  - function: `expand\_single\_rawjson\_row` (line 1691)
  - function: `\_norm\_key` (line 1717)
  - function: `\_build\_colmap` (line 1722)
  - function: `\_read\_ndjson\_record` (line 1726)
  - function: `\_pick\_contest\_from\_obj` (line 1764)
  - function: `\_load\_contest\_from\_rows` (line 1821)
  - function: `pivot\_candidate\_groups\_from\_rawjson` (line 1863)
- Imports:
  - **Standard Library** (11):
    - `import hashlib as hashlib` (line 25)
    - `import math as math` (line 26)
    - `import os as os` (line 27)
    - `import re as re` (line 28)
    - `from collections import defaultdict` (line 29)
    - `from typing import Any` (line 30)
    - `from typing import Dict` (line 30)
    - `from typing import List` (line 30)
    - `from typing import Optional` (line 30)
    - `from typing import Set` (line 30)
    - `from typing import Tuple` (line 30)
  - **Third-party** (1):
    - `import orjson as orjson` (line 32)
  - **Local/Project** (23):
    - `from __future__ import annotations` (line 23)
    - `from Context_Integration.Context_Library.constants import BALLOT_TYPES`
      (line 34)
    - `from Context_Integration.Context_Library.constants import
      BALLOT_TYPES_SORT_ORDER` (line 34)
    - `from Context_Integration.Context_Library.constants import
      CANDIDATE_BALLOT_SPLIT_PATTERN` (line 34)
    - `from Context_Integration.Context_Library.constants import
      DIVISION_HEURISTIC_TERMS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      DIVISION_SUFFIXES` (line 34)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_ABBREVIATIONS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_KEYWORDS` (line 34)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_SYNONYM_MAP` (line 34)
    - `from Context_Integration.Context_Library.constants import
      PARTY_NORMALIZATION_MAP` (line 34)
    - `from Context_Integration.Context_Library.constants import
      STATE_TO_DIVISION_TYPE_MAP` (line 34)
    - `from Context_Integration.Context_Library.constants import TOTAL_KEYWORDS`
      (line 34)
    - `from Context_Integration.Context_Library.constants import
      canonical_ballot_group` (line 34)
    - `from Context_Integration.Context_Library.constants import
      normalize_party_code` (line 34)
    - `from Context_Integration.Context_Library.constants import
      normalize_party_label` (line 34)
    - `from detect import dynamic_detect_location_header` (line 50)
    - `from detect import normalize_header` (line 50)
    - `from detect import parse_numeric` (line 50)
    - `from logger_singleton import logger` (line 51)
    - `from shared_logic import lookup_precinct_aliases_for_county` (line 52)
    - `from shared_logic import normalize_county_key` (line 52)
    - `from shared_logic import safe_get` (line 52)
    - `from shared_logic import safe_strip` (line 52)
- Task markers:
  - L1353 **WARNING**: ("\[PIVOT\] No candidates detected – verify headers and
    candidate column extraction.")
- Outgoing cross-module calls (sample):
  - re.compile (line 59)
  - detect.normalize\_header (line 60)
  - detect.normalize\_header (line 61)
  - detect.normalize\_header (line 62)
  - detect.normalize\_header (line 64)
  - detect.normalize\_header (line 65)
  - Context\_Integration.Context\_Library.constants.PARTY\_NORMALIZATION\_MAP.values
    (line 70)
  - val.strip (line 71)
  - re.compile (line 107)
  - re.compile (line 110)
  - token.strip (line 150)
  - re.escape (line 151)
  - escaped.replace (line 152)
  - escaped.replace (line 153)
  - dtype.lower (line 158)
  - base\_types.update (line 159)
  - kw.strip (line 175)
  - Context\_Integration.Context\_Library.constants.LOCATION\_SYNONYM\_MAP.items
    (line 179)
  - canonical.strip (line 180)
  - alias.strip (line 181)
  - Context\_Integration.Context\_Library.constants.LOCATION\_ABBREVIATIONS.items
    (line 185)
  - abbr.strip (line 186)
  - expansion.strip (line 188)
  - token\_map.items (line 194)
  - clean\_tokens.sort (line 198)
  - patterns.append (line 204)
  - re.compile (line 204)
  - priority\_map.get (line 208)
  - patterns.sort (line 211)
  - re.compile (line 218)
  - math.isnan (line 244)
  - math.isinf (line 244)
  - val.is\_integer (line 246)
  - val.strip (line 248)
  - text.lower (line 251)
  - text.replace (line 254)
  - cleaned.strip (line 255)
  - math.isnan (line 262)
  - math.isinf (line 262)
  - num.is\_integer (line 264)
  - val.is\_integer (line 274)
  - \_NUM\_CLEAN\_RE.sub (line 276)
  - val.strip (line 276)
  - s.isdigit (line 277)
  - s.startswith (line 277)
  - detect.normalize\_header (line 286)
  - s.lower (line 291)
  - \_SPLIT\_NUM\_RE.split (line 293)
  - s.lower (line 293)
  - key.append (line 296)
- Inbound references:
  - \_token\_to\_pattern ← pivot.py:195
  - \_build\_division\_token\_patterns ← pivot.py:215
  - \_parse\_numeric\_token ← pivot.py:1468
  - \_parse\_numeric\_token ← pivot.py:1473
  - \_coerce\_int ← pivot.py:1257
  - \_coerce\_int ← pivot.py:1260
  - \_coerce\_int ← pivot.py:1266
  - \_coerce\_int ← pivot.py:1272
  - \_coerce\_int ← pivot.py:1303
  - \_coerce\_int ← pivot.py:1960
  - \_coerce\_int ← pivot.py:2177
  - \_coerce\_int ← pivot.py:2182
  - \_coerce\_int ← pivot.py:2186
  - \_coerce\_int ← pivot.py:2203
  - \_coerce\_int ← pivot.py:2212
  - \_coerce\_int ← pivot.py:2218
  - \_coerce\_int ← pivot.py:2221
  - \_normalized\_header\_cache ← pivot.py:921
  - \_sort\_precincts ← pivot.py:1141
  - \_sort\_precincts ← pivot.py:1208
  - \_infer\_division\_type\_by\_suffix ← pivot.py:564
  - \_extract\_municipality ← pivot.py:679
  - \_extract\_municipality ← pivot.py:2078
  - \_extract\_municipality ← pivot.py:2086
  - \_numeric\_ratio ← pivot.py:365
  - \_is\_numeric\_column ← pivot.py:394
  - \_is\_numeric\_column ← pivot.py:519
  - \_fast\_path\_already\_wide ← pivot.py:1028
  - \_fast\_path\_already\_wide ← pivot.py:1093
  - debug\_dump\_pivot\_state ← pivot.py:1089
  - \_strip\_party\_fragment ← pivot.py:489
  - \_strip\_party\_fragment ← pivot.py:809
  - \_collect\_ballot\_types ← pivot.py:1122
  - \_derive\_party\_map ← pivot.py:1123
  - \_normalize\_division\_name ← pivot.py:555
  - \_division\_type\_for ← pivot.py:654
  - \_division\_type\_for ← pivot.py:979
  - \_division\_type\_for ← pivot.py:1144
  - \_s ← pivot.py:578
  - \_s ← table_builder.py:637
  - \_s ← table_builder.py:639
  - \_s ← table_builder.py:639
  - \_s ← table_builder.py:641
  - \_safe\_col\_name ← pivot.py:1195
  - \_safe\_col\_name ← pivot.py:1197
  - \_safe\_col\_name ← pivot.py:1198
  - \_safe\_col\_name ← pivot.py:1200
  - \_safe\_col\_name ← pivot.py:1202
  - \_safe\_col\_name ← pivot.py:1204
  - \_safe\_col\_name ← pivot.py:1259

### utils/privilege\_tiers.py {#webapp-parser-utils-privilege-tiers-py}

> 4-Tier Privilege System for Election Results Parser

- Definitions:
  - class: `PrivilegeTier` (line 23)
  - function: `get\_tier\_trust\_thresholds` (line 48)
  - function: `get\_principal\_tier` (line 71)
  - function: `should\_apply\_admin\_boost` (line 187)
  - function: `is\_domain\_in\_allowlist` (line 228)
  - function: `\_parse\_env\_list` (line 242)
  - function: `require\_minimum\_tier` (line 254)
- Imports:
  - **Standard Library** (4):
    - `import os as os` (line 16)
    - `from enum import IntEnum` (line 17)
    - `from typing import Dict` (line 18)
    - `from typing import Optional` (line 18)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 14)
    - `from utils.logger_singleton import logger` (line 20)
- Task markers:
  - L167 **WARNING**: ({
  - L168 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - names.get (line 39)
  - boosts.get (line 45)
  - thresholds.get (line 68)
  - principal.startswith (line 86)
  - utils.logger\_singleton.logger.info (line 92)
  - utils.logger\_singleton.logger.info (line 104)
  - utils.logger\_singleton.logger.info (line 116)
  - principal.startswith (line 126)
  - rc.lower (line 131)
  - utils.logger\_singleton.logger.info (line 132)
  - fc.lower (line 143)
  - utils.logger\_singleton.logger.info (line 144)
  - rc.lower (line 155)
  - utils.logger\_singleton.logger.info (line 156)
  - utils.logger\_singleton.logger.warning (line 167)
  - utils.logger\_singleton.logger.debug (line 177)
  - domain.endswith (line 215)
  - trust\_factors.get (line 220)
  - trust\_factors.get (line 221)
  - domain.lower (line 231)
  - allowed.lower (line 235)
  - domain\_lower.endswith (line 236)
  - item.strip (line 247)
  - raw.split (line 247)
  - item.strip (line 247)
  - kwargs.get (line 258)
- Inbound references:
  - PrivilegeTier ← privilege_tiers.py:260
  - PrivilegeTier ← privilege_tiers.py:261
  - get\_principal\_tier ← Smart_Elections_Parser_Webapp.py:6544
  - get\_principal\_tier ← Smart_Elections_Parser_Webapp.py:9166
  - get\_principal\_tier ← html_election_parser.py:1524
  - get\_principal\_tier ← web_pipeline.py:268
  - is\_domain\_in\_allowlist ← privilege_tiers.py:222
  - \_parse\_env\_list ← privilege_tiers.py:90
  - \_parse\_env\_list ← privilege_tiers.py:102
  - \_parse\_env\_list ← privilege_tiers.py:114
  - \_parse\_env\_list ← privilege_tiers.py:130
  - \_parse\_env\_list ← privilege_tiers.py:142
  - \_parse\_env\_list ← privilege_tiers.py:154
  - \_parse\_env\_list ← privilege_tiers.py:230

### utils/rawjson\_utils.py {#webapp-parser-utils-rawjson-utils-py}

- Definitions:
  - function: `\_rj\_first` (line 17)
  - function: `\_rj\_as\_dict` (line 29)
  - function: `\_rj\_ensure\_list` (line 44)
  - function: `\_infer\_party\_from\_name` (line 49)
  - function: `extract\_rawjson\_enrichment\_from\_rows` (line 58)
  - function: `offload\_rawjson\_to\_ndjson` (line 183)
- Imports:
  - **Standard Library** (2):
    - `import os as os` (line 3)
    - `from typing import Iterable` (line 4)
  - **Third-party** (1):
    - `import orjson as orjson` (line 6)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - raw.strip (line 33)
  - orjson.loads (line 35)
  - json.loads (line 39)
  - label.lower (line 52)
  - p.title (line 55)
  - r.get (line 68)
  - group\_totals.get (line 130)
  - enr\_candidates.append (line 132)
  - group\_totals.values (line 153)
  - group\_totals.items (line 156)
  - group\_totals.keys (line 179)
  - r.get (line 190)
  - os.makedirs (line 194)
  - out\_rows.append (line 206)
  - r.pop (line 208)
  - out\_rows.append (line 210)
  - r.get (line 213)
  - f.write (line 215)
  - orjson.dumps (line 215)
  - out\_rows.append (line 221)
- Inbound references:
  - \_rj\_first ← rawjson_utils.py:76
  - \_rj\_first ← rawjson_utils.py:77
  - \_rj\_first ← rawjson_utils.py:78
  - \_rj\_first ← rawjson_utils.py:79
  - \_rj\_first ← rawjson_utils.py:80
  - \_rj\_first ← rawjson_utils.py:81
  - \_rj\_first ← rawjson_utils.py:82
  - \_rj\_first ← rawjson_utils.py:85
  - \_rj\_first ← rawjson_utils.py:98
  - \_rj\_first ← rawjson_utils.py:99
  - \_rj\_first ← rawjson_utils.py:100
  - \_rj\_first ← rawjson_utils.py:101
  - \_rj\_first ← rawjson_utils.py:112
  - \_rj\_first ← rawjson_utils.py:118
  - \_rj\_first ← rawjson_utils.py:119
  - \_rj\_first ← rawjson_utils.py:141
  - \_rj\_as\_dict ← rawjson_utils.py:69
  - \_rj\_as\_dict ← rawjson_utils.py:212
  - \_infer\_party\_from\_name ← rawjson_utils.py:100

### utils/retry\_utils.py {#webapp-parser-utils-retry-utils-py}

> Retry Utilities with Snapshot Mode

- Definitions:
  - function: `retry\_with\_snapshot` (line 30)
  - function: `\_get\_html\_context` (line 115)
  - function: `\_save\_failure\_snapshot` (line 139)
  - function: `\_log\_extraction\_failure` (line 208)
  - function: `\_get\_traceback\_str` (line 246)
  - function: `example\_handler\_with\_retry` (line 260)
- Imports:
  - **Standard Library** (7):
    - `import functools as functools` (line 21)
    - `import time as time` (line 22)
    - `from datetime import datetime` (line 23)
    - `from pathlib import Path` (line 24)
    - `from typing import Callable` (line 25)
    - `from typing import Optional` (line 25)
    - `from typing import Tuple` (line 25)
  - **Local/Project** (1):
    - `from logger_singleton import logger` (line 27)
- Task markers:
  - L84 **WARNING**: (f"\[yellow\]\[retry\] Attempt {attempt}/{max_attempts}
    failed: {e}\[/yellow\]")
  - L173 **WARNING**: (f"\[snapshot\] Could not save HTML: {e}")
  - L184 **WARNING**: (f"\[snapshot\] Could not save context: {e}")
  - L243 **WARNING**: (f"\[retry\] Could not log failure: {e}")
- Outgoing cross-module calls (sample):
  - logger\_singleton.logger.info (line 65)
  - logger\_singleton.logger.info (line 71)
  - logger\_singleton.logger.info (line 78)
  - logger\_singleton.logger.warning (line 84)
  - logger\_singleton.logger.info (line 89)
  - time.sleep (line 90)
  - logger\_singleton.logger.error (line 93)
  - functools.wraps (line 57)
  - pathlib.Path (line 152)
  - snapshot\_dir.mkdir (line 153)
  - kwargs.get (line 156)
  - datetime.datetime.now (line 161)
  - kwargs.get (line 165)
  - page.content (line 168)
  - html\_file.write\_text (line 170)
  - logger\_singleton.logger.info (line 171)
  - logger\_singleton.logger.warning (line 173)
  - context\_file.write\_bytes (line 181)
  - orjson.dumps (line 181)
  - logger\_singleton.logger.info (line 182)
  - logger\_singleton.logger.warning (line 184)
  - datetime.datetime.now (line 189)
  - html\_context.get (line 194)
  - html\_context.get (line 195)
  - html\_context.get (line 196)
  - error\_file.write\_text (line 201)
  - logger\_singleton.logger.info (line 202)
  - logger\_singleton.logger.error (line 205)
  - pathlib.Path (line 217)
  - kwargs.get (line 221)
  - datetime.datetime.now (line 224)
  - html\_context.get (line 229)
  - html\_context.get (line 230)
  - html\_context.get (line 231)
  - html\_context.get (line 232)
  - f.write (line 237)
  - orjson.dumps (line 237)
  - f.write (line 238)
  - logger\_singleton.logger.debug (line 240)
  - logger\_singleton.logger.warning (line 243)
  - traceback.format\_exc (line 250)
- Inbound references:
  - retry\_with\_snapshot ← retry_utils.py:256
  - retry\_with\_snapshot ← retry_utils.py:264
  - \_get\_html\_context ← retry_utils.py:67
  - \_get\_html\_context ← retry_utils.py:176
  - \_get\_html\_context ← retry_utils.py:220
  - \_save\_failure\_snapshot ← retry_utils.py:97
  - \_log\_extraction\_failure ← retry_utils.py:100
  - \_get\_traceback\_str ← retry_utils.py:199

### utils/root\_admin\_session.py {#webapp-parser-utils-root-admin-session-py}

> Root Admin Session Management for Smart Elections Parser

- Definitions:
  - function: `generate\_root\_admin\_token` (line 39)
  - function: `hash\_token` (line 62)
  - function: `verify\_root\_admin\_token` (line 74)
  - function: `check\_is\_root\_uid` (line 101)
  - function: `create\_root\_admin\_session` (line 119)
  - function: `is\_root\_admin\_session` (line 185)
  - function: `get\_root\_admin\_session\_info` (line 214)
  - function: `revoke\_root\_admin\_session` (line 238)
  - function: `cleanup\_expired\_root\_admin\_sessions` (line 268)
  - function: `list\_active\_root\_admin\_sessions` (line 294)
- Imports:
  - **Standard Library** (5):
    - `import hashlib as hashlib` (line 20)
    - `import os as os` (line 21)
    - `import time as time` (line 23)
    - `from typing import Any` (line 24)
    - `from typing import Dict` (line 24)
  - **Local/Project** (3):
    - `from __future__ import annotations` (line 18)
    - `import secrets as secrets` (line 22)
    - `from logger_singleton import logger` (line 26)
- Task markers:
  - L84 **NOTE**:     Note:
  - L107 **NOTE**:     Note:
  - L258 **WARNING**: ({
  - L259 **WARNING**: ",
  - L300 **WARNING**:     WARNING:
- Outgoing cross-module calls (sample):
  - secrets.token\_bytes (line 50)
  - token\_bytes.hex (line 51)
  - logger\_singleton.logger.error (line 53)
  - hashlib.sha256 (line 71)
  - token.encode (line 71)
  - hmac.compare\_digest (line 96)
  - os.getuid (line 113)
  - logger\_singleton.logger.error (line 140)
  - logger\_singleton.logger.error (line 151)
  - time.time (line 162)
  - secrets.token\_hex (line 163)
  - time.time (line 167)
  - logger\_singleton.logger.info (line 172)
  - time.time (line 200)
  - \_ROOT\_ADMIN\_SESSIONS.pop (line 208)
  - \_ROOT\_ADMIN\_SESSIONS.get (line 226)
  - time.time (line 234)
  - \_ROOT\_ADMIN\_SESSIONS.pop (line 256)
  - logger\_singleton.logger.warning (line 258)
  - time.time (line 274)
  - \_ROOT\_ADMIN\_SESSIONS.items (line 278)
  - \_ROOT\_ADMIN\_SESSIONS.pop (line 280)
  - logger\_singleton.logger.info (line 284)
  - time.time (line 303)
  - \_ROOT\_ADMIN\_SESSIONS.items (line 307)
  - sessions.append (line 309)
- Inbound references:
  - hash\_token ← root_admin_session.py:96
  - hash\_token ← root_admin_session.py:96
  - check\_is\_root\_uid ← root_admin_session.py:139
  - is\_root\_admin\_session ← root_admin_session.py:223

### utils/safe\_decide.py {#webapp-parser-utils-safe-decide-py}

> Safe Decision Helpers: Confidence/Caution Gates for Election Entities

- Definitions:
  - function: `\_emit\_decision\_log` (line 27)
  - function: `safe\_decide\_jurisdiction` (line 76)
  - function: `safe\_decide\_office` (line 124)
  - function: `safe\_decide\_party` (line 159)
  - function: `safe\_decide\_source` (line 193)
  - function: `should\_proceed` (line 227)
  - function: `should\_caution` (line 232)
  - function: `should\_stop` (line 237)
- Imports:
  - **Standard Library** (5):
    - `from datetime import datetime` (line 14)
    - `from datetime import timezone` (line 14)
    - `from typing import List` (line 15)
    - `from typing import Optional` (line 15)
    - `from typing import Tuple` (line 15)
  - **Third-party** (6):
    - `from webapp.parser.Context_Integration.library.entity_confidence_map
      import AnomalyType` (line 17)
    - `from webapp.parser.Context_Integration.library.entity_confidence_map
      import OverrideTrigger` (line 17)
    - `from webapp.parser.Context_Integration.library.entity_confidence_map
      import SignalType` (line 17)
    - `from webapp.parser.Context_Integration.library.entity_confidence_map
      import get_confidence_map` (line 17)
    - `from webapp.parser.utils.logger_singleton import logger` (line 23)
    - `from webapp.parser.utils.shared_logic import DecisionTuple` (line 24)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 12)
- Task markers:
  - L66 **WARNING**: ({
  - L67 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - decision\_tuple.get (line 52)
  - decision\_tuple.get (line 54)
  - decision\_tuple.get (line 55)
  - decision\_tuple.get (line 56)
  - decision\_tuple.get (line 57)
  - decision\_tuple.get (line 58)
  - decision\_tuple.get (line 59)
  - decision\_tuple.get (line 60)
  - datetime.datetime.now (line 60)
  - webapp.parser.utils.logger\_singleton.logger.info (line 63)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 66)
  - webapp.parser.Context\_Integration.library.entity\_confidence\_map.get\_confidence\_map
    (line 98)
  - confidence\_map.calculate\_confidence\_caution (line 99)
  - datetime.datetime.now (line 116)
  - webapp.parser.Context\_Integration.library.entity\_confidence\_map.get\_confidence\_map
    (line 133)
  - confidence\_map.calculate\_confidence\_caution (line 134)
  - datetime.datetime.now (line 151)
  - webapp.parser.Context\_Integration.library.entity\_confidence\_map.get\_confidence\_map
    (line 167)
  - confidence\_map.calculate\_confidence\_caution (line 168)
  - datetime.datetime.now (line 185)
  - webapp.parser.Context\_Integration.library.entity\_confidence\_map.get\_confidence\_map
    (line 201)
  - confidence\_map.calculate\_confidence\_caution (line 202)
  - datetime.datetime.now (line 219)
  - decision\_tuple.get (line 229)
  - decision\_tuple.get (line 234)
  - decision\_tuple.get (line 239)
- Inbound references:
  - \_emit\_decision\_log ← safe_decide.py:120
  - \_emit\_decision\_log ← safe_decide.py:155
  - \_emit\_decision\_log ← safe_decide.py:189
  - \_emit\_decision\_log ← safe_decide.py:223

### utils/salvage.py {#webapp-parser-utils-salvage-py}

> salvage.py

- Definitions:
  - function: `\_to\_int\_or\_none` (line 35)
  - function: `normalize\_ballot\_column\_name` (line 39)
  - function: `collapse\_ballot\_synonym\_columns` (line 96)
  - function: `merge\_multiline\_candidate\_rows` (line 183)
  - function: `combine\_panel\_tables\_by\_precinct` (line 216)
  - function: `\_salvage\_rows\_from\_rawjson` (line 237)
  - function: `remove\_footer\_and\_summary\_rows` (line 333)
  - function: `remove\_outlier\_and\_empty\_rows` (line 354)
- Imports:
  - **Standard Library** (5):
    - `import re as re` (line 7)
    - `from typing import Any` (line 8)
    - `from typing import Dict` (line 8)
    - `from typing import List` (line 8)
    - `from typing import Tuple` (line 8)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 5)
    - `from detect import parse_numeric` (line 10)
- Outgoing cross-module calls (sample):
  - detect.parse\_numeric (line 36)
  - raw.lower (line 50)
  - raw.split (line 60)
  - left.strip (line 61)
  - right.strip (line 62)
  - b.lower (line 76)
  - right\_norm.endswith (line 78)
  - rename.get (line 116)
  - new\_headers.append (line 118)
  - seen.add (line 119)
  - r.items (line 127)
  - rename.get (line 128)
  - acc.get (line 131)
  - acc.get (line 134)
  - acc.get (line 145)
  - acc.items (line 148)
  - out\_rows.append (line 159)
  - present.update (line 164)
  - r.keys (line 164)
  - final\_headers.append (line 169)
  - logger.debug (line 171)
  - r.get (line 204)
  - r.items (line 206)
  - r.items (line 209)
  - prev.get (line 210)
  - out.append (line 213)
  - seen.add (line 228)
  - union\_headers.append (line 229)
  - r.get (line 233)
  - out\_rows.append (line 234)
  - h.lower (line 261)
  - h.lower (line 267)
  - ctx.get (line 285)
  - headers.append (line 295)
  - out.append (line 300)
  - rec.get (line 303)
  - \_orjson.loads (line 307)
  - blob.strip (line 311)
  - s.startswith (line 312)
  - s.startswith (line 312)
  - \_orjson.loads (line 314)
  - obj.get (line 323)
  - obj.get (line 323)
  - obj.get (line 325)
  - rec.get (line 326)
  - out.append (line 329)
  - re.compile (line 340)
  - r.values (line 345)
  - r.get (line 347)
  - r.get (line 348)
- Inbound references:
  - \_to\_int\_or\_none ← salvage.py:131
  - \_to\_int\_or\_none ← salvage.py:132
  - \_to\_int\_or\_none ← salvage.py:145
  - \_to\_int\_or\_none ← salvage.py:152
  - normalize\_ballot\_column\_name ← salvage.py:108
  - normalize\_ballot\_column\_name ← salvage.py:150

### utils/seleniumbase\_launcher.py {#webapp-parser-utils-seleniumbase-launcher-py}

- Definitions:
  - class: `\_MissingDriver` (line 20)
  - function: `launch\_browser` (line 38)
  - function: `relaunch\_browser\_fullscreen\_if\_needed` (line 55)
  - function: `relaunch\_browser\_stealth` (line 102)
  - function: `close\_driver` (line 119)
  - function: `\_capture\_post\_captcha\_dom\_metadata` (line 129)
  - function: `\_log\_captcha\_resolution\_data` (line 163)
- Imports:
  - **Standard Library** (2):
    - `import time as time` (line 3)
    - `from typing import Optional` (line 4)
  - **Local/Project** (3):
    - `from __future__ import annotations` (line 1)
    - `from config import HEADLESS_DEFAULT` (line 34)
    - `from logger_singleton import logger` (line 35)
- Outgoing cross-module calls (sample):
  - driver.get (line 70)
  - driver.maximize\_window (line 72)
  - logger\_singleton.logger.info (line 75)
  - logger\_singleton.logger.info (line 76)
  - time.time (line 77)
  - time.time (line 78)
  - logger\_singleton.logger.info (line 90)
  - logger\_singleton.logger.debug (line 97)
  - time.sleep (line 99)
  - driver.get (line 116)
  - driver.quit (line 124)
  - driver.execute\_script (line 143)
  - logger\_singleton.logger.debug (line 159)
  - time.time (line 189)
  - time.time (line 191)
  - os.makedirs (line 195)
  - f.write (line 198)
  - orjson.dumps (line 198)
  - f.write (line 199)
  - logger\_singleton.logger.info (line 201)
  - dom\_metadata.get (line 201)
  - dom\_metadata.get (line 202)
  - logger\_singleton.logger.debug (line 205)
- Inbound references:
  - \_capture\_post\_captcha\_dom\_metadata ← seleniumbase_launcher.py:93
  - \_log\_captcha\_resolution\_data ← seleniumbase_launcher.py:95

### utils/session\_state.py {#webapp-parser-utils-session-state-py}

- Definitions:
  - class: `SessionState` (line 7)
  - class: `PipelinePhase` (line 21)
  - function: `export\_session\_enums` (line 44)
- Imports:
  - **Standard Library** (2):
    - `from enum import Enum` (line 3)
    - `from typing import Dict` (line 4)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - SessionState.as\_dict (line 47)
  - PipelinePhase.ordered (line 49)

### utils/shared\_logger.py {#webapp-parser-utils-shared-logger-py}

- Definitions:
  - function: `safe\_getvalue` (line 39)
  - class: `RichConsoleProxy` (line 50)
  - class: `SQLAlchemyToSharedLoggerHandler` (line 150)
  - class: `SharedLogger` (line 167)
- Imports:
  - **Standard Library** (18):
    - `import inspect as inspect` (line 3)
    - `import logging as logging` (line 4)
    - `import os as os` (line 12)
    - `import re as re` (line 13)
    - `import threading as threading` (line 14)
    - `import time as time` (line 15)
    - `import traceback as traceback` (line 16)
    - `from contextlib import contextmanager` (line 17)
    - `from io import StringIO` (line 18)
    - `from pathlib import Path` (line 19)
    - `from typing import Any` (line 20)
    - `from typing import Callable` (line 20)
    - `from typing import Dict` (line 20)
    - `from typing import Generator` (line 20)
    - `from typing import List` (line 20)
    - `from typing import Optional` (line 20)
    - `from typing import Set` (line 20)
    - `from typing import Tuple` (line 20)
  - **Third-party** (1):
    - `import orjson as orjson` (line 22)
  - **Local/Project** (14):
    - `from __future__ import annotations` (line 1)
    - `from rich import print as rprint` (line 23)
    - `from rich.console import Console` (line 24)
    - `from rich.console import RenderableType` (line 24)
    - `from rich.json import JSON` (line 25)
    - `from rich.logging import RichHandler` (line 26)
    - `from rich.panel import Panel` (line 27)
    - `from rich.progress import BarColumn` (line 28)
    - `from rich.progress import Progress` (line 28)
    - `from rich.progress import SpinnerColumn` (line 28)
    - `from rich.progress import TextColumn` (line 28)
    - `from rich.progress import TimeElapsedColumn` (line 28)
    - `from rich.progress import TimeRemainingColumn` (line 28)
    - `from rich.table import Table` (line 36)
- Task markers:
  - L160 **WARNING**:         elif record.levelno &gt;= logging.WARNING:
  - L161 **WARNING**: (msg)
  - L286 **WARNING**: ": logging.WARNING,
  - L357 **WARNING**: ": "yellow",
  - L419 **WARNING**: (self, msg, context=None, exc_info=None):
  - L421 **WARNING**: ", msg, context, color="yellow")
  - L435 **WARNING**: ": "yellow",
  - L667 **WARNING**: (f"Log directory does not exist: {log_dir}")
  - L684 **WARNING**: (f"Corrupt line in {path}: {e}")
- Outgoing cross-module calls (sample):
  - file\_obj.getvalue (line 45)
  - logging.error (line 47)
  - rich.console.Console (line 58)
  - renderables.append (line 69)
  - rich.json.JSON (line 69)
  - renderables.append (line 73)
  - rich.json.JSON (line 73)
  - obj.decode (line 73)
  - renderables.append (line 75)
  - obj.strip (line 77)
  - renderables.append (line 79)
  - rich.json.JSON (line 79)
  - renderables.append (line 81)
  - renderables.append (line 83)
  - self.\_render\_to\_text (line 88)
  - rich.console.Console (line 96)
  - io.StringIO (line 96)
  - temp\_console.print (line 97)
  - rich.panel.Panel (line 111)
  - self.print (line 112)
  - rich.table.Table (line 123)
  - self.\_render\_to\_text (line 126)
  - self.\_render\_to\_text (line 139)
  - self.format (line 156)
  - re.compile (line 174)
  - re.compile (line 175)
  - level.upper (line 201)
  - self.\_setup\_python\_logger (line 207)
  - threading.RLock (line 234)
  - time.time (line 257)
  - logging.addLevelName (line 290)
  - logging.getLogger (line 291)
  - rich.logging.RichHandler (line 301)
  - logging.FileHandler (line 304)
  - file\_handler.setLevel (line 305)
  - logging.Formatter (line 306)
  - file\_handler.setFormatter (line 307)
  - level.upper (line 324)
  - self.\_setup\_python\_logger (line 334)
  - match.group (line 348)
  - match.group (line 349)
  - label.split (line 351)
  - color\_map.get (line 370)
  - color.upper (line 370)
  - orjson.dumps (line 385)
  - inspect.currentframe (line 392)
  - inspect.getouterframes (line 394)
  - self.\_append\_traceback (line 408)
  - self.\_log (line 409)
  - self.\_append\_traceback (line 412)
- Inbound references:
  - safe\_getvalue ← shared_logger.py:98
  - SharedLogger ← shared_logger.py:57

### utils/shared\_logic.py {#webapp-parser-utils-shared-logic-py}

- Definitions:
  - class: `DecisionTuple` (line 80)
  - class: `ExtractPlugin` (line 111)
  - class: `Saveable` (line 114)
  - class: `GCModule` (line 117)
  - class: `ShutilModule` (line 120)
  - class: `TimeModule` (line 124)
  - class: `HasItem` (line 128)
  - class: `HasAllMethod` (line 133)
  - class: `PredictionResult` (line 140)
  - class: `EventLike` (line 162)
  - class: `Predictable` (line 171)
  - function: `safe\_filename` (line 197)
  - function: `is\_path\_safe` (line 283)
  - function: `safe\_resolve\_path` (line 316)
  - function: `safe\_join\_path` (line 347)
  - function: `validate\_directory\_path` (line 375)
  - function: `safe\_slug` (line 391)
  - function: `safe\_query` (line 407)
  - function: `safe\_key` (line 418)
  - function: `\_filter\_valid\_kwargs` (line 429)
  - function: `safe\_filter\_by` (line 447)
  - function: `safe\_first` (line 461)
  - function: `get\_or\_create` (line 474)
  - function: `safe\_translate` (line 497)
  - function: `safe\_scheme` (line 509)
  - function: `safe\_netloc` (line 517)
  - function: `safe\_geturl` (line 525)
  - function: `\_resolve\_host\_ips` (line 533)
  - function: `safe\_validate\_external\_url` (line 545)
  - function: `safe\_extract` (line 648)
  - function: `safe\_isalpha` (line 662)
  - function: `safe\_pop` (line 672)
  - function: `safe\_merge\_defaults` (line 680)
  - function: `safe\_strip` (line 696)
  - function: `safe\_setdefault` (line 702)
  - function: `safe\_tolist` (line 713)
  - function: `safe\_execute` (line 735)
  - function: `safe\_commit` (line 747)
  - function: `safe\_scalar\_one\_or\_none` (line 756)
  - function: `safe\_model\_save` (line 767)
  - function: `safe\_all` (line 812)
  - function: `safe\_copy` (line 824)
  - function: `safe\_isalnum` (line 847)
  - function: `safe\_keys` (line 857)
  - function: `safe\_attr\_keys` (line 870)
  - function: `safe\_replace` (line 884)
  - function: `safe\_isdigit` (line 897)
  - function: `safe\_get` (line 904)
  - function: `safe\_values` (line 913)
  - function: `safe\_is\_set` (line 925)
  - function: `safe\_set` (line 937)
  - function: `safe\_clear` (line 947)
  - function: `safe\_append\_cached\_segment` (line 957)
  - function: `safe\_db\_call` (line 975)
  - function: `safe\_append` (line 996)
  - function: `safe\_update` (line 1017)
  - function: `safe\_extend` (line 1042)
  - function: `convert\_ndarrays` (line 1063)
  - function: `normalize\_html\_for\_hash` (line 1073)
  - function: `clean\_cache\_inplace` (line 1081)
  - function: `\_to\_json\_safe` (line 1093)
  - function: `sync\_type\_and\_election\_types` (line 1102)
  - function: `keyword\_in\_text` (line 1128)
  - function: `safe\_lower` (line 1136)
  - function: `safe\_encode` (line 1142)
  - function: `safe\_startswith` (line 1150)
  - function: `safe\_add` (line 1165)
  - function: `safe\_predict` (line 1181)
  - function: `safe\_split` (line 1192)
  - function: `safe\_capitalize` (line 1212)
  - function: `safe\_item` (line 1216)
  - function: `safe\_items` (line 1230)
  - function: `safe\_similarity` (line 1249)
  - function: `safe\_model\_encode` (line 1275)
  - function: `safe\_get\_first` (line 1369)
  - function: `validate\_handler\_result` (line 1395)
  - function: `safe\_parse` (line 1427)
  - function: `safe\_endswith` (line 1587)
  - function: `safe\_isupper` (line 1598)
  - function: `resolve\_county\_alias` (line 1609)
  - function: `safe\_sid` (line 1638)
  - function: `safe\_rsplit` (line 1660)
  - function: `normalize\_county\_name` (line 1674)
  - function: `flatten\_raw\_field` (line 1696)
  - function: `normalize\_state\_name` (line 1711)
  - function: `infer\_state\_county\_from\_url` (line 1747)
  - function: `resolve\_state\_county\_from\_context` (line 1830)
  - function: `format\_state\_label` (line 1856)
  - function: `canonicalize\_county\_label` (line 1871)
  - function: `format\_county\_label` (line 1883)
  - function: `\_table\_sample\_text` (line 1913)
  - function: `derive\_state\_county\_from\_table` (line 1944)
  - function: `derive\_candidate\_party\_metadata` (line 2070)
  - function: `build\_camelot\_row\_filter\_for\_context` (line 2178)
  - function: `record\_noise\_suggestion` (line 2186)
  - function: `get\_county\_precincts` (line 2217)
  - function: `normalize\_county\_key` (line 2222)
  - function: `lookup\_precinct\_aliases\_for\_county` (line 2234)
  - function: `get\_state\_counties` (line 2244)
  - function: `scan\_environment` (line 2248)
  - function: `get\_title\_embedding\_features` (line 2256)
  - function: `show\_progress\_bar` (line 2265)
  - function: `coordinator\_feedback` (line 2275)
  - function: `normalize\_text` (line 2278)
  - function: `match\_any` (line 2281)
  - function: `build\_csv\_headers` (line 2285)
  - function: `keyphrase\_match` (line 2292)
  - function: `normalize\_label` (line 2313)
  - function: `infer\_contest\_fields` (line 2318)
  - function: `\_infer\_category` (line 2455)
  - function: `\_read\_module\_summary` (line 2480)
  - function: `\_is\_ignored\_dir` (line 2503)
  - function: `generate\_project\_inventory` (line 2507)
  - function: `\_render\_inventory\_md` (line 2542)
  - function: `\_finalize\_markdown\_lines` (line 2572)
  - function: `update\_architecture\_md` (line 2659)
  - function: `generate\_project\_map` (line 2682)
  - function: `\_posix` (line 2692)
  - function: `\_read\_file\_text` (line 2695)
  - function: `\_extract\_top\_comment\_block` (line 2701)
  - function: `\_harvest\_todos` (line 2731)
  - function: `\_module\_info\_from\_ast` (line 2748)
  - function: `\_scan\_webapp\_modules` (line 2842)
  - function: `\_index\_defs` (line 2861)
  - function: `\_resolve\_targets` (line 2881)
  - function: `\_render\_audit\_md` (line 2917)
  - function: `generate\_project\_audit` (line 3465)
  - function: `generate\_todos\_index` (line 3488)
  - function: `generate\_noise\_override\_suggestions` (line 3614)
  - function: `generate\_pipeline\_map` (line 3738)
  - function: `generate\_docs\_artifacts` (line 4068)
  - function: `log\_rejection\_reason` (line 4088)
  - function: `batch\_log\_rejections` (line 4171)
- Imports:
  - **Standard Library** (31):
    - `import copy as copy` (line 3)
    - `import inspect as inspect` (line 11)
    - `import os as os` (line 13)
    - `import platform as platform` (line 14)
    - `import re as re` (line 15)
    - `import shutil as shutil` (line 16)
    - `import socket as socket` (line 17)
    - `import time as time` (line 19)
    - `import traceback as traceback` (line 20)
    - `from pathlib import Path` (line 21)
    - `from typing import TYPE_CHECKING` (line 22)
    - `from typing import Any` (line 22)
    - `from typing import Awaitable` (line 22)
    - `from typing import Callable` (line 22)
    - `from typing import Dict` (line 22)
    - `from typing import Generator` (line 22)
    - `from typing import Iterable` (line 22)
    - `from typing import List` (line 22)
    - `from typing import Mapping` (line 22)
    - `from typing import Optional` (line 22)
    - `from typing import Protocol` (line 22)
    - `from typing import Sequence` (line 22)
    - `from typing import Set` (line 22)
    - `from typing import Type` (line 22)
    - `from typing import TypedDict` (line 22)
    - `from typing import TypeVar` (line 22)
    - `from typing import Union` (line 22)
    - `from typing import runtime_checkable` (line 22)
    - `from urllib.parse import ParseResult` (line 42)
    - `from urllib.parse import SplitResult` (line 42)
    - `from urllib.parse import urlparse` (line 42)
  - **Third-party** (7):
    - `import numpy as np` (line 44)
    - `import orjson as orjson` (line 45)
    - `from flask import request` (line 46)
    - `from flask import session` (line 46)
    - `from sqlalchemy.engine import ScalarResult` (line 56)
    - `from sqlalchemy.orm import Query` (line 57)
    - `from sqlalchemy.orm import Session` (line 57)
  - **Local/Project** (12):
    - `from __future__ import annotations` (line 1)
    - `import difflib as difflib` (line 9)
    - `import gc as gc` (line 10)
    - `import ipaddress as ipaddress` (line 12)
    - `import textwrap as textwrap` (line 18)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_COUNTY_TO_PRECINCTS_MAP` (line 59)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_STATE_TO_COUNTY_MAP` (line 59)
    - `from Context_Integration.Context_Library.constants import STATE_ABBR`
      (line 59)
    - `from Context_Integration.Context_Library.constants import
      STATE_MODULE_MAP` (line 59)
    - `from Context_Integration.Context_Library.constants import
      build_camelot_row_filter` (line 59)
    - `from Context_Integration.Context_Library.constants import
      normalize_party_label` (line 59)
    - `from utils.logger_singleton import logger` (line 67)
- Task markers:
  - L415 **WARNING**: (f"\[safe_query\] session.query({model}) failed: {e}")
  - L438 **WARNING**: (f"\[safe_filter_by\] No mapper found for model {model}")
  - L444 **WARNING**: (f"\[safe_filter_by\] Could not inspect model {model}:
    {e}")
  - L458 **WARNING**: (f"\[safe_filter_by\] filter_by failed: {e}")
  - L471 **WARNING**: (f"\[safe_first\] query.first() failed: {e}")
  - L629 **WARNING**: ({
  - L630 **WARNING**: ",
  - L656 **WARNING**: (f"\[PLUGIN EXTRACTION\] Plugin {plugin} has no callable
    'extract' method.")
  - L790 **WARNING**: (f"\[WARN\] Model save failed (attempt {attempt}): {e}")
  - L1004 **WARNING**: (f"\[safe_append\] Target is not a list: {type(lst)};
    coercing to list.")
  - L1026 **WARNING**: (f"\[safe_update\] Target is not a dict: {type(dct)}")
  - L1030 **WARNING**: (f"\[safe_update\] Updates is not a dict:
    {type(updates)}")
  - L1050 **WARNING**: (f"\[safe_extend\] Target is not a list: {type(lst)};
    coercing to list.")
  - L1390 **WARNING**: (f"\[DOM_PARTS\] '{label}' is not a list for URL: {url}
    (type: {type(lst).\_\_name\_\_})")
  - L1799 **WARNING**: (f"State '{state_norm}' not found in county map")
  - L2665 **WARNING**: (f"\[inventory\] architecture.md not found at {md_file}")
  - L2671 **WARNING**: ("\[inventory\] Markers not found in architecture.md;
    aborting replace.")
  - L2686 **WARNING**: ("\[inventory\] generate_project_map completed with
    warnings; check markers and path.")
  - L2732 **WARN**: ) and return their metadata."""
  - L2734 **WARN**: ", "WARNING", "NOTE", "HA" + "CK", "X"_3, "BUG")
  - L3502 **BUG**: '\]
  - L3504 **WARN**: ', 'WARNING', 'NOTE'\]
  - L3633 **WARNING**: (f"\[noise\] No suggestions file found at {path}")
  - L4073 **TODO**: index."""
  - L4165 **WARNING**: (f"Failed to log rejection reason: {e}")
- Outgoing cross-module calls (sample):
  - os.getenv (line 49)
  - Context\_Integration.Context\_Library.constants.STATE\_MODULE\_MAP.keys
    (line 72)
  - Context\_Integration.Context\_Library.constants.KNOWN\_STATE\_TO\_COUNTY\_MAP.keys
    (line 72)
  - name.encode (line 222)
  - name.replace (line 225)
  - re.sub (line 228)
  - re.sub (line 231)
  - name.replace (line 234)
  - re.sub (line 237)
  - re.sub (line 238)
  - name.strip (line 241)
  - name.rsplit (line 244)
  - base.endswith (line 247)
  - base.rstrip (line 248)
  - re.sub (line 257)
  - re.sub (line 258)
  - name.strip (line 259)
  - name.upper (line 267)
  - pathlib.Path (line 276)
  - pathlib.Path (line 286)
  - pathlib.Path (line 293)
  - pathlib.Path (line 296)
  - target.is\_relative\_to (line 301)
  - base.resolve (line 301)
  - base.resolve (line 305)
  - pathlib.Path (line 324)
  - pathlib.Path (line 325)
  - raw\_path.is\_absolute (line 326)
  - pathlib.Path.cwd (line 326)
  - target.resolve (line 328)
  - target.resolve (line 333)
  - resolved.exists (line 338)
  - resolved.mkdir (line 342)
  - pathlib.Path (line 349)
  - re.split (line 356)
  - sanitized\_parts.append (line 361)
  - base\_path.joinpath (line 363)
  - candidate.resolve (line 365)
  - pathlib.Path (line 377)
  - candidate.exists (line 379)
  - candidate.is\_dir (line 379)
  - candidate.exists (line 382)
  - candidate.mkdir (line 385)
  - typing.TypeVar (line 389)
  - c.isalnum (line 401)
  - re.sub (line 402)
  - s.replace (line 403)
  - re.sub (line 404)
  - flask.session.query (line 413)
  - utils.logger\_singleton.logger.warning (line 415)
- Inbound references:
  - safe\_filename ← shared_logic.py:359
  - is\_path\_safe ← shared_logic.py:335
  - is\_path\_safe ← shared_logic.py:369
  - safe\_slug ← shared_logic.py:2164
  - safe\_query ← shared_logic.py:485
  - safe\_key ← shared_logic.py:441
  - \_filter\_valid\_kwargs ← shared_logic.py:454
  - \_filter\_valid\_kwargs ← shared_logic.py:490
  - safe\_filter\_by ← shared_logic.py:486
  - safe\_first ← shared_logic.py:487
  - \_resolve\_host\_ips ← shared_logic.py:625
  - safe\_merge\_defaults ← shared_logic.py:692
  - safe\_strip ← shared_logic.py:1617
  - safe\_strip ← shared_logic.py:1682
  - safe\_strip ← shared_logic.py:1719
  - safe\_strip ← shared_logic.py:2279
  - safe\_strip ← shared_logic.py:2297
  - safe\_strip ← shared_logic.py:2298
  - safe\_strip ← user_prompt.py:581
  - safe\_strip ← user_prompt.py:651
  - safe\_strip ← user_prompt.py:687
  - safe\_strip ← user_prompt.py:689
  - safe\_commit ← shared_logic.py:494
  - safe\_replace ← shared_logic.py:1617
  - safe\_replace ← shared_logic.py:1683
  - safe\_replace ← shared_logic.py:1684
  - safe\_replace ← shared_logic.py:1719
  - safe\_replace ← shared_logic.py:1753
  - safe\_replace ← shared_logic.py:1753
  - safe\_get ← shared_logic.py:688
  - safe\_get ← shared_logic.py:707
  - safe\_append ← shared_logic.py:2395
  - safe\_append ← shared_logic.py:2421
  - safe\_append ← shared_logic.py:2436
  - safe\_append ← shared_logic.py:2440
  - safe\_update ← shared_logic.py:1035
  - convert\_ndarrays ← shared_logic.py:1065
  - convert\_ndarrays ← shared_logic.py:1067
  - \_to\_json\_safe ← shared_logic.py:1097
  - \_to\_json\_safe ← shared_logic.py:1099
  - sync\_type\_and\_election\_types ← shared_logic.py:2447
  - safe\_lower ← shared_logger.py:481
  - safe\_lower ← shared_logic.py:878
  - safe\_lower ← shared_logic.py:880
  - safe\_lower ← shared_logic.py:1130
  - safe\_lower ← shared_logic.py:1132
  - safe\_lower ← shared_logic.py:1617
  - safe\_lower ← shared_logic.py:1682
  - safe\_lower ← shared_logic.py:1719
  - safe\_lower ← shared_logic.py:1752

### utils/spacy\_utils.py {#webapp-parser-utils-spacy-utils-py}

- Definitions:
  - function: `\_get\_nlp` (line 27)
  - function: `extract\_entities` (line 45)
  - function: `get\_sentences` (line 94)
  - function: `clean\_text` (line 101)
  - function: `extract\_entities\_from\_list` (line 104)
  - function: `extract\_entity\_labels` (line 107)
  - function: `is\_location\_entity` (line 114)
  - function: `extract\_locations` (line 117)
  - function: `extract\_dates` (line 124)
  - function: `filter\_entities\_by\_type` (line 131)
  - function: `entity\_frequency` (line 138)
  - function: `get\_entity\_context` (line 150)
  - function: `similarity\_score` (line 160)
  - function: `extract\_persons` (line 170)
  - function: `extract\_organizations` (line 177)
  - function: `extract\_money` (line 184)
  - function: `extract\_emails` (line 191)
  - function: `extract\_urls` (line 194)
  - function: `load\_known\_states\_counties` (line 200)
  - function: `normalize\_location` (line 211)
  - function: `is\_known\_state` (line 219)
  - function: `is\_known\_county` (line 222)
  - function: `detect\_noisy\_or\_ambiguous\_entities` (line 225)
  - function: `canonicalize\_entity` (line 245)
  - function: `validate\_contest` (line 251)
  - function: `flag\_suspicious\_contests` (line 278)
  - function: `demo\_analysis` (line 331)
- Imports:
  - **Standard Library** (9):
    - `import os as os` (line 12)
    - `import re as re` (line 13)
    - `import sys as sys` (line 14)
    - `from collections import Counter` (line 15)
    - `from typing import Any` (line 16)
    - `from typing import Dict` (line 16)
    - `from typing import List` (line 16)
    - `from typing import Set` (line 16)
    - `from typing import Tuple` (line 16)
  - **Third-party** (1):
    - `import orjson as orjson` (line 18)
  - **Local/Project** (5):
    - `from __future__ import annotations` (line 1)
    - `from Context_Integration.Context_Library.constants import
      KNOWN_STATE_TO_COUNTY_MAP` (line 22)
    - `from logger_singleton import logger` (line 23)
    - `from shared_logic import safe_get` (line 24)
    - `from shared_logic import safe_lower` (line 24)
- Task markers:
  - L40 **WARNING**: (f"spaCy unavailable or model load failed: {e}")
- Outgoing cross-module calls (sample):
  - spacy.load (line 38)
  - logger\_singleton.logger.warning (line 40)
  - text.strip (line 50)
  - logger\_singleton.logger.error (line 51)
  - text.lower (line 57)
  - Context\_Integration.Context\_Library.constants.KNOWN\_STATE\_TO\_COUNTY\_MAP.keys
    (line 60)
  - st.replace (line 63)
  - matches.append (line 66)
  - st\_norm.title (line 66)
  - Context\_Integration.Context\_Library.constants.KNOWN\_STATE\_TO\_COUNTY\_MAP.items
    (line 68)
  - c.lower (line 73)
  - matches.append (line 74)
  - ent.lower (line 79)
  - seen.add (line 82)
  - out.append (line 83)
  - logger\_singleton.logger.error (line 91)
  - text.lower (line 102)
  - collections.Counter (line 139)
  - counter.most\_common (line 148)
  - text.lower (line 152)
  - entity.lower (line 152)
  - contexts.append (line 156)
  - text.lower (line 157)
  - entity.lower (line 157)
  - doc1.similarity (line 167)
  - re.findall (line 192)
  - re.findall (line 196)
  - Context\_Integration.Context\_Library.constants.KNOWN\_STATE\_TO\_COUNTY\_MAP.keys
    (line 205)
  - Context\_Integration.Context\_Library.constants.KNOWN\_STATE\_TO\_COUNTY\_MAP.values
    (line 207)
  - counties.update (line 208)
  - c.lower (line 208)
  - s.lower (line 209)
  - name.lower (line 215)
  - re.sub (line 216)
  - re.search (line 241)
  - noisy.append (line 242)
  - re.sub (line 249)
  - entity.strip (line 249)
  - orjson.loads (line 290)
  - f.read (line 290)
  - logger\_singleton.logger.error (line 292)
  - shared\_logic.safe\_get (line 298)
  - context\_library.keys (line 304)
  - shared\_logic.safe\_lower (line 304)
  - shared\_logic.safe\_lower (line 304)
  - context\_library.get (line 308)
  - shared\_logic.safe\_get (line 314)
  - result.setdefault (line 316)
  - flagged.append (line 318)
  - result.get (line 321)
- Inbound references:
  - \_get\_nlp ← spacy_utils.py:53
  - \_get\_nlp ← spacy_utils.py:95
  - \_get\_nlp ← spacy_utils.py:108
  - \_get\_nlp ← spacy_utils.py:118
  - \_get\_nlp ← spacy_utils.py:125
  - \_get\_nlp ← spacy_utils.py:132
  - \_get\_nlp ← spacy_utils.py:141
  - \_get\_nlp ← spacy_utils.py:161
  - \_get\_nlp ← spacy_utils.py:171
  - \_get\_nlp ← spacy_utils.py:178
  - \_get\_nlp ← spacy_utils.py:185
  - \_get\_nlp ← spacy_utils.py:234
  - extract\_entities ← html_election_parser.py:2923
  - extract\_entities ← spacy_utils.py:105
  - extract\_entities ← spacy_utils.py:258
  - extract\_entities ← spacy_utils.py:332
  - get\_sentences ← spacy_utils.py:333
  - is\_location\_entity ← spacy_utils.py:122
  - extract\_locations ← spacy_utils.py:259
  - extract\_locations ← spacy_utils.py:334
  - extract\_dates ← spacy_utils.py:260
  - extract\_dates ← spacy_utils.py:335
  - entity\_frequency ← spacy_utils.py:341
  - similarity\_score ← spacy_utils.py:342
  - extract\_persons ← spacy_utils.py:261
  - extract\_persons ← spacy_utils.py:336
  - extract\_organizations ← spacy_utils.py:262
  - extract\_organizations ← spacy_utils.py:337
  - extract\_money ← spacy_utils.py:338
  - extract\_emails ← spacy_utils.py:339
  - extract\_urls ← spacy_utils.py:340
  - load\_known\_states\_counties ← spacy_utils.py:295
  - load\_known\_states\_counties ← spacy_utils.py:344
  - normalize\_location ← spacy_utils.py:220
  - normalize\_location ← spacy_utils.py:223
  - is\_known\_state ← spacy_utils.py:264
  - is\_known\_county ← spacy_utils.py:265
  - detect\_noisy\_or\_ambiguous\_entities ← spacy_utils.py:263
  - validate\_contest ← spacy_utils.py:309
  - validate\_contest ← spacy_utils.py:345
  - demo\_analysis ← spacy_utils.py:351

### utils/status\_reconciliation.py {#webapp-parser-utils-status-reconciliation-py}

> Status Reconciliation System

- Definitions:
  - class: `StatusReconciliation` (line 15)
  - class: `WorklistParser` (line 197)
  - function: `\_normalize\_state` (line 260)
- Imports:
  - **Standard Library** (4):
    - `from typing import Any` (line 12)
    - `from typing import Dict` (line 12)
    - `from typing import Optional` (line 12)
    - `from typing import Tuple` (line 12)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 10)
- Task markers:
  - L58 **WARNING**: ', 'priority': 4},
  - L59 **WARNING**: ', 'priority': 7},
  - L61 **WARNING**: ', 'priority': 9},
  - L72 **WARNING**: ', 'priority': 6},
- Outgoing cross-module calls (sample):
  - StatusReconciliation.\_normalize\_worklist\_status (line 114)
  - worklist\_status.strip (line 146)
  - mappings.get (line 162)
  - badge\_info.get (line 171)
  - row.items (line 220)
  - state.strip (line 236)
  - race.strip (line 237)
  - race.lower (line 243)
  - state.strip (line 264)
  - state\_lower.replace (line 288)
- Inbound references:
  - \_normalize\_state ← status_reconciliation.py:240

### utils/strategy\_concurrency.py {#webapp-parser-utils-strategy-concurrency-py}

> strategy_concurrency.py

- Definitions:
  - function: `run\_strategies\_concurrently` (line 19)
  - function: `\_safe\_run\_strategy` (line 68)
  - async_function: `run\_strategies\_concurrently\_async` (line 76)
- Imports:
  - **Standard Library** (7):
    - `import asyncio as asyncio` (line 8)
    - `from functools import partial` (line 10)
    - `from typing import Any` (line 11)
    - `from typing import Callable` (line 11)
    - `from typing import Dict` (line 11)
    - `from typing import List` (line 11)
    - `from typing import Tuple` (line 11)
  - **Local/Project** (5):
    - `from __future__ import annotations` (line 6)
    - `from concurrent.futures import ThreadPoolExecutor` (line 9)
    - `from concurrent.futures import as_completed` (line 9)
    - `from browser_utils import safe_content` (line 13)
    - `from logger_singleton import logger` (line 14)
- Task markers:
  - L37 **WARNING**: (f"\[CONCURRENCY\] DOM strategy {name} failed: {e}")
  - L65 **WARNING**: (f"\[CONCURRENCY\] Strategy {name} error: {e}")
  - L73 **WARNING**: (f"\[CONCURRENCY\] {_safe_run_strategy.\_\_name\_\_} {name}
    failed: {e}")
  - L102 **WARNING**: (f"\[CONCURRENCY\]\[ASYNC\] DOM strategy {name} failed:
    {e}")
  - L120 **WARNING**: (f"\[CONCURRENCY\]\[ASYNC\] Strategy {name} error: {e}")
- Outgoing cross-module calls (sample):
  - results.append (line 35)
  - logger\_singleton.logger.warning (line 37)
  - browser\_utils.safe\_content (line 42)
  - concurrent.futures.ThreadPoolExecutor (line 51)
  - pool.submit (line 55)
  - concurrent.futures.as\_completed (line 57)
  - fut.result (line 60)
  - results.append (line 63)
  - logger\_singleton.logger.warning (line 65)
  - logger\_singleton.logger.warning (line 73)
  - asyncio.get\_running\_loop (line 90)
  - loop.run\_in\_executor (line 97)
  - functools.partial (line 97)
  - results.append (line 100)
  - logger\_singleton.logger.warning (line 102)
  - browser\_utils.safe\_content (line 106)
  - loop.run\_in\_executor (line 118)
  - functools.partial (line 118)
  - logger\_singleton.logger.warning (line 120)
  - asyncio.create\_task (line 123)
  - asyncio.as\_completed (line 124)
  - results.append (line 128)

### utils/structure\_cache.py {#webapp-parser-utils-structure-cache-py}

> structure_cache.py

- Definitions:
  - function: `table\_signature` (line 14)
  - function: `cache\_table\_structure` (line 19)
  - function: `get\_cached\_structure` (line 25)
- Imports:
  - **Standard Library** (4):
    - `import hashlib as hashlib` (line 7)
    - `from typing import Any` (line 8)
    - `from typing import Dict` (line 8)
    - `from typing import List` (line 8)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 5)
    - `from detect import normalize_header` (line 10)
- Outgoing cross-module calls (sample):
  - detect.normalize\_header (line 15)
  - hashlib.sha1 (line 17)
  - sig\_str.encode (line 17)
  - \_STRUCTURE\_CACHE.setdefault (line 22)
  - \_STRUCTURE\_CACHE.get (line 26)

### utils/table\_builder.py {#webapp-parser-utils-table-builder-py}

- Definitions:
  - function: `\_normalize\_header\_cached` (line 76)
  - function: `\_norm\_header` (line 81)
  - function: `\_percent\_norms` (line 91)
  - function: `\_percent\_reported\_norm` (line 105)
  - function: `\_looks\_like\_location\_header` (line 175)
  - function: `\_location\_priority\_score` (line 183)
  - function: `\_candidate\_header\_info` (line 194)
  - function: `\_extract\_candidate\_blocks` (line 213)
  - function: `\_coerce\_int\_for\_total` (line 224)
  - function: `\_ensure\_division\_totals` (line 247)
  - function: `\_apply\_canonical\_order` (line 324)
  - function: `\_emit` (line 406)
  - function: `\_salvage\_promote\_best\_row\_as\_header` (line 425)
  - function: `\_salvage\_promote\_first\_row\_as\_header` (line 480)
  - function: `\_sanitize\_headers\_and\_rows` (line 509)
  - function: `\_stringify\_for\_pivot` (line 600)
  - function: `\_stringify\_entity\_info` (line 623)
  - function: `\_drop\_title\_noise\_rows` (line 648)
  - function: `build\_dynamic\_table` (line 751)
  - function: `build\_table\_noninteractive` (line 1042)
  - function: `\_get\_table\_builder\_cache\_dir` (line 1076)
  - function: `\_save\_table\_builder\_cache` (line 1084)
  - function: `\_list\_table\_builder\_cache` (line 1108)
  - function: `\_load\_table\_builder\_cache` (line 1121)
  - function: `prompt\_user\_to\_confirm\_table\_structure` (line 1143)
  - function: `interactive\_batch\_operations` (line 1426)
  - function: `auto\_suggest\_corrections` (line 1483)
  - function: `dynamic\_confidence\_threshold` (line 1540)
  - function: `\_unify\_percent\_columns` (line 1582)
- Imports:
  - **Standard Library** (12):
    - `import copy as copy` (line 3)
    - `import os as os` (line 4)
    - `import re as re` (line 5)
    - `import time as time` (line 6)
    - `from collections import OrderedDict` (line 7)
    - `from functools import lru_cache` (line 8)
    - `from typing import TYPE_CHECKING` (line 9)
    - `from typing import Any` (line 9)
    - `from typing import Dict` (line 9)
    - `from typing import List` (line 9)
    - `from typing import Optional` (line 9)
    - `from typing import Tuple` (line 9)
  - **Third-party** (1):
    - `import orjson as orjson` (line 16)
  - **Local/Project** (39):
    - `from __future__ import annotations` (line 1)
    - `from rich.table import Table` (line 17)
    - `from config import CACHE_DIR` (line 19)
    - `from config import TABLE_BUILDER_AUTO_ACCEPT_THRESHOLD` (line 19)
    - `from config import TABLE_BUILDER_LOW_CONFIDENCE_THRESHOLD` (line 19)
    - `from Context_Integration.Context_Library.constants import
      BALLOT_TYPES_SORT_ORDER` (line 24)
    - `from Context_Integration.Context_Library.constants import
      LOCATION_KEYWORDS` (line 24)
    - `from Context_Integration.Context_Library.constants import
      PERCENT_KEYWORDS` (line 24)
    - `from Context_Integration.Context_Library.constants import
      TABLE_BUILDER_CANDIDATE_SUFFIXES` (line 24)
    - `from Context_Integration.Context_Library.constants import
      TABLE_BUILDER_LOCATION_PRIORITY` (line 24)
    - `from Context_Integration.Context_Library.constants import
      TABLE_BUILDER_LOCATION_TOKENS` (line 24)
    - `from Context_Integration.Context_Library.constants import TOTAL_KEYWORDS`
      (line 24)
    - `from Context_Integration.Context_Library.constants import
      get_camelot_row_regex` (line 24)
    - `from Context_Integration.Context_Library.constants import
      get_camelot_title_regex` (line 24)
    - `from Context_Integration.Context_Library.constants import
      is_pseudo_result_party` (line 24)
    - `from coordinator_protocol import CoordinatorProtocol` (line 36)
    - `from detect import emit_metric` (line 37)
    - `from detect import harmonize_headers_and_data` (line 37)
    - `from detect import nlp_entity_annotate_table` (line 37)
    - `from detect import normalize_header` (line 37)
    - `from logger_singleton import logger` (line 43)
    - `from merge_utils import merge_table_data` (line 44)
    - `from pivot import pivot_candidate_groups_from_rawjson` (line 45)
    - `from pivot import pivot_to_wide as pivot_to_wide_format` (line 46)
    - `from salvage import collapse_ballot_synonym_columns` (line 47)
    - `from shared_logic import build_camelot_row_filter_for_context` (line 48)
    - `from shared_logic import log_rejection_reason` (line 48)
    - `from shared_logic import record_noise_suggestion` (line 48)
    - `from shared_logic import resolve_state_county_from_context` (line 48)
    - `from shared_logic import safe_append` (line 48)
    - `from shared_logic import safe_copy` (line 48)
    - `from shared_logic import safe_get` (line 48)
    - `from shared_logic import safe_isalnum` (line 48)
    - `from shared_logic import safe_lower` (line 48)
    - `from shared_logic import safe_replace` (line 48)
    - `from shared_logic import safe_strip` (line 48)
    - `from shared_logic import safe_values` (line 48)
    - `from structure_cache import cache_table_structure` (line 62)
    - `from structure_cache import table_signature` (line 62)
- Task markers:
  - L821 **WARNING**: ", "builder", "\[TABLE_BUILDER\] dynamic_table_extractor
    failed for panel table", session_id, error=str(e))
  - L833 **WARNING**: ", "builder", "\[TABLE_BUILDER\] dynamic_table_extractor
    failed (no panels path)", session_id, error=str(e))
  - L841 **WARNING**: ", "builder", "\[TABLE_BUILDER\] all_panel_tables was not
    a list; coercing to empty list", session_id,
    got_type=str(type(all_panel_tables)))
  - L850 **WARNING**: ", "builder", "\[TABLE_BUILDER\] Dropping invalid table
    entry", session_id, entry_type=str(type(item)))
  - L867 **WARNING**: ", "builder", "\[TABLE_BUILDER\] sanitize failed",
    session_id, error=str(e))
  - L872 **WARNING**: ", "builder", "\[TABLE_BUILDER\] harmonize failed",
    session_id, error=str(e))
  - L878 **WARNING**: ", "builder", "\[TABLE_BUILDER\]
    collapse_ballot_synonym_columns failed", session_id, error=str(e))
  - L930 **WARNING**: ",
  - L955 **WARNING**: ", "builder", "\[TABLE_BUILDER\] entity annotate failed",
    session_id, error=str(e))
  - L960 **WARNING**: ", "builder", "\[TABLE_BUILDER\] stringify entity_info
    failed", session_id, error=str(e))
  - L980 **WARNING**: ", "builder", "\[TABLE_BUILDER\] pivot_to_wide failed",
    session_id, error=str(e))
  - L1000 **WARNING**: ", "builder", "\[TABLE_BUILDER\] ensure division totals
    failed", session_id, error=str(e))
  - L1326 **WARNING**: ", "builder", f"\[TABLE_BUILDER\] Column marked
    incorrect: {col_name}", session_id, contest=contest)
  - L1399 **WARNING**: ", "builder", "\[TABLE_BUILDER\] Failed to persist table
    structure logs", session_id, error=str(e))
  - L1414 **WARNING**: ", "builder", "\[TABLE_BUILDER\] Failed to persist
    coordinator DB log", session_id, error=str(e))
- Outgoing cross-module calls (sample):
  - detect.normalize\_header (line 78)
  - functools.lru\_cache (line 75)
  - norms.update (line 99)
  - norms.add (line 100)
  - functools.lru\_cache (line 90)
  - functools.lru\_cache (line 104)
  - dict.fromkeys (line 123)
  - tok.lower (line 123)
  - dict.fromkeys (line 127)
  - tok.lower (line 127)
  - dict.fromkeys (line 155)
  - term.lower (line 162)
  - \_TOTAL\_KEYWORD\_NORMS.update (line 164)
  - \_CANDIDATE\_SUFFIX\_NORMS.update (line 170)
  - header.lower (line 179)
  - \_LOCATION\_PRIORITY\_NORMS.index (line 186)
  - header.lower (line 187)
  - header.split (line 197)
  - left.strip (line 198)
  - right.strip (line 199)
  - bt.lower (line 208)
  - right.lower (line 208)
  - collections.OrderedDict (line 214)
  - blocks.setdefault (line 220)
  - val.is\_integer (line 232)
  - val.replace (line 234)
  - s.endswith (line 235)
  - s.lstrip (line 239)
  - row.get (line 258)
  - candidate\_blocks.values (line 265)
  - candidate\_total\_cols.append (line 273)
  - ballot\_value\_cols.append (line 275)
  - bt.lower (line 278)
  - suffix.lower (line 278)
  - ballot\_value\_cols.append (line 279)
  - row.get (line 289)
  - row.get (line 296)
  - row.get (line 298)
  - row.items (line 302)
  - row.setdefault (line 319)
  - headers.index (line 336)
  - ordered.append (line 337)
  - seen.add (line 338)
  - ordered.append (line 342)
  - seen.add (line 343)
  - ordered.append (line 349)
  - seen.add (line 350)
  - headers.index (line 353)
  - ordered.append (line 356)
  - seen.add (line 357)
- Inbound references:
  - \_normalize\_header\_cached ← table_builder.py:84
  - \_normalize\_header\_cached ← table_builder.py:86
  - \_normalize\_header\_cached ← table_builder.py:87
  - \_norm\_header ← table_builder.py:98
  - \_norm\_header ← table_builder.py:99
  - \_norm\_header ← table_builder.py:100
  - \_norm\_header ← table_builder.py:106
  - \_norm\_header ← table_builder.py:120
  - \_norm\_header ← table_builder.py:141
  - \_norm\_header ← table_builder.py:156
  - \_norm\_header ← table_builder.py:159
  - \_norm\_header ← table_builder.py:165
  - \_norm\_header ← table_builder.py:166
  - \_norm\_header ← table_builder.py:167
  - \_norm\_header ← table_builder.py:168
  - \_norm\_header ← table_builder.py:172
  - \_norm\_header ← table_builder.py:176
  - \_norm\_header ← table_builder.py:184
  - \_norm\_header ← table_builder.py:202
  - \_norm\_header ← table_builder.py:250
  - \_norm\_header ← table_builder.py:251
  - \_norm\_header ← table_builder.py:271
  - \_norm\_header ← table_builder.py:303
  - \_norm\_header ← table_builder.py:348
  - \_norm\_header ← table_builder.py:371
  - \_norm\_header ← table_builder.py:381
  - \_norm\_header ← table_builder.py:382
  - \_norm\_header ← table_builder.py:390
  - \_norm\_header ← table_builder.py:391
  - \_norm\_header ← table_builder.py:392
  - \_norm\_header ← table_builder.py:393
  - \_norm\_header ← table_builder.py:441
  - \_norm\_header ← table_builder.py:457
  - \_norm\_header ← table_builder.py:461
  - \_norm\_header ← table_builder.py:491
  - \_norm\_header ← table_builder.py:495
  - \_norm\_header ← table_builder.py:677
  - \_norm\_header ← table_builder.py:678
  - \_norm\_header ← table_builder.py:679
  - \_norm\_header ← table_builder.py:680
  - \_norm\_header ← table_builder.py:681
  - \_norm\_header ← table_builder.py:711
  - \_norm\_header ← table_builder.py:892
  - \_norm\_header ← table_builder.py:893
  - \_norm\_header ← table_builder.py:894
  - \_norm\_header ← table_builder.py:895
  - \_norm\_header ← table_builder.py:896
  - \_norm\_header ← table_builder.py:945
  - \_norm\_header ← table_builder.py:946
  - \_norm\_header ← table_builder.py:948

### utils/table\_core.py {#webapp-parser-utils-table-core-py}

> table_core.py (refactored orchestrator)

- Definitions:
  - function: `\_stringify\_for\_pivot` (line 83)
  - function: `\_deduplicate\_tables` (line 100)
  - function: `\_log\_extraction\_summary` (line 114)
  - function: `\_annotate\_entities\_via\_detector` (line 123)
  - function: `robust\_table\_extraction` (line 142)
  - function: `\_sanitize\_headers` (line 317)
  - function: `build\_table\_from\_page` (line 333)
  - async_function: `robust\_table\_extraction\_async` (line 352)
  - async_function: `build\_table\_from\_page\_async` (line 464)
  - function: `auto\_table\_build` (line 480)
- Imports:
  - **Standard Library** (8):
    - `import asyncio as asyncio` (line 41)
    - `import re as re` (line 42)
    - `import time as time` (line 43)
    - `from typing import Any` (line 44)
    - `from typing import Dict` (line 44)
    - `from typing import List` (line 44)
    - `from typing import Optional` (line 44)
    - `from typing import Tuple` (line 44)
  - **Local/Project** (23):
    - `from __future__ import annotations` (line 39)
    - `from detect import emit_metric` (line 47)
    - `from detect import harmonize_headers_and_data` (line 47)
    - `from detect import normalize_header` (line 47)
    - `from detector import Detector` (line 54)
    - `from extraction_strategies import strategy_dom_repetition` (line 55)
    - `from extraction_strategies import strategy_heading_associated` (line 55)
    - `from extraction_strategies import strategy_html_tables` (line 55)
    - `from extraction_strategies import strategy_ml_detection` (line 55)
    - `from extraction_strategies import strategy_nlp_fallback` (line 55)
    - `from extraction_strategies import strategy_pattern_based` (line 55)
    - `from extraction_strategies import strategy_selectolax_fallback` (line 55)
    - `from logger_singleton import logger` (line 64)
    - `from pivot import pivot_candidate_groups_from_rawjson` (line 65)
    - `from pivot import pivot_to_wide as pivot_to_wide_unified` (line 66)
    - `from salvage import _salvage_rows_from_rawjson` (line 69)
    - `from salvage import combine_panel_tables_by_precinct` (line 69)
    - `from salvage import merge_multiline_candidate_rows` (line 69)
    - `from salvage import remove_footer_and_summary_rows` (line 69)
    - `from salvage import remove_outlier_and_empty_rows` (line 69)
    - `from shared_logic import safe_get` (line 76)
    - `from strategy_concurrency import run_strategies_concurrently` (line 79)
    - `from strategy_concurrency import run_strategies_concurrently_async` (line
      79)
- Task markers:
  - L231 **WARNING**: (f"\[TABLE BUILDER\] Concurrent strategies execution
    failed: {e}")
  - L288 **WARNING**: (f"\[TABLE BUILDER\] RawJSON pivot failed: {e}")
  - L296 **WARNING**: (f"\[TABLE BUILDER\] pivot_to_wide signature mismatch
    (skipped): {e}")
  - L298 **WARNING**: (f"\[TABLE BUILDER\] pivot_to_wide failed (skipped): {e}")
  - L349 **WARNING**: (f"\[TABLE BUILDER\] finalize output failed: {e}")
  - L414 **WARNING**: (f"\[TABLE BUILDER\]\[ASYNC\] Concurrent strategies
    execution failed: {e}")
  - L477 **WARNING**: (f"\[TABLE BUILDER\]\[ASYNC\] finalize output failed:
    {e}")
- Outgoing cross-module calls (sample):
  - r.items (line 88)
  - out.append (line 97)
  - sig\_map.values (line 112)
  - e.get (line 116)
  - e.get (line 118)
  - logger\_singleton.logger.info (line 121)
  - detector.annotate\_entities (line 127)
  - time.time (line 151)
  - shared\_logic.safe\_get (line 153)
  - shared\_logic.safe\_get (line 154)
  - logger\_singleton.logger.info (line 156)
  - detector.Detector (line 163)
  - collected.append (line 171)
  - extraction\_logs.append (line 172)
  - shared\_logic.safe\_get (line 181)
  - collected.append (line 187)
  - extraction\_logs.append (line 188)
  - strategy\_concurrency.run\_strategies\_concurrently (line 214)
  - collected.append (line 223)
  - extraction\_logs.append (line 224)
  - logger\_singleton.logger.warning (line 231)
  - detect.emit\_metric (line 234)
  - salvage.combine\_panel\_tables\_by\_precinct (line 243)
  - salvage.merge\_multiline\_candidate\_rows (line 248)
  - salvage.\_salvage\_rows\_from\_rawjson (line 251)
  - salvage.remove\_footer\_and\_summary\_rows (line 254)
  - salvage.remove\_outlier\_and\_empty\_rows (line 255)
  - detect.emit\_metric (line 258)
  - detect.harmonize\_headers\_and\_data (line 266)
  - shared\_logic.safe\_get (line 269)
  - shared\_logic.safe\_get (line 270)
  - shared\_logic.safe\_get (line 271)
  - re.sub (line 275)
  - re.sub (line 276)
  - n.lower (line 276)
  - pivot.pivot\_candidate\_groups\_from\_rawjson (line 282)
  - detect.emit\_metric (line 286)
  - logger\_singleton.logger.warning (line 288)
  - shared\_logic.safe\_get (line 290)
  - pivot.pivot\_to\_wide (line 294)
  - logger\_singleton.logger.warning (line 296)
  - logger\_singleton.logger.warning (line 298)
  - detect.emit\_metric (line 302)
  - logger\_singleton.logger.info (line 305)
  - time.time (line 311)
  - detect.normalize\_header (line 326)
  - seen.add (line 329)
  - cleaned.append (line 330)
  - shared\_logic.safe\_get (line 342)
  - shared\_logic.safe\_get (line 347)
- Inbound references:
  - \_deduplicate\_tables ← table_core.py:239
  - \_deduplicate\_tables ← table_core.py:421
  - \_log\_extraction\_summary ← table_core.py:235
  - \_log\_extraction\_summary ← table_core.py:259
  - \_log\_extraction\_summary ← table_core.py:303
  - \_log\_extraction\_summary ← table_core.py:418
  - \_log\_extraction\_summary ← table_core.py:436
  - \_log\_extraction\_summary ← table_core.py:451
  - \_annotate\_entities\_via\_detector ← table_core.py:263
  - \_annotate\_entities\_via\_detector ← table_core.py:439
  - robust\_table\_extraction ← table_core.py:339
  - robust\_table\_extraction ← table_core.py:487
  - robust\_table\_extraction ← test_table_builder.py:70
  - \_sanitize\_headers ← table_core.py:300
  - \_sanitize\_headers ← table_core.py:448
  - robust\_table\_extraction\_async ← table_core.py:468
  - robust\_table\_extraction\_async ← table_core.py:486

### utils/telemetry.py {#webapp-parser-utils-telemetry-py}

- Definitions:
  - function: `\_derive\_url\_fields` (line 23)
  - function: `emit\_telemetry\_event` (line 35)
- Imports:
  - **Standard Library** (6):
    - `import hashlib as hashlib` (line 1)
    - `import json as json` (line 2)
    - `import os as os` (line 3)
    - `import time as time` (line 4)
    - `from typing import Any` (line 5)
    - `from typing import Dict` (line 5)
- Outgoing cross-module calls (sample):
  - os.getcwd (line 11)
  - os.makedirs (line 20)
  - hashlib.sha1 (line 29)
  - payload.setdefault (line 37)
  - payload.setdefault (line 38)
  - time.time (line 38)
  - payload.setdefault (line 39)
  - time.strftime (line 39)
  - time.gmtime (line 39)
  - payload.get (line 42)
  - payload.get (line 42)
  - payload.get (line 42)
  - payload.setdefault (line 45)
  - derived.get (line 45)
  - payload.setdefault (line 46)
  - derived.get (line 46)
  - payload.setdefault (line 49)
  - logger.info (line 52)
  - json.dumps (line 59)
  - f.write (line 60)
- Inbound references:
  - \_derive\_url\_fields ← telemetry.py:44

### utils/telemetry\_agg.py {#webapp-parser-utils-telemetry-agg-py}

- Definitions:
  - function: `\_read` (line 14)
  - function: `\_write` (line 23)
  - function: `get\_counters` (line 38)
  - function: `increment\_counter` (line 41)
  - function: `set\_counter` (line 63)
  - function: `reset\_counters` (line 69)
- Imports:
  - **Standard Library** (5):
    - `import json as json` (line 1)
    - `import os as os` (line 2)
    - `import time as time` (line 3)
    - `from typing import Any` (line 4)
    - `from typing import Dict` (line 4)
- Outgoing cross-module calls (sample):
  - os.getcwd (line 9)
  - os.makedirs (line 12)
  - json.load (line 19)
  - json.dump (line 27)
  - f.flush (line 28)
  - os.fsync (line 29)
  - f.fileno (line 29)
  - os.replace (line 30)
  - os.remove (line 34)
  - data.get (line 46)
  - data.setdefault (line 50)
  - time.time (line 50)
  - data.setdefault (line 66)
  - time.time (line 66)
- Inbound references:
  - \_read ← telemetry_agg.py:39
  - \_read ← telemetry_agg.py:44
  - \_read ← telemetry_agg.py:64
  - \_write ← telemetry_agg.py:51
  - \_write ← telemetry_agg.py:67
  - \_write ← telemetry_agg.py:70

### utils/url\_ingestion.py {#webapp-parser-utils-url-ingestion-py}

- Definitions:
  - function: `url\_already\_listed` (line 9)
- Imports:
  - **Standard Library** (1):
    - `import os as os` (line 3)
  - **Third-party** (2):
    - `from webapp.parser.utils.misc_utils import extract_url_and_label` (line
      5)
    - `from webapp.parser.utils.shared_logic import safe_strip` (line 6)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.shared\_logic.safe\_strip (line 15)
  - s.startswith (line 16)
  - webapp.parser.utils.misc\_utils.extract\_url\_and\_label (line 18)

### utils/url\_trust\_scorer.py {#webapp-parser-utils-url-trust-scorer-py}

> URL Trust Scoring System for Smart Elections Parser

- Definitions:
  - function: `\_load\_verified\_domains` (line 85)
  - function: `\_load\_trust\_history` (line 113)
  - function: `\_log\_trust\_decision` (line 184)
  - function: `get\_domain\_trust\_factors` (line 211)
  - function: `detect\_domain\_mimicry` (line 310)
  - function: `compute\_trust\_score` (line 376)
  - function: `should\_use\_snapshot\_mode` (line 586)
  - function: `should\_quarantine` (line 599)
  - function: `should\_reject` (line 630)
- Imports:
  - **Standard Library** (9):
    - `import json as json` (line 20)
    - `import re as re` (line 21)
    - `import time as time` (line 22)
    - `from pathlib import Path` (line 23)
    - `from typing import Any` (line 24)
    - `from typing import Dict` (line 24)
    - `from typing import List` (line 24)
    - `from typing import Tuple` (line 24)
    - `from urllib.parse import urlparse` (line 25)
  - **Local/Project** (10):
    - `from __future__ import annotations` (line 18)
    - `from config import LOG_DIR` (line 33)
    - `from config import PROJECT_ROOT` (line 33)
    - `from config import URL_ALLOWLIST_HOSTS` (line 33)
    - `from config import URL_ALLOWLIST_SUFFIXES` (line 33)
    - `from logger_singleton import logger` (line 39)
    - `from privilege_tiers import PrivilegeTier` (line 40)
    - `from privilege_tiers import get_principal_tier` (line 40)
    - `from privilege_tiers import should_apply_admin_boost` (line 40)
    - `from telemetry import emit_telemetry_event` (line 45)
- Task markers:
  - L104 **WARNING**: ({
  - L105 **WARNING**: ",
  - L459 **WARNING**: ({
  - L460 **WARNING**: ",
  - L470 **WARNING**: ({
  - L471 **WARNING**: ",
  - L484 **WARNING**: ({
  - L485 **WARNING**: ",
  - L497 **WARNING**: ({
  - L498 **WARNING**: ",
  - L648 **NOTE**:     Security Note:
  - L658 **WARNING**: ({
  - L659 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 54)
  - VERIFIED\_DOMAINS\_FILE.exists (line 95)
  - json.load (line 99)
  - logger\_singleton.logger.warning (line 104)
  - urllib.parse.urlparse (line 126)
  - TRUST\_HISTORY\_FILE.exists (line 131)
  - time.time (line 134)
  - line.strip (line 143)
  - orjson.loads (line 148)
  - json.loads (line 151)
  - entry.get (line 155)
  - entry.get (line 159)
  - entry.get (line 164)
  - entry.get (line 166)
  - urllib.parse.urlparse (line 187)
  - time.time (line 193)
  - orjson.dumps (line 204)
  - f.write (line 206)
  - urllib.parse.urlparse (line 243)
  - verified\_data.get (line 251)
  - verified\_data.get (line 255)
  - re.search (line 257)
  - re.search (line 266)
  - domain.endswith (line 275)
  - suffix.lower (line 275)
  - h.lower (line 278)
  - history.get (line 283)
  - domain.split (line 287)
  - re.search (line 296)
  - urllib.parse.urlparse (line 326)
  - u.startswith (line 334)
  - urllib.parse.urlparse (line 334)
  - u.lower (line 334)
  - verified\_data.get (line 335)
  - u.startswith (line 337)
  - urllib.parse.urlparse (line 337)
  - u.lower (line 337)
  - Levenshtein.distance (line 351)
  - logger\_singleton.logger.debug (line 414)
  - logger\_singleton.logger.debug (line 425)
  - logger\_singleton.logger.debug (line 436)
  - logger\_singleton.logger.debug (line 448)
  - logger\_singleton.logger.warning (line 459)
  - logger\_singleton.logger.warning (line 470)
  - logger\_singleton.logger.warning (line 484)
  - logger\_singleton.logger.warning (line 497)
  - privilege\_tiers.get\_principal\_tier (line 510)
  - urllib.parse.urlparse (line 511)
  - privilege\_tiers.should\_apply\_admin\_boost (line 515)
  - logger\_singleton.logger.info (line 520)
- Inbound references:
  - \_load\_verified\_domains ← url_trust_scorer.py:250
  - \_load\_verified\_domains ← url_trust_scorer.py:333
  - \_load\_trust\_history ← url_trust_scorer.py:282
  - \_log\_trust\_decision ← url_trust_scorer.py:555
  - get\_domain\_trust\_factors ← url_trust_scorer.py:406
  - detect\_domain\_mimicry ← url_trust_scorer.py:480
  - compute\_trust\_score ← promotion_helpers.py:46

### utils/user\_prompt.py {#webapp-parser-utils-user-prompt-py}

- Definitions:
  - function: `safe\_lower` (line 33)
  - function: `safe\_strip` (line 39)
  - class: `PromptCancelled` (line 50)
  - class: `PromptSession` (line 54)
  - class: `UserPrompt` (line 135)
- Imports:
  - **Standard Library** (17):
    - `import datetime as datetime` (line 3)
    - `import inspect as inspect` (line 4)
    - `import os as os` (line 5)
    - `import re as re` (line 6)
    - `import threading as threading` (line 7)
    - `import time as time` (line 16)
    - `import traceback as traceback` (line 17)
    - `from contextlib import contextmanager` (line 18)
    - `from datetime import timezone` (line 19)
    - `from typing import Any` (line 20)
    - `from typing import Callable` (line 20)
    - `from typing import ContextManager` (line 20)
    - `from typing import Dict` (line 20)
    - `from typing import Generator` (line 20)
    - `from typing import List` (line 20)
    - `from typing import Optional` (line 20)
    - `from typing import Union` (line 20)
  - **Third-party** (1):
    - `import orjson as orjson` (line 22)
  - **Local/Project** (7):
    - `from __future__ import annotations` (line 1)
    - `from rich.progress import BarColumn` (line 23)
    - `from rich.progress import Progress` (line 23)
    - `from rich.progress import SpinnerColumn` (line 23)
    - `from rich.progress import TextColumn` (line 23)
    - `from rich.progress import TimeElapsedColumn` (line 23)
    - `from rich.progress import TimeRemainingColumn` (line 23)
- Task markers:
  - L318 **WARNING**: ("\[UserPrompt\] Webapp mode active but no
    socketio_emit_func set!")
  - L355 **WARNING**: ("\[CLI Prompt\] EOFError encountered.")
  - L376 **WARNING**: ("\[Webapp Prompt\] socketio_emit_func not set.")
  - L434 **WARNING**: ": 30,
  - L515 **WARNING**: ("\n\[Prompt\] Timed out.")
  - L566 **WARNING**: ("\n\[Prompt\] No input available (EOF). Exiting prompt.")
  - L600 **WARNING**: ("Invalid input. Please try again.")
  - L602 **WARNING**: ("\[Prompt\] Too many invalid attempts.")
  - L667 **WARNING**: ("\[Prompt Queue\] Invalid queued yes/no response; falling
    back to interactive prompt.")
  - L682 **WARNING**: ("\n\[Prompt\] Timed out.")
  - L889 **WARNING**: ("\[yellow\]\[FEEDBACK\] Skipped manual
    correction.\[/yellow\]")
  - L921 **WARNING**: ("\[yellow\]Button confirmation cancelled by
    user.\[/yellow\]")
- Outgoing cross-module calls (sample):
  - val.lower (line 35)
  - val.strip (line 41)
  - threading.Event (line 62)
  - traceback.print\_tb (line 85)
  - self.is\_expired (line 94)
  - self.is\_expired (line 131)
  - re.compile (line 143)
  - re.compile (line 144)
  - threading.Lock (line 159)
  - threading.Lock (line 161)
  - self.\_start\_cleanup\_thread (line 163)
  - time.time (line 171)
  - ps.is\_expired (line 179)
  - time.sleep (line 182)
  - threading.Thread (line 183)
  - t.start (line 184)
  - entry.get (line 212)
  - entry.get (line 220)
  - entry.get (line 222)
  - entry.get (line 230)
  - entry.get (line 231)
  - session.is\_expired (line 256)
  - session.is\_resolved (line 256)
  - time.time (line 265)
  - time.time (line 269)
  - sess.is\_expired (line 273)
  - logger.info (line 289)
  - logger.info (line 310)
  - logger.error (line 312)
  - logger.warning (line 318)
  - logger.error (line 328)
  - traceback.print\_tb (line 329)
  - logger.info (line 350)
  - logger.warning (line 355)
  - self.socketio\_emit\_func (line 374)
  - orjson.dumps (line 374)
  - logger.warning (line 376)
  - logger.info (line 381)
  - logger.info (line 382)
  - title.center (line 382)
  - logger.info (line 383)
  - logger.info (line 410)
  - orjson.dumps (line 410)
  - self.socketio\_emit\_func (line 413)
  - orjson.dumps (line 413)
  - console.print (line 417)
  - console.print (line 419)
  - console.print (line 421)
  - level.upper (line 439)
  - logger\_level.upper (line 440)
- Inbound references:
  - PromptCancelled ← user_prompt.py:585
  - PromptCancelled ← user_prompt.py:607
  - PromptCancelled ← user_prompt.py:656
  - PromptCancelled ← user_prompt.py:694
  - PromptSession ← user_prompt.py:243
  - PromptSession ← user_prompt.py:257
  - UserPrompt ← logger_singleton.py:29

### utils/verification\_framework.py {#webapp-parser-utils-verification-framework-py}

> Dual-Truth Verification Framework for Smart Elections Parser

- Definitions:
  - class: `VerificationStatus` (line 41)
  - class: `VerificationConfidence` (line 49)
  - class: `AnomalyType` (line 57)
  - class: `VerificationLineageEntry` (line 69)
  - class: `VerificationLog` (line 162)
  - function: `classify\_anomaly` (line 295)
- Imports:
  - **Standard Library** (11):
    - `import hashlib as hashlib` (line 31)
    - `import json as json` (line 32)
    - `from datetime import datetime` (line 33)
    - `from datetime import timezone` (line 33)
    - `from enum import Enum` (line 34)
    - `from pathlib import Path` (line 35)
    - `from typing import Any` (line 36)
    - `from typing import Dict` (line 36)
    - `from typing import List` (line 36)
    - `from typing import Optional` (line 36)
    - `from typing import Tuple` (line 36)
  - **Local/Project** (2):
    - `from __future__ import annotations` (line 29)
    - `from logger_singleton import logger` (line 38)
- Outgoing cross-module calls (sample):
  - datetime.datetime.now (line 110)
  - self.\_compute\_hash (line 111)
  - json.dumps (line 127)
  - hashlib.sha256 (line 128)
  - blob.encode (line 128)
  - data.get (line 150)
  - data.get (line 151)
  - data.get (line 152)
  - data.get (line 153)
  - data.get (line 154)
  - data.get (line 155)
  - data.get (line 156)
  - data.get (line 157)
  - data.get (line 158)
  - pathlib.Path (line 175)
  - f.write (line 190)
  - orjson.dumps (line 190)
  - entry.to\_dict (line 190)
  - f.flush (line 191)
  - logger\_singleton.logger.info (line 192)
  - logger\_singleton.logger.error (line 203)
  - line.strip (line 228)
  - orjson.loads (line 231)
  - entries.append (line 232)
  - VerificationLineageEntry.from\_dict (line 232)
  - logger\_singleton.logger.error (line 238)
  - self.read\_all (line 256)
  - self.read\_all (line 267)
  - anom.get (line 289)
  - str\_dl2.lower (line 317)
  - str\_dl1.lower (line 317)
- Inbound references:
  - VerificationStatus ← verification_framework.py:154
  - VerificationConfidence ← verification_framework.py:155
  - VerificationLineageEntry ← verification_framework.py:149

### utils/xlsx\_exporter.py {#webapp-parser-utils-xlsx-exporter-py}

- Definitions:
  - function: `\_auto\_width` (line 13)
  - function: `\_apply\_styles` (line 26)
  - function: `export\_candidate\_group\_pivot\_xlsx` (line 50)
- Imports:
  - **Standard Library** (5):
    - `import re as re` (line 3)
    - `from typing import Any` (line 4)
    - `from typing import Dict` (line 4)
    - `from typing import List` (line 4)
    - `from typing import Optional` (line 4)
  - **Local/Project** (9):
    - `from __future__ import annotations` (line 1)
    - `from openpyxl import Workbook` (line 6)
    - `from openpyxl.formatting.rule import ColorScaleRule` (line 7)
    - `from openpyxl.styles import Alignment` (line 8)
    - `from openpyxl.styles import Border` (line 8)
    - `from openpyxl.styles import Font` (line 8)
    - `from openpyxl.styles import PatternFill` (line 8)
    - `from openpyxl.styles import Side` (line 8)
    - `from openpyxl.utils import get_column_letter` (line 9)
- Outgoing cross-module calls (sample):
  - openpyxl.utils.get\_column\_letter (line 16)
  - openpyxl.styles.Font (line 27)
  - openpyxl.styles.Alignment (line 28)
  - openpyxl.styles.Side (line 29)
  - openpyxl.styles.Border (line 30)
  - openpyxl.styles.PatternFill (line 31)
  - openpyxl.styles.PatternFill (line 32)
  - ws.cell (line 36)
  - ws.cell (line 45)
  - openpyxl.styles.Alignment (line 48)
  - openpyxl.Workbook (line 65)
  - ws.append (line 78)
  - ws.append (line 79)
  - ws.merge\_cells (line 90)
  - ws.merge\_cells (line 101)
  - ws.merge\_cells (line 107)
  - ws.append (line 113)
  - re.compile (line 118)
  - re.compile (line 119)
  - ws.append (line 121)
  - r.get (line 121)
  - percent\_col\_re.search (line 126)
  - numeric\_candidate\_re.search (line 127)
  - ws.cell (line 133)
  - re.fullmatch (line 137)
  - val.strip (line 137)
  - ws.cell (line 143)
  - val.strip (line 150)
  - sv.replace (line 151)
  - val.replace (line 163)
  - re.fullmatch (line 164)
  - re.fullmatch (line 170)
  - percent\_col\_re.search (line 187)
  - percent\_fill\_cols.append (line 188)
  - numeric\_candidate\_re.search (line 189)
  - count\_fill\_cols.append (line 190)
  - openpyxl.utils.get\_column\_letter (line 193)
  - openpyxl.utils.get\_column\_letter (line 193)
  - openpyxl.formatting.rule.ColorScaleRule (line 196)
  - openpyxl.utils.get\_column\_letter (line 204)
  - openpyxl.utils.get\_column\_letter (line 204)
  - openpyxl.formatting.rule.ColorScaleRule (line 207)
  - ws.cell (line 218)
  - wb.create\_sheet (line 221)
  - meta\_ws.append (line 222)
  - context.get (line 224)
  - context.get (line 225)
  - context.get (line 226)
  - context.get (line 227)
  - context.get (line 228)
- Inbound references:
  - \_auto\_width ← xlsx_exporter.py:215
  - \_apply\_styles ← xlsx_exporter.py:214
  - export\_candidate\_group\_pivot\_xlsx ← output_utils.py:672

### verification/local\_dl\_sync.py {#webapp-parser-verification-local-dl-sync-py}

> Local DL1/DL2 File System Sync Implementation

- Definitions:
  - class: `LocalStorageSync` (line 22)
- Imports:
  - **Standard Library** (10):
    - `import hashlib as hashlib` (line 11)
    - `import os as os` (line 12)
    - `import shutil as shutil` (line 13)
    - `import threading as threading` (line 14)
    - `from datetime import datetime` (line 15)
    - `from datetime import timezone` (line 15)
    - `from pathlib import Path` (line 16)
    - `from typing import Any` (line 17)
    - `from typing import Dict` (line 17)
    - `from typing import List` (line 17)
  - **Third-party** (1):
    - `import orjson as orjson` (line 19)
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 41)
  - threading.RLock (line 46)
  - pathlib.Path (line 67)
  - hashlib.new (line 68)
  - f.read (line 71)
  - hasher.update (line 72)
  - hasher.hexdigest (line 73)
  - datetime.datetime.now (line 82)
  - orjson.loads (line 90)
  - f.read (line 90)
  - datetime.datetime.now (line 94)
  - f.write (line 105)
  - orjson.dumps (line 105)
  - f.flush (line 106)
  - os.fsync (line 107)
  - f.fileno (line 107)
  - tmp\_path.replace (line 108)
  - tmp\_path.unlink (line 110)
  - self.\_load\_metadata (line 118)
  - self.compute\_file\_hash (line 120)
  - file\_path.stat (line 125)
  - datetime.datetime.fromtimestamp (line 127)
  - file\_path.stat (line 128)
  - metadata.get (line 130)
  - files.append (line 132)
  - self.compute\_file\_hash (line 147)
  - file\_path.stat (line 152)
  - datetime.datetime.fromtimestamp (line 154)
  - file\_path.stat (line 155)
  - files.append (line 158)
  - pathlib.Path (line 184)
  - source\_path.exists (line 185)
  - self.compute\_file\_hash (line 190)
  - datetime.datetime.now (line 191)
  - dest\_path.exists (line 194)
  - shutil.copy2 (line 199)
  - self.compute\_file\_hash (line 202)
  - self.\_load\_metadata (line 205)
  - datetime.datetime.now (line 209)
  - sync\_metadata.setdefault (line 214)
  - self.\_save\_metadata (line 220)
  - dl2\_path.exists (line 242)
  - dl1\_path.exists (line 246)
  - shutil.copy2 (line 251)
  - datetime.datetime.now (line 258)
  - self.compute\_file\_hash (line 261)
  - self.compute\_file\_hash (line 262)
  - self.\_load\_metadata (line 266)
  - self.\_save\_metadata (line 276)
  - self.\_append\_promotion\_log (line 279)
- Inbound references:
  - LocalStorageSync ← verification_endpoints.py:572
  - LocalStorageSync ← verification_endpoints.py:624
  - LocalStorageSync ← verification_endpoints.py:678
  - LocalStorageSync ← verification_endpoints.py:740
  - LocalStorageSync ← verification_endpoints.py:796

### verification\_endpoints.py {#webapp-parser-verification-endpoints-py}

> Verification Framework API Endpoints

- Definitions:
  - function: `\_require\_verification\_enabled` (line 50)
  - function: `\_get\_verifier\_principal` (line 60)
  - function: `\_get\_verifier\_identity` (line 70)
  - function: `\_normalize\_required\_tier` (line 81)
  - function: `\_require\_verifier\_tier` (line 94)
  - function: `\_require\_principal` (line 134)
  - function: `get\_system\_mission` (line 145)
  - function: `get\_verification\_stats` (line 163)
  - function: `get\_verification\_entries` (line 200)
  - function: `submit\_verification` (line 268)
  - function: `compare\_dl1\_dl2` (line 381)
  - function: `export\_dl1\_verified` (line 467)
  - function: `sync\_status` (line 548)
  - function: `sync\_list\_dl2` (line 593)
  - function: `sync\_list\_dl1` (line 648)
  - function: `sync\_stage\_dl2` (line 702)
  - function: `sync\_promote` (line 762)
- Imports:
  - **Standard Library** (5):
    - `import os as os` (line 12)
    - `from datetime import datetime` (line 13)
    - `from datetime import timezone` (line 13)
    - `from functools import wraps` (line 14)
    - `from typing import Optional` (line 15)
  - **Third-party** (18):
    - `from flask import Blueprint` (line 17)
    - `from flask import Response` (line 17)
    - `from flask import jsonify` (line 17)
    - `from flask import request` (line 17)
    - `from webapp.parser.config import ENABLE_VERIFICATION_FRAMEWORK` (line 18)
    - `from webapp.parser.config import SYSTEM_AUTHOR` (line 18)
    - `from webapp.parser.config import SYSTEM_MISSION` (line 18)
    - `from webapp.parser.config import VERIFICATION_LOG_FILE` (line 18)
    - `from webapp.parser.utils.logger_singleton import logger` (line 24)
    - `from webapp.parser.utils.privilege_tiers import PrivilegeTier` (line 25)
    - `from webapp.parser.utils.privilege_tiers import get_principal_tier` (line
      25)
    - `from webapp.parser.utils.shared_logic import safe_get` (line 26)
    - `from webapp.parser.utils.shared_logic import safe_strip` (line 26)
    - `from webapp.parser.utils.verification_framework import
      VerificationConfidence` (line 27)
    - `from webapp.parser.utils.verification_framework import
      VerificationLineageEntry` (line 27)
    - `from webapp.parser.utils.verification_framework import VerificationLog`
      (line 27)
    - `from webapp.parser.utils.verification_framework import
      VerificationStatus` (line 27)
    - `from webapp.parser.utils.verification_framework import classify_anomaly`
      (line 27)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 10)
- Task markers:
  - L110 **WARNING**: ({
  - L111 **WARNING**: ",
  - L810 **WARNING**: ({
  - L811 **WARNING**: ",
  - L817 **WARNING**: ({
  - L818 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - flask.Blueprint (line 43)
  - flask.jsonify (line 55)
  - functools.wraps (line 52)
  - tier\_map.get (line 91)
  - flask.jsonify (line 105)
  - webapp.parser.utils.privilege\_tiers.get\_principal\_tier (line 108)
  - webapp.parser.utils.logger\_singleton.logger.warning (line 110)
  - flask.jsonify (line 121)
  - functools.wraps (line 101)
  - tier.lower (line 136)
  - flask.jsonify (line 151)
  - verification\_bp.route (line 143)
  - webapp.parser.utils.verification\_framework.VerificationLog (line 171)
  - vlog.get\_stats (line 172)
  - datetime.datetime.now (line 173)
  - webapp.parser.utils.logger\_singleton.logger.info (line 176)
  - stats.get (line 182)
  - flask.jsonify (line 185)
  - webapp.parser.utils.logger\_singleton.logger.error (line 187)
  - flask.jsonify (line 194)
  - verification\_bp.route (line 160)
  - webapp.parser.utils.verification\_framework.VerificationLog (line 222)
  - vlog.read\_all (line 223)
  - filtered.append (line 231)
  - entry.to\_dict (line 231)
  - webapp.parser.utils.logger\_singleton.logger.info (line 235)
  - flask.jsonify (line 244)
  - webapp.parser.utils.logger\_singleton.logger.error (line 255)
  - flask.jsonify (line 262)
  - verification\_bp.route (line 197)
  - flask.jsonify (line 290)
  - flask.request.get\_json (line 293)
  - webapp.parser.utils.logger\_singleton.logger.error (line 295)
  - flask.jsonify (line 302)
  - webapp.parser.utils.shared\_logic.safe\_strip (line 305)
  - webapp.parser.utils.shared\_logic.safe\_get (line 305)
  - webapp.parser.utils.shared\_logic.safe\_get (line 306)
  - webapp.parser.utils.shared\_logic.safe\_strip (line 307)
  - webapp.parser.utils.shared\_logic.safe\_get (line 307)
  - webapp.parser.utils.shared\_logic.safe\_strip (line 308)
  - webapp.parser.utils.shared\_logic.safe\_get (line 308)
  - webapp.parser.utils.shared\_logic.safe\_strip (line 309)
  - webapp.parser.utils.shared\_logic.safe\_get (line 309)
  - webapp.parser.utils.shared\_logic.safe\_strip (line 310)
  - webapp.parser.utils.shared\_logic.safe\_get (line 310)
  - webapp.parser.utils.shared\_logic.safe\_get (line 311)
  - webapp.parser.utils.shared\_logic.safe\_get (line 312)
  - flask.jsonify (line 315)
  - webapp.parser.utils.verification\_framework.VerificationStatus (line 318)
  - flask.jsonify (line 320)
- Inbound references:
  - \_get\_verifier\_principal ← verification_endpoints.py:169
  - \_get\_verifier\_principal ← verification_endpoints.py:211
  - \_get\_verifier\_principal ← verification_endpoints.py:288
  - \_get\_verifier\_principal ← verification_endpoints.py:394
  - \_get\_verifier\_principal ← verification_endpoints.py:479
  - \_get\_verifier\_identity ← verification_endpoints.py:103
  - \_normalize\_required\_tier ← verification_endpoints.py:107
  - \_normalize\_required\_tier ← qa_endpoints.py:110
  - \_require\_verifier\_tier ← verification_endpoints.py:136
  - \_require\_verifier\_tier ← verification_endpoints.py:162
  - \_require\_verifier\_tier ← verification_endpoints.py:199
  - \_require\_verifier\_tier ← verification_endpoints.py:267
  - \_require\_verifier\_tier ← verification_endpoints.py:380
  - \_require\_verifier\_tier ← verification_endpoints.py:466
  - \_require\_principal ← verification_endpoints.py:547
  - \_require\_principal ← verification_endpoints.py:592
  - \_require\_principal ← verification_endpoints.py:647
  - \_require\_principal ← verification_endpoints.py:701
  - \_require\_principal ← verification_endpoints.py:761

### web\_pipeline.py {#webapp-parser-web-pipeline-py}

- Definitions:
  - class: `CancellationManager` (line 22)
  - function: `heartbeat` (line 97)
  - function: `save\_pipeline\_report` (line 111)
  - function: `\_collect\_output\_artifacts` (line 123)
  - function: `process\_urls\_for\_web` (line 211)
  - function: `cancel\_processing` (line 910)
- Imports:
  - **Standard Library** (5):
    - `import os as os` (line 1)
    - `import threading as threading` (line 2)
    - `import time as time` (line 3)
    - `import traceback as traceback` (line 4)
    - `from pathlib import Path` (line 5)
  - **Third-party** (1):
    - `import orjson as orjson` (line 7)
  - **Local/Project** (12):
    - `from config import PIPELINE_HEARTBEAT_INTERVAL` (line 9)
    - `from config import PIPELINE_MAX_WORKERS` (line 9)
    - `from config import PROCESSED_URLS_FILE` (line 9)
    - `from config import SLOW_NLP_AUDIT_MIN_HITS` (line 9)
    - `from config import SLOW_NLP_AUDIT_THRESHOLD` (line 9)
    - `from config import URL_LIST_FILE` (line 9)
    - `from html_election_parser import main` (line 17)
    - `from utils.logger_singleton import logger` (line 18)
    - `from utils.logger_singleton import prompt` (line 18)
    - `from utils.shared_logic import safe_clear` (line 19)
    - `from utils.shared_logic import safe_is_set` (line 19)
    - `from utils.shared_logic import safe_set` (line 19)
- Task markers:
  - L53 **WARNING**: ({
  - L54 **WARNING**: ",
  - L70 **WARNING**: ({
  - L71 **WARNING**: ",
  - L87 **WARNING**: ({
  - L88 **WARNING**: ",
  - L279 **WARNING**: ({
  - L280 **WARNING**: ",
  - L440 **WARNING**: ({
  - L441 **WARNING**: ",
  - L451 **WARNING**: ({
  - L452 **WARNING**: ",
  - L514 **WARNING**: ({
  - L515 **WARNING**: ",
  - L525 **WARNING**: ({
  - L526 **WARNING**: ",
  - L731 **WARNING**: ({
  - L732 **WARNING**: ",
  - L749 **WARNING**: ({
  - L750 **WARNING**: ",
  - L812 **WARNING**: ({
  - L813 **WARNING**: ",
  - L880 **WARNING**: ({
  - L881 **WARNING**: ",
  - L900 **WARNING**: ({
  - L901 **WARNING**: ",
- Outgoing cross-module calls (sample):
  - threading.Lock (line 29)
  - threading.Event (line 35)
  - utils.shared\_logic.safe\_set (line 45)
  - utils.logger\_singleton.logger.info (line 46)
  - utils.logger\_singleton.logger.warning (line 53)
  - ev.is\_set (line 64)
  - utils.shared\_logic.safe\_clear (line 65)
  - utils.logger\_singleton.logger.warning (line 70)
  - utils.logger\_singleton.logger.warning (line 87)
  - time.sleep (line 99)
  - time.time (line 106)
  - utils.shared\_logic.safe\_is\_set (line 108)
  - os.makedirs (line 113)
  - f.write (line 116)
  - orjson.dumps (line 116)
  - pathlib.Path (line 124)
  - path\_value.strip (line 134)
  - raw.replace (line 137)
  - normalized.startswith (line 138)
  - abs\_path.startswith (line 145)
  - normalized.startswith (line 149)
  - rel.lower (line 157)
  - low.endswith (line 158)
  - csv\_paths.add (line 159)
  - low.endswith (line 160)
  - low.endswith (line 160)
  - xlsx\_paths.add (line 161)
  - low.endswith (line 162)
  - low.endswith (line 162)
  - metadata\_paths.add (line 163)
  - low.endswith (line 164)
  - other\_paths.add (line 165)
  - entry.get (line 171)
  - entry.get (line 171)
  - entry.get (line 173)
  - metadata.get (line 174)
  - entry.get (line 175)
  - metadata.get (line 175)
  - output\_dirs.add (line 177)
  - rel\_dir.rstrip (line 177)
  - abs\_dir.is\_dir (line 182)
  - abs\_dir.iterdir (line 184)
  - artifact.is\_file (line 185)
  - cancellation\_manager.reset (line 235)
  - utils.logger\_singleton.logger.set\_mode (line 237)
  - utils.logger\_singleton.logger.set\_format (line 238)
  - utils.logger\_singleton.prompt.set\_mode (line 240)
  - utils.logger\_singleton.prompt.set\_socketio\_emit\_func (line 241)
  - utils.logger\_singleton.logger.info (line 247)
  - kwargs.setdefault (line 254)
- Inbound references:
  - CancellationManager ← web_pipeline.py:95
  - save\_pipeline\_report ← web_pipeline.py:791
  - save\_pipeline\_report ← web_pipeline.py:864
  - \_collect\_output\_artifacts ← web_pipeline.py:689
  - \_collect\_output\_artifacts ← web_pipeline.py:861

### webapp/tests/\_\_init\_\_.py {#webapp-tests-init-py}

> Unit tests for the Smart Elections Parser

### webapp/tests/conftest.py {#webapp-tests-conftest-py}

> Pytest configuration and shared fixtures for all tests.

- Definitions:
  - function: `test\_db\_engine` (line 47)
  - function: `db\_session` (line 56)
  - function: `temp\_output\_dir` (line 72)
  - function: `sample\_html\_content` (line 79)
  - function: `sample\_csv\_data` (line 100)
  - function: `sample\_contest\_data` (line 109)
  - function: `mock\_coordinator` (line 121)
  - function: `mock\_page` (line 130)
- Imports:
  - **Standard Library** (8):
    - `import os as os` (line 2)
    - `import sys as sys` (line 3)
    - `import tempfile as tempfile` (line 4)
    - `import warnings as warnings` (line 5)
    - `from pathlib import Path` (line 6)
    - `from typing import Generator` (line 7)
    - `from unittest.mock import Mock` (line 8)
    - `from unittest.mock import patch` (line 8)
  - **Third-party** (3):
    - `import pytest as pytest` (line 12)
    - `from sqlalchemy import create_engine` (line 13)
    - `from sqlalchemy.orm import Session` (line 14)
  - **Local/Project** (2):
    - `import importlib.machinery as importlib` (line 9)
    - `import types as types` (line 10)
- Task markers:
  - L20 **WARNING**: suppression
- Outgoing cross-module calls (sample):
  - pathlib.Path (line 17)
  - warnings.filterwarnings (line 26)
  - warnings.filterwarnings (line 27)
  - warnings.filterwarnings (line 28)
  - warnings.filterwarnings (line 29)
  - warnings.filterwarnings (line 30)
  - types.SimpleNamespace (line 36)
  - unittest.mock.patch (line 41)
  - unittest.mock.Mock (line 41)
  - sqlalchemy.create\_engine (line 49)
  - engine.dispose (line 52)
  - pytest.fixture (line 46)
  - test\_db\_engine.connect (line 60)
  - connection.begin (line 61)
  - session.close (line 66)
  - transaction.rollback (line 67)
  - connection.close (line 68)
  - tempfile.TemporaryDirectory (line 74)
  - pathlib.Path (line 75)
  - unittest.mock.Mock (line 123)
  - unittest.mock.Mock (line 124)
  - unittest.mock.Mock (line 125)
  - unittest.mock.Mock (line 132)
  - unittest.mock.Mock (line 134)
  - unittest.mock.Mock (line 135)

### webapp/tests/test\_ballot\_lens\_pathways.py {#webapp-tests-test-ballot-lens-pathways-py}

> Multi-Pathway Integration Tests for Ballot Lens

- Definitions:
  - class: `ExecutionPathway` (line 61)
  - class: `DataValidationResult` (line 68)
  - class: `CSVValidation` (line 78)
  - class: `PathwayExecutionResult` (line 92)
  - function: `webapp\_client` (line 112)
  - function: `temp\_output\_dir` (line 120)
  - function: `sample\_election\_urls` (line 127)
  - function: `sample\_html\_fixture` (line 146)
  - function: `validate\_csv` (line 171)
  - function: `read\_csv\_content` (line 275)
  - function: `hash\_csv\_content` (line 285)
  - function: `execute\_via\_cli` (line 294)
  - function: `execute\_via\_direct\_api` (line 373)
  - function: `execute\_via\_webapp\_api` (line 470)
  - class: `TestPathwayConsistency` (line 558)
  - class: `TestCSVValidation` (line 603)
  - class: `TestEdgeCases` (line 639)
  - class: `TestDataComparison` (line 671)
- Imports:
  - **Standard Library** (19):
    - `import csv as csv` (line 33)
    - `import json as json` (line 34)
    - `import tempfile as tempfile` (line 35)
    - `from pathlib import Path` (line 36)
    - `from typing import Dict` (line 37)
    - `from typing import List` (line 37)
    - `from typing import Optional` (line 37)
    - `from typing import Tuple` (line 37)
    - `from typing import Generator` (line 37)
    - `from typing import Any` (line 37)
    - `from dataclasses import dataclass` (line 38)
    - `from enum import Enum` (line 39)
    - `from unittest.mock import patch` (line 40)
    - `from unittest.mock import Mock` (line 40)
    - `from unittest.mock import MagicMock` (line 40)
    - `import subprocess as subprocess` (line 41)
    - `import sys as sys` (line 42)
    - `import time as time` (line 43)
    - `import hashlib as hashlib` (line 44)
  - **Third-party** (1):
    - `import pytest as pytest` (line 46)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 31)
- Task markers:
  - L130 **NOTE**: These are representative URLs. Real testing should use:
  - L312 **NOTE**: Adjust based on actual CLI interface of
    html_election_parser.py
  - L392 **NOTE**: This is a simplified test version
  - L489 **NOTE**: Adjust based on actual Flask route
  - L565 **NOTE**: For real tests, either:
- Outgoing cross-module calls (sample):
  - pytest.skip (line 54)
  - app.test\_client (line 115)
  - tempfile.TemporaryDirectory (line 122)
  - pathlib.Path (line 123)
  - csv\_path.exists (line 186)
  - csv\_path.stat (line 195)
  - csv.DictReader (line 206)
  - errors.append (line 223)
  - errors.append (line 243)
  - csv\_path.exists (line 278)
  - csv\_path.read\_text (line 279)
  - hashlib.sha256 (line 287)
  - csv\_content.encode (line 287)
  - time.time (line 304)
  - subprocess.run (line 320)
  - output\_dir.glob (line 331)
  - time.time (line 352)
  - time.time (line 384)
  - time.time (line 394)
  - result.get (line 407)
  - result.get (line 408)
  - csv.writer (line 422)
  - writer.writerow (line 423)
  - writer.writerows (line 424)
  - time.time (line 449)
  - time.time (line 481)
  - client.post (line 490)
  - response.get\_data (line 497)
  - response.get\_json (line 507)
  - result\_data.get (line 508)
  - result\_data.get (line 508)
  - csv\_path.write\_text (line 512)
  - time.time (line 541)
  - csv\_path.touch (line 609)
  - csv\_path.write\_text (line 618)
  - csv\_path.write\_text (line 632)
  - pytest.main (line 715)
- Inbound references:
  - CSVValidation ← test_ballot_lens_pathways.py:187
  - CSVValidation ← test_ballot_lens_pathways.py:197
  - CSVValidation ← test_ballot_lens_pathways.py:209
  - CSVValidation ← test_ballot_lens_pathways.py:230
  - CSVValidation ← test_ballot_lens_pathways.py:251
  - CSVValidation ← test_ballot_lens_pathways.py:260
  - CSVValidation ← test_ballot_lens_pathways.py:267
  - CSVValidation ← test_ballot_lens_pathways.py:338
  - CSVValidation ← test_ballot_lens_pathways.py:355
  - CSVValidation ← test_ballot_lens_pathways.py:412
  - CSVValidation ← test_ballot_lens_pathways.py:430
  - CSVValidation ← test_ballot_lens_pathways.py:439
  - CSVValidation ← test_ballot_lens_pathways.py:452
  - CSVValidation ← test_ballot_lens_pathways.py:498
  - CSVValidation ← test_ballot_lens_pathways.py:517
  - CSVValidation ← test_ballot_lens_pathways.py:525
  - CSVValidation ← test_ballot_lens_pathways.py:534
  - PathwayExecutionResult ← test_ballot_lens_pathways.py:362
  - PathwayExecutionResult ← test_ballot_lens_pathways.py:459
  - PathwayExecutionResult ← test_ballot_lens_pathways.py:543
  - validate\_csv ← test_ballot_lens_pathways.py:335
  - validate\_csv ← test_ballot_lens_pathways.py:427
  - validate\_csv ← test_ballot_lens_pathways.py:514
  - validate\_csv ← test_ballot_lens_pathways.py:611
  - validate\_csv ← test_ballot_lens_pathways.py:624
  - validate\_csv ← test_ballot_lens_pathways.py:634
  - read\_csv\_content ← test_ballot_lens_pathways.py:334
  - read\_csv\_content ← test_ballot_lens_pathways.py:426
  - hash\_csv\_content ← test_ballot_lens_pathways.py:693
  - hash\_csv\_content ← test_ballot_lens_pathways.py:694
  - execute\_via\_direct\_api ← test_ballot_lens_pathways.py:571
  - execute\_via\_direct\_api ← test_ballot_lens_pathways.py:590
  - execute\_via\_direct\_api ← test_ballot_lens_pathways.py:646
  - execute\_via\_direct\_api ← test_ballot_lens_pathways.py:658
  - execute\_via\_direct\_api ← test_ballot_lens_pathways.py:679
  - execute\_via\_direct\_api ← test_ballot_lens_pathways.py:685
  - execute\_via\_direct\_api ← test_ballot_lens_pathways.py:701

### webapp/tests/test\_batch\_processor.py {#webapp-tests-test-batch-processor-py}

> Tests for handlers/batch_handler.py

- Definitions:
  - class: `TestBatchProcessor` (line 7)
- Imports:
  - **Standard Library** (1):
    - `from unittest.mock import Mock` (line 3)
  - **Third-party** (2):
    - `import pytest as pytest` (line 2)
    - `from webapp.parser.handlers.batch_handler import BatchProcessor` (line 4)
- Outgoing cross-module calls (sample):
  - webapp.parser.handlers.batch\_handler.BatchProcessor (line 12)
  - unittest.mock.Mock (line 14)
  - unittest.mock.Mock (line 22)
  - unittest.mock.Mock (line 23)
  - unittest.mock.Mock (line 24)
  - unittest.mock.Mock (line 25)

### webapp/tests/test\_context\_coordinator.py {#webapp-tests-test-context-coordinator-py}

> Tests for Context_Integration/context_coordinator.py

- Definitions:
  - class: `TestContextCoordinator` (line 10)
  - class: `TestSemanticScore` (line 30)
  - class: `TestStateCountyDetection` (line 49)
- Imports:
  - **Third-party** (4):
    - `import pytest as pytest` (line 2)
    - `from webapp.parser.Context_Integration.context_coordinator import
      ContextCoordinator` (line 3)
    - `from webapp.parser.Context_Integration.context_coordinator import
      get_semantic_score` (line 3)
    - `from webapp.parser.Context_Integration.context_coordinator import
      dynamic_state_county_detection` (line 3)
- Outgoing cross-module calls (sample):
  - webapp.parser.Context\_Integration.context\_coordinator.ContextCoordinator
    (line 15)
  - webapp.parser.Context\_Integration.context\_coordinator.ContextCoordinator
    (line 21)
  - coordinator.extract\_entities (line 23)
  - webapp.parser.Context\_Integration.context\_coordinator.get\_semantic\_score
    (line 35)
  - webapp.parser.Context\_Integration.context\_coordinator.get\_semantic\_score
    (line 40)
  - webapp.parser.Context\_Integration.context\_coordinator.get\_semantic\_score
    (line 45)
  - webapp.parser.Context\_Integration.context\_coordinator.dynamic\_state\_county\_detection
    (line 55)

### webapp/tests/test\_csv\_handler.py {#webapp-tests-test-csv-handler-py}

> Tests for webapp/parser/handlers/formats/csv_handler.py

- Definitions:
  - class: `TestCSVHandler` (line 8)
- Imports:
  - **Standard Library** (3):
    - `import csv as csv` (line 2)
    - `import tempfile as tempfile` (line 4)
    - `from pathlib import Path` (line 5)
  - **Third-party** (1):
    - `import pytest as pytest` (line 3)
- Outgoing cross-module calls (sample):
  - csv.DictWriter (line 18)
  - writer.writeheader (line 19)
  - writer.writerow (line 20)
  - writer.writerow (line 21)
  - csv.DictWriter (line 41)
  - writer.writeheader (line 42)
  - writer.writerow (line 43)
  - writer.writerow (line 44)

### webapp/tests/test\_data\_comparator.py {#webapp-tests-test-data-comparator-py}

- Definitions:
  - function: `test\_data\_comparator\_exact\_near\_missing\_extra` (line 6)
  - function: `test\_data\_comparator\_detects\_mismatch\_over\_tolerance` (line
    32)
  - function: `test\_regression\_report\_contract\_gate\_fails` (line 45)
- Imports:
  - **Third-party** (1):
    - `from webapp.parser.utils.data_comparator import DataComparator` (line 3)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.data\_comparator.DataComparator (line 21)
  - comparator.compare\_datasets (line 22)
  - webapp.parser.utils.data\_comparator.DataComparator (line 36)
  - comparator.compare\_datasets (line 37)
  - webapp.parser.utils.data\_comparator.DataComparator (line 49)
  - comparator.compare\_datasets (line 50)
  - comparator.build\_regression\_report (line 51)

### webapp/tests/test\_detect.py {#webapp-tests-test-detect-py}

> Tests for webapp/parser/utils/detect.py

- Definitions:
  - class: `TestNormalization` (line 15)
  - class: `TestLocationDetection` (line 31)
  - class: `TestCandidateDetection` (line 49)
  - class: `TestHarmonization` (line 67)
  - class: `TestNumericParsing` (line 87)
  - class: `TestHeaderDeduplication` (line 109)
- Imports:
  - **Third-party** (9):
    - `import pytest as pytest` (line 2)
    - `from webapp.parser.utils.detect import normalize_text` (line 3)
    - `from webapp.parser.utils.detect import normalize_header` (line 3)
    - `from webapp.parser.utils.detect import is_location_header` (line 3)
    - `from webapp.parser.utils.detect import dynamic_detect_location_header`
      (line 3)
    - `from webapp.parser.utils.detect import detect_candidate_column` (line 3)
    - `from webapp.parser.utils.detect import harmonize_headers_and_data` (line
      3)
    - `from webapp.parser.utils.detect import parse_numeric` (line 3)
    - `from webapp.parser.utils.detect import dedupe_headers_with_suffix` (line
      3)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.detect.normalize\_text (line 20)
  - webapp.parser.utils.detect.normalize\_text (line 21)
  - webapp.parser.utils.detect.normalize\_text (line 22)
  - webapp.parser.utils.detect.normalize\_header (line 26)
  - webapp.parser.utils.detect.normalize\_header (line 27)
  - webapp.parser.utils.detect.normalize\_header (line 28)
  - webapp.parser.utils.detect.is\_location\_header (line 36)
  - webapp.parser.utils.detect.is\_location\_header (line 37)
  - webapp.parser.utils.detect.is\_location\_header (line 38)
  - webapp.parser.utils.detect.is\_location\_header (line 39)
  - webapp.parser.utils.detect.dynamic\_detect\_location\_header (line 44)
  - webapp.parser.utils.detect.detect\_candidate\_column (line 56)
  - webapp.parser.utils.detect.detect\_candidate\_column (line 63)
  - webapp.parser.utils.detect.harmonize\_headers\_and\_data (line 77)
  - webapp.parser.utils.detect.parse\_numeric (line 92)
  - webapp.parser.utils.detect.parse\_numeric (line 98)
  - webapp.parser.utils.detect.parse\_numeric (line 104)
  - webapp.parser.utils.detect.dedupe\_headers\_with\_suffix (line 115)
  - webapp.parser.utils.detect.dedupe\_headers\_with\_suffix (line 121)

### webapp/tests/test\_integrity\_api\_structure.py {#webapp-tests-test-integrity-api-structure-py}

- Definitions:
  - function: `client` (line 17)
  - function: `\_sample\_trends` (line 23)
  - function: `test\_integrity\_trends\_response\_shape` (line 48)
  - function: `test\_integrity\_signal\_response\_shape\_and\_status` (line 62)
  - function: `test\_integrity\_signal\_insufficient\_data` (line 87)
  - function:
    `test\_integrity\_trends\_recovers\_from\_malformed\_primary\_file` (line
    97)
  - function: `test\_integrity\_endpoints\_entry\_count\_consistency` (line 147)
- Imports:
  - **Standard Library** (4):
    - `import json as json` (line 3)
    - `from pathlib import Path` (line 4)
    - `from typing import Any` (line 5)
    - `from unittest.mock import patch` (line 6)
  - **Third-party** (1):
    - `import pytest as pytest` (line 8)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - pytest.skip (line 13)
  - app.test\_client (line 19)
  - unittest.mock.patch (line 50)
  - client.get (line 51)
  - resp.get\_json (line 54)
  - unittest.mock.patch (line 64)
  - client.post (line 65)
  - resp.get\_json (line 77)
  - unittest.mock.patch (line 88)
  - client.post (line 89)
  - resp.get\_json (line 92)
  - pathlib.Path (line 98)
  - primary\_path.exists (line 105)
  - primary\_path.read\_text (line 105)
  - cache\_path.exists (line 106)
  - cache\_path.read\_text (line 106)
  - primary\_path.write\_text (line 122)
  - cache\_path.write\_text (line 123)
  - json.dumps (line 123)
  - client.get (line 125)
  - resp.get\_json (line 128)
  - primary\_path.exists (line 135)
  - primary\_path.unlink (line 136)
  - primary\_path.write\_text (line 138)
  - cache\_path.exists (line 141)
  - cache\_path.unlink (line 142)
  - cache\_path.write\_text (line 144)
  - unittest.mock.patch (line 149)
  - client.get (line 150)
  - client.post (line 151)
  - trends\_resp.get\_json (line 165)
  - signal\_resp.get\_json (line 166)
- Inbound references:
  - \_sample\_trends ← test_integrity_api_structure.py:49
  - \_sample\_trends ← test_integrity_api_structure.py:63
  - \_sample\_trends ← test_integrity_api_structure.py:148

### webapp/tests/test\_librarian.py {#webapp-tests-test-librarian-py}

> Tests for Context_Integration/librarian.py

- Definitions:
  - class: `TestLibrarian` (line 12)
  - class: `TestContextLibrary` (line 29)
- Imports:
  - **Standard Library** (2):
    - `import tempfile as tempfile` (line 3)
    - `from pathlib import Path` (line 4)
  - **Third-party** (4):
    - `import pytest as pytest` (line 2)
    - `from webapp.parser.Context_Integration.librarian import
      load_context_library` (line 5)
    - `from webapp.parser.Context_Integration.librarian import
      update_context_library` (line 5)
    - `from webapp.parser.Context_Integration.librarian import
      parse_filename_for_location` (line 5)
- Outgoing cross-module calls (sample):
  - webapp.parser.Context\_Integration.librarian.parse\_filename\_for\_location
    (line 24)
  - expected.items (line 25)
  - result.get (line 26)
  - webapp.parser.Context\_Integration.librarian.load\_context\_library (line
    34)

### webapp/tests/test\_librarian\_security.py {#webapp-tests-test-librarian-security-py}

> Security Test Suite for librarian.py

- Definitions:
  - class: `TestSafePathLibrarian` (line 31)
  - class: `TestGetSafeLogPath` (line 65)
  - class: `TestAtomicWriteJsonLibrarian` (line 104)
  - class: `TestLoadContextLibrary` (line 152)
  - class: `TestSaveContextLibrary` (line 195)
  - class: `TestBackupContextLibrary` (line 245)
  - class: `TestGetLogPath` (line 283)
  - class: `TestDeduplicateJsonlLog` (line 300)
  - class: `TestLogUnknownTag` (line 330)
  - class: `TestLogUnknownAttr` (line 348)
  - class: `TestSelfHealSecurity` (line 366)
  - class: `TestIntegrationScenarios` (line 387)
- Imports:
  - **Standard Library** (5):
    - `import os as os` (line 5)
    - `import tempfile as tempfile` (line 6)
    - `from pathlib import Path` (line 7)
    - `from unittest.mock import MagicMock` (line 8)
    - `from unittest.mock import patch` (line 8)
  - **Third-party** (15):
    - `import pytest as pytest` (line 10)
    - `from webapp.parser.Context_Integration.librarian import ALLOWED_ROOTS`
      (line 13)
    - `from webapp.parser.Context_Integration.librarian import LOG_DIR_PATH`
      (line 13)
    - `from webapp.parser.Context_Integration.librarian import
      CONTEXT_LIBRARY_DIR` (line 13)
    - `from webapp.parser.Context_Integration.librarian import
      PROJECT_ROOT_PATH` (line 13)
    - `from webapp.parser.Context_Integration.librarian import safe_path` (line
      13)
    - `from webapp.parser.Context_Integration.librarian import
      get_safe_log_path` (line 13)
    - `from webapp.parser.Context_Integration.librarian import
      atomic_write_json` (line 13)
    - `from webapp.parser.Context_Integration.librarian import
      load_context_library` (line 13)
    - `from webapp.parser.Context_Integration.librarian import
      save_context_library` (line 13)
    - `from webapp.parser.Context_Integration.librarian import
      backup_context_library` (line 13)
    - `from webapp.parser.Context_Integration.librarian import _get_log_path`
      (line 13)
    - `from webapp.parser.Context_Integration.librarian import
      _deduplicate_jsonl_log` (line 13)
    - `from webapp.parser.Context_Integration.librarian import log_unknown_tag`
      (line 13)
    - `from webapp.parser.Context_Integration.librarian import log_unknown_attr`
      (line 13)
- Outgoing cross-module calls (sample):
  - test\_root.mkdir (line 37)
  - webapp.parser.Context\_Integration.librarian.safe\_path (line 40)
  - test\_file.resolve (line 41)
  - d.mkdir (line 49)
  - pytest.raises (line 53)
  - webapp.parser.Context\_Integration.librarian.safe\_path (line 54)
  - pathlib.Path (line 59)
  - webapp.parser.Context\_Integration.librarian.safe\_path (line 60)
  - result.is\_relative\_to (line 62)
  - pathlib.Path (line 62)
  - unittest.mock.patch (line 70)
  - unittest.mock.patch (line 71)
  - webapp.parser.Context\_Integration.librarian.get\_safe\_log\_path (line 73)
  - result.is\_relative\_to (line 76)
  - tmp\_path.resolve (line 76)
  - log\_dir.mkdir (line 82)
  - unittest.mock.patch (line 84)
  - unittest.mock.patch (line 85)
  - webapp.parser.Context\_Integration.librarian.get\_safe\_log\_path (line 86)
  - result.is\_relative\_to (line 89)
  - log\_dir.resolve (line 89)
  - log\_dir.mkdir (line 94)
  - unittest.mock.patch (line 96)
  - unittest.mock.patch (line 97)
  - webapp.parser.Context\_Integration.librarian.get\_safe\_log\_path (line 98)
  - lib\_dir.mkdir (line 110)
  - unittest.mock.patch (line 115)
  - webapp.parser.Context\_Integration.librarian.atomic\_write\_json (line 116)
  - test\_file.exists (line 117)
  - lib\_dir.mkdir (line 122)
  - test\_file.write\_text (line 124)
  - unittest.mock.patch (line 128)
  - webapp.parser.Context\_Integration.librarian.atomic\_write\_json (line 129)
  - test\_file.with\_suffix (line 132)
  - backup.exists (line 133)
  - backup.is\_relative\_to (line 134)
  - lib\_dir.resolve (line 134)
  - lib\_dir.mkdir (line 139)
  - unittest.mock.patch (line 144)
  - webapp.parser.Context\_Integration.librarian.atomic\_write\_json (line 145)
  - lib\_dir.glob (line 148)
  - lib\_dir.mkdir (line 158)
  - lib\_file.write\_text (line 160)
  - unittest.mock.patch (line 162)
  - webapp.parser.Context\_Integration.librarian.load\_context\_library (line
    163)
  - lib\_dir.mkdir (line 169)
  - unittest.mock.patch (line 174)
  - pytest.raises (line 175)
  - webapp.parser.Context\_Integration.librarian.load\_context\_library (line
    176)
  - lib\_dir.mkdir (line 181)

### webapp/tests/test\_manual\_correction\_security.py {#webapp-tests-test-manual-correction-security-py}

> Security Test Suite for manual_correction_bot.py

- Definitions:
  - class: `TestSafePathValidation` (line 29)
  - class: `TestLoadJsonlSecurity` (line 73)
  - class: `TestSaveJsonlSecurity` (line 105)
  - class: `TestAtomicWriteJsonSecurity` (line 150)
  - class: `TestFindLogFilesSecurity` (line 197)
  - class: `TestCheckAndFixJsonFilesSecurity` (line 228)
  - class: `TestExportImportSecurity` (line 269)
  - class: `TestSubprocessSecurity` (line 325)
  - class: `TestIntegrationScenarios` (line 344)
- Imports:
  - **Standard Library** (5):
    - `import os as os` (line 5)
    - `import tempfile as tempfile` (line 6)
    - `from pathlib import Path` (line 7)
    - `from unittest.mock import MagicMock` (line 8)
    - `from unittest.mock import patch` (line 8)
  - **Third-party** (13):
    - `import pytest as pytest` (line 10)
    - `from webapp.parser.health.manual_correction_bot import ALLOWED_ROOTS`
      (line 13)
    - `from webapp.parser.health.manual_correction_bot import LOG_DIR` (line 13)
    - `from webapp.parser.health.manual_correction_bot import
      CONTEXT_LIBRARY_DIR` (line 13)
    - `from webapp.parser.health.manual_correction_bot import CACHE_DIR` (line
      13)
    - `from webapp.parser.health.manual_correction_bot import safe_path` (line
      13)
    - `from webapp.parser.health.manual_correction_bot import find_log_files`
      (line 13)
    - `from webapp.parser.health.manual_correction_bot import load_jsonl` (line
      13)
    - `from webapp.parser.health.manual_correction_bot import save_jsonl` (line
      13)
    - `from webapp.parser.health.manual_correction_bot import atomic_write_json`
      (line 13)
    - `from webapp.parser.health.manual_correction_bot import
      check_and_fix_json_files` (line 13)
    - `from webapp.parser.health.manual_correction_bot import
      export_correction_session` (line 13)
    - `from webapp.parser.health.manual_correction_bot import
      import_correction_session` (line 13)
- Task markers:
  - L212 **WARNING**:             # Forbidden dir should be skipped with warning
- Outgoing cross-module calls (sample):
  - test\_root.mkdir (line 35)
  - webapp.parser.health.manual\_correction\_bot.safe\_path (line 38)
  - test\_file.resolve (line 39)
  - root.mkdir (line 47)
  - pytest.raises (line 51)
  - webapp.parser.health.manual\_correction\_bot.safe\_path (line 52)
  - allowed\_root.mkdir (line 57)
  - pytest.raises (line 62)
  - webapp.parser.health.manual\_correction\_bot.safe\_path (line 63)
  - pathlib.Path (line 68)
  - webapp.parser.health.manual\_correction\_bot.safe\_path (line 69)
  - result.is\_relative\_to (line 70)
  - pathlib.Path (line 70)
  - log\_dir.mkdir (line 79)
  - test\_file.write\_text (line 81)
  - unittest.mock.patch (line 83)
  - unittest.mock.patch (line 84)
  - webapp.parser.health.manual\_correction\_bot.load\_jsonl (line 85)
  - d.mkdir (line 95)
  - forbidden\_file.write\_text (line 98)
  - unittest.mock.patch (line 100)
  - pytest.raises (line 101)
  - webapp.parser.health.manual\_correction\_bot.load\_jsonl (line 102)
  - log\_dir.mkdir (line 111)
  - unittest.mock.patch (line 116)
  - unittest.mock.patch (line 117)
  - webapp.parser.health.manual\_correction\_bot.save\_jsonl (line 118)
  - test\_file.exists (line 119)
  - d.mkdir (line 127)
  - unittest.mock.patch (line 132)
  - pytest.raises (line 133)
  - webapp.parser.health.manual\_correction\_bot.save\_jsonl (line 134)
  - log\_dir.mkdir (line 139)
  - unittest.mock.patch (line 144)
  - webapp.parser.health.manual\_correction\_bot.save\_jsonl (line 145)
  - test\_file.exists (line 147)
  - allowed\_dir.mkdir (line 156)
  - unittest.mock.patch (line 161)
  - webapp.parser.health.manual\_correction\_bot.atomic\_write\_json (line 162)
  - test\_file.exists (line 163)
  - allowed\_dir.mkdir (line 168)
  - test\_file.write\_text (line 172)
  - unittest.mock.patch (line 176)
  - webapp.parser.health.manual\_correction\_bot.atomic\_write\_json (line 177)
  - test\_file.with\_suffix (line 179)
  - backup.exists (line 180)
  - allowed\_dir.mkdir (line 185)
  - unittest.mock.patch (line 190)
  - webapp.parser.health.manual\_correction\_bot.atomic\_write\_json (line 191)
  - allowed\_dir.glob (line 193)

### webapp/tests/test\_mobile\_ui.py {#webapp-tests-test-mobile-ui-py}

- Top-of-file comments:

```python

#!/usr/bin/env python3

```

- Imports:
  - **Standard Library** (3):
    - `import json as json` (line 2)
    - `import subprocess as subprocess` (line 3)
    - `import sys as sys` (line 4)
- Outgoing cross-module calls (sample):
  - subprocess.run (line 6)
  - l.strip (line 13)
  - json.loads (line 17)
  - test.get (line 21)
  - t.get (line 25)
  - sys.exit (line 28)
  - sys.exit (line 32)

### webapp/tests/test\_models.py {#webapp-tests-test-models-py}

> Tests for database models in utils/models.py

- Definitions:
  - class: `TestModels` (line 12)
- Imports:
  - **Third-party** (6):
    - `import pytest as pytest` (line 2)
    - `from webapp.parser.utils.models import Contest` (line 3)
    - `from webapp.parser.utils.models import Candidate` (line 3)
    - `from webapp.parser.utils.models import Party` (line 3)
    - `from webapp.parser.utils.models import State` (line 3)
    - `from webapp.parser.utils.models import County` (line 3)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.models.Contest (line 17)
  - db\_session.add (line 22)
  - db\_session.commit (line 23)
  - webapp.parser.utils.models.Party (line 30)
  - db\_session.add (line 31)
  - db\_session.commit (line 32)
  - webapp.parser.utils.models.State (line 39)
  - webapp.parser.utils.models.County (line 40)
  - db\_session.add (line 42)
  - db\_session.add (line 43)
  - db\_session.commit (line 44)

### webapp/tests/test\_party\_codes.py {#webapp-tests-test-party-codes-py}

- Definitions:
  - function: `test\_get\_party\_code\_info\_known` (line 6)
  - function: `test\_normalize\_party\_code\_variants` (line 12)
  - function: `test\_get\_party\_code\_info\_unknown` (line 29)
- Imports:
  - **Third-party** (2):
    - `import pytest as pytest` (line 1)
    - `from webapp.parser.Context_Integration.Context_Library import constants
      as C` (line 3)
- Outgoing cross-module calls (sample):
  - webapp.parser.Context\_Integration.Context\_Library.constants.get\_party\_code\_info
    (line 7)
  - info.get (line 9)
  - webapp.parser.Context\_Integration.Context\_Library.constants.normalize\_party\_code
    (line 13)
  - dem\_code.lower (line 14)
  - webapp.parser.Context\_Integration.Context\_Library.constants.normalize\_party\_code
    (line 16)
  - dem\_lower.lower (line 17)
  - webapp.parser.Context\_Integration.Context\_Library.constants.normalize\_party\_code
    (line 19)
  - d\_code.lower (line 20)
  - webapp.parser.Context\_Integration.Context\_Library.constants.normalize\_party\_code
    (line 22)
  - w\_code.lower (line 23)
  - webapp.parser.Context\_Integration.Context\_Library.constants.normalize\_party\_code
    (line 25)
  - dc.lower (line 26)
  - dc.lower (line 26)
  - webapp.parser.Context\_Integration.Context\_Library.constants.get\_party\_code\_info
    (line 30)

### webapp/tests/test\_path\_security.py {#webapp-tests-test-path-security-py}

> Path Security Test Suite

- Definitions:
  - class: `TestSafeFilename` (line 20)
  - class: `TestSafeResolvePath` (line 81)
  - class: `TestIsPathSafe` (line 155)
  - class: `TestSafeJoinPath` (line 200)
  - class: `TestValidateDirectoryPath` (line 237)
  - class: `TestPathTraversalAttacks` (line 272)
  - class: `TestIntegrationScenarios` (line 325)
- Imports:
  - **Standard Library** (3):
    - `import os as os` (line 5)
    - `import tempfile as tempfile` (line 6)
    - `from pathlib import Path` (line 7)
  - **Third-party** (6):
    - `import pytest as pytest` (line 9)
    - `from webapp.parser.utils.shared_logic import is_path_safe` (line 11)
    - `from webapp.parser.utils.shared_logic import safe_filename` (line 11)
    - `from webapp.parser.utils.shared_logic import safe_join_path` (line 11)
    - `from webapp.parser.utils.shared_logic import safe_resolve_path` (line 11)
    - `from webapp.parser.utils.shared_logic import validate_directory_path`
      (line 11)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.shared\_logic.safe\_filename (line 25)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 26)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 30)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 31)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 32)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 33)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 37)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 38)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 39)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 40)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 44)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 45)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 49)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 50)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 51)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 52)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 56)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 57)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 58)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 63)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 69)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 76)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 78)
  - target.touch (line 89)
  - webapp.parser.utils.shared\_logic.safe\_resolve\_path (line 91)
  - resolved.is\_relative\_to (line 93)
  - webapp.parser.utils.shared\_logic.safe\_resolve\_path (line 100)
  - base.mkdir (line 106)
  - pytest.raises (line 109)
  - webapp.parser.utils.shared\_logic.safe\_resolve\_path (line 110)
  - base.mkdir (line 115)
  - outside.mkdir (line 118)
  - pytest.raises (line 120)
  - webapp.parser.utils.shared\_logic.safe\_resolve\_path (line 121)
  - pytest.skip (line 126)
  - base.mkdir (line 129)
  - outside.mkdir (line 131)
  - link.symlink\_to (line 134)
  - pytest.raises (line 137)
  - webapp.parser.utils.shared\_logic.safe\_resolve\_path (line 138)
  - pytest.raises (line 145)
  - webapp.parser.utils.shared\_logic.safe\_resolve\_path (line 146)
  - existing.touch (line 150)
  - webapp.parser.utils.shared\_logic.safe\_resolve\_path (line 151)
  - webapp.parser.utils.shared\_logic.is\_path\_safe (line 163)
  - d.mkdir (line 172)
  - webapp.parser.utils.shared\_logic.is\_path\_safe (line 174)
  - d.mkdir (line 182)
  - webapp.parser.utils.shared\_logic.is\_path\_safe (line 187)
  - webapp.parser.utils.shared\_logic.is\_path\_safe (line 188)

### webapp/tests/test\_phase\_a\_integration.py {#webapp-tests-test-phase-a-integration-py}

> Integration Tests for Phase A: Confidence/Caution Decision Gates

- Definitions:
  - function: `confidence\_map` (line 65)
  - function: `temp\_vocab\_dir` (line 71)
  - function: `vocab\_loader` (line 105)
  - function: `mock\_logger` (line 113)
  - function: `mock\_metrics` (line 120)
  - class: `TestEntityConfidenceMap` (line 130)
  - class: `TestSafeDecideAPI` (line 244)
  - class: `TestVocabLoader` (line 331)
  - class: `TestLoggerDecisionFiltering` (line 393)
  - class: `TestPrometheusMetrics` (line 475)
  - class: `TestPhaseAIntegration` (line 518)
- Imports:
  - **Standard Library** (10):
    - `import os as os` (line 26)
    - `import tempfile as tempfile` (line 27)
    - `import time as time` (line 28)
    - `from pathlib import Path` (line 29)
    - `from typing import Dict` (line 30)
    - `from typing import Generator` (line 30)
    - `from typing import Any` (line 30)
    - `from unittest.mock import Mock` (line 31)
    - `from unittest.mock import patch` (line 31)
    - `from unittest.mock import MagicMock` (line 31)
  - **Third-party** (17):
    - `import pytest as pytest` (line 33)
    - `from webapp.parser.Context_Integration.library.entity_confidence_map
      import EntityConfidenceMap` (line 36)
    - `from webapp.parser.Context_Integration.library.entity_confidence_map
      import SignalType` (line 36)
    - `from webapp.parser.Context_Integration.library.entity_confidence_map
      import AnomalyType` (line 36)
    - `from webapp.parser.Context_Integration.library.entity_confidence_map
      import OverrideTrigger` (line 36)
    - `from webapp.parser.Context_Integration.library.entity_confidence_map
      import get_confidence_map` (line 36)
    - `from webapp.parser.utils.safe_decide import safe_decide_jurisdiction`
      (line 43)
    - `from webapp.parser.utils.safe_decide import safe_decide_office` (line 43)
    - `from webapp.parser.utils.safe_decide import safe_decide_party` (line 43)
    - `from webapp.parser.utils.safe_decide import safe_decide_source` (line 43)
    - `from webapp.parser.Context_Integration.vocab.loader import VocabLoader`
      (line 49)
    - `from webapp.parser.Context_Integration.vocab.loader import
      VocabLoaderError` (line 49)
    - `from webapp.parser.Context_Integration.vocab.loader import
      VocabFileNotFound` (line 49)
    - `from webapp.parser.Context_Integration.vocab.loader import
      VocabSecurityError` (line 49)
    - `from webapp.parser.utils.shared_logic import DecisionTuple` (line 55)
    - `from webapp.parser.utils.logger_singleton import logger` (line 56)
    - `from webapp.parser.utils.metrics_prom import increment_prom_counter`
      (line 57)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 24)
- Outgoing cross-module calls (sample):
  - webapp.parser.Context\_Integration.library.entity\_confidence\_map.get\_confidence\_map
    (line 67)
  - tempfile.TemporaryDirectory (line 73)
  - pathlib.Path (line 74)
  - webapp.parser.Context\_Integration.vocab.loader.VocabLoader (line 107)
  - loader.clear\_cache (line 109)
  - unittest.mock.patch (line 115)
  - unittest.mock.patch (line 122)
  - webapp.parser.Context\_Integration.library.entity\_confidence\_map.get\_confidence\_map
    (line 135)
  - webapp.parser.Context\_Integration.library.entity\_confidence\_map.get\_confidence\_map
    (line 136)
  - confidence\_map.calculate\_confidence\_caution (line 168)
  - confidence\_map.calculate\_confidence\_caution (line 189)
  - confidence\_map.calculate\_confidence\_caution (line 211)
  - confidence\_map.calculate\_confidence\_caution (line 228)
  - webapp.parser.utils.safe\_decide.safe\_decide\_jurisdiction (line 251)
  - result.get (line 259)
  - result.get (line 261)
  - webapp.parser.utils.safe\_decide.safe\_decide\_office (line 267)
  - result.get (line 276)
  - webapp.parser.utils.safe\_decide.safe\_decide\_party (line 283)
  - result.get (line 291)
  - webapp.parser.utils.safe\_decide.safe\_decide\_source (line 298)
  - result.get (line 305)
  - vocab\_loader.load\_canonical (line 336)
  - vocab\_loader.load\_mapping (line 345)
  - aliases.get (line 348)
  - aliases.get (line 349)
  - vocab\_loader.get\_load\_count (line 354)
  - vocab\_loader.load\_canonical (line 355)
  - vocab\_loader.get\_load\_count (line 356)
  - vocab\_loader.load\_canonical (line 359)
  - vocab\_loader.get\_load\_count (line 360)
  - vocab\_loader.load\_canonical (line 367)
  - vocab\_loader.load\_canonical (line 368)
  - pytest.raises (line 375)
  - vocab\_loader.load\_canonical (line 376)
  - pytest.raises (line 380)
  - vocab\_loader.load\_canonical (line 381)
  - pytest.raises (line 385)
  - vocab\_loader.load\_canonical (line 386)
  - webapp.parser.utils.logger\_singleton.logger.\_filter\_decision\_noise (line
    398)
  - time.time (line 412)
  - webapp.parser.utils.logger\_singleton.logger.\_filter\_decision\_noise (line
    415)
  - webapp.parser.utils.logger\_singleton.logger.\_filter\_decision\_noise (line
    419)
  - time.time (line 428)
  - webapp.parser.utils.logger\_singleton.logger.\_filter\_decision\_noise (line
    431)
  - webapp.parser.utils.logger\_singleton.logger.\_filter\_decision\_noise (line
    435)
  - time.time (line 443)
  - webapp.parser.utils.logger\_singleton.logger.\_filter\_decision\_noise (line
    446)
  - webapp.parser.utils.logger\_singleton.logger.\_filter\_decision\_noise (line
    450)
  - time.time (line 457)

### webapp/tests/test\_qa\_authority\_and\_stats.py {#webapp-tests-test-qa-authority-and-stats-py}

- Definitions:
  - function: `\_build\_app` (line 8)
  - function: `test\_verify\_and\_promote\_requires\_admin\_reviewer\_tier`
    (line 14)
  - function: `test\_verify\_and\_promote\_allows\_admin\_reviewer\_tier` (line
    41)
  - function: `test\_stats\_reports\_rejected\_count` (line 68)
- Imports:
  - **Third-party** (2):
    - `from flask import Flask` (line 3)
    - `from webapp.parser.quality_assurance import qa_endpoints as qae` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - flask.Flask (line 9)
  - app.register\_blueprint (line 10)
  - monkeypatch.setattr (line 17)
  - monkeypatch.setattr (line 18)
  - monkeypatch.setattr (line 23)
  - app.test\_client (line 25)
  - client.post (line 26)
  - resp.get\_json (line 35)
  - monkeypatch.setattr (line 44)
  - monkeypatch.setenv (line 45)
  - monkeypatch.setattr (line 46)
  - monkeypatch.setattr (line 51)
  - app.test\_client (line 53)
  - client.post (line 54)
  - resp.get\_json (line 63)
  - monkeypatch.setattr (line 71)
  - monkeypatch.setattr (line 72)
  - monkeypatch.setattr (line 74)
  - monkeypatch.setattr (line 75)
  - monkeypatch.setattr (line 83)
  - app.test\_client (line 85)
  - client.get (line 86)
  - resp.get\_json (line 89)
- Inbound references:
  - \_build\_app ← test_qa_authority_and_stats.py:15
  - \_build\_app ← test_qa_authority_and_stats.py:42
  - \_build\_app ← test_qa_authority_and_stats.py:69
  - \_build\_app ← test_verification_tier_enforcement.py:20
  - \_build\_app ← test_verification_tier_enforcement.py:39
  - \_build\_app ← test_verification_tier_enforcement.py:60

### webapp/tests/test\_risk\_gates\_runtime.py {#webapp-tests-test-risk-gates-runtime-py}

- Definitions:
  - function:
    `test\_risk\_gate\_evaluator\_initializes\_with\_valid\_boundaries` (line 8)
  - function:
    `test\_calculus\_evaluator\_supports\_fallback\_verification\_score` (line
    21)
  - function: `test\_calculus\_evaluator\_blocks\_high\_suspicion\_data` (line
    36)
  - function: `test\_apply\_risk\_assessment\_enriches\_metadata` (line 49)
- Imports:
  - **Third-party** (3):
    - `from webapp.parser.health.risk_gates import RiskGateEvaluator` (line 3)
    - `from webapp.parser.health.risk_gates_calculus import
      CalculusRiskEvaluator` (line 4)
    - `from webapp.parser.html_election_parser import _apply_risk_assessment`
      (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Task markers:
  - L67 **WARN**: ", "block"}
- Outgoing cross-module calls (sample):
  - webapp.parser.health.risk\_gates.RiskGateEvaluator (line 9)
  - evaluator.evaluate (line 10)
  - webapp.parser.health.risk\_gates\_calculus.CalculusRiskEvaluator (line 22)
  - evaluator.evaluate\_with\_derivatives (line 23)
  - webapp.parser.health.risk\_gates\_calculus.CalculusRiskEvaluator (line 37)
  - evaluator.evaluate\_with\_derivatives (line 38)
  - webapp.parser.html\_election\_parser.\_apply\_risk\_assessment (line 58)
  - enriched.get (line 66)
  - enriched.get (line 67)
  - enriched.get (line 68)
  - enriched.get (line 69)

### webapp/tests/test\_schema\_validation.py {#webapp-tests-test-schema-validation-py}

> Schema validation and regression tests for parser output

- Definitions:
  - class: `TestSchemaValidation` (line 12)
  - class: `TestRegressionFixtures` (line 194)
  - class: `TestSchemaDocumentation` (line 299)
- Imports:
  - **Third-party** (3):
    - `import pytest as pytest` (line 7)
    - `from webapp.parser.utils.table_builder import build_table_noninteractive`
      (line 8)
    - `from webapp.parser.utils.shared_logic import safe_get` (line 9)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.table\_builder.build\_table\_noninteractive (line 29)
  - row.get (line 44)
  - webapp.parser.utils.table\_builder.build\_table\_noninteractive (line 69)
  - expected\_canonical.lower (line 83)
  - party\_value.lower (line 83)
  - webapp.parser.utils.table\_builder.build\_table\_noninteractive (line 109)
  - webapp.parser.utils.table\_builder.build\_table\_noninteractive (line 142)
  - context.get (line 155)
  - context.get (line 156)
  - webapp.parser.utils.table\_builder.build\_table\_noninteractive (line 175)
  - all\_headers.append (line 184)
  - webapp.parser.utils.table\_builder.build\_table\_noninteractive (line 242)
  - webapp.parser.utils.table\_builder.build\_table\_noninteractive (line 284)

### webapp/tests/test\_session\_manager.py {#webapp-tests-test-session-manager-py}

> Tests for health/session_manager.py

- Definitions:
  - class: `TestSessionManager` (line 7)
- Imports:
  - **Third-party** (4):
    - `import pytest as pytest` (line 2)
    - `from webapp.parser.health.session_manager import SessionManager` (line 3)
    - `from webapp.parser.utils.session_state import SessionState` (line 4)
    - `from webapp.parser.utils.session_state import PipelinePhase` (line 4)
- Outgoing cross-module calls (sample):
  - webapp.parser.health.session\_manager.SessionManager (line 12)
  - manager.ensure\_session (line 14)
  - webapp.parser.health.session\_manager.SessionManager (line 22)
  - manager.ensure\_session (line 24)
  - manager.set\_state (line 27)
  - webapp.parser.health.session\_manager.SessionManager (line 39)
  - manager.ensure\_session (line 41)
  - manager.set\_manual\_source (line 43)
  - manager.get\_manual\_source (line 45)
  - manager.get\_manual\_source\_origin (line 46)
  - webapp.parser.health.session\_manager.SessionManager (line 50)
  - manager.ensure\_session (line 52)
  - manager.has\_session (line 54)
  - manager.delete\_session (line 56)
  - manager.has\_session (line 58)

### webapp/tests/test\_shared\_logic.py {#webapp-tests-test-shared-logic-py}

> Tests for webapp/parser/utils/shared_logic.py

- Top-of-file comments:

```python

# -\*- coding: utf-8 -\*-

```

- Definitions:
  - class: `TestSafeFilename` (line 18)
  - class: `TestSafeSlug` (line 51)
  - class: `TestSafeAccessors` (line 71)
  - class: `TestLocationNormalization` (line 94)
  - class: `TestSafeParse` (line 122)
- Imports:
  - **Third-party** (11):
    - `import pytest as pytest` (line 3)
    - `from webapp.parser.utils.shared_logic import safe_filename` (line 4)
    - `from webapp.parser.utils.shared_logic import safe_slug` (line 4)
    - `from webapp.parser.utils.shared_logic import safe_get` (line 4)
    - `from webapp.parser.utils.shared_logic import safe_strip` (line 4)
    - `from webapp.parser.utils.shared_logic import safe_lower` (line 4)
    - `from webapp.parser.utils.shared_logic import normalize_county_name` (line
      4)
    - `from webapp.parser.utils.shared_logic import normalize_state_name` (line
      4)
    - `from webapp.parser.utils.shared_logic import format_county_label` (line
      4)
    - `from webapp.parser.utils.shared_logic import format_state_label` (line 4)
    - `from webapp.parser.utils.shared_logic import safe_parse` (line 4)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.shared\_logic.safe\_filename (line 23)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 24)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 29)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 34)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 35)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 36)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 41)
  - result.endswith (line 43)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 47)
  - webapp.parser.utils.shared\_logic.safe\_filename (line 48)
  - webapp.parser.utils.shared\_logic.safe\_slug (line 56)
  - webapp.parser.utils.shared\_logic.safe\_slug (line 57)
  - webapp.parser.utils.shared\_logic.safe\_slug (line 61)
  - webapp.parser.utils.shared\_logic.safe\_slug (line 62)
  - webapp.parser.utils.shared\_logic.safe\_slug (line 67)
  - webapp.parser.utils.shared\_logic.safe\_get (line 77)
  - webapp.parser.utils.shared\_logic.safe\_get (line 78)
  - webapp.parser.utils.shared\_logic.safe\_get (line 79)
  - webapp.parser.utils.shared\_logic.safe\_strip (line 83)
  - webapp.parser.utils.shared\_logic.safe\_strip (line 84)
  - webapp.parser.utils.shared\_logic.safe\_strip (line 85)
  - webapp.parser.utils.shared\_logic.safe\_lower (line 89)
  - webapp.parser.utils.shared\_logic.safe\_lower (line 90)
  - webapp.parser.utils.shared\_logic.safe\_lower (line 91)
  - webapp.parser.utils.shared\_logic.normalize\_county\_name (line 99)
  - webapp.parser.utils.shared\_logic.normalize\_county\_name (line 100)
  - webapp.parser.utils.shared\_logic.normalize\_county\_name (line 101)
  - webapp.parser.utils.shared\_logic.normalize\_state\_name (line 105)
  - webapp.parser.utils.shared\_logic.normalize\_state\_name (line 106)
  - webapp.parser.utils.shared\_logic.normalize\_state\_name (line 107)
  - webapp.parser.utils.shared\_logic.format\_county\_label (line 111)
  - webapp.parser.utils.shared\_logic.format\_county\_label (line 112)
  - webapp.parser.utils.shared\_logic.format\_county\_label (line 113)
  - webapp.parser.utils.shared\_logic.format\_state\_label (line 117)
  - webapp.parser.utils.shared\_logic.format\_state\_label (line 118)
  - webapp.parser.utils.shared\_logic.format\_state\_label (line 119)
  - webapp.parser.utils.shared\_logic.safe\_parse (line 138)
  - webapp.parser.utils.shared\_logic.safe\_parse (line 149)
  - webapp.parser.utils.shared\_logic.safe\_parse (line 158)

### webapp/tests/test\_table\_builder.py {#webapp-tests-test-table-builder-py}

> Tests for webapp/parser/utils/table_builder.py

- Definitions:
  - class: `TestTableBuilder` (line 6)
  - class: `TestTableCore` (line 57)
- Imports:
  - **Third-party** (2):
    - `import pytest as pytest` (line 2)
    - `from webapp.parser.utils.table_builder import build_table_noninteractive`
      (line 3)
- Outgoing cross-module calls (sample):
  - webapp.parser.utils.table\_builder.build\_table\_noninteractive (line 18)
  - webapp.parser.utils.table\_builder.build\_table\_noninteractive (line 43)

### webapp/tests/test\_verification\_tier\_enforcement.py {#webapp-tests-test-verification-tier-enforcement-py}

- Definitions:
  - function: `\_build\_app` (line 8)
  - function: `test\_verifier\_tier\_requires\_authenticated\_principal` (line
    19)
  - function: `test\_verifier\_tier\_blocks\_insufficient\_tier` (line 38)
  - function: `test\_verifier\_tier\_allows\_sufficient\_tier` (line 59)
- Imports:
  - **Third-party** (3):
    - `from flask import Flask` (line 3)
    - `from flask import jsonify` (line 3)
    - `from webapp.parser import verification_endpoints as ve` (line 5)
  - **Local/Project** (1):
    - `from __future__ import annotations` (line 1)
- Outgoing cross-module calls (sample):
  - flask.Flask (line 9)
  - flask.jsonify (line 14)
  - app.route (line 11)
  - webapp.parser.verification\_endpoints.\_require\_verifier\_tier (line 12)
  - monkeypatch.setattr (line 25)
  - app.test\_client (line 30)
  - client.get (line 31)
  - resp.get\_json (line 34)
  - monkeypatch.setattr (line 44)
  - app.test\_client (line 49)
  - client.get (line 50)
  - resp.get\_json (line 53)
  - monkeypatch.setenv (line 62)
  - monkeypatch.setattr (line 67)
  - app.test\_client (line 72)
  - client.get (line 73)
  - resp.get\_json (line 76)
