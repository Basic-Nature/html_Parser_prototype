# Pipeline map — compact

```mermaid
graph LR
  webapp_parser_handlers_states_example state_example_county_example_county[webapp.parser.handlers.states.example state.example_county.example_county] -->|2| webapp_parser_handlers_states_example state_example_state[webapp.parser.handlers.states.example state.example_state]
  webapp_parser_handlers_formats_json_handler[webapp.parser.handlers.formats.json_handler] -->|1| webapp_parser_handlers_formats_csv_handler[webapp.parser.handlers.formats.csv_handler]
  webapp_parser_handlers_formats_pdf_handler[webapp.parser.handlers.formats.pdf_handler] -->|1| webapp_parser_handlers_formats_csv_handler[webapp.parser.handlers.formats.csv_handler]
```
