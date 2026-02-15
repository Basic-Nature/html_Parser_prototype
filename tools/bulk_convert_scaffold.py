import ast
from pathlib import Path

root = Path("webapp/parser/handlers/states")
source = Path("scripts/generate_state_handler.py").read_text(encoding="utf-8")
module = ast.parse(source)
state_codes = {}
for node in module.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "STATE_CODES":
                state_codes = ast.literal_eval(node.value)
                break
        if state_codes:
            break

if not state_codes:
    raise SystemExit("STATE_CODES not found in generate_state_handler.py")

state_codes.update(
    {
        "American Samoa": "AS",
        "Guam": "GU",
        "Northern Mariana Islands": "MP",
        "Puerto Rico": "PR",
        "US Virgin Islands": "VI",
    }
)


def normalize(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace(".", "")


slug_to_name = {normalize(name): name for name in state_codes}

updated = 0
unknown = []

for path in root.rglob("*.py"):
    if path.name == "__init__.py":
        continue
    content = path.read_text(encoding="utf-8")
    if "state_scaffold import parse as scaffold_parse" not in content:
        continue

    if path.parent == root:
        slug = path.stem
    else:
        slug = path.parent.name

    slug_norm = normalize(slug)
    state_name = slug_to_name.get(slug_norm)
    if not state_name:
        slug_norm = normalize(path.stem)
        state_name = slug_to_name.get(slug_norm)

    if not state_name:
        unknown.append(str(path))
        continue

    state_code = state_codes[state_name]
    class_name = state_name.replace(" ", "") + "Handler"

    lines = content.splitlines()
    out_lines = []
    inserted = False
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith(
            "from webapp.parser.handlers.shared.state_scaffold import parse as scaffold_parse"
        ):
            line = (
                "from webapp.parser.handlers.shared.state_handler_base "
                "import SimpleTableHandler"
            )

        if line.startswith("def parse("):
            if not inserted:
                out_lines.append("")
                out_lines.append(f"class {class_name}(SimpleTableHandler):")
                out_lines.append(
                    f"    \"\"\"Handler for {state_name} election data.\"\"\""
                )
                out_lines.append("")
                out_lines.append(f"    STATE_NAME = \"{state_name}\"")
                out_lines.append(f"    STATE_CODE = \"{state_code}\"")
                out_lines.append("")
                out_lines.append(
                    "# Create module-level parse function for router compatibility"
                )
                out_lines.append(f"_handler_instance = {class_name}()")
                out_lines.append("")
                inserted = True

            out_lines.append(
                "def parse(page: Any = None, html_context: Dict[str, Any] | None = None, "
                "coordinator: Any = None, context: Dict[str, Any] | None = None, "
                "session_id: str | None = None, **kwargs):"
            )
            out_lines.append(f"    \"\"\"State handler for {state_name}.\"\"\"")
            out_lines.append("    return _handler_instance.parse(")
            out_lines.append("        page=page,")
            out_lines.append("        html_context=html_context,")
            out_lines.append("        coordinator=coordinator,")
            out_lines.append("        context=context,")
            out_lines.append("        session_id=session_id,")
            out_lines.append("        **kwargs,")
            out_lines.append("    )")

            i += 1
            while i < len(lines):
                if lines[i].startswith("def ") and not lines[i].startswith("    "):
                    break
                i += 1
            continue

        out_lines.append(line)
        i += 1

    new_content = "\n".join(out_lines).rstrip() + "\n"
    path.write_text(new_content, encoding="utf-8")
    updated += 1

if unknown:
    print("Unknown state slugs (skipped):")
    for item in unknown:
        print(f"  - {item}")

print(f"Updated scaffold stubs: {updated}")
