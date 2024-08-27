"""Generate the code reference pages and navigation."""
from pathlib import Path
import sys

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
import_name = "uqtils"
nav_file = "SUMMARY.md"
root = Path(__file__).parent.parent
src = root / "src" / import_name

if not src.exists():
    sys.exit(0)

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    parts = tuple(module_path.parts)

    top_level_init = False  # Catch the top-level package's __init__ file and handle differently
    if parts[-1] == "__init__":
        parts = parts[:-1]
        top_level_init = len(parts) == 0
        new_name = "index.md"
        doc_path = doc_path.with_name(new_name)
        full_doc_path = full_doc_path.with_name(new_name)
    elif parts[-1] == "__main__":
        continue

    if not top_level_init:
        nav[parts] = doc_path.as_posix()

    # For testing locally:
    # full_doc_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(full_doc_path, 'w') as fd:

    # Creates virtual docs/reference/*.md files that are loaded by mkdocs
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)

        # Change md header for subpackage __init__ files
        if full_doc_path.name == 'index.md' and not top_level_init:
            fd.write(f'# {import_name}.{ident}\n')

        # Write mkdocstrings import mechanism at top of each reference file
        fd.write(f"::: {import_name}.{ident}" if not top_level_init else f"::: {import_name}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# For testing locally:
# with open('reference/mkdocs-nav.md', 'w') as nav_file:

# Here's where you would write the SUMMARY.md file for literate-nav
# with mkdocs_gen_files.open(f"reference/{nav_file}", "w") as nav_file:
#     nav_file.writelines(nav.build_literate_nav())
