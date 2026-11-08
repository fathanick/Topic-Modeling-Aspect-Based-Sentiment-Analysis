"""Module-level names must be defined before use.

py_compile only parses; it does not execute. A forward reference in the
constant block (MIN_TOPIC_SIZE = MIN_CLUSTER_SIZE before MIN_CLUSTER_SIZE
exists) therefore compiles fine and only explodes on the cluster -- twice
in this project's history. Catch it locally instead.
"""

import ast
from pathlib import Path

BAD = []


def module_level_names_used_before_definition(path):
    tree = ast.parse(Path(path).read_text())
    tree_src = Path(path).read_text()
    imported = set()
    for n in ast.walk(ast.parse(tree_src)):
        if isinstance(n, ast.Import):
            imported.update((a.asname or a.name.split(".")[0]) for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            imported.update((a.asname or a.name) for a in n.names)
    defined = set(dir(__builtins__)) | imported
    problems = []
    for node in tree.body:
        targets, value = [], None
        if isinstance(node, ast.Try):
            for sub in node.body + node.handlers + node.orelse + node.finalbody:
                for n2 in ast.walk(sub):
                    if isinstance(n2, ast.Assign):
                        defined.update(t.id for t in n2.targets if isinstance(t, ast.Name))
            continue
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value

        if value is not None:
            for sub in ast.walk(value):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id not in defined and sub.id not in ("__file__", "__name__"):
                        problems.append(f"{path}:{sub.lineno} {sub.id} used before definition")
        defined.update(targets)
    return problems


def main():
    files = [
        "src/topic_modeling/topic_modeling_bertopic.py",
        "src/topic_modeling/sweep_topic_params.py",
        "src/sentiment_analysis/absa_indobertweet.py",
        "src/sentiment_analysis/absa_llm_local_gpu.py",
        "src/preprocessing/preprocessing.py",
    ]
    root = Path(__file__).resolve().parents[1]
    for f in files:
        p = root / f
        if p.exists():
            BAD.extend(module_level_names_used_before_definition(p))

    # Attributes/functions may legitimately reference globals; only the
    # module-level constant block is checked, which is where this bug lives.
    if BAD:
        print("\n".join(BAD))
        raise SystemExit(f"FAILED: {len(BAD)} forward reference(s)")
    print(f"OK: no forward references in {len(files)} modules")


if __name__ == "__main__":
    main()
