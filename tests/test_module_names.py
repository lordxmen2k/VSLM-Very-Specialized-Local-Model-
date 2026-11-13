"""Every name a module uses has to be bound somewhere it can see.

A name that is used but never defined is a runtime error that no syntax
check catches, and it bites at the moment the code path is reached. The
speed test hit exactly that: a helper module was written, the import was
meant to be added with a string replacement, the string did not match the
file, and the replacement silently did nothing. The error surfaced hours
later, on the owner's machine, as a NameError two lines into a function.

This walks every module in the package and reports names that are loaded but
never bound anywhere in that module. It is not a full linter: it does not
distinguish scopes, so a name bound only inside one function and used in
another would pass. That is an acceptable trade for catching the case that
matters.
"""
import ast
import builtins
import pathlib
import unittest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "la"

# names the interpreter provides in every module
_DUNDERS = {"__name__", "__file__", "__doc__", "__package__", "__spec__",
            "__loader__", "__builtins__", "__debug__"}
_BUILTINS = set(dir(builtins)) | _DUNDERS


def _bound(tree: ast.AST) -> set[str]:
    """Every name bound anywhere in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.MatchAs) and node.name:
            names.add(node.name)
    return names


def _used(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


class TestNoUndefinedNames(unittest.TestCase):
    def test_every_module_binds_the_names_it_uses(self):
        modules = sorted(PACKAGE.glob("*.py"))
        self.assertTrue(modules, "no modules found")
        problems = []
        for path in modules:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            bound = _bound(tree)
            missing = sorted(_used(tree) - bound - _BUILTINS)
            if missing:
                problems.append(f"{path.name}: {', '.join(missing)}")
        self.assertEqual([], problems,
                         "names used but never bound:\n  " + "\n  ".join(problems))

    def test_every_module_compiles(self):
        for path in sorted(PACKAGE.glob("*.py")):
            with self.subTest(module=path.name):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
