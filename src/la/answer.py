"""Parse and compare the model's answer.

The model emits LaTeX. To check that answer against ground truth we need to
turn LaTeX back into something sympy can compare, and doing that with a
general-purpose LaTeX parser (sympy's `parse_latex` needs antlr, is slow, and
mis-parses a lot) is a poor trade for a project with a fixed answer grammar.

So this module defines the grammar explicitly, writes its own printer and its
own parser, and unit-tests the round trip. The grammar is deliberately small:

    answer   := pairs | set | matrix | scalar
    pairs    := label '=' value (',' label '=' value)*
    set      := '{' value (',' value)* '}'
    matrix   := '\\begin{pmatrix}' row ('\\\\' row)* '\\end{pmatrix}'
    value    := scalar | matrix | set | pairs
    scalar   := term (('+' | '-') term)*
    term     := factor (('*' | '/') factor)*
    factor   := '-' factor
             | number
             | '\\frac' '{' scalar '}' '{' scalar '}'
             | '\\sqrt' '{' scalar '}'
             | name

`name` is a symbol: `i` for the imaginary unit, `x_1` for a labelled
component, or any other identifier.

The printer produces exactly this grammar, so printing then parsing is a
round trip that the test suite checks on every generated example.
"""
from __future__ import annotations

import re
from typing import Any

import sympy as sp


# --------------------------------------------------------------- printing --

def _fmt_number(x: sp.Expr) -> str:
    if x.is_Integer:
        return str(int(x))
    if x.is_Rational:
        p, q = int(sp.numer(x)), int(sp.denom(x))
        if q == 1:
            return str(p)
        return rf"\frac{{{p}}}{{{q}}}"
    if x.is_Float:
        return f"{float(x):.10g}"
    return str(x)


def fmt_scalar(expr: sp.Expr) -> str:
    """Print a scalar in the answer grammar."""
    expr = sp.nsimplify(expr) if expr.is_number else expr
    expr = sp.expand(expr)

    if expr.is_Add:
        parts = []
        for term in sp.Add.make_args(expr):
            s = fmt_scalar(term)
            if s.startswith("-"):
                parts.append(" - " + s[1:])
            else:
                parts.append(" + " + s)
        out = "".join(parts).strip()
        return out.lstrip("+ ")
    if expr.is_Mul:
        args = sp.Mul.make_args(expr)
        lead = ""
        if expr.could_extract_minus_sign():
            lead = "-"
            args = [-a for a in args]
        rendered = []
        for i, a in enumerate(args):
            s = fmt_scalar(a)
            if i > 0 and s.startswith("-"):
                rendered.append(" - " + s[1:])
            elif i > 0 and (s.startswith(r"\sqrt") or s.startswith(r"\frac")):
                rendered.append(s)      # 2\sqrt{3}, not 2 * \sqrt{3}
            else:
                rendered.append((" * " if i else " * ") + s)
        return lead + "".join(rendered).lstrip(" *")
    if expr.is_Pow:
        base, exp = expr.as_base_exp()
        if exp == sp.Rational(1, 2) and base.is_positive:
            return rf"\sqrt{{{fmt_scalar(base)}}}"
        return f"({fmt_scalar(base)})^{{{fmt_scalar(exp)}}}"
    if expr == sp.I:
        return "i"
    if expr.is_Number:
        return _fmt_number(expr)
    if expr.is_Symbol:
        return str(expr)
    if expr.is_Matrix:
        return fmt_matrix(expr)
    return str(expr)


def fmt_matrix(M: sp.Matrix) -> str:
    rows = []
    for i in range(M.rows):
        rows.append(" & ".join(fmt_scalar(M[i, j]) for j in range(M.cols)))
    body = r" \\ ".join(rows)
    return rf"\begin{{pmatrix}} {body} \end{{pmatrix}}"


def fmt_set(values) -> str:
    return r"\{ " + ", ".join(fmt_scalar(v) for v in values) + r" \}"


def fmt_pairs(pairs: dict) -> str:
    return ", ".join(f"{k} = {fmt_scalar(v)}" for k, v in pairs.items())


# --------------------------------------------------------------- parsing --

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<frac>\\frac)
      | (?P<sqrt>\\sqrt)
      | (?P<pmatrix>\\begin\{pmatrix\})
      | (?P<endpmatrix>\\end\{pmatrix\})
      | (?P<rowsep>\\\\)
      | (?P<lbrace>\\{)
      | (?P<rbrace>\\})
      | (?P<num>\d+(?:\.\d+)?)
      | (?P<name>[A-Za-z][A-Za-z0-9_]*)
      | (?P<op>[+\-*/=,&{}\[\]().])
    )""",
    re.VERBOSE,
)


class _Lexer:
    def __init__(self, text: str):
        self.tokens: list[tuple[str, str]] = []
        pos = 0
        while pos < len(text):
            if text[pos].isspace():
                pos += 1
                continue
            m = _TOKEN.match(text, pos)
            if not m:
                raise ValueError(f"cannot tokenise at {text[pos:pos+20]!r}")
            kind = m.lastgroup
            self.tokens.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, kind):
        k, v = self.next()
        if k != kind:
            raise ValueError(f"expected {kind}, got {k} ({v!r})")
        return v


def _parse_scalar(lex: _Lexer) -> sp.Expr:
    value = _parse_term(lex)
    while True:
        k, v = lex.peek()
        if k == "op" and v in "+-":
            lex.next()
            rhs = _parse_term(lex)
            value = value + rhs if v == "+" else value - rhs
        else:
            return value


# a factor may follow another with no operator between them: 2\sqrt{3}, 3x
_IMPLICIT_FACTORS = ("frac", "sqrt", "num", "name")


def _parse_term(lex: _Lexer) -> sp.Expr:
    value = _parse_factor(lex)
    while True:
        k, v = lex.peek()
        if k == "op" and v in "*/":
            lex.next()
            rhs = _parse_factor(lex)
            value = value * rhs if v == "*" else value / rhs
        elif k in _IMPLICIT_FACTORS:
            # juxtaposition means multiplication
            value = value * _parse_factor(lex)
        else:
            return value


def _parse_factor(lex: _Lexer) -> sp.Expr:
    k, v = lex.next()
    if k == "op" and v == "-":
        return -_parse_factor(lex)
    if k == "op" and v == "+":
        return _parse_factor(lex)
    if k == "frac":
        lex.expect("op")          # '{'
        num = _parse_scalar(lex)
        lex.expect("op")          # '}'
        lex.expect("op")          # '{'
        den = _parse_scalar(lex)
        lex.expect("op")          # '}'
        return sp.cancel(num / den)
    if k == "sqrt":
        lex.expect("op")          # '{'
        inner = _parse_scalar(lex)
        lex.expect("op")          # '}'
        return sp.sqrt(inner)
    if k == "num":
        return sp.Rational(v) if "." not in v else sp.Float(v)
    if k == "name":
        if v == "i":
            return sp.I
        return sp.Symbol(v)
    raise ValueError(f"unexpected token {k} ({v!r}) in a scalar")


def _parse_matrix(lex: _Lexer) -> sp.Matrix:
    rows: list[list[sp.Expr]] = []
    row: list[sp.Expr] = []
    while True:
        k, v = lex.next()
        if k == "endpmatrix":
            break
        if k == "rowsep":
            rows.append(row)
            row = []
            continue
        if k == "op" and v == "&":
            continue
        if k == "op" and v == "{":
            row.append(_parse_scalar(lex))
            lex.expect("op")      # '}'
            continue
        # a bare entry
        if k == "num":
            row.append(sp.Rational(v) if "." not in v else sp.Float(v))
            continue
        if k == "name":
            row.append(sp.I if v == "i" else sp.Symbol(v))
            continue
        if k == "frac":
            lex.i -= 1
            row.append(_parse_factor(lex))
            continue
        if k == "sqrt":
            lex.i -= 1
            row.append(_parse_factor(lex))
            continue
        if k == "op" and v == "-":
            lex.i -= 1
            row.append(_parse_factor(lex))
            continue
        raise ValueError(f"unexpected token {k} ({v!r}) in a matrix")
    rows.append(row)
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("ragged matrix")
    return sp.Matrix(rows)


def _parse_set(lex: _Lexer) -> list[sp.Expr]:
    k, _v = lex.next()            # '{' or '\{'
    if not ((k == "op" and _v == "{") or k == "lbrace"):
        raise ValueError(f"expected '{{', got {k} ({_v!r})")
    items = []
    while True:
        k, v = lex.peek()
        if (k == "op" and v == "}") or k == "rbrace":
            lex.next()
            return items
        items.append(_parse_value(lex))
        k, v = lex.peek()
        if k == "op" and v == ",":
            lex.next()


def _parse_value(lex: _Lexer) -> Any:
    k, v = lex.peek()
    if k == "pmatrix":
        lex.next()
        return _parse_matrix(lex)
    if (k == "op" and v == "{") or k == "lbrace":
        return _parse_set(lex)
    return _parse_scalar(lex)


def parse_answer(text: str) -> Any:
    """Parse an answer string into a comparable object.

    Returns one of:
        sp.Expr                     a scalar
        sp.Matrix                   a matrix or vector
        list[sp.Expr]               a set
        dict[str, sp.Expr]          labelled components
    """
    text = text.strip()
    if not text:
        raise ValueError("empty answer")
    lex = _Lexer(text)
    value = _parse_value(lex)
    # a labelled list: "x_1 = 3, x_2 = -1"
    if isinstance(value, sp.Expr) and value.is_Symbol:
        pairs = {}
        while True:
            k, v = lex.peek()
            if k is None:
                break
            if k == "op" and v == "=":
                lex.next()
                label = str(value)
                pairs[label] = _parse_value(lex)
            elif k == "op" and v == ",":
                lex.next()
                k2, v2 = lex.peek()
                if k2 != "name":
                    raise ValueError("expected a label after ','")
                lex.next()
                value = sp.Symbol(v2)
                continue
            else:
                raise ValueError(f"trailing tokens in answer: {lex.tokens[lex.i:]}")
        if pairs:
            return pairs
    if lex.peek()[0] is not None:
        raise ValueError(f"trailing tokens in answer: {lex.tokens[lex.i:]}")
    return value


# ------------------------------------------------------------- comparison --

def _scalar_equal(a: sp.Expr, b: sp.Expr, tol: float = 1e-9) -> bool:
    try:
        d = sp.simplify(sp.expand(a - b))
        if d == 0:
            return True
        if d.is_number:
            return bool(abs(complex(d)) < tol)
    except Exception:
        pass
    try:
        return bool(sp.Abs(a - b).evalf() < tol)
    except Exception:
        return False


def answers_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    """Compare two parsed answers.

    Sets are compared as sets, so order does not matter; eigenvalues written
    in a different order still count as correct.
    """
    if isinstance(a, sp.Matrix) or isinstance(b, sp.Matrix):
        if not (isinstance(a, sp.Matrix) and isinstance(b, sp.Matrix)):
            return False
        if a.shape != b.shape:
            return False
        return all(_scalar_equal(a[i, j], b[i, j], tol)
                   for i in range(a.rows) for j in range(a.cols))
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        used = [False] * len(b)
        for x in a:
            hit = False
            for j, y in enumerate(b):
                if not used[j] and _scalar_equal(x, y, tol):
                    used[j] = True
                    hit = True
                    break
            if not hit:
                return False
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_scalar_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, dict) or isinstance(b, dict):
        return False
    return _scalar_equal(a, b, tol)


def check_answer(answer_latex: str, truth: Any, tol: float = 1e-9) -> dict:
    """Check a model's answer against a ground truth value.

    `truth` may be a parsed object or a string to be parsed.
    """
    try:
        got = parse_answer(answer_latex)
    except Exception as exc:
        return {"correct": False, "reason": f"unparseable: {exc}"}
    if isinstance(truth, str):
        try:
            truth = parse_answer(truth)
        except Exception as exc:
            return {"correct": False, "reason": f"bad truth: {exc}"}
    ok = answers_equal(got, truth, tol)
    return {"correct": bool(ok), "reason": "" if ok else "value mismatch"}
