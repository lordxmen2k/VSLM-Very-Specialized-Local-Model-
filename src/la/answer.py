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
    matrix   := '\\begin{pmatrix}'|'\\begin{bmatrix}' row ('\\\\' row)*
                ('\\end{pmatrix}'|'\\end{bmatrix}')
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

import zlib

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


def fmt_scalar(expr) -> str:
    """Print a value in the answer grammar.

    A matrix is not a scalar but it is printable here, so that a pair such as
    "L = ..., U = ..." can be written by one function.
    """
    if isinstance(expr, sp.Matrix):
        return fmt_matrix(expr)
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
        # The sign is carried by the real numeric coefficient. sympy's own
        # could_extract_minus_sign recurses on a product that contains the
        # imaginary unit, so the arithmetic is done here instead.
        coef = sp.Integer(1)
        rest = []
        for a in sp.Mul.make_args(expr):
            # Only a RATIONAL factor becomes part of the coefficient. sqrt(3)
            # is a number and is real, but folding it into the coefficient
            # leaves the coefficient as a product again and the printer calls
            # itself for ever.
            if a.is_Rational:
                coef *= a
            else:
                rest.append(a)
        lead = ""
        if coef < 0:
            lead = "-"
            coef = -coef
        if coef != 1:
            rest.insert(0, coef)
        rendered = []
        for i, a in enumerate(rest):
            s = fmt_scalar(a)
            if i > 0 and (s.startswith(r"\sqrt") or s.startswith(r"\frac")):
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


def matrix_environment(body: str) -> str:
    """Choose the LaTeX environment a matrix is written in.

    LinAlg-Bench writes every matrix as \\begin{bmatrix}. Most textbooks,
    and this corpus originally, use \\begin{pmatrix}. A model that has only
    ever seen one of them meets the other for the first time on benchmark
    day, which is a distribution shift nobody needs.

    The choice is derived from the contents of the matrix, so the same matrix
    is always written the same way within an example - the question and the
    scratchpad agree - while different matrices in the same example may use
    different environments. That variety is the point: the model should be
    indifferent to the environment.
    """
    return "bmatrix" if (zlib.crc32(body.encode("utf-8")) >> 16) & 1 else "pmatrix"


def fmt_matrix(M: sp.Matrix) -> str:
    rows = []
    for i in range(M.rows):
        rows.append(" & ".join(fmt_scalar(M[i, j]) for j in range(M.cols)))
    body = r" \\ ".join(rows)
    env = matrix_environment(body)
    return rf"\begin{{{env}}} {body} \end{{{env}}}"


def fmt_set(values) -> str:
    return r"\{ " + ", ".join(fmt_scalar(v) for v in values) + r" \}"


def fmt_pairs(pairs: dict) -> str:
    return ", ".join(f"{k} = {fmt_scalar(v)}" for k, v in pairs.items())


# --------------------------------------------------------------- parsing --

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<frac>\\frac)
      | (?P<sqrt>\\sqrt)
      | (?P<mbegin>\\begin\{(?:p|b)matrix\})
      | (?P<mend>\\end\{(?:p|b)matrix\})
      | (?P<rowsep>\\\\)
      | (?P<lbrace>\\{)
      | (?P<rbrace>\\})
      | (?P<num>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
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
        if k == "mend":
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
    if k == "mbegin":
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
        list[sp.Expr]               a set, or a bare list of values
        dict[str, sp.Expr]          labelled components
    """
    text = text.strip()
    if not text:
        raise ValueError("empty answer")
    lex = _Lexer(text)
    value = _parse_value(lex)

    # a bare list: "-1, 0, 3". LinAlg-Bench writes the eigenvalues this way,
    # and a grammar that cannot read it scores every eigenvalue problem zero
    # however well the model does.
    if isinstance(value, sp.Expr) and not value.is_Symbol:
        items = [value]
        while True:
            k, v = lex.peek()
            if k == "op" and v == ",":
                lex.next()
                items.append(_parse_value(lex))
                continue
            break
        if len(items) > 1:
            if lex.peek()[0] is not None:
                raise ValueError(f"trailing tokens in answer: {lex.tokens[lex.i:]}")
            return items

    # a labelled list or a labelled value: "x_1 = 3, x_2 = -1" or "λ = 1, 2"
    if isinstance(value, sp.Expr) and value.is_Symbol:
        pairs = {}
        while True:
            k, v = lex.peek()
            if k is None:
                break
            if k == "op" and v == "=":
                lex.next()
                label = str(value)
                first = _parse_value(lex)
                pairs[label] = first
                # the label may cover a whole list rather than one value
                items = [first]
                while True:
                    k2, v2 = lex.peek()
                    if k2 == "op" and v2 == ",":
                        lex.next()
                        k3, v3 = lex.peek()
                        if k3 == "name":
                            # a second labelled pair, not a list
                            lex.next()
                            value = sp.Symbol(v3)
                            break
                        items.append(_parse_value(lex))
                        continue
                    break
                if len(items) > 1:
                    if lex.peek()[0] is not None:
                        raise ValueError(
                            f"trailing tokens in answer: {lex.tokens[lex.i:]}")
                    return items
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

def _vec_equal(u, v, tol: float) -> bool:
    """Element-wise comparison of two column vectors."""
    if u.shape != v.shape:
        return False
    return all(_scalar_equal(u[i, 0], v[i, 0], tol) for i in range(u.rows))


def _mat_equal(u: sp.Matrix, v: sp.Matrix, tol: float) -> bool:
    if u.shape != v.shape:
        return False
    return all(_scalar_equal(u[i, j], v[i, j], tol)
               for i in range(u.rows) for j in range(u.cols))


def _scalar_equal(a, b, tol: float = 1e-9) -> bool:
    if isinstance(a, sp.Matrix) and isinstance(b, sp.Matrix):
        return _mat_equal(a, b, tol)
    if isinstance(a, sp.Matrix) or isinstance(b, sp.Matrix):
        return False
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


def answers_equal(a: Any, b: Any, tol: float = 1e-9, mode: str = "exact") -> bool:
    """Compare two parsed answers.

    Sets are compared as sets, so order does not matter; eigenvalues written
    in a different order still count as correct.

    `mode="basis"` is for an orthonormal basis, which is only determined up to
    the sign of each vector and the orientation of the set. A basis is
    therefore compared column by column, each column up to sign, and either
    orientation of the other matrix is accepted.
    """
    if mode == "basis":
        return _basis_equal(a, b, tol)
    if mode == "diag":
        return _diag_equal(a, b, tol)
    if isinstance(a, sp.Matrix) or isinstance(b, sp.Matrix):
        if not (isinstance(a, sp.Matrix) and isinstance(b, sp.Matrix)):
            return False
        return _mat_equal(a, b, tol)
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


def _diag_equal(a: Any, b: Any, tol: float) -> bool:
    """Compare two diagonal matrices by their diagonal entries, in any order."""
    if not (isinstance(a, sp.Matrix) and isinstance(b, sp.Matrix)):
        return answers_equal(a, b, tol)
    if a.shape != b.shape:
        return False
    da = sorted([sp.nsimplify(a[i, i]) for i in range(a.rows)],
                key=lambda e: float(sp.re(e)) if e.is_number else 0.0)
    db = sorted([sp.nsimplify(b[i, i]) for i in range(b.rows)],
                key=lambda e: float(sp.re(e)) if e.is_number else 0.0)
    return all(_scalar_equal(x, y, tol) for x, y in zip(da, db))


def _basis_equal(a: Any, b: Any, tol: float) -> bool:
    if not (isinstance(a, sp.Matrix) and isinstance(b, sp.Matrix)):
        return answers_equal(a, b, tol)
    if a.shape != b.shape:
        return False
    for other in (b, b.T):
        if other.shape != a.shape:
            continue
        used = [False] * other.cols
        ok = True
        for i in range(a.cols):
            hit = False
            for j in range(other.cols):
                if used[j]:
                    continue
                v = other[:, j]
                if _vec_equal(a[:, i], v, tol) or _vec_equal(a[:, i], -v, tol):
                    used[j] = True
                    hit = True
                    break
            if not hit:
                ok = False
                break
        if ok and all(used):
            return True
    return False


def strip_boxed(text: str) -> str:
    """Return the contents of a \\boxed{...}, with the braces balanced.

    A model trained on this corpus answers inside \\boxed{}, because the
    benchmark asks for it that way. A \\boxed{} around a matrix contains its
    own braces, so the closing brace cannot be found by counting naively.
    """
    marker = "\\boxed{"
    start = text.find(marker)
    if start < 0:
        return text
    i = start + len(marker)
    depth = 1
    while i < len(text) and depth:
        c = text[i]
        if c in "{}" and (i == 0 or text[i - 1] != "\\"):
            depth += 1 if c == "{" else -1
        i += 1
    if depth:
        return text
    return text[start + len(marker): i - 1]


def check_answer(answer_latex: str, truth: Any, tol: float = 1e-9,
                 mode: str = "exact") -> dict:
    """Check a model's answer against a ground truth value.

    `truth` may be a parsed object or a string to be parsed.
    """
    try:
        got = parse_answer(strip_boxed(answer_latex))
    except Exception as exc:
        return {"correct": False, "reason": f"unparseable: {exc}"}
    if isinstance(truth, str):
        try:
            truth = parse_answer(truth)
        except Exception as exc:
            return {"correct": False, "reason": f"bad truth: {exc}"}
    ok = answers_equal(got, truth, tol, mode=mode)
    return {"correct": bool(ok), "reason": "" if ok else "value mismatch"}
