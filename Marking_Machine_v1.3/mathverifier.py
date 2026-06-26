from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import sympy as sp
from sympy import Eq, Matrix, simplify
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


@dataclass
class ValidationResult:
    step_index: int
    previous_step: str
    current_step: str
    valid: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedStep:
    raw: str
    kind: str
    value: Any
    symbols: List[sp.Symbol] = field(default_factory=list)


def format_results(results: List[ValidationResult], show_details: bool = True) -> str:
    """Generates a readable verification report from ValidationResults.

    show_details defaults to True.
    """
    if not results:
        return "No verification results available."

    total = len(results)
    valid_count = sum(r.valid for r in results)
    invalid_count = total - valid_count

    lines = [
        "Math Step Verification Report",
        "=" * 34,
        f"Total transitions checked: {total}",
        f"Valid: {valid_count}",
        f"Invalid: {invalid_count}",
        "",
    ]

    for r in results:
        icon = "✅" if r.valid else "❌"
        status = "Valid" if r.valid else "Invalid"

        lines.extend([
            f"{icon} Step {r.step_index}: {status}",
            f"   Previous: {r.previous_step}",
            f"   Current : {r.current_step}",
            f"   Reason  : {r.reason}",
        ])

        if show_details and r.details:
            lines.append("   Details :")
            for key, value in r.details.items():
                lines.append(f"      - {key}: {value}")

        lines.append("")

    if invalid_count == 0:
        lines.append("Overall result: ✅ All steps are mathematically valid.")
    else:
        lines.append("Overall result: ❌ Some steps may contain errors.")

    return "\n".join(lines)


class MathStepVerifier:
    def __init__(
        self,
        allow_numeric_substitution: bool = True,
        allow_matrix_row_operations: bool = True,
    ):
        self.allow_numeric_substitution = allow_numeric_substitution

        # Row operations are enabled by default.
        # _validate_matrix_to_matrix below ALWAYS checks row operations.
        self.allow_matrix_row_operations = allow_matrix_row_operations

        self.validators: Dict[
            Tuple[str, str],
            Callable[[ParsedStep, ParsedStep], Tuple[bool, str, Dict[str, Any]]]
        ] = {
            ("expr", "or_eq"): self._validate_expr_to_or_eq,
            ("expr", "expr"): self._validate_expr_to_expr,
            ("eq", "eq"): self._validate_eq_to_eq,
            ("eq", "or_eq"): self._validate_eq_to_or_eq,
            ("or_eq", "or_eq"): self._validate_or_eq_to_or_eq,
            ("or_eq", "eq"): self._validate_or_eq_to_eq,
            ("expr", "eq"): self._validate_expr_to_eq,
            ("eq", "expr"): self._validate_eq_to_expr,
            ("boolean", "boolean"): self._validate_boolean_to_boolean,
            ("matrix", "matrix"): self._validate_matrix_to_matrix,
            ("derivative", "expr"): self._validate_derivative_to_expr,
            ("derivative", "eq"): self._validate_derivative_to_eq,
            ("integral", "expr"): self._validate_integral_to_expr,
            ("integral", "eq"): self._validate_integral_to_eq,
            ("system", "system"): self._validate_system_to_system,
            ("chain_eq", "eq"): self._validate_chain_eq_to_eq,
            ("chain_eq", "chain_eq"): self._validate_chain_eq_to_chain_eq,
            ("eq", "chain_eq"): self._validate_eq_to_chain_eq,
        }

    def verify_steps(self, latex_steps: Union[Sequence[str], str]) -> List[ValidationResult]:
        if isinstance(latex_steps, str):
            latex_steps = [
                part.strip()
                for part in latex_steps.split(",")
                if part.strip()
            ]

        if len(latex_steps) < 2:
            return []

        parsed_steps = [self.parse_step(s) for s in latex_steps]
        results: List[ValidationResult] = []

        for i in range(1, len(parsed_steps)):
            prev_step = parsed_steps[i - 1]
            curr_step = parsed_steps[i]

            valid, reason, details = self._validate_pair(prev_step, curr_step)

            results.append(
                ValidationResult(
                    step_index=i,
                    previous_step=prev_step.raw,
                    current_step=curr_step.raw,
                    valid=valid,
                    reason=reason,
                    details=details,
                )
            )

        return results

    def parse_step(self, latex: str) -> ParsedStep:
        s = self._normalize_latex(latex)

        if self._looks_like_system(s):
            system_eqs = self._parse_system(s)
            syms = sorted(
                self._extract_symbols_from_sequence(system_eqs),
                key=lambda x: x.name
            )
            return ParsedStep(raw=latex, kind="system", value=system_eqs, symbols=syms)

        if self._looks_like_or_equation(s):
            equations = self._parse_or_equations(s)
            syms = sorted(
                self._extract_symbols_from_sequence(equations),
                key=lambda x: x.name
            )
            return ParsedStep(raw=latex, kind="or_eq", value=equations, symbols=syms)

        if self._count_top_level_equals(s) > 1:
            try:
                parts = self._split_all_top_level_equals(s)
                parsed_parts = [self._parse_expression(p) for p in parts]
                syms = sorted(
                    list(self._extract_symbols_from_sequence(parsed_parts)),
                    key=lambda x: x.name
                )
                return ParsedStep(raw=latex, kind="chain_eq", value=parsed_parts, symbols=syms)
            except Exception:
                pass

        if self._is_single_equation(s):
            eq = self._parse_equation(s)
            syms = sorted(list(eq.free_symbols), key=lambda x: x.name)
            return ParsedStep(raw=latex, kind="eq", value=eq, symbols=syms)

        if self._looks_like_matrix(s):
            matrix_expr = self._parse_matrix(s)
            syms = sorted(
                list(getattr(matrix_expr, "free_symbols", set())),
                key=lambda x: x.name
            )
            return ParsedStep(raw=latex, kind="matrix", value=matrix_expr, symbols=syms)

        expr = self._parse_expression(s)
        syms = sorted(list(getattr(expr, "free_symbols", set())), key=lambda x: x.name)
        kind = self._classify_expr(expr)

        return ParsedStep(raw=latex, kind=kind, value=expr, symbols=syms)

    def _normalize_latex(self, s: str) -> str:
        s = (s or "").strip()

        s = s.replace("\r\n", "\n").replace("\r", "\n")

        wrapper_patterns = [
            r"^\s*\$\$(.*?)\$\$\s*$",
            r"^\s*\$(.*?)\$\s*$",
            r"^\s*\\$$(.*?)\\$$\s*$",
            r"^\s*\\$$(.*?)\\$$\s*$",
        ]

        changed = True
        while changed:
            changed = False
            for pat in wrapper_patterns:
                m = re.match(pat, s, flags=re.DOTALL)
                if m:
                    s = m.group(1).strip()
                    changed = True

        s = s.replace("\n", " ")
        s = re.sub(r"\s+", " ", s).strip()

        s = re.sub(
            r"\\frac\{d\}\{d\s*([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)\}",
            r"\\frac{d}{d \1}",
            s
        )

        s = re.sub(
            r"\\frac\{\\partial\}\{\\partial\s*([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)\}",
            r"\\frac{\\partial}{\\partial \1}",
            s
        )

        s = re.sub(
            r"\\Delta\s*([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)",
            r"Delta\1",
            s
        )

        s = re.sub(
            r"(?<=[0-9}$$$$])\s*(?=(?:\\left\s*)?$$)",
            r" \\cdot ",
            s
        )

        s = re.sub(
            r"(\\frac\{d\}\{d\s*[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?\}|"
            r"\\frac\{\\partial\}\{\\partial\s*[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?\})\s*\\cdot\s*(?=(?:\\left\s*)?$$)",
            r"\1 ",
            s
        )

        s = re.sub(r"\s*\\cdot\s*", r" \\cdot ", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Remove dangling trailing operators that can appear after normalization/input mistakes.
        # Example:
        #   "\begin{bmatrix}1&2\\3&4\end{bmatrix} \cdot"
        # becomes:
        #   "\begin{bmatrix}1&2\\3&4\end{bmatrix}"
        s = re.sub(r"\s*(\\cdot|\\times|\+|-|/|\*)\s*$", "", s).strip()

        return s

    def _looks_like_system(self, s: str) -> bool:
        return "\\begin{cases}" in s and "\\end{cases}" in s

    def _parse_system(self, s: str) -> List[Eq]:
        match = re.search(r"\\begin\{cases\}(.*?)\\end\{cases\}", s, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse system of equations: {s}")

        content = match.group(1).strip()
        rows = [row.strip() for row in content.split("\\\\") if row.strip()]

        equations = []
        for row in rows:
            row = row.replace("&", "").strip()
            if "=" in row:
                equations.append(self._parse_equation(row))

        if not equations:
            raise ValueError("No equations found inside cases environment.")

        return equations

    def _looks_like_or_equation(self, s: str) -> bool:
        return (
            "\\text{ or }" in s
            or "\\text{or}" in s
            or "\\lor" in s
            or re.search(r"\s+or\s+", s) is not None
        ) and "=" in s

    def _parse_or_equations(self, s: str) -> List[Eq]:
        parts = re.split(r"\\text\{\s*or\s*\}|\\lor|\s+or\s+", s)
        equations: List[Eq] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Each 'or' part must be an equation. Problem part: {part}")
            equations.append(self._parse_equation(part))

        if len(equations) < 2:
            raise ValueError(f"Could not parse 'or' equation statement: {s}")

        return equations

    def _looks_like_matrix(self, s: str) -> bool:
        return any(env in s for env in [
            "\\begin{bmatrix}",
            "\\begin{pmatrix}",
            "\\begin{matrix}",
            "\\begin{Bmatrix}",
            "\\begin{vmatrix}",
        ])

    def _is_single_equation(self, s: str) -> bool:
        return self._count_top_level_equals(s) == 1

    def _count_top_level_equals(self, s: str) -> int:
        depth = 0
        count = 0
        i = 0

        while i < len(s):
            if s.startswith(r"\begin{", i):
                end = s.find("}", i + len(r"\begin{"))
                if end != -1:
                    depth += 1
                    i = end + 1
                    continue

            if s.startswith(r"\end{", i):
                end = s.find("}", i + len(r"\end{"))
                if end != -1:
                    depth = max(0, depth - 1)
                    i = end + 1
                    continue

            ch = s[i]
            if ch in "({[":
                depth += 1
            elif ch in ")}]":
                depth = max(0, depth - 1)
            elif ch == "=" and depth == 0:
                count += 1

            i += 1

        return count

    def _split_all_top_level_equals(self, s: str) -> List[str]:
        parts = []
        depth = 0
        last_idx = 0
        i = 0

        while i < len(s):
            if s.startswith(r"\begin{", i):
                end = s.find("}", i + len(r"\begin{"))
                if end != -1:
                    depth += 1
                    i = end + 1
                    continue

            if s.startswith(r"\end{", i):
                end = s.find("}", i + len(r"\end{"))
                if end != -1:
                    depth = max(0, depth - 1)
                    i = end + 1
                    continue

            ch = s[i]
            if ch in "({[":
                depth += 1
            elif ch in ")}]":
                depth = max(0, depth - 1)
            elif ch == "=" and depth == 0:
                parts.append(s[last_idx:i].strip())
                last_idx = i + 1

            i += 1

        parts.append(s[last_idx:].strip())
        return parts

    def _split_top_level_equation(self, s: str) -> Tuple[str, str]:
        parts = self._split_all_top_level_equals(s)
        if len(parts) >= 2:
            return parts[0], parts[1]
        raise ValueError(f"Could not split equation: {s}")

    def _parse_equation(self, s: str) -> Eq:
        lhs_s, rhs_s = self._split_top_level_equation(s)
        lhs = self._parse_expression(lhs_s)
        rhs = self._parse_expression(rhs_s)
        return Eq(lhs, rhs)

    def _parse_expression(self, s: str) -> Any:
        if self._looks_like_matrix(s):
            return self._parse_matrix(s)
        return parse_latex(s)

    def _parse_matrix(self, s: str) -> Any:
        """
        Parses a single LaTeX matrix or a full matrix expression.
        It replaces every LaTeX matrix environment with Matrix(...)
        and then parses/evaluates the whole expression.

        Also safely removes dangling trailing operators like:
            \cdot
            \times
            +
            -
            *
            /
        """

        matrix_env_pattern = re.compile(
            r"\\begin\{(bmatrix|pmatrix|matrix|Bmatrix|vmatrix)\}(.*?)\\end\{\1\}",
            re.DOTALL
        )

        def convert_matrix(match: re.Match) -> str:
            env = match.group(1)
            content = match.group(2).strip()

            rows = [row.strip() for row in content.split("\\\\") if row.strip()]
            data = []

            for row in rows:
                cols = [c.strip() for c in row.split("&")]
                parsed_cols = []

                for c in cols:
                    if not c:
                        raise ValueError(f"Empty matrix entry in: {s}")
                    parsed_cols.append(self._parse_expression(c))

                data.append(parsed_cols)

            widths = {len(r) for r in data}
            if len(widths) != 1:
                raise ValueError("Matrix rows have inconsistent lengths.")

            if env == "vmatrix":
                return f"Matrix({data}).det()"

            return f"Matrix({data})"

        # Remove accidental trailing LaTeX/Python-style operators before conversion.
        s = re.sub(r"\s*(\\cdot|\\times|\+|-|/|\*)\s*$", "", s).strip()

        converted = matrix_env_pattern.sub(convert_matrix, s)

        converted = converted.replace(r"\cdot", "*")
        converted = converted.replace(r"\times", "*")
        converted = converted.replace(r"\left", "")
        converted = converted.replace(r"\right", "")

        converted = re.sub(r"\s+", " ", converted).strip()

        # Remove dangling Python-style operators after conversion too.
        converted = re.sub(r"\s*(\*|\+|-|/)\s*$", "", converted).strip()

        transformations = standard_transformations + (implicit_multiplication_application,)

        try:
            return parse_expr(
                converted,
                local_dict={"Matrix": Matrix},
                global_dict=sp.__dict__,
                transformations=transformations,
                evaluate=True,
            )
        except Exception as e:
            raise ValueError(
                f"Could not parse matrix expression: {s}. "
                f"Converted form: {converted}. Error: {e}"
            )

    def _classify_expr(self, expr: Any) -> str:
        if isinstance(expr, Relational):
            return "eq"
        if isinstance(expr, Boolean):
            return "boolean"
        if isinstance(expr, sp.Derivative):
            return "derivative"
        if isinstance(expr, sp.Integral):
            return "integral"
        if isinstance(expr, sp.MatrixBase):
            return "matrix"
        return "expr"

    def _extract_symbols_from_sequence(self, objs: Sequence[Any]) -> set:
        out = set()
        for obj in objs:
            out |= set(getattr(obj, "free_symbols", set()))
        return out

    def _expressions_numerically_close(self, a: Any, b: Any, rel_tol=5e-3, abs_tol=1e-9) -> bool:
        try:
            a_f = complex(sp.N(a))
            b_f = complex(sp.N(b))

            real_close = math.isclose(a_f.real, b_f.real, rel_tol=rel_tol, abs_tol=abs_tol)
            imag_close = math.isclose(a_f.imag, b_f.imag, rel_tol=rel_tol, abs_tol=abs_tol)

            return real_close and imag_close
        except Exception:
            return False

    def _expressions_equivalent(self, a: Any, b: Any) -> bool:
        if type(a) != type(b) and (isinstance(a, list) or isinstance(b, list)):
            return False

        if isinstance(a, sp.MatrixBase) and isinstance(b, sp.MatrixBase):
            if a.shape != b.shape:
                return False

            try:
                diff = a - b
                return all(simplify(entry) == 0 for entry in diff)
            except Exception:
                try:
                    return bool(a.equals(b))
                except Exception:
                    return False

        try:
            diff = simplify(a - b)
            if diff == 0:
                return True
        except Exception:
            pass

        try:
            if bool(a.equals(b)):
                return True
        except Exception:
            pass

        if self.allow_numeric_substitution:
            if len(getattr(a, "free_symbols", set())) == 0 and len(getattr(b, "free_symbols", set())) == 0:
                if self._expressions_numerically_close(a, b):
                    return True

        return False

    def _solution_sets_equivalent(self, sol1: sp.Set, sol2: sp.Set) -> bool:
        if sol1 == sol2:
            return True

        if isinstance(sol1, sp.FiniteSet) and isinstance(sol2, sp.FiniteSet):
            if len(sol1) != len(sol2):
                return False

            list1 = list(sol1)
            list2 = list(sol2)

            for perm in itertools.permutations(list2):
                match_all = True
                for a, b in zip(list1, perm):
                    if not self._expressions_numerically_close(a, b) and not self._expressions_equivalent(a, b):
                        match_all = False
                        break
                if match_all:
                    return True

        return False

    def _system_solutions_equivalent(self, sol1: set, sol2: set) -> bool:
        if sol1 == sol2:
            return True

        if len(sol1) != len(sol2):
            return False

        list1 = list(sol1)
        list2 = list(sol2)

        for perm in itertools.permutations(list2):
            match_all = True

            for fs1, fs2 in zip(list1, perm):
                dict1 = dict(fs1)
                dict2 = dict(fs2)

                if dict1.keys() != dict2.keys():
                    match_all = False
                    break

                for k in dict1:
                    if not self._expressions_numerically_close(dict1[k], dict2[k]) and not self._expressions_equivalent(dict1[k], dict2[k]):
                        match_all = False
                        break

                if not match_all:
                    break

            if match_all:
                return True

        return False

    def _validate_pair(self, prev_step: ParsedStep, curr_step: ParsedStep) -> Tuple[bool, str, Dict[str, Any]]:
        key = (prev_step.kind, curr_step.kind)

        if key in self.validators:
            return self.validators[key](prev_step, curr_step)

        if prev_step.kind not in ("chain_eq",) and curr_step.kind not in ("chain_eq",):
            try:
                if self._expressions_equivalent(prev_step.value, curr_step.value):
                    return True, "Equivalent by symbolic simplification.", {}
            except Exception:
                pass

        return (
            False,
            f"Transition from {prev_step.kind!r} to {curr_step.kind!r} is not supported or mathematically invalid.",
            {"from_kind": prev_step.kind, "to_kind": curr_step.kind},
        )

    def _are_multivariable_equations_equivalent(self, eq1: Eq, eq2: Eq) -> bool:
        syms = eq1.free_symbols.intersection(eq2.free_symbols)
        if not syms:
            return False

        e1 = eq1.lhs - eq1.rhs
        e2 = eq2.lhs - eq2.rhs

        for sym in syms:
            try:
                sols1 = sp.solve(e1, sym)
                sols2 = sp.solve(e2, sym)
                if sols1 and sols2:
                    if isinstance(sols1, list) and isinstance(sols2, list):
                        try:
                            if set(sols1) == set(sols2):
                                return True
                        except TypeError:
                            if len(sols1) == len(sols2) and all(s in sols2 for s in sols1):
                                return True
            except Exception:
                continue

        return False

    def _equations_identical(self, eq1: Eq, eq2: Eq) -> bool:
        if (
            isinstance(eq1.lhs, sp.MatrixBase)
            or isinstance(eq1.rhs, sp.MatrixBase)
            or isinstance(eq2.lhs, sp.MatrixBase)
            or isinstance(eq2.rhs, sp.MatrixBase)
        ):
            return (
                self._expressions_equivalent(eq1.lhs, eq2.lhs)
                and self._expressions_equivalent(eq1.rhs, eq2.rhs)
            ) or (
                self._expressions_equivalent(eq1.lhs, eq1.rhs)
                and self._expressions_equivalent(eq2.lhs, eq2.rhs)
            )

        e1 = simplify(eq1.lhs - eq1.rhs)
        e2 = simplify(eq2.lhs - eq2.rhs)

        if simplify(e1 - e2) == 0:
            return True

        try:
            ratio = simplify(e2 / e1)
            if ratio.free_symbols == set() and ratio != 0:
                return True
        except Exception:
            pass

        if self._are_multivariable_equations_equivalent(eq1, eq2):
            return True

        return False

    def _equation_solution_set(self, eq: Eq):
        syms = sorted(list(eq.free_symbols), key=lambda x: x.name)
        if len(syms) != 1:
            raise ValueError("Only single-variable equation solution sets are supported.")

        var = syms[0]
        return sp.solveset(eq.lhs - eq.rhs, var, domain=sp.S.Complexes)

    def _or_equations_solution_set(self, equations: List[Eq]):
        if not equations:
            raise ValueError("No equations found in 'or' statement.")

        symbols_in_all = sorted(
            list(self._extract_symbols_from_sequence(equations)),
            key=lambda x: x.name
        )

        if len(symbols_in_all) != 1:
            raise ValueError("Only single-variable 'or' solution statements are supported.")

        combined = sp.EmptySet

        for eq in equations:
            combined = combined.union(self._equation_solution_set(eq))

        return combined

    def _is_numeric_expression(self, expr: Any) -> bool:
        try:
            expr = sp.sympify(expr)
            return len(expr.free_symbols) == 0
        except Exception:
            return False

    def _is_substitution_instance(self, template: Any, candidate: Any) -> Tuple[bool, Dict[str, Any]]:
        if isinstance(template, sp.MatrixBase) or isinstance(candidate, sp.MatrixBase):
            if self._expressions_equivalent(template, candidate):
                return True, {}
            return False, {}

        try:
            template = sp.sympify(template)
            candidate = sp.sympify(candidate)
        except Exception:
            return False, {}

        if self._expressions_equivalent(template, candidate):
            return True, {}

        template_symbols = sorted(list(template.free_symbols), key=lambda x: x.name)
        if not template_symbols:
            return False, {}

        try:
            wild_map = {sym: sp.Wild(f"{sym.name}_wild") for sym in template_symbols}
            wild_template = template.xreplace(wild_map)
            match = candidate.match(wild_template)

            if match is not None:
                substitutions = {}

                for sym, wild in wild_map.items():
                    if wild in match:
                        substitutions[sym] = match[wild]

                if substitutions:
                    substituted = template.subs(substitutions)
                    if self._expressions_equivalent(substituted, candidate):
                        return True, {str(k): str(v) for k, v in substitutions.items()}
        except Exception:
            pass

        if self.allow_numeric_substitution:
            if self._is_numeric_expression(candidate):
                return True, {str(sym): "numeric value" for sym in template_symbols}

        try:
            candidate_symbols = set(candidate.free_symbols)
            if candidate_symbols.issubset(set(template_symbols)):
                removed_symbols = set(template_symbols) - candidate_symbols

                if removed_symbols:
                    return True, {
                        str(sym): "substituted expression or value"
                        for sym in sorted(removed_symbols, key=lambda x: x.name)
                    }
        except Exception:
            pass

        return False, {}

    def _equation_is_substitution_instance(self, previous_eq: Eq, current_eq: Eq) -> Tuple[bool, Dict[str, Any]]:
        lhs_same = self._expressions_equivalent(previous_eq.lhs, current_eq.lhs)
        rhs_same = self._expressions_equivalent(previous_eq.rhs, current_eq.rhs)

        lhs_sub_ok, lhs_subs = self._is_substitution_instance(previous_eq.lhs, current_eq.lhs)
        rhs_sub_ok, rhs_subs = self._is_substitution_instance(previous_eq.rhs, current_eq.rhs)

        valid_lhs = lhs_same or lhs_sub_ok
        valid_rhs = rhs_same or rhs_sub_ok

        if valid_lhs and valid_rhs:
            substitutions = {}
            if lhs_subs:
                substitutions["lhs"] = lhs_subs
            if rhs_subs:
                substitutions["rhs"] = rhs_subs
            return True, substitutions

        return False, {
            "lhs": {"same": lhs_same, "substitution_valid": lhs_sub_ok, "substitutions": lhs_subs},
            "rhs": {"same": rhs_same, "substitution_valid": rhs_sub_ok, "substitutions": rhs_subs},
        }

    def _system_solution(self, equations: List[Eq]) -> set:
        exprs = [eq.lhs - eq.rhs for eq in equations]
        sols = sp.solve(exprs, dict=True)
        return set(frozenset((k, simplify(v)) for k, v in s.items()) for s in sols)

    def _validate_expr_to_or_eq(self, prev: ParsedStep, curr: ParsedStep):
        try:
            syms = sorted(list(prev.value.free_symbols), key=lambda x: x.name)
            if len(syms) != 1:
                return False, "Expression-to-'or' validation requires a single variable expression interpreted as = 0.", {}

            var = syms[0]
            prev_solution_set = sp.solveset(prev.value, var, domain=sp.S.Complexes)
            curr_solution_set = self._or_equations_solution_set(curr.value)

            if self._solution_sets_equivalent(prev_solution_set, curr_solution_set):
                return True, "The 'or' statement matches the solution set of the expression interpreted as = 0.", {
                    "previous_solution_set": str(prev_solution_set),
                    "current_solution_set": str(curr_solution_set),
                }

            return False, "The 'or' statement does not match the expression's =0 solution set.", {
                "previous_solution_set": str(prev_solution_set),
                "current_solution_set": str(curr_solution_set),
            }
        except Exception as e:
            return False, f"Could not compare expression with 'or' statement: {e}", {}

    def _validate_expr_to_expr(self, prev: ParsedStep, curr: ParsedStep):
        if self._expressions_equivalent(prev.value, curr.value):
            return True, "Expression was correctly simplified or rearranged algebraically.", {}

        substitution_ok, substitution_details = self._is_substitution_instance(prev.value, curr.value)

        if substitution_ok:
            return True, "Expression correctly substituted known values or sub-expressions.", {
                "substitutions": substitution_details
            }

        return False, "Expressions are not mathematically equivalent. Check for algebraic or arithmetic errors.", {
            "prev_simplified": str(simplify(prev.value)),
            "curr_simplified": str(simplify(curr.value)),
            "substitution_check": substitution_details,
        }

    def _validate_eq_to_eq(self, prev: ParsedStep, curr: ParsedStep):
        eq1: Eq = prev.value
        eq2: Eq = curr.value

        if self._equations_identical(eq1, eq2):
            return True, "The equations are symbolically identical or equivalent.", {}

        substitution_ok, substitution_details = self._equation_is_substitution_instance(eq1, eq2)

        if substitution_ok:
            return True, "Equation step is justified by valid variable substitution or evaluation.", {
                "substitutions": substitution_details
            }

        try:
            sol1 = self._equation_solution_set(eq1)
            sol2 = self._equation_solution_set(eq2)

            if self._solution_sets_equivalent(sol1, sol2):
                return True, "Both single-variable equations share equivalent solution sets.", {
                    "previous_solution_set": str(sol1),
                    "current_solution_set": str(sol2),
                }

            return False, "The solution sets differ. The transformation changed the mathematical meaning.", {
                "previous_solution_set": str(sol1),
                "current_solution_set": str(sol2),
            }
        except Exception:
            pass

        previous_normal_form = str(eq1)
        current_normal_form = str(eq2)

        try:
            if not isinstance(eq1.lhs, sp.MatrixBase) and not isinstance(eq1.rhs, sp.MatrixBase):
                previous_normal_form = str(simplify(eq1.lhs - eq1.rhs))
        except Exception:
            pass

        try:
            if not isinstance(eq2.lhs, sp.MatrixBase) and not isinstance(eq2.rhs, sp.MatrixBase):
                current_normal_form = str(simplify(eq2.lhs - eq2.rhs))
        except Exception:
            pass

        return False, "Equation transformation is neither logically equivalent nor a valid substitution. Ensure algebraic operations are applied equally to both sides.", {
            "previous_normal_form": previous_normal_form,
            "current_normal_form": current_normal_form,
            "substitution_check": substitution_details,
        }

    def _validate_eq_to_or_eq(self, prev: ParsedStep, curr: ParsedStep):
        try:
            prev_solution_set = self._equation_solution_set(prev.value)
            curr_solution_set = self._or_equations_solution_set(curr.value)

            if self._solution_sets_equivalent(prev_solution_set, curr_solution_set):
                return True, "The 'or' statement precisely captures the exact solution set of the previous equation.", {
                    "previous_solution_set": str(prev_solution_set),
                    "current_solution_set": str(curr_solution_set),
                }

            return False, "The 'or' statement's solution set does not logically match the equation's true solution set.", {
                "previous_solution_set": str(prev_solution_set),
                "current_solution_set": str(curr_solution_set),
            }
        except Exception as e:
            return False, f"Could not mathematically compare equation with 'or' solution statement: {e}", {}

    def _validate_or_eq_to_or_eq(self, prev: ParsedStep, curr: ParsedStep):
        try:
            prev_solution_set = self._or_equations_solution_set(prev.value)
            curr_solution_set = self._or_equations_solution_set(curr.value)

            if self._solution_sets_equivalent(prev_solution_set, curr_solution_set):
                return True, "Both 'or' solution statements have the exact same valid solution set.", {
                    "previous_solution_set": str(prev_solution_set),
                    "current_solution_set": str(curr_solution_set),
                }

            return False, "The solution sets for the 'or' statements differ.", {
                "previous_solution_set": str(prev_solution_set),
                "current_solution_set": str(curr_solution_set),
            }
        except Exception as e:
            return False, f"Could not accurately compare 'or' solution statements: {e}", {}

    def _validate_or_eq_to_eq(self, prev: ParsedStep, curr: ParsedStep):
        try:
            prev_solution_set = self._or_equations_solution_set(prev.value)
            curr_solution_set = self._equation_solution_set(curr.value)

            if self._solution_sets_equivalent(prev_solution_set, curr_solution_set):
                return True, "The equation perfectly represents the same full solution set as the previous 'or' statement.", {
                    "previous_solution_set": str(prev_solution_set),
                    "current_solution_set": str(curr_solution_set),
                }

            return False, "The equation's solution set does not accurately match the 'or' statement's original set.", {
                "previous_solution_set": str(prev_solution_set),
                "current_solution_set": str(curr_solution_set),
            }
        except Exception as e:
            return False, f"Could not accurately compare 'or' statement with equation: {e}", {}

    def _validate_expr_to_eq(self, prev: ParsedStep, curr: ParsedStep):
        eq: Eq = curr.value

        if self._expressions_equivalent(prev.value, eq.lhs) or self._expressions_equivalent(prev.value, eq.rhs):
            return True, "The expression was correctly expanded into an equation matching an equivalent side.", {}

        lhs_sub_ok, lhs_subs = self._is_substitution_instance(prev.value, eq.lhs)
        rhs_sub_ok, rhs_subs = self._is_substitution_instance(prev.value, eq.rhs)

        if lhs_sub_ok or rhs_sub_ok:
            return True, "The expression matches one side of the equation by a valid logical substitution.", {
                "lhs_substitutions": lhs_subs,
                "rhs_substitutions": rhs_subs,
            }

        return False, "The prior expression is mathematically unequivalent to either side of the new equation.", {
            "expression": str(prev.value),
            "equation": str(eq),
        }

    def _validate_eq_to_expr(self, prev: ParsedStep, curr: ParsedStep):
        eq: Eq = prev.value

        if self._expressions_equivalent(eq.lhs, curr.value) or self._expressions_equivalent(eq.rhs, curr.value):
            return True, "The expression accurately matches one side of the previous equation.", {}

        lhs_sub_ok, lhs_subs = self._is_substitution_instance(eq.lhs, curr.value)
        rhs_sub_ok, rhs_subs = self._is_substitution_instance(eq.rhs, curr.value)

        if lhs_sub_ok or rhs_sub_ok:
            return True, "The expression successfully matches one side of the equation by a valid substitution.", {
                "lhs_substitutions": lhs_subs,
                "rhs_substitutions": rhs_subs,
            }

        return False, "The expression does not logically follow from either side of the previous equation.", {
            "equation": str(eq),
            "expression": str(curr.value),
        }

    def _validate_boolean_to_boolean(self, prev: ParsedStep, curr: ParsedStep):
        if prev.value == curr.value:
            return True, "Boolean logic step preserved its exact truth value.", {}

        return False, f"Boolean truth value incorrectly changed from {prev.value} to {curr.value}.", {
            "previous": str(prev.value),
            "current": str(curr.value),
        }

    def _validate_system_to_system(self, prev: ParsedStep, curr: ParsedStep):
        try:
            sol1 = self._system_solution(prev.value)
            sol2 = self._system_solution(curr.value)

            if self._system_solutions_equivalent(sol1, sol2):
                return True, "Systems of equations possess identical mathematical solution sets.", {
                    "solution_set": str(sol1)
                }

            return False, "The solution sets between systems differ.", {
                "previous_solution_set": str(sol1),
                "current_solution_set": str(sol2),
            }
        except Exception as e:
            return False, f"Could not mathematically solve or compare systems: {e}", {}

    def _validate_chain_eq_to_eq(self, prev: ParsedStep, curr: ParsedStep):
        chain_exprs = prev.value
        best_details = {}

        for i in range(len(chain_exprs)):
            for j in range(i + 1, len(chain_exprs)):
                candidate_eq = Eq(chain_exprs[i], chain_exprs[j])
                candidate_step = ParsedStep(raw="", kind="eq", value=candidate_eq)

                valid, reason, details = self._validate_eq_to_eq(candidate_step, curr)
                if valid:
                    return True, f"Derived correctly from parts of the multiple equality. {reason}", details
                else:
                    best_details = details

        return False, "The equation does not logically follow from any pair of expressions in the previous multiple equality chain.", best_details

    def _validate_chain_eq_to_chain_eq(self, prev: ParsedStep, curr: ParsedStep):
        curr_exprs = curr.value

        for i in range(len(curr_exprs) - 1):
            curr_eq = Eq(curr_exprs[i], curr_exprs[i + 1])
            curr_eq_step = ParsedStep(raw="", kind="eq", value=curr_eq)

            valid, reason, details = self._validate_chain_eq_to_eq(prev, curr_eq_step)

            if not valid:
                return False, f"Transition for chain link {i + 1} to {i + 2} is not valid. {reason}", details

        return True, "All parts of the multiple equality were correctly transformed.", {}

    def _validate_eq_to_chain_eq(self, prev: ParsedStep, curr: ParsedStep):
        curr_exprs = curr.value

        for i in range(len(curr_exprs) - 1):
            curr_eq = Eq(curr_exprs[i], curr_exprs[i + 1])
            curr_eq_step = ParsedStep(raw="", kind="eq", value=curr_eq)

            if self._expressions_equivalent(curr_eq.lhs, curr_eq.rhs):
                continue

            valid, reason, details = self._validate_eq_to_eq(prev, curr_eq_step)

            if not valid:
                return False, f"Chain link {i + 1} to {i + 2} is neither an identity nor logically derived from the previous equation. {reason}", details

        return True, "Multiple equality statement is logically derived from the previous equation.", {}

    def _is_single_elementary_row_operation(self, A: Matrix, B: Matrix) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Checks whether B can be obtained from A by one elementary row operation.

        Supported:
        1. Row swap: Ri <-> Rj
        2. Row scaling: Ri -> c Ri
        3. Row replacement: Ri -> Ri + c Rj
        """

        if A.shape != B.shape:
            return False, "Matrix shapes differ.", {}

        rows, cols = A.shape

        if self._expressions_equivalent(A, B):
            return True, "Matrices are identical.", {
                "operation": "identity"
            }

        changed_rows = []

        for i in range(rows):
            if not self._expressions_equivalent(A.row(i), B.row(i)):
                changed_rows.append(i)

        # One changed row: row scaling or row replacement.
        if len(changed_rows) == 1:
            i = changed_rows[0]
            old_row = A.row(i)
            new_row = B.row(i)

            # Check row scaling: Ri -> c Ri.
            scale_candidate = None
            possible = True

            for k in range(cols):
                old_entry = simplify(old_row[k])
                new_entry = simplify(new_row[k])

                if old_entry == 0:
                    if new_entry != 0:
                        possible = False
                        break
                else:
                    c = simplify(new_entry / old_entry)

                    if scale_candidate is None:
                        scale_candidate = c
                    elif simplify(c - scale_candidate) != 0:
                        possible = False
                        break

            if possible and scale_candidate is not None and simplify(scale_candidate) != 0:
                scaled = old_row.applyfunc(lambda x: simplify(scale_candidate * x))

                if self._expressions_equivalent(scaled, new_row):
                    return True, f"Valid elementary row operation: R{i + 1} -> ({scale_candidate}) R{i + 1}.", {
                        "operation": "row_scaling",
                        "row": i + 1,
                        "scale": str(scale_candidate),
                    }

            # Check row replacement: Ri -> Ri + c Rj.
            for j in range(rows):
                if j == i:
                    continue

                other_row = A.row(j)
                diff = new_row - old_row

                coeff_candidate = None
                possible = True

                for k in range(cols):
                    source_entry = simplify(other_row[k])
                    diff_entry = simplify(diff[k])

                    if source_entry == 0:
                        if diff_entry != 0:
                            possible = False
                            break
                    else:
                        c = simplify(diff_entry / source_entry)

                        if coeff_candidate is None:
                            coeff_candidate = c
                        elif simplify(c - coeff_candidate) != 0:
                            possible = False
                            break

                if possible and coeff_candidate is not None:
                    c = simplify(coeff_candidate)
                    candidate = old_row + c * other_row

                    if self._expressions_equivalent(candidate, new_row):
                        if c == 1:
                            op_text = f"R{i + 1} -> R{i + 1} + R{j + 1}"
                        elif c == -1:
                            op_text = f"R{i + 1} -> R{i + 1} - R{j + 1}"
                        elif str(c).startswith("-"):
                            op_text = f"R{i + 1} -> R{i + 1} - {simplify(-c)} R{j + 1}"
                        else:
                            op_text = f"R{i + 1} -> R{i + 1} + {c} R{j + 1}"

                        return True, f"Valid elementary row operation: {op_text}.", {
                            "operation": "row_replacement",
                            "target_row": i + 1,
                            "source_row": j + 1,
                            "coefficient": str(c),
                        }

        # Two changed rows: possible row swap.
        if len(changed_rows) == 2:
            i, j = changed_rows

            swapped = A.copy()
            swapped.row_swap(i, j)

            if self._expressions_equivalent(swapped, B):
                return True, f"Valid elementary row operation: R{i + 1} <-> R{j + 1}.", {
                    "operation": "row_swap",
                    "row_1": i + 1,
                    "row_2": j + 1,
                }

        return False, "No single elementary row operation transforms the previous matrix into the current matrix.", {
            "changed_rows": [i + 1 for i in changed_rows],
        }

    def _validate_matrix_to_matrix(self, prev: ParsedStep, curr: ParsedStep):
        A = prev.value
        B = curr.value

        if not isinstance(A, sp.MatrixBase) or not isinstance(B, sp.MatrixBase):
            if self._expressions_equivalent(A, B):
                return True, "Matrix expressions are mathematically equivalent.", {}

            return False, "Matrix expressions are not equivalent.", {
                "previous": str(A),
                "current": str(B),
            }

        if A.shape != B.shape:
            return False, f"Matrix shape changed improperly from {A.shape} to {B.shape}.", {
                "prev_shape": str(A.shape),
                "curr_shape": str(B.shape),
            }

        if self._expressions_equivalent(A, B):
            return True, "Matrices are mathematically equivalent and identical.", {
                "operation": "identity",
                "previous_matrix": str(A.tolist()),
                "current_matrix": str(B.tolist()),
            }

        # IMPORTANT:
        # Elementary row operations are ALWAYS checked here.
        # This avoids relying on an option being passed correctly.
        row_op_valid, row_op_reason, row_op_details = self._is_single_elementary_row_operation(A, B)

        if row_op_valid:
            return True, row_op_reason, row_op_details

        try:
            diff = A - B
            simplified_diff = diff.applyfunc(simplify)
        except Exception:
            simplified_diff = None

        return False, "Matrices are mathematically unequivalent and cannot be obtained by one elementary row operation.", {
            "previous_matrix": str(A.tolist()),
            "current_matrix": str(B.tolist()),
            "difference": str(simplified_diff),
        }

    def _validate_derivative_to_expr(self, prev: ParsedStep, curr: ParsedStep):
        deriv: sp.Derivative = prev.value
        evaluated = sp.simplify(deriv.doit())

        if self._expressions_equivalent(evaluated, curr.value):
            return True, "Derivative was evaluated correctly.", {
                "computed_derivative": str(evaluated)
            }

        return False, f"Incorrect derivative logic. Mathematically expected an equivalent to {evaluated}.", {
            "computed_derivative": str(evaluated),
            "provided_expression": str(curr.value),
        }

    def _validate_derivative_to_eq(self, prev: ParsedStep, curr: ParsedStep):
        deriv: sp.Derivative = prev.value
        evaluated = sp.simplify(deriv.doit())
        eq: Eq = curr.value

        if self._expressions_equivalent(evaluated, eq.rhs) or self._expressions_equivalent(evaluated, eq.lhs):
            return True, "Derivative evaluation strictly matches one valid side of the equation.", {
                "computed_derivative": str(evaluated)
            }

        return False, "Evaluated derivative result does not precisely match either side of the equation.", {
            "computed_derivative": str(evaluated),
            "equation": str(eq),
        }

    def _validate_integral_to_expr(self, prev: ParsedStep, curr: ParsedStep):
        integral: sp.Integral = prev.value

        if len(integral.limits[0]) == 3:
            evaluated = sp.simplify(integral.doit())

            if self._expressions_equivalent(evaluated, curr.value):
                return True, "Definite integral was successfully evaluated.", {
                    "computed_integral": str(evaluated)
                }

            return False, f"Incorrect definite integral evaluation. Mathematically expected an equivalent to {evaluated}.", {
                "computed_integral": str(evaluated),
                "provided_expression": str(curr.value),
            }

        var = integral.limits[0][0]
        integrand = integral.function
        candidate_derivative = sp.simplify(sp.diff(curr.value, var))

        if self._expressions_equivalent(candidate_derivative, integrand):
            return True, "Indefinite integral computed is structurally correct up to an additive constant.", {
                "integrand": str(integrand),
                "derivative_of_candidate": str(candidate_derivative),
            }

        return False, "Incorrect antiderivative applied. Differentiating the result does not yield the required integrand.", {
            "integrand": str(integrand),
            "derivative_of_candidate": str(candidate_derivative),
            "provided_expression": str(curr.value),
        }

    def _validate_integral_to_eq(self, prev: ParsedStep, curr: ParsedStep):
        eq: Eq = curr.value

        left_ok, _, _ = self._validate_integral_to_expr(prev, ParsedStep(curr.raw, "expr", eq.lhs))
        right_ok, _, _ = self._validate_integral_to_expr(prev, ParsedStep(curr.raw, "expr", eq.rhs))

        if left_ok or right_ok:
            return True, "Integral evaluation matches one side of the equation appropriately.", {}

        return False, "Integral result evaluation does not correspond to either side of the equation.", {
            "equation": str(eq)
        }

