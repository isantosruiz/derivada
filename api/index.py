from __future__ import annotations

import re
from typing import Any

import sympy as sp
from flask import Flask, jsonify, render_template_string, request
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

app = Flask(__name__)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
GLOBAL_DICT = dict(sp.__dict__)
GLOBAL_DICT["__builtins__"] = {}


def _log10(expr: sp.Expr) -> sp.Expr:
    return sp.log(expr, 10)


COMMON_LOCALS = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "sen": sp.sin,
    "ln": sp.log,
    "log10": _log10,
    "oo": sp.oo,
}

DELTA = sp.Symbol("h")
VALID_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

INDEX_HTML = r"""
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Derivación paso a paso</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet" />
    <style>
      :root {
        --bg: #f6f4ec;
        --panel: #fffdf7;
        --ink: #1f1b16;
        --accent: #b35b2f;
        --border: #d7d2c5;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        color: var(--ink);
        line-height: 1.55;
        background:
          radial-gradient(circle at 15% 20%, #fff 0, #f6f4ec 40%),
          radial-gradient(circle at 85% 0%, #f2e9d5 0, #f6f4ec 55%);
        min-height: 100vh;
      }

      main {
        max-width: 980px;
        margin: 0 auto;
        padding: 2rem 1rem 3rem;
      }

      .card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 10px 28px rgba(95, 71, 48, 0.08);
      }

      .steps-card {
        margin-top: 1rem;
      }

      h1 {
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.5rem;
        letter-spacing: 0.005em;
        font-size: clamp(1.55rem, 4vw, 2rem);
      }

      p.note {
        margin-top: 0;
        margin-bottom: 1.2rem;
        color: #5b4b3a;
        font-size: 0.98rem;
      }

      form {
        display: grid;
        gap: 0.95rem;
      }

      label {
        font-weight: 600;
        letter-spacing: 0.01em;
      }

      .input-hint {
        margin: 0.5rem 0 0;
        font-size: 0.92rem;
        color: #5b4b3a;
      }

      input {
        width: 100%;
        border: 1px solid var(--border);
        background: #fff;
        border-radius: 10px;
        padding: 0.72rem 0.8rem;
        font-size: 1rem;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      }

      code {
        font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-size: 0.92em;
      }

      button {
        justify-self: start;
        border: none;
        border-radius: 10px;
        background: var(--accent);
        color: #fff;
        font-size: 1rem;
        padding: 0.7rem 1.1rem;
        cursor: pointer;
      }

      button:hover {
        background: #944a25;
      }

      .result {
        margin-top: 1.3rem;
      }

      .error {
        color: #ab1f25;
        font-weight: 600;
      }

      .math-line {
        margin: 0.6rem 0;
        font-size: 1.01rem;
      }

      ol {
        padding-left: 1.1rem;
        font-size: 0.99rem;
      }

      li {
        margin: 0.8rem 0;
      }

      .step-title {
        font-weight: 500;
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        letter-spacing: 0.01em;
      }

      #copyright { margin-top: 0.95rem; color: var(--muted); font-size: 0.95rem; display: flex; justify-content: center; text-align: center; }
      #copyright p { margin: 0; }

      .context-block h2,
      .steps-block h2 {
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        font-size: 1.05rem;
        margin: 1.2rem 0 0.4rem;
      }
    </style>
    <script>
      window.MathJax = {
        tex: {
          inlineMath: [["$", "$"], ["\\(", "\\)"]],
          displayMath: [["\\[", "\\]"]]
        },
        svg: { fontCache: "global" }
      };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  </head>
  <body>
    <main>
      <section class="card">
        <h1>Derivación paso a paso</h1>
        <p class="note">Introduce una función. El asistente mostrará el cociente de incrementos, su límite y la derivada.</p>

        <form id="derive-form">
          <div>
            <label for="function">Función \(f\)</label>
            <input id="function" name="function" placeholder="Ej: x^2 + 5*x + 6" required />
            <p class="input-hint">Usa <code>pi</code> para \( π \) y <code>e</code> para el número de Euler \( e \). Escribe funciones en minúsculas: <code>exp()</code>, <code>sqrt()</code>, <code>sin()</code> o <code>sen()</code>, <code>log()</code>, etc. Tanto <code>ln()</code> como <code>log()</code> representan el logaritmo natural; <code>log10()</code> representa el logaritmo decimal.</p>
          </div>

          <div>
            <label for="variable">Variable independiente (opcional)</label>
            <input id="variable" name="variable" placeholder="Ej: x" />
          </div>

          <div>
            <label for="point">Punto (opcional)</label>
            <input id="point" name="point" placeholder="Ej: 1" />
          </div>

          <button type="submit">Derivar</button>
        </form>
      </section>

      <section class="card steps-card" id="steps-card" hidden>
        <section class="result" id="result"></section>
      </section>
      <div id="copyright">
        <p>&copy; 2026, <a href="https://isantosruiz.github.io/home/" style="text-decoration: none;">Ildeberto de los Santos Ruiz</a></p>
      </div>

    </main>

    <script>
      const form = document.getElementById("derive-form");
      const stepsCard = document.getElementById("steps-card");
      const result = document.getElementById("result");

      function renderMath() {
        if (window.MathJax && window.MathJax.typesetPromise) {
          window.MathJax.typesetPromise([result]);
        }
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const payload = {
          function: form.function.value,
          variable: form.variable.value,
          point: form.point.value,
        };

        stepsCard.hidden = false;
        result.innerHTML = "<p>Calculando...</p>";

        try {
          const response = await fetch("/derive", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await response.json();

          if (!response.ok) {
            throw new Error(data.error || "No se pudo procesar la solicitud.");
          }

          const inferredText = data.inferred_variable
            ? `<p><em>Variable deducida automáticamente: <code>${data.variable}</code>.</em></p>`
            : "";

          const requestLine = data.point_latex
            ? `\\[ \\text{Se solicita } f'(${data.point_symbol_latex}) \\text{ para } ${data.point_symbol_latex} = ${data.point_latex} \\]`
            : `\\[ \\text{Se solicita } f'(${data.variable_latex}) \\]`;

          const stepsHtml = data.steps
            .map(
              (step) => `
                <li>
                  <div class="step-title">${step.title}:</div>
                  <div class="math-line">\\[ ${step.latex} \\]</div>
                </li>
              `
            )
            .join("");

          result.innerHTML = `
            <section class="context-block">
              <h2>Planteamiento</h2>
              ${inferredText}
              <div class="math-line">\\[ f(${data.variable_latex}) = ${data.input_function_latex} \\]</div>
              <div class="math-line">${requestLine}</div>
            </section>
            <section class="steps-block">
              <h2>Pasos</h2>
              <ol>${stepsHtml}</ol>
            </section>
          `;

          renderMath();
        } catch (error) {
          result.innerHTML = `<p class="error">${error.message}</p>`;
          renderMath();
        }
      });
    </script>
  </body>
</html>
"""


def _parse_math(
    text: str,
    extra_locals: dict[str, Any] | None = None,
    evaluate: bool = True,
) -> sp.Expr:
    local_dict = dict(COMMON_LOCALS)
    if extra_locals:
        local_dict.update(extra_locals)
    return parse_expr(
        text,
        local_dict=local_dict,
        global_dict=GLOBAL_DICT,
        transformations=TRANSFORMATIONS,
        evaluate=evaluate,
    )


def _choose_variable(expr: sp.Expr, raw_variable: str | None) -> tuple[sp.Symbol, bool]:
    variable_text = (raw_variable or "").strip()
    if variable_text:
        if not VALID_SYMBOL_RE.match(variable_text):
            raise ValueError(
                "La variable independiente debe ser un símbolo válido, por ejemplo x o t."
            )
        return sp.Symbol(variable_text), False

    symbols = sorted(expr.free_symbols, key=lambda symbol: symbol.name)
    if symbols:
        return symbols[0], True
    return sp.Symbol("x"), True


def _delta_increment_latex(variable: sp.Symbol) -> str:
    return rf"\Delta {sp.latex(variable)}"


def _point_symbol(variable: sp.Symbol) -> sp.Symbol:
    return sp.Symbol(f"{variable.name}_0")


def _latex(expr: sp.Expr, delta_latex: str | None = None) -> str:
    if delta_latex is None:
        return sp.latex(expr)
    return sp.latex(expr, symbol_names={DELTA: delta_latex})


def _disambiguate_delta_terms(
    latex_text: str,
    delta_latex: str,
    variable_latex: str,
) -> str:
    # Case 1: (\Delta v)^n instead of \Delta v^n
    formatted = latex_text.replace(
        f"{delta_latex}^{{",
        rf"\left({delta_latex}\right)^{{",
    )
    # Case 2: (\Delta v) v or (\Delta v) v^n instead of \Delta v v or \Delta v v^n
    already_parenthesized = re.escape(r"\left(")
    factor_pattern = re.compile(
        rf"(?<!{already_parenthesized}){re.escape(delta_latex)}(?=\s+{re.escape(variable_latex)}(?:\^\{{[^}}]+\}})?)"
    )
    return factor_pattern.sub(lambda _: rf"\left({delta_latex}\right)", formatted)


def _build_derivation(
    expr: sp.Expr,
    variable: sp.Symbol,
    raw_point: str | None,
) -> dict[str, Any]:
    point_text = (raw_point or "").strip()
    point_expr = _parse_math(point_text, {variable.name: variable}) if point_text else None

    expr_general_plus_delta = sp.simplify(expr.subs(variable, variable + DELTA))
    quotient_general = (expr_general_plus_delta - expr) / DELTA
    quotient_general_simplified = sp.simplify(sp.cancel(sp.factor(quotient_general)))

    try:
        limit_general = sp.simplify(sp.limit(quotient_general_simplified, DELTA, 0))
    except Exception:
        limit_general = sp.Limit(quotient_general_simplified, DELTA, 0)

    derivative_by_diff = sp.simplify(sp.diff(expr, variable))
    derivative_from_definition = sp.simplify(limit_general)
    derivative_function = (
        derivative_by_diff if derivative_from_definition.has(sp.Limit) else derivative_from_definition
    )

    delta_latex = _delta_increment_latex(variable)

    point_symbol = _point_symbol(variable)
    display_anchor = point_symbol if point_expr is not None else variable
    calc_anchor = point_expr if point_expr is not None else variable

    expr_anchor = sp.simplify(expr.subs(variable, calc_anchor))
    expr_anchor_plus_delta = sp.simplify(expr.subs(variable, calc_anchor + DELTA))

    quotient_raw = (expr_anchor_plus_delta - expr_anchor) / DELTA
    quotient_simplified = sp.simplify(sp.cancel(sp.factor(quotient_raw)))

    try:
        limit_result = sp.simplify(sp.limit(quotient_simplified, DELTA, 0))
    except Exception:
        limit_result = sp.Limit(quotient_simplified, DELTA, 0)

    display_anchor_latex = sp.latex(display_anchor)
    calc_anchor_latex = sp.latex(calc_anchor)
    variable_latex = sp.latex(variable)
    definition_minuend_latex = rf"\textcolor{{red}}{{f({display_anchor_latex}+{delta_latex})}}"
    definition_subtrahend_latex = rf"\textcolor{{blue}}{{f({display_anchor_latex})}}"
    minuend_latex = rf"f({calc_anchor_latex}+{delta_latex})"
    subtrahend_latex = rf"f({calc_anchor_latex})"
    substitution_minuend_latex = rf"\textcolor{{red}}{{{minuend_latex}}}"
    substitution_subtrahend_latex = rf"\textcolor{{blue}}{{{subtrahend_latex}}}"
    evaluated_minuend_latex = _disambiguate_delta_terms(
        _latex(expr_anchor_plus_delta, delta_latex),
        delta_latex,
        variable_latex,
    )
    evaluated_subtrahend_latex = _disambiguate_delta_terms(
        _latex(expr_anchor, delta_latex),
        delta_latex,
        variable_latex,
    )
    colored_evaluated_minuend_latex = rf"\color{{red}}{{{evaluated_minuend_latex}}}"
    colored_evaluated_subtrahend_latex = rf"\color{{blue}}{{{evaluated_subtrahend_latex}}}"
    quotient_evaluated_unsimplified_latex = (
        rf"\frac{{\left.{colored_evaluated_minuend_latex}\right.-\left({colored_evaluated_subtrahend_latex}\right)}}{{{delta_latex}}}"
    )
    quotient_simplified_latex = _disambiguate_delta_terms(
        _latex(quotient_simplified, delta_latex),
        delta_latex,
        variable_latex,
    )
    limit_result_latex = _disambiguate_delta_terms(
        _latex(limit_result, delta_latex),
        delta_latex,
        variable_latex,
    )

    steps: list[dict[str, str]] = [
        {
            "title": "Definición de la derivada",
            "latex": (
                rf"f'({display_anchor_latex}) = "
                rf"\lim_{{{delta_latex} \to 0}}"
                rf"\frac{{\Delta f}}{{{delta_latex}}} = "
                rf"\lim_{{{delta_latex} \to 0}} "
                rf"\frac{{{definition_minuend_latex}\textcolor{{black}}{{-}}{definition_subtrahend_latex}}}{{{delta_latex}}}"
            ),
        }
    ]

    if point_expr is not None:
        steps.append(
            {
                "title": "Sustitución en el cociente de incrementos",
                "latex": (
                    rf"f'({calc_anchor_latex}) = "
                    rf"\lim_{{{delta_latex} \to 0}}"
                    rf"\frac{{\Delta f}}{{{delta_latex}}} = "
                    rf"\lim_{{{delta_latex} \to 0}}"
                    rf"\frac{{{substitution_minuend_latex}\textcolor{{black}}{{-}}{substitution_subtrahend_latex}}}{{{delta_latex}}}"
                ),
            }
        )

    steps.extend(
        [
            {
                "title": "Simplificación del cociente de incrementos",
                "latex": (
                    rf"\frac{{\Delta f}}{{ {delta_latex} }} = "
                    rf"{quotient_evaluated_unsimplified_latex} = {quotient_simplified_latex}"
                ),
            },
            {
                "title": "Aplicación del límite",
                "latex": (
                    rf"f'({calc_anchor_latex}) = "
                    rf"\lim_{{{delta_latex} \to 0}} \left[{quotient_simplified_latex}\right] = {limit_result_latex}"
                ),
            },
        ]
    )

    if point_expr is None:
        steps.append(
            {
                "title": "Función derivada",
                "latex": rf"f'({sp.latex(variable)}) = {_latex(derivative_function)}",
            }
        )

    derivative_at_point_latex = None
    point_latex = None

    if point_expr is not None:
        derivative_at_point = sp.simplify(limit_result)
        if derivative_at_point.has(sp.Limit):
            derivative_at_point = sp.simplify(derivative_by_diff.subs(variable, point_expr))
        derivative_at_point_latex = _latex(derivative_at_point)
        point_latex = sp.latex(point_expr)

    return {
        "steps": steps,
        "derivative_function": derivative_function,
        "derivative_at_point_latex": derivative_at_point_latex,
        "point_latex": point_latex,
        "point_symbol_latex": sp.latex(point_symbol) if point_expr is not None else None,
    }


@app.get("/")
def index() -> str:
    return render_template_string(INDEX_HTML)


@app.post("/derive")
def derive():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    function_text = str(payload.get("function", "")).strip()
    variable_text = str(payload.get("variable", "")).strip()
    point_text = str(payload.get("point", "")).strip()

    if not function_text:
        return jsonify({"error": "Debes indicar una función."}), 400

    try:
        function_expr_display = _parse_math(function_text, evaluate=False)
        function_expr = sp.simplify(_parse_math(function_text, evaluate=True))
        variable, inferred = _choose_variable(function_expr, variable_text)
        derivation = _build_derivation(function_expr, variable, point_text)
    except Exception as error:
        return jsonify({"error": f"Entrada inválida: {error}"}), 400

    return jsonify(
        {
            "input_function_latex": _latex(function_expr_display),
            "variable": variable.name,
            "variable_latex": sp.latex(variable),
            "inferred_variable": inferred,
            "derivative_function_latex": _latex(derivation["derivative_function"]),
            "derivative_at_point_latex": derivation["derivative_at_point_latex"],
            "point_latex": derivation["point_latex"],
            "point_symbol_latex": derivation["point_symbol_latex"],
            "steps": derivation["steps"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
