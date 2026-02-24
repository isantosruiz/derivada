# Derivación por definición con SymPy

Aplicación web en Python (Flask) para obtener la derivada paso a paso usando la definición con cociente incremental y límite.

## Características

- Captura una función en texto (ej: `x^3 + 2x*sin(x)`).
- Variable independiente opcional:
  - Si se omite, se deduce automáticamente de la función.
- Punto `x_0` opcional:
  - Si se omite, se trabaja en punto arbitrario (`x`).
  - Si se indica, también evalúa `f'(x_0)`.
- Muestra resultados y pasos renderizados en LaTeX (MathJax).

## Estructura

- `api/index.py`: app Flask + lógica de derivación con SymPy.
- `requirements.txt`: dependencias Python.
- `vercel.json`: configuración para desplegar en Vercel.

## Ejecutar local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 api/index.py
```

Luego abre [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Despliegue en Vercel

1. Sube este proyecto a un repositorio Git.
2. Importa el repositorio en Vercel.
3. Vercel detectará `vercel.json` y ejecutará `api/index.py` con `@vercel/python`.

