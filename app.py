from flask import Flask, render_template, request, jsonify
import numpy as np
import math
from typing import Callable, Dict

app = Flask(__name__)

class ZOFSolver:
    """Zero of Functions Solver - NO SYMPY VERSION"""

    def __init__(self):
        # Allowed math functions for safe eval
        self.allowed = {k: getattr(np, k) for k in dir(np) if not k.startswith("_")}
        self.allowed.update({k: getattr(math, k) for k in dir(math) if not k.startswith("_")})

    def parse_function(self, func_str: str) -> Callable:
        """Convert string expression to a safe numerical function."""

        def f(x):
            local_scope = {"x": x}
            return eval(func_str, {"__builtins__": {}}, {**self.allowed, **local_scope})

        # quick test
        try:
            f(1.0)
        except:
            raise ValueError("Invalid function format")
        
        return f

    def parse_derivative(self, func_str: str) -> Callable:
        """Numerical derivative instead of SymPy derivative."""

        f = self.parse_function(func_str)

        def df(x, h=1e-6):
            return (f(x + h) - f(x - h)) / (2*h)

        return df

    # ----------------------------------------------------
    # BISECTION
    # ----------------------------------------------------
    def bisection_method(self, func_str: str, a: float, b: float,
                         tol: float = 1e-6, max_iter: int = 100) -> Dict:

        f = self.parse_function(func_str)
        iterations = []

        if f(a) * f(b) >= 0:
            return {"error": "f(a) and f(b) must have opposite signs"}

        for i in range(max_iter):
            c = (a + b) / 2
            fc = f(c)
            fa = f(a)
            error = abs(b - a) / 2

            iterations.append({
                'iteration': i + 1,
                'a': round(a, 10),
                'b': round(b, 10),
                'c': round(c, 10),
                'f(c)': round(float(fc), 10),
                'error': round(error, 10)
            })

            if error < tol or abs(fc) < tol:
                return {
                    'root': round(c, 10),
                    'iterations': iterations,
                    'final_error': round(error, 10),
                    'converged': True
                }

            if fa * fc < 0:
                b = c
            else:
                a = c

        return {
            'root': round(c, 10),
            'iterations': iterations,
            'final_error': round(error, 10),
            'converged': False,
            'message': "Max iterations reached"
        }

    # ----------------------------------------------------
    # REGULA FALSI
    # ----------------------------------------------------
    def regula_falsi_method(self, func_str: str, a: float, b: float,
                            tol: float = 1e-6, max_iter: int = 100) -> Dict:

        f = self.parse_function(func_str)
        iterations = []

        if f(a) * f(b) >= 0:
            return {"error": "f(a) and f(b) must have opposite signs"}

        c_old = a

        for i in range(max_iter):
            fa = f(a)
            fb = f(b)
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            error = abs(c - c_old) if i > 0 else abs(b - a)

            iterations.append({
                'iteration': i + 1,
                'a': round(a, 10),
                'b': round(b, 10),
                'c': round(c, 10),
                'f(c)': round(float(fc), 10),
                'error': round(error, 10)
            })

            if error < tol or abs(fc) < tol:
                return {
                    'root': round(c, 10),
                    'iterations': iterations,
                    'final_error': round(error, 10),
                    'converged': True
                }

            if fa * fc < 0:
                b = c
            else:
                a = c

            c_old = c

        return {
            'root': round(c, 10),
            'iterations': iterations,
            'final_error': round(error, 10),
            'converged': False,
            'message': "Max iterations reached"
        }

    # ----------------------------------------------------
    # SECANT
    # ----------------------------------------------------
    def secant_method(self, func_str: str, x0: float, x1: float,
                      tol: float = 1e-6, max_iter: int = 100) -> Dict:

        f = self.parse_function(func_str)
        iterations = []

        for i in range(max_iter):
            f0 = f(x0)
            f1 = f(x1)

            if abs(f1 - f0) < 1e-12:
                return {"error": "Division by zero in secant method"}

            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            f2 = f(x2)
            error = abs(x2 - x1)

            iterations.append({
                'iteration': i + 1,
                'x0': round(x0, 10),
                'x1': round(x1, 10),
                'x2': round(x2, 10),
                'f(x2)': round(float(f2), 10),
                'error': round(error, 10)
            })

            if error < tol or abs(f2) < tol:
                return {
                    'root': round(x2, 10),
                    'iterations': iterations,
                    'final_error': round(error, 10),
                    'converged': True
                }

            x0, x1 = x1, x2

        return {
            'root': round(x2, 10),
            'iterations': iterations,
            'final_error': round(error, 10),
            'converged': False,
            'message': "Max iterations reached"
        }

    # ----------------------------------------------------
    # NEWTON-RAPHSON
    # ----------------------------------------------------
    def newton_raphson_method(self, func_str: str, x0: float,
                              tol: float = 1e-6, max_iter: int = 100) -> Dict:

        f = self.parse_function(func_str)
        df = self.parse_derivative(func_str)
        iterations = []

        x = x0

        for i in range(max_iter):
            fx = f(x)
            dfx = df(x)

            if abs(dfx) < 1e-12:
                return {"error": "Derivative too small → division by zero"}

            x_new = x - fx / dfx
            fx_new = f(x_new)
            error = abs(x_new - x)

            iterations.append({
                'iteration': i + 1,
                'x': round(x, 10),
                'f(x)': round(float(fx), 10),
                "f'(x)": round(float(dfx), 10),
                'x_new': round(x_new, 10),
                'error': round(error, 10)
            })

            if error < tol or abs(fx_new) < tol:
                return {
                    'root': round(x_new, 10),
                    'iterations': iterations,
                    'final_error': round(error, 10),
                    'converged': True
                }

            x = x_new

        return {
            'root': round(x_new, 10),
            'iterations': iterations,
            'final_error': round(error, 10),
            'converged': False,
            'message': "Max iterations reached"
        }

    # ----------------------------------------------------
    # FIXED POINT
    # ----------------------------------------------------
    def fixed_point_iteration(self, func_str: str, x0: float,
                              tol: float = 1e-6, max_iter: int = 100) -> Dict:

        g = self.parse_function(func_str)
        iterations = []
        x = x0

        for i in range(max_iter):
            x_new = g(x)
            error = abs(x_new - x)

            iterations.append({
                'iteration': i + 1,
                'x': round(x, 10),
                'g(x)': round(float(x_new), 10),
                'error': round(error, 10)
            })

            if error < tol:
                return {
                    'root': round(x_new, 10),
                    'iterations': iterations,
                    'final_error': round(error, 10),
                    'converged': True
                }

            if abs(x_new) > 1e10:
                return {
                    'error': "Method diverging",
                    'iterations': iterations
                }

            x = x_new

        return {
            'root': round(x_new, 10),
            'iterations': iterations,
            'final_error': round(error, 10),
            'converged': False,
            'message': "Max iterations reached"
        }

    # ----------------------------------------------------
    # MODIFIED SECANT
    # ----------------------------------------------------
    def modified_secant_method(self, func_str: str, x0: float, delta: float = 0.01,
                               tol: float = 1e-6, max_iter: int = 100) -> Dict:

        f = self.parse_function(func_str)
        iterations = []
        x = x0

        for i in range(max_iter):
            fx = f(x)
            fx_delta = f(x + delta * x)

            if abs(fx_delta - fx) < 1e-12:
                return {"error": "Division by zero in modified secant"}

            x_new = x - fx * (delta * x) / (fx_delta - fx)
            fx_new = f(x_new)
            error = abs(x_new - x)

            iterations.append({
                'iteration': i + 1,
                'x': round(x, 10),
                'f(x)': round(float(fx), 10),
                'x_new': round(x_new, 10),
                'error': round(error, 10)
            })

            if error < tol or abs(fx_new) < tol:
                return {
                    'root': round(x_new, 10),
                    'iterations': iterations,
                    'final_error': round(error, 10),
                    'converged': True
                }

            x = x_new

        return {
            'root': round(x_new, 10),
            'iterations': iterations,
            'final_error': round(error, 10),
            'converged': False,
            'message': "Max iterations reached"
        }


solver = ZOFSolver()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        method = data.get("method")
        func_str = data.get("function")
        tol = float(data.get("tolerance", 1e-6))
        max_iter = int(data.get("max_iterations", 100))

        if method == "bisection":
            a = float(data.get("a"))
            b = float(data.get("b"))
            result = solver.bisection_method(func_str, a, b, tol, max_iter)

        elif method == "regula_falsi":
            a = float(data.get("a"))
            b = float(data.get("b"))
            result = solver.regula_falsi_method(func_str, a, b, tol, max_iter)

        elif method == "secant":
            x0 = float(data.get("x0"))
            x1 = float(data.get("x1"))
            result = solver.secant_method(func_str, x0, x1, tol, max_iter)

        elif method == "newton_raphson":
            x0 = float(data.get("x0"))
            result = solver.newton_raphson_method(func_str, x0, tol, max_iter)

        elif method == "fixed_point":
            x0 = float(data.get("x0"))
            result = solver.fixed_point_iteration(func_str, x0, tol, max_iter)

        elif method == "modified_secant":
            x0 = float(data.get("x0"))
            delta = float(data.get("delta", 0.01))
            result = solver.modified_secant_method(func_str, x0, delta, tol, max_iter)

        else:
            return jsonify({"error": "Invalid method selected"}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
