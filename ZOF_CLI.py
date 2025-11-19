import numpy as np
import math
from typing import Callable, Dict


# ========== SAFE NUMERIC EXPRESSION PARSER ==========
def safe_eval(func_str: str) -> Callable:
    """Convert string math expression to a numerical function using NumPy."""

    # Allowed math functions from numpy and math
    allowed = {k: getattr(np, k) for k in dir(np) if not k.startswith("_")}
    allowed.update({k: getattr(math, k) for k in dir(math) if not k.startswith("_")})

    def f(x):
        scope = {"x": x}
        return eval(func_str, {"__builtins__": {}}, {**allowed, **scope})

    return f


# ========== NUMERICAL DERIVATIVE ==========
def numerical_derivative(f: Callable, h: float = 1e-6) -> Callable:
    """Central difference derivative approximation"""
    return lambda x: (f(x + h) - f(x - h)) / (2 * h)


# ========== MAIN SOLVER CLASS ==========
class ZOFSolver:
    """Zero of Functions Solver - Implements 6 numerical root-finding methods"""

    def parse_function(self, func_str: str) -> Callable:
        """Replace SymPy with safe numeric parsing"""
        try:
            return safe_eval(func_str)
        except:
            raise ValueError("Invalid function format")

    def parse_derivative(self, func_str: str) -> Callable:
        """Replace symbolic derivative with numerical derivative"""
        try:
            f = self.parse_function(func_str)
            return numerical_derivative(f)
        except:
            raise ValueError("Cannot compute numerical derivative")

    # ==========================================
    # BISECTION METHOD
    # ==========================================
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
                'a': a,
                'b': b,
                'c': c,
                'f(c)': fc,
                'error': error
            })

            if error < tol or abs(fc) < tol:
                return {
                    'root': c,
                    'iterations': iterations,
                    'final_error': error,
                    'converged': True
                }

            if fa * fc < 0:
                b = c
            else:
                a = c

        return {
            'root': c,
            'iterations': iterations,
            'final_error': error,
            'converged': False,
            'message': 'Max iterations reached'
        }

    # ==========================================
    # REGULA FALSI METHOD
    # ==========================================
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
                'a': a,
                'b': b,
                'c': c,
                'f(c)': fc,
                'error': error
            })

            if error < tol or abs(fc) < tol:
                return {
                    'root': c,
                    'iterations': iterations,
                    'final_error': error,
                    'converged': True
                }

            if fa * fc < 0:
                b = c
            else:
                a = c

            c_old = c

        return {
            'root': c,
            'iterations': iterations,
            'final_error': error,
            'converged': False,
            'message': 'Max iterations reached'
        }

    # ==========================================
    # SECANT METHOD
    # ==========================================
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
                'x0': x0,
                'x1': x1,
                'x2': x2,
                'f(x2)': f2,
                'error': error
            })

            if error < tol or abs(f2) < tol:
                return {
                    'root': x2,
                    'iterations': iterations,
                    'final_error': error,
                    'converged': True
                }

            x0, x1 = x1, x2

        return {
            'root': x2,
            'iterations': iterations,
            'final_error': error,
            'converged': False,
            'message': 'Max iterations reached'
        }

    # ==========================================
    # NEWTON–RAPHSON METHOD
    # ==========================================
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
                return {"error": "Derivative too small — division by zero"}

            x_new = x - fx / dfx
            fx_new = f(x_new)

            error = abs(x_new - x)

            iterations.append({
                'iteration': i + 1,
                'x': x,
                'f(x)': fx,
                "f'(x)": dfx,
                'x_new': x_new,
                'error': error
            })

            if error < tol or abs(fx_new) < tol:
                return {
                    'root': x_new,
                    'iterations': iterations,
                    'final_error': error,
                    'converged': True
                }

            x = x_new

        return {
            'root': x_new,
            'iterations': iterations,
            'final_error': error,
            'converged': False,
            'message': 'Max iterations reached'
        }

    # ==========================================
    # FIXED POINT ITERATION
    # ==========================================
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
                'x': x,
                'g(x)': x_new,
                'error': error
            })

            if error < tol:
                return {
                    'root': x_new,
                    'iterations': iterations,
                    'final_error': error,
                    'converged': True
                }

            if abs(x_new) > 1e10:
                return {
                    'error': 'Method diverging',
                    'iterations': iterations
                }

            x = x_new

        return {
            'root': x_new,
            'iterations': iterations,
            'final_error': error,
            'converged': False,
            'message': 'Max iterations reached'
        }

    # ==========================================
    # MODIFIED SECANT METHOD
    # ==========================================
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
                'x': x,
                'f(x)': fx,
                'x_new': x_new,
                'error': error
            })

            if error < tol or abs(fx_new) < tol:
                return {
                    'root': x_new,
                    'iterations': iterations,
                    'final_error': error,
                    'converged': True
                }

            x = x_new

        return {
            'root': x_new,
            'iterations': iterations,
            'final_error': error,
            'converged': False,
            'message': 'Max iterations reached'
        }
