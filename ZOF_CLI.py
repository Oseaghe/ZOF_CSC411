import numpy as np
from typing import Callable, Tuple, List, Dict
import sympy as sp

class ZOFSolver:
    """Zero of Functions Solver - Implements 6 numerical root-finding methods"""
    
    def __init__(self):
        self.x = sp.Symbol('x')
    
    def parse_function(self, func_str: str) -> Callable:
        """Parse string function to executable function"""
        try:
            expr = sp.sympify(func_str)
            return sp.lambdify(self.x, expr, 'numpy')
        except:
            raise ValueError("Invalid function format")
    
    def parse_derivative(self, func_str: str) -> Callable:
        """Parse and differentiate function"""
        try:
            expr = sp.sympify(func_str)
            derivative = sp.diff(expr, self.x)
            return sp.lambdify(self.x, derivative, 'numpy')
        except:
            raise ValueError("Cannot compute derivative")
    
    def bisection_method(self, func_str: str, a: float, b: float, 
                        tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Bisection Method"""
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
    
    def regula_falsi_method(self, func_str: str, a: float, b: float,
                           tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Regula Falsi (False Position) Method"""
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
    
    def secant_method(self, func_str: str, x0: float, x1: float,
                     tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Secant Method"""
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
    
    def newton_raphson_method(self, func_str: str, x0: float,
                             tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Newton-Raphson Method"""
        f = self.parse_function(func_str)
        df = self.parse_derivative(func_str)
        iterations = []
        
        x = x0
        
        for i in range(max_iter):
            fx = f(x)
            dfx = df(x)
            
            if abs(dfx) < 1e-12:
                return {"error": "Derivative too small, division by zero"}
            
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
    
    def fixed_point_iteration(self, func_str: str, x0: float,
                             tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Fixed Point Iteration Method - Solves x = g(x)"""
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
    
    def modified_secant_method(self, func_str: str, x0: float, delta: float = 0.01,
                              tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """Modified Secant Method"""
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


def display_results(result: Dict, method_name: str):
    """Display results in formatted way"""
    print(f"\n{'='*60}")
    print(f"{method_name}")
    print(f"{'='*60}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print("\nIteration Details:")
    print("-" * 60)
    
    if result['iterations']:
        # Print header
        headers = list(result['iterations'][0].keys())
        header_str = " | ".join(f"{h:>12}" for h in headers)
        print(header_str)
        print("-" * len(header_str))
        
        # Print rows (show first 10 and last 5 if too many)
        iterations = result['iterations']
        if len(iterations) <= 15:
            display_iterations = iterations
        else:
            display_iterations = iterations[:10] + [None] + iterations[-5:]
        
        for iteration in display_iterations:
            if iteration is None:
                print("..." + " " * (len(header_str) - 3))
                continue
            row_str = " | ".join(f"{iteration[h]:>12.6e}" if isinstance(iteration[h], float) 
                                else f"{iteration[h]:>12}" for h in headers)
            print(row_str)
    
    print("\n" + "="*60)
    print(f"Final Root: {result['root']:.10f}")
    print(f"Final Error: {result['final_error']:.10e}")
    print(f"Total Iterations: {len(result['iterations'])}")
    print(f"Converged: {'Yes' if result['converged'] else 'No'}")
    if 'message' in result:
        print(f"Note: {result['message']}")
    print("="*60)


def main():
    """Main CLI interface"""
    solver = ZOFSolver()
    
    print("\n" + "="*60)
    print("ZERO OF FUNCTIONS (ZOF) SOLVER")
    print("="*60)
    
    methods = {
        '1': ('Bisection Method', solver.bisection_method),
        '2': ('Regula Falsi Method', solver.regula_falsi_method),
        '3': ('Secant Method', solver.secant_method),
        '4': ('Newton-Raphson Method', solver.newton_raphson_method),
        '5': ('Fixed Point Iteration', solver.fixed_point_iteration),
        '6': ('Modified Secant Method', solver.modified_secant_method)
    }
    
    print("\nAvailable Methods:")
    for key, (name, _) in methods.items():
        print(f"{key}. {name}")
    
    choice = input("\nSelect method (1-6): ").strip()
    
    if choice not in methods:
        print("Invalid choice!")
        return
    
    method_name, method_func = methods[choice]
    
    print(f"\n{method_name} Selected")
    print("-" * 60)
    
    func_str = input("Enter function f(x) (e.g., x**3 - x - 2): ").strip()
    
    try:
        if choice in ['1', '2']:  # Bisection or Regula Falsi
            a = float(input("Enter a (left bound): "))
            b = float(input("Enter b (right bound): "))
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = method_func(func_str, a, b, tol, max_iter)
        
        elif choice == '3':  # Secant
            x0 = float(input("Enter x0 (first guess): "))
            x1 = float(input("Enter x1 (second guess): "))
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = method_func(func_str, x0, x1, tol, max_iter)
        
        elif choice == '4':  # Newton-Raphson
            x0 = float(input("Enter x0 (initial guess): "))
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = method_func(func_str, x0, tol, max_iter)
        
        elif choice == '5':  # Fixed Point (enter g(x))
            print("Note: Enter g(x) such that x = g(x)")
            print("Example: For x^3 - x - 2 = 0, use g(x) = (x + 2)**(1/3)")
            func_str = input("Enter g(x): ").strip()
            x0 = float(input("Enter x0 (initial guess): "))
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = method_func(func_str, x0, tol, max_iter)
        
        elif choice == '6':  # Modified Secant
            x0 = float(input("Enter x0 (initial guess): "))
            delta = float(input("Enter delta (default 0.01): ") or 0.01)
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = method_func(func_str, x0, delta, tol, max_iter)
        
        display_results(result, method_name)
        
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    # Ask to continue
    again = input("\nSolve another equation? (y/n): ").strip().lower()
    if again == 'y':
        main()


if __name__ == "__main__":
    main()
