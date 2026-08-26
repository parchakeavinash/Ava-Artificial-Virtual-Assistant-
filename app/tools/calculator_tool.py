import ast
import math
import operator
from typing import Any

# Supported binary operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions and constants
MATH_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "pow": math.pow,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Expression):
        return eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.UnaryOp):
        op = OPERATORS.get(type(node.op))
        if not op:
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        return op(eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        op = OPERATORS.get(type(node.op))
        if not op:
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")
        left = eval_node(node.left)
        right = eval_node(node.right)
        return op(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only standard mathematical functions are supported.")
        func_name = node.func.id.lower()
        if func_name not in MATH_FUNCTIONS:
            raise ValueError(f"Unknown or unsupported math function: {func_name}")
        args = [eval_node(arg) for arg in node.args]
        return MATH_FUNCTIONS[func_name](*args)

    if isinstance(node, ast.Name):
        var_name = node.id.lower()
        if var_name in MATH_FUNCTIONS and isinstance(MATH_FUNCTIONS[var_name], (int, float)):
            return MATH_FUNCTIONS[var_name]
        raise ValueError(f"Undefined variable: {node.id}")

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Safely and deterministically evaluate a mathematical expression.

    Args:
        expression: A mathematical expression string, e.g. '8.5 * (1 + 0.175)',
                    '850000 * 1.175', 'sqrt(144) + 10', '2 ** 8'.

    Returns:
        The evaluated result as a clean formatted string.
    """
    if not expression or not expression.strip():
        return "Error: Expression is empty."

    cleaned_expr = expression.strip()
    
    # Replace common symbol variants
    cleaned_expr = cleaned_expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    
    # Handle percentages like 17.5% -> (17.5/100)
    # If expression contains '%', replace with /100 when used as percentage
    # (e.g. 17.5% -> (17.5/100))
    import re
    cleaned_expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', cleaned_expr)

    try:
        parsed = ast.parse(cleaned_expr, mode="eval")
        result = eval_node(parsed)

        # Format clean integer or rounded float
        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            # Round float nicely to avoid floating point precision issues like 9.987500000000001
            return str(round(result, 6)).rstrip("0").rstrip(".") if "." in str(round(result, 6)) else str(round(result, 6))
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"
