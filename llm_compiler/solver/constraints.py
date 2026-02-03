"""
Constraint System
================

Implements constraint solving for architecture dimensions.
Handles equality, inequality, divisibility, and range constraints.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp
from sympy import symbols, Eq, Le, Ge, Mod, solve

class ConstraintType(Enum):
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    DIVISIBILITY = "divisibility"
    RANGE = "range"
    LINEAR = "linear"

@dataclass
class Constraint:
    """Base constraint class"""
    name: str
    constraint_type: ConstraintType = field(init=False)
    
    def to_sympy(self) -> List[sp.Basic]:
        """Convert to sympy expressions"""
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"{self.name}: {self.constraint_type}"

@dataclass
class EqualityConstraint(Constraint):
    """Equality constraint: var1 == var2 or var == value"""
    var1: str
    var2: str  # Can be another variable or a constant expression
    
    def __post_init__(self):
        self.constraint_type = ConstraintType.EQUALITY
    
    def to_sympy(self) -> List[sp.Basic]:
        var1_sym = symbols(self.var1)
        
        # Try to parse var2 as expression
        try:
            # Check if var2 is a simple number
            if self.var2.replace('.', '').replace('-', '').isdigit():
                var2_val = float(self.var2) if '.' in self.var2 else int(self.var2)
                return [Eq(var1_sym, var2_val)]
            
            # Check if var2 is another variable
            if self.var2.isidentifier():
                var2_sym = symbols(self.var2)
                return [Eq(var1_sym, var2_sym)]
            
            # Try to parse as expression
            # Simple expression parser - in practice would use sympy parsing
            if '*' in self.var2:
                parts = self.var2.split('*')
                if len(parts) == 2:
                    coef, var = parts
                    coef = float(coef.strip())
                    var_sym = symbols(var.strip())
                    return [Eq(var1_sym, coef * var_sym)]
            
        except:
            pass
        
        # Default: treat as variable
        var2_sym = symbols(self.var2)
        return [Eq(var1_sym, var2_sym)]
    
    def __str__(self) -> str:
        return f"{self.name}: {self.var1} == {self.var2}"

@dataclass
class InequalityConstraint(Constraint):
    """Inequality constraint: var1 <= var2 or var1 >= var2"""
    var1: str
    var2: str
    greater_than: bool = False  # True for >=, False for <=
    
    def __post_init__(self):
        self.constraint_type = ConstraintType.INEQUALITY
    
    def to_sympy(self) -> List[sp.Basic]:
        var1_sym = symbols(self.var1)
        
        try:
            if self.var2.replace('.', '').replace('-', '').isdigit():
                var2_val = float(self.var2) if '.' in self.var2 else int(self.var2)
                if self.greater_than:
                    return [Ge(var1_sym, var2_val)]
                else:
                    return [Le(var1_sym, var2_val)]
        except:
            pass
        
        var2_sym = symbols(self.var2)
        if self.greater_than:
            return [Ge(var1_sym, var2_sym)]
        else:
            return [Le(var1_sym, var2_sym)]
    
    def __str__(self) -> str:
        op = ">=" if self.greater_than else "<="
        return f"{self.name}: {self.var1} {op} {self.var2}"

@dataclass  
class DivisibilityConstraint(Constraint):
    """Divisibility constraint: var1 % var2 == 0"""
    var1: str
    var2: str
    
    def __post_init__(self):
        self.constraint_type = ConstraintType.DIVISIBILITY
    
    def to_sympy(self) -> List[sp.Basic]:
        var1_sym = symbols(self.var1)
        
        try:
            if self.var2.replace('.', '').replace('-', '').isdigit():
                var2_val = int(self.var2)
                return [Eq(Mod(var1_sym, var2_val), 0)]
        except:
            pass
        
        var2_sym = symbols(self.var2)
        return [Eq(Mod(var1_sym, var2_sym), 0)]
    
    def __str__(self) -> str:
        return f"{self.name}: {self.var1} % {self.var2} == 0"

@dataclass
class RangeConstraint(Constraint):
    """Range constraint: min <= var <= max"""
    var: str
    min_val: Optional[Union[int, float, str]] = None
    max_val: Optional[Union[int, float, str]] = None
    
    def __post_init__(self):
        self.constraint_type = ConstraintType.RANGE
    
    def to_sympy(self) -> List[sp.Basic]:
        var_sym = symbols(self.var)
        constraints = []
        
        if self.min_val is not None:
            try:
                if isinstance(self.min_val, str) and self.min_val.replace('.', '').replace('-', '').isdigit():
                    min_val = float(self.min_val) if '.' in self.min_val else int(self.min_val)
                    constraints.append(Ge(var_sym, min_val))
                elif isinstance(self.min_val, (int, float)):
                    constraints.append(Ge(var_sym, self.min_val))
                else:
                    min_sym = symbols(self.min_val)
                    constraints.append(Ge(var_sym, min_sym))
            except:
                pass
        
        if self.max_val is not None:
            try:
                if isinstance(self.max_val, str) and self.max_val.replace('.', '').replace('-', '').isdigit():
                    max_val = float(self.max_val) if '.' in self.max_val else int(self.max_val)
                    constraints.append(Le(var_sym, max_val))
                elif isinstance(self.max_val, (int, float)):
                    constraints.append(Le(var_sym, self.max_val))
                else:
                    max_sym = symbols(self.max_val)
                    constraints.append(Le(var_sym, max_sym))
            except:
                pass
        
        return constraints
    
    def __str__(self) -> str:
        parts = [self.var]
        if self.min_val is not None:
            parts.insert(0, f"{self.min_val} <=")
        if self.max_val is not None:
            parts.append(f"<= {self.max_val}")
        return f"{self.name}: {' '.join(parts)}"

@dataclass
class LinearConstraint(Constraint):
    """Linear constraint: a*x + b*y + ... == c"""
    coefficients: Dict[str, float]  # variable -> coefficient
    constant: float
    equality: bool = True  # True for ==, False for <=
    
    def __post_init__(self):
        self.constraint_type = ConstraintType.LINEAR
    
    def to_sympy(self) -> List[sp.Basic]:
        expr = 0
        for var, coeff in self.coefficients.items():
            var_sym = symbols(var)
            expr += coeff * var_sym
        
        if self.equality:
            return [Eq(expr, self.constant)]
        else:
            return [Le(expr, self.constant)]
    
    def __str__(self) -> str:
        terms = []
        for var, coeff in self.coefficients.items():
            if coeff == 1:
                terms.append(var)
            elif coeff == -1:
                terms.append(f"-{var}")
            else:
                terms.append(f"{coeff}*{var}")
        
        expr = " + ".join(terms).replace("+ -", "- ")
        op = "==" if self.equality else "<="
        return f"{self.name}: {expr} {op} {self.constant}"

class ConstraintSystem:
    """System of constraints to solve"""
    
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.variables: Dict[str, Any] = {}
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the system"""
        self.constraints.append(constraint)
        
        # Extract variables from constraint
        if isinstance(constraint, EqualityConstraint):
            self._add_variable(constraint.var1)
            if constraint.var2.isidentifier():
                self._add_variable(constraint.var2)
        elif isinstance(constraint, InequalityConstraint):
            self._add_variable(constraint.var1)
            if constraint.var2.isidentifier():
                self._add_variable(constraint.var2)
        elif isinstance(constraint, DivisibilityConstraint):
            self._add_variable(constraint.var1)
            self._add_variable(constraint.var2)
        elif isinstance(constraint, RangeConstraint):
            self._add_variable(constraint.var)
        elif isinstance(constraint, LinearConstraint):
            for var in constraint.coefficients.keys():
                self._add_variable(var)
    
    def _add_variable(self, var: str):
        """Add variable to tracking dict"""
        if var not in self.variables:
            self.variables[var] = {
                'min': None,
                'max': None,
                'fixed': False,
                'value': None
            }
    
    def solve(self, fixed_vars: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Solve constraint system.
        
        Args:
            fixed_vars: Variables with fixed values
            
        Returns:
            Dictionary of variable values
        """
        if fixed_vars:
            for var, value in fixed_vars.items():
                self._add_variable(var)
                self.variables[var]['fixed'] = True
                self.variables[var]['value'] = value
        
        # Convert to sympy
        sympy_constraints = []
        variables_set = set()
        
        for constraint in self.constraints:
            sympy_exprs = constraint.to_sympy()
            sympy_constraints.extend(sympy_exprs)
            
            # Collect variables
            for expr in sympy_exprs:
                variables_set.update(expr.free_symbols)
        
        # Create symbol list
        variables = list(variables_set)
        
        # Try to solve
        try:
            solution = solve(sympy_constraints, variables, dict=True)
            
            if not solution:
                raise ValueError("No solution found")
            
            # Convert to simple dict
            result = {}
            for sol in solution:
                for var, value in sol.items():
                    result[str(var)] = float(value) if value.is_Float else int(value)
            
            return result
            
        except Exception as e:
            # Fall back to iterative solving
            return self._solve_iteratively()
    
    def _solve_iteratively(self) -> Dict[str, Any]:
        """Iterative constraint solving"""
        # Initialize with reasonable defaults
        solution = {}
        for var in self.variables:
            if self.variables[var]['fixed']:
                solution[var] = self.variables[var]['value']
            else:
                # Set to midpoint of bounds if available
                if (self.variables[var]['min'] is not None and 
                    self.variables[var]['max'] is not None):
                    min_val = self.variables[var]['min']
                    max_val = self.variables[var]['max']
                    solution[var] = (min_val + max_val) // 2
                else:
                    solution[var] = 1  # Default
        
        # Apply constraints iteratively
        changed = True
        max_iter = 100
        iteration = 0
        
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            
            for constraint in self.constraints:
                if isinstance(constraint, EqualityConstraint):
                    result = self._apply_equality(constraint, solution)
                    if result:
                        solution.update(result)
                        changed = True
                
                elif isinstance(constraint, DivisibilityConstraint):
                    result = self._apply_divisibility(constraint, solution)
                    if result:
                        solution.update(result)
                        changed = True
                
                elif isinstance(constraint, RangeConstraint):
                    result = self._apply_range(constraint, solution)
                    if result:
                        solution.update(result)
                        changed = True
        
        return solution
    
    def _apply_equality(self, constraint: EqualityConstraint, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Apply equality constraint"""
        updates = {}
        
        # Check if we can evaluate
        var1 = constraint.var1
        var2 = constraint.var2
        
        # Try to parse var2 as number
        try:
            if var2.replace('.', '').replace('-', '').isdigit():
                value = float(var2) if '.' in var2 else int(var2)
                updates[var1] = value
                return updates
        except:
            pass
        
        # Check if it's an expression
        if '*' in var2:
            parts = var2.split('*')
            if len(parts) == 2:
                try:
                    coef = float(parts[0].strip())
                    other_var = parts[1].strip()
                    if other_var in solution:
                        updates[var1] = coef * solution[other_var]
                        return updates
                except:
                    pass
        
        # Variable equality
        if var2 in solution and var1 not in solution:
            updates[var1] = solution[var2]
        elif var1 in solution and var2 not in solution:
            updates[var2] = solution[var1]
        
        return updates
    
    def _apply_divisibility(self, constraint: DivisibilityConstraint, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Apply divisibility constraint"""
        updates = {}
        var1 = constraint.var1
        var2 = constraint.var2
        
        # Try to get divisor value
        divisor = None
        try:
            if var2.replace('.', '').replace('-', '').isdigit():
                divisor = int(var2)
        except:
            if var2 in solution:
                divisor = solution[var2]
        
        if divisor is None:
            return updates
        
        # Check if var1 needs adjustment
        if var1 in solution:
            value = solution[var1]
            if value % divisor != 0:
                # Round up to next multiple
                updates[var1] = ((value + divisor - 1) // divisor) * divisor
        
        return updates
    
    def _apply_range(self, constraint: RangeConstraint, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Apply range constraint"""
        updates = {}
        var = constraint.var
        
        if var not in solution:
            return updates
        
        value = solution[var]
        
        # Apply min bound
        if constraint.min_val is not None:
            try:
                if isinstance(constraint.min_val, (int, float)):
                    min_val = constraint.min_val
                elif constraint.min_val.replace('.', '').replace('-', '').isdigit():
                    min_val = float(constraint.min_val) if '.' in constraint.min_val else int(constraint.min_val)
                else:
                    min_val = solution.get(constraint.min_val, value)
                
                if value < min_val:
                    updates[var] = min_val
            except:
                pass
        
        # Apply max bound
        if constraint.max_val is not None:
            try:
                if isinstance(constraint.max_val, (int, float)):
                    max_val = constraint.max_val
                elif constraint.max_val.replace('.', '').replace('-', '').isdigit():
                    max_val = float(constraint.max_val) if '.' in constraint.max_val else int(constraint.max_val)
                else:
                    max_val = solution.get(constraint.max_val, value)
                
                if value > max_val:
                    updates[var] = max_val
            except:
                pass
        
        return updates
    
    def __str__(self) -> str:
        lines = ["Constraint System:"]
        for constraint in self.constraints:
            lines.append(f"  {constraint}")
        return "\n".join(lines)
