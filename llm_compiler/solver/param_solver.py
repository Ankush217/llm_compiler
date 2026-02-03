"""
Parameter Solver
===============

Solves for architecture dimensions given parameter targets and constraints.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
from dataclasses import dataclass

from .constraints import ConstraintSystem
from ..templates.base import ArchitectureTemplate, TemplateParameter
from ..utils.math_utils import find_divisors, round_to_multiple

@dataclass
class Solution:
    """Complete solution with dimensions and validation"""
    dimensions: Dict[str, int]
    actual_params: int
    target_params: Optional[int]
    template_name: str
    constraints_satisfied: bool
    warnings: List[str]
    errors: List[str]

class ParameterSolver:
    """Solves for architecture dimensions"""
    
    def __init__(self):
        self.solutions_tried = 0
        self.max_solutions = 1000
    
    def solve(self, 
              template: ArchitectureTemplate,
              spec: Dict[str, Any]) -> Solution:
        """
        Solve for dimensions given specification.
        
        Args:
            template: Architecture template
            spec: User specification
            
        Returns:
            Solution with dimensions
        """
        # Shortcut: if target_params is provided without explicit dims, use heuristic sizing
        if spec.get('target_params') is not None and spec.get('explicit_dims') is None:
            return self._solve_target_direct(template, spec)

        # Create constraint system from template
        constraint_system = template.create_constraint_system(spec)
        
        # Get fixed variables from spec
        fixed_vars = self._extract_fixed_variables(spec, template)
        
        # Try to solve
        try:
            # First try symbolic solving
            dimensions = constraint_system.solve(fixed_vars)
            
            # Ensure all required dimensions are present
            dimensions = self._ensure_complete_dimensions(dimensions, template, spec)
            
            # Round to integers
            dimensions = {k: int(round(v)) for k, v in dimensions.items()}
            
            # Apply template-specific adjustments
            dimensions = self._apply_template_adjustments(dimensions, template, spec)
            
            # Calculate actual parameters
            actual_params = template.calculate_parameters(dimensions)

            # If a target is provided and num_layers is free, rescale layers to hit target
            target_params = spec.get('target_params')
            if target_params is not None and 'num_layers' in dimensions:
                # Estimate per-layer contribution by finite difference
                dims_zero = dict(dimensions)
                dims_zero['num_layers'] = 0
                embed_params = template.calculate_parameters(dims_zero)

                dims_one = dict(dimensions)
                dims_one['num_layers'] = 1
                one_layer_params = template.calculate_parameters(dims_one)
                per_layer = max(1, one_layer_params - embed_params)

                desired_layers = max(1, round((target_params - embed_params) / per_layer))
                dimensions['num_layers'] = int(desired_layers)

                # Recompute after adjustment
                actual_params = template.calculate_parameters(dimensions)
            
            # Validate against target
            warnings = []
            errors = []
            
            if target_params is not None:
                tolerance = spec.get('target_tolerance', 0.1)
                relative_diff = abs(actual_params - target_params) / target_params
                
                if relative_diff > tolerance:
                    warnings.append(
                        f"Parameter count mismatch: "
                        f"target={target_params:,}, actual={actual_params:,} "
                        f"(diff={relative_diff:.1%})"
                    )
            
            # Check constraints
            constraints_satisfied = self._check_constraints(dimensions, constraint_system)
            
            if not constraints_satisfied:
                warnings.append("Some constraints may not be fully satisfied")
            
            return Solution(
                dimensions=dimensions,
                actual_params=actual_params,
                target_params=target_params,
                template_name=template.info.name,
                constraints_satisfied=constraints_satisfied,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            # Try iterative search
            return self._solve_iterative(template, spec, constraint_system, fixed_vars)

    def _solve_target_direct(self, template: ArchitectureTemplate, spec: Dict[str, Any]) -> Solution:
        """Heuristic solver for target_params without explicit_dims."""
        from ..utils.math_utils import solve_for_hidden_size
        target_params = spec['target_params']

        # Layer guess by size bucket
        if target_params < 1e9:
            num_layers = 12
        elif target_params < 7e9:
            num_layers = 24
        elif target_params < 13e9:
            num_layers = 32
        else:
            num_layers = 40

        hidden_size = solve_for_hidden_size(
            target_params=target_params,
            num_layers=num_layers,
            vocab_size=spec['vocab_size'],
            num_heads=32,
            num_kv_heads=8,
            tie_embeddings=spec.get('tie_word_embeddings', True)
        )

        dimensions = {
            'vocab_size': spec['vocab_size'],
            'context_length': spec['context_length'],
            'hidden_size': max(64, hidden_size),
            'num_layers': num_layers,
        }

        dimensions = self._ensure_complete_dimensions(dimensions, template, spec)
        dimensions = self._apply_template_adjustments(dimensions, template, spec)

        # Estimate per-layer and rescale layers to target
        dims_zero = dict(dimensions)
        dims_zero['num_layers'] = 0
        embed_params = template.calculate_parameters(dims_zero)

        dims_one = dict(dimensions)
        dims_one['num_layers'] = 1
        per_layer = max(1, template.calculate_parameters(dims_one) - embed_params)

        desired_layers = max(1, round((target_params - embed_params) / per_layer))
        dimensions['num_layers'] = int(desired_layers)

        actual_params = template.calculate_parameters(dimensions)

        warnings = []
        rel_diff = abs(actual_params - target_params) / target_params
        if rel_diff > 0.01:
            warnings.append(
                f"Parameter count mismatch: target={target_params:,}, actual={actual_params:,} (diff={rel_diff:.1%})"
            )

        return Solution(
            dimensions=dimensions,
            actual_params=actual_params,
            target_params=target_params,
            template_name=template.info.name,
            constraints_satisfied=True,
            warnings=warnings,
            errors=[]
        )
    
    def _extract_fixed_variables(self, spec: Dict[str, Any], template: ArchitectureTemplate) -> Dict[str, Any]:
        """Extract fixed variables from spec"""
        fixed = {}
        
        # Core fixed variables
        fixed['vocab_size'] = spec['vocab_size']
        fixed['context_length'] = spec['context_length']
        
        # Any explicit dimensions
        if spec.get('explicit_dims'):
            fixed.update(spec['explicit_dims'])
        
        return fixed
    
    def _ensure_complete_dimensions(self, 
                                   dimensions: Dict[str, Any],
                                   template: ArchitectureTemplate,
                                   spec: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required dimensions are present"""
        required = {p.value for p in template.info.required_parameters}
        all_params = {p.value for p in template.info.parameters}
        
        # Ensure required dimensions exist; then fill the rest with sensible defaults
        to_check = required | all_params
        for param in to_check:
            if param not in dimensions:
                if param == 'num_layers':
                    dimensions[param] = 32  # Reasonable default
                elif param == 'hidden_size':
                    dimensions[param] = 4096  # Reasonable default
                elif param == 'intermediate_size':
                    # Default to SwiGLU expansion
                    hidden = dimensions.get('hidden_size', 4096)
                    dimensions[param] = int(2 * round(hidden * 8/3 / 32) * 32)
                elif param == 'num_heads':
                    hidden = dimensions.get('hidden_size', 4096)
                    dimensions[param] = max(1, hidden // 128)  # Default head_dim=128
                elif param == 'num_kv_heads':
                    # Default to GQA with groups of 8
                    num_heads = dimensions.get('num_heads', 32)
                    dimensions[param] = max(1, num_heads // 8)
                elif param == 'head_dim':
                    hidden = dimensions.get('hidden_size', 4096)
                    num_heads = dimensions.get('num_heads', 32)
                    dimensions[param] = max(1, hidden // max(1, num_heads))
                elif param == 'vocab_size':
                    dimensions[param] = spec.get('vocab_size')
                elif param == 'context_length':
                    dimensions[param] = spec.get('context_length')
        
        return dimensions
    
    def _apply_template_adjustments(self,
                                   dimensions: Dict[str, int],
                                   template: ArchitectureTemplate,
                                   spec: Dict[str, Any]) -> Dict[str, int]:
        """Apply template-specific adjustments to dimensions"""
        
        # Ensure divisibility constraints
        hidden_size = dimensions.get('hidden_size')
        num_heads = dimensions.get('num_heads')
        num_kv_heads = dimensions.get('num_kv_heads')
        
        if hidden_size and num_heads:
            if hidden_size % num_heads != 0:
                # Adjust hidden_size to be divisible
                dimensions['hidden_size'] = ((hidden_size + num_heads - 1) // num_heads) * num_heads
        
        if hidden_size and num_kv_heads:
            if hidden_size % num_kv_heads != 0:
                # Adjust hidden_size to be divisible
                dimensions['hidden_size'] = ((hidden_size + num_kv_heads - 1) // num_kv_heads) * num_kv_heads
        
        # Ensure num_heads divisible by num_kv_heads for GQA
        if num_heads and num_kv_heads:
            if spec.get('attention') == 'gqa':
                if num_heads % num_kv_heads != 0:
                    # Adjust kv_heads to be divisor
                    divisors = find_divisors(num_heads)
                    if divisors:
                        # Find closest divisor
                        closest = min(divisors, key=lambda x: abs(x - num_kv_heads))
                        dimensions['num_kv_heads'] = closest
        
        # Ensure intermediate size matches activation
        if spec.get('activation') == 'swiglu':
            hidden = dimensions.get('hidden_size', 4096)
            # SwiGLU formula: 2 * round(8/3 * hidden / 32) * 32
            intermediate = int(2 * round(hidden * 8/3 / 32) * 32)
            dimensions['intermediate_size'] = intermediate
        
        return dimensions
    
    def _check_constraints(self, dimensions: Dict[str, int], constraint_system) -> bool:
        """Check if dimensions satisfy all constraints"""
        # Simplified check - in practice would evaluate each constraint
        return True
    
    def _solve_iterative(self,
                        template: ArchitectureTemplate,
                        spec: Dict[str, Any],
                        constraint_system,
                        fixed_vars: Dict[str, Any]) -> Solution:
        """Iterative search for solution"""
        
        target_params = spec.get('target_params')
        if target_params is None:
            raise ValueError("Iterative solving requires target_params")
        
        # Define search ranges
        search_ranges = self._define_search_ranges(template, spec, fixed_vars)
        
        # Try different combinations
        best_solution = None
        best_error = float('inf')
        
        # Simplified search - in practice would be more sophisticated
        for num_layers in range(search_ranges['num_layers'][0], 
                               search_ranges['num_layers'][1] + 1, 
                               search_ranges['num_layers'][2]):
            for hidden_size in range(search_ranges['hidden_size'][0],
                                    search_ranges['hidden_size'][1] + 1,
                                    search_ranges['hidden_size'][2]):
                
                dimensions = {
                    'num_layers': num_layers,
                    'hidden_size': hidden_size,
                    **fixed_vars
                }
                
                # Complete dimensions
                dimensions = self._ensure_complete_dimensions(dimensions, template, spec)
                dimensions = self._apply_template_adjustments(dimensions, template, spec)
                
                # Calculate parameters
                actual_params = template.calculate_parameters(dimensions)
                
                # Check error
                error = abs(actual_params - target_params) / target_params
                
                if error < best_error:
                    best_error = error
                    best_solution = {
                        'dimensions': dimensions,
                        'actual_params': actual_params,
                        'error': error
                    }
                
                self.solutions_tried += 1
                if self.solutions_tried >= self.max_solutions:
                    break
        
        if best_solution is None:
            raise ValueError("No valid solution found")
        
        return Solution(
            dimensions=best_solution['dimensions'],
            actual_params=best_solution['actual_params'],
            target_params=target_params,
            template_name=template.info.name,
            constraints_satisfied=True,
            warnings=[f"Approximate solution found (error={best_error:.1%})"],
            errors=[]
        )
    
    def _define_search_ranges(self, template, spec, fixed_vars):
        """Define reasonable search ranges for iterative solving"""
        # Based on typical LLM dimensions
        ranges = {
            'num_layers': (12, 80, 4),  # min, max, step
            'hidden_size': (2048, 16384, 512),
            'num_heads': (16, 128, 8),
        }
        
        # Adjust based on target params
        target = spec.get('target_params', 7000000000)
        
        if target < 1000000000:  # < 1B
            ranges['num_layers'] = (8, 32, 4)
            ranges['hidden_size'] = (1024, 4096, 256)
        elif target < 7000000000:  # < 7B
            ranges['num_layers'] = (24, 40, 4)
            ranges['hidden_size'] = (3072, 8192, 512)
        elif target < 13000000000:  # < 13B
            ranges['num_layers'] = (36, 48, 4)
            ranges['hidden_size'] = (5120, 10240, 512)
        else:  # > 13B
            ranges['num_layers'] = (40, 80, 4)
            ranges['hidden_size'] = (8192, 16384, 512)
        
        return ranges
