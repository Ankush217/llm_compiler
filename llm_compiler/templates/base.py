"""
Base Template Class
===================

Defines the interface that all architecture templates must implement.
Templates are responsible for:
1. Declaring their degrees of freedom
2. Defining parameter calculation formulas
3. Generating IR graphs
4. Enforcing architecture-specific constraints
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..ir.graph import IRGraph, Tensor
from ..ir.builder import GraphBuilder
from ..solver.constraints import Constraint, ConstraintSystem

class TemplateParameter(Enum):
    """Standard template parameters"""
    NUM_LAYERS = "num_layers"
    HIDDEN_SIZE = "hidden_size"
    INTERMEDIATE_SIZE = "intermediate_size"
    NUM_HEADS = "num_heads"
    NUM_KV_HEADS = "num_kv_heads"
    HEAD_DIM = "head_dim"
    VOCAB_SIZE = "vocab_size"
    CONTEXT_LENGTH = "context_length"

@dataclass
class TemplateInfo:
    """Metadata about a template"""
    name: str
    description: str
    version: str
    parameters: List[TemplateParameter]
    required_parameters: Set[TemplateParameter]
    optional_parameters: Set[TemplateParameter]
    default_constraints: List[Constraint]

class ArchitectureTemplate(ABC):
    """
    Abstract base class for all architecture templates.
    
    Templates must be deterministic and explicit - they cannot make
    arbitrary choices on behalf of the user.
    """
    
    @property
    @abstractmethod
    def info(self) -> TemplateInfo:
        """Return template metadata"""
        pass
    
    @abstractmethod
    def create_constraint_system(self, spec: Dict[str, Any]) -> ConstraintSystem:
        """
        Create constraint system for this template.
        
        Args:
            spec: User specification with some parameters fixed
            
        Returns:
            ConstraintSystem with all architectural constraints
        """
        pass
    
    @abstractmethod
    def calculate_parameters(self, dims: Dict[str, int]) -> int:
        """
        Calculate total parameter count for given dimensions.
        
        Must match exactly what the generated model will have.
        
        Args:
            dims: Complete dimension dictionary
            
        Returns:
            Total number of trainable parameters
        """
        pass
    
    @abstractmethod
    def build_ir(self, 
                 dims: Dict[str, int],
                 builder: GraphBuilder,
                 spec: Dict[str, Any]) -> IRGraph:
        """
        Build the IR graph for this architecture.
        
        Args:
            dims: Solved dimensions
            builder: Graph builder helper
            spec: Complete specification
            
        Returns:
            Complete IR graph
        """
        pass
    
    @abstractmethod
    def validate_spec(self, spec: Dict[str, Any]) -> List[str]:
        """
        Validate specification against template requirements.
        
        Args:
            spec: User specification
            
        Returns:
            List of error messages (empty if valid)
        """
        pass
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
        """
        Get reasonable bounds for each parameter.
        
        Returns:
            Dict mapping parameter name to (min, max) bounds
        """
        return {
            TemplateParameter.NUM_LAYERS.value: (1, 1000),
            TemplateParameter.HIDDEN_SIZE.value: (128, 65536),
            TemplateParameter.INTERMEDIATE_SIZE.value: (128, 262144),
            TemplateParameter.NUM_HEADS.value: (1, 256),
            TemplateParameter.NUM_KV_HEADS.value: (1, 256),
            TemplateParameter.HEAD_DIM.value: (32, 256),
            TemplateParameter.VOCAB_SIZE.value: (32, 1000000),
            TemplateParameter.CONTEXT_LENGTH.value: (1, 1000000),
        }