"""
LLM Architecture Compiler
=========================

A deterministic, explicit system for defining, generating, and emitting
Large Language Models from declarative specifications.

Philosophy:
- Architecture-first, not checkpoint-first
- Templates, not ad-hoc classes
- Constraint solving instead of guessing
- Compiler mindset, not framework mindset
- Everything reproducible from spec alone

Key Components:
1. Templates: Canonical architecture definitions
2. Specification: Single declarative model spec
3. Solver: Architecture constraint solver
4. IR: Framework-agnostic intermediate representation
5. Emitters: Backend-specific code generation
"""

__version__ = "1.0.0"
__all__ = ['LLM', 'compile_spec', 'validate_spec', 'list_templates',
           'DatasetSpec', 'DatasetCompiler', 'DatasetIR',
           'TokenizerSpec', 'TokenizerIR', 'TokenizerCompiler',
           'TrainingSpec', 'TrainingIR', 'OptimizerSpec', 'OptimizerIR',
           'run_training']

from .spec import LLM
from .compile import compile_spec
from .templates.registry import list_templates
from .utils.validation import validate_spec
from .dataset.spec import DatasetSpec
from .dataset import DatasetCompiler, DatasetIR
from .tokenizer import TokenizerSpec, TokenizerIR, TokenizerCompiler
from .training import TrainingSpec, TrainingIR, OptimizerSpec, OptimizerIR
from .execution.run import run_training
