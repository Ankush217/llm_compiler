"""
Template Registry
================

Manages available architecture templates.
Central registry for all template implementations.
"""

from typing import Dict, Type, List
import inspect

from .base import ArchitectureTemplate
from .decoder_only import DecoderOnlyTemplate
from .encoder_decoder import EncoderDecoderTemplate

class TemplateRegistry:
    """Registry for all architecture templates"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._templates = {}
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize with built-in templates"""
        self.register(DecoderOnlyTemplate())
        self.register(DecoderOnlyTemplate(version="v2"))
        self.register(EncoderDecoderTemplate())
    
    def register(self, template: ArchitectureTemplate):
        """Register a template"""
        self._templates[template.info.name] = template
    
    def get(self, name: str) -> ArchitectureTemplate:
        """Get template by name"""
        if name in self._templates:
            return self._templates[name]
        # Fallback: append _v1 if not provided
        alt = f"{name}_v1"
        if alt in self._templates:
            return self._templates[alt]
        raise ValueError(f"Template not found: {name}. "
                       f"Available: {list(self._templates.keys())}")
    
    def list(self) -> List[str]:
        """List all registered template names"""
        return list(self._templates.keys())
    
    def list_with_info(self) -> Dict[str, dict]:
        """List templates with their metadata"""
        return {
            name: {
                'description': template.info.description,
                'version': template.info.version,
                'parameters': [p.value for p in template.info.parameters],
                'required': [p.value for p in template.info.required_parameters],
            }
            for name, template in self._templates.items()
        }

# Global registry instance
registry = TemplateRegistry()

def get_template(name: str) -> ArchitectureTemplate:
    """Get template from global registry"""
    return registry.get(name)

def list_templates() -> List[str]:
    """List all available templates"""
    return registry.list()

def list_templates_with_info() -> Dict[str, dict]:
    """List templates with metadata"""
    return registry.list_with_info()
