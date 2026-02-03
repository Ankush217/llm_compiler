"""
Main Compilation Entry Point
============================

Orchestrates the entire compilation pipeline.
"""

from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path
import shutil
import sys
import hashlib

from .spec import LLM
from .templates.registry import get_template
from .solver.param_solver import ParameterSolver
from .ir.builder import GraphBuilder
from .emitters.pytorch import PyTorchEmitter
from .emitters.safetensors import SafetensorsEmitter
from .utils.validation import validate_spec
from .utils.parameters import count_parameters_from_ir

class CompileResult(dict):
    """Dictionary report with IR helpers for convenience access."""
    def __init__(self, data: Dict[str, Any], ir_graph):
        super().__init__(data)
        self.ir_graph = ir_graph
        blob = ir_graph.to_json().encode()
        self._fingerprint = hashlib.sha256(blob).hexdigest()
        self.parameter_count = ir_graph.get_parameter_count()

    def fingerprint(self):
        return self._fingerprint


class LLMCompiler:
    """Main compiler class"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.solver = ParameterSolver()
        self.pt_emitter = PyTorchEmitter()
        self.st_emitter = SafetensorsEmitter()
        
    def compile(self, spec: LLM, output_dir: Path, dataset=None) -> Dict[str, Any]:
        """
        Compile LLM from specification.
        
        Args:
            spec: LLM specification
            output_dir: Output directory
            
        Returns:
            Compilation report
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start compilation report
        report = {
            "spec": spec.to_dict(),
            "steps": [],
            "success": False,
            "output_dir": str(output_dir),
            "files": []
        }
        
        try:
            # Step 1: Get template
            self._log("Step 1: Getting template")
            template = get_template(spec.template)
            report["steps"].append({
                "step": "template_selection",
                "template": template.info.name,
                "status": "success"
            })
            
            # Step 2: Validate specification
            self._log("Step 2: Validating specification")
            validation_errors = template.validate_spec(spec.to_dict())
            if validation_errors:
                raise ValueError(f"Specification validation failed: {validation_errors}")
            # Cross-check tokenizer/dataset coupling
            if dataset is not None:
                ds_stats = getattr(dataset, "tokenizer_stats", None)
                _, ds_errors = validate_spec(spec, tokenizer_stats=ds_stats, dataset=dataset)
                if ds_errors:
                    raise ValueError(f"Dataset validation failed: {ds_errors}")
            
            report["steps"].append({
                "step": "spec_validation",
                "status": "success"
            })
            if dataset is not None:
                report["dataset"] = getattr(dataset, "to_dict", lambda: {})()
            
            # Step 3: Solve for dimensions
            self._log("Step 3: Solving architecture dimensions")
            spec_dict = spec.to_dict()
            solution = self.solver.solve(template, spec_dict)
            
            if solution.errors:
                raise ValueError(f"Constraint solving failed: {solution.errors}")
            
            report["solution"] = {
                "dimensions": solution.dimensions,
                "actual_params": solution.actual_params,
                "target_params": solution.target_params,
                "warnings": solution.warnings,
                "constraints_satisfied": solution.constraints_satisfied
            }
            
            report["steps"].append({
                "step": "dimension_solving",
                "status": "success",
                "parameters": solution.actual_params,
                "warnings": solution.warnings
            })
            
            # Step 4: Build IR graph
            self._log("Step 4: Building IR graph")
            builder = GraphBuilder()
            ir_graph = template.build_ir(
                solution.dimensions,
                builder,
                spec_dict
            )
            
            # Validate graph
            graph_errors = ir_graph.validate()
            if graph_errors:
                raise ValueError(f"IR graph validation failed: {graph_errors}")
            
            report["steps"].append({
                "step": "ir_generation",
                "status": "success",
                "nodes": len(ir_graph.operations),
                "tensors": len(ir_graph.tensors)
            })

            # Step 5: Canonical parameter count from IR
            self._log("Step 5: Computing IR parameter count")
            ir_param_count = count_parameters_from_ir(ir_graph)

            # Cross-validate against template estimate
            template_param_estimate = solution.actual_params
            relative_diff = 0.0
            if template_param_estimate:
                relative_diff = abs(ir_param_count - template_param_estimate) / template_param_estimate
                if relative_diff > 0.01:
                    note_msg = (
                        f"Template estimate differs from IR "
                        f"({template_param_estimate:,} vs {ir_param_count:,}); "
                        f"IR is canonical."
                    )
                    report.setdefault("notes", []).append(note_msg)

            # Promote IR count to canonical value
            solution.actual_params = ir_param_count
            solution.dimensions["total_params"] = ir_param_count
            report["solution"]["actual_params"] = ir_param_count
            report["solution"]["param_source"] = "ir_graph"

            report["steps"].append({
                "step": "parameter_accounting",
                "status": "success",
                "parameters_ir": ir_param_count,
                "template_estimate": template_param_estimate,
                "relative_diff": relative_diff
            })
            
            # Step 6: Emit code
            self._log("Step 6: Emitting code")
            
            # Generate model name
            model_name = self._generate_model_name(spec, solution.dimensions)
            
            # Emit PyTorch code
            pt_files = self.pt_emitter.emit(
                ir_graph,
                model_name,
                output_dir / "model"
            )
            
            # Write files
            model_dir = output_dir / "model"
            model_dir.mkdir(exist_ok=True)
            
            for filename, content in pt_files.items():
                file_path = model_dir / filename
                file_path.write_text(content)
                report["files"].append({
                    "path": str(file_path.relative_to(output_dir)),
                    "type": "code",
                    "size": len(content)
                })
            
            # Step 7: Generate weights (placeholder)
            self._log("Step 7: Generating weights structure")
            
            # Create weights directory
            weights_dir = output_dir / "weights"
            weights_dir.mkdir(exist_ok=True)
            
            # Generate weight metadata
            weight_names = self.st_emitter.generate_weight_names(
                model_name,
                solution.dimensions.get('num_layers', 32)
            )
            
            # Create weight manifest
            weight_manifest = {
                "model_name": model_name,
                "total_parameters": solution.actual_params,
                "weights": weight_names,
                "dtype": spec.precision,
                "quantization": spec.quantize
            }
            
            manifest_path = weights_dir / "manifest.json"
            manifest_path.write_text(json.dumps(weight_manifest, indent=2))
            
            report["files"].append({
                "path": str(manifest_path.relative_to(output_dir)),
                "type": "manifest",
                "size": manifest_path.stat().st_size
            })
            
            # Step 8: Generate compilation report
            self._log("Step 8: Generating final report")
            
            # Save spec
            spec_path = output_dir / "spec.json"
            spec_path.write_text(json.dumps(spec.to_dict(), indent=2))
            
            # Save solution
            solution_path = output_dir / "solution.json"
            solution_path.write_text(json.dumps({
                "dimensions": solution.dimensions,
                "parameters": solution.actual_params
            }, indent=2))
            
            # Save IR graph
            ir_path = output_dir / "ir_graph.json"
            ir_path.write_text(ir_graph.to_json())
            
            # Update report
            report.update({
                "success": True,
                "model_name": model_name,
                "model_dir": str(model_dir.relative_to(output_dir)),
                "weights_dir": str(weights_dir.relative_to(output_dir)),
                "total_files": len(report["files"]),
                "compilation_time": "N/A"  # Would track actual time
            })
            
            report["steps"].append({
                "step": "finalization",
                "status": "success"
            })
            
            # Write final report
            report_path = output_dir / "compilation_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            
            self._log(f"\nCompilation successful!")
            self._log(f"Model: {model_name}")
            self._log(f"Parameters: {solution.actual_params:,}")
            self._log(f"Output directory: {output_dir}")
            
            if solution.warnings:
                self._log("\nWarnings:")
                for warning in solution.warnings:
                    self._log(f"  • {warning}")
            
            return CompileResult(report, ir_graph)
            
        except Exception as e:
            report.update({
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            report["steps"].append({
                "step": "error",
                "status": "failed",
                "error": str(e)
            })
            
            # Write error report
            error_report = output_dir / "error_report.json"
            try:
                error_report.write_text(json.dumps(report, indent=2))
            except Exception:
                pass
            
            self._log(f"\nCompilation failed: {e}")
            raise
    
    def _generate_model_name(self, spec: LLM, dims: Dict[str, int]) -> str:
        """Generate meaningful model name"""
        # Extract key information
        template = spec.template.replace("_", "-")
        
        # Get parameter count in billions
        param_b = dims.get('total_params', 0) / 1e9
        
        # Round to nearest decimal
        if param_b < 1:
            param_str = f"{param_b * 1000:.0f}M"
        else:
            param_str = f"{param_b:.1f}B".replace(".", "")
        
        # Architecture features
        att = getattr(spec.attention, "value", spec.attention)
        norm = getattr(spec.norm, "value", spec.norm)
        act = getattr(spec.activation, "value", spec.activation)

        features = []
        if att != "gqa":
            features.append(att)
        if norm != "rmsnorm":
            features.append(norm)
        if act != "swiglu":
            features.append(act)
        
        # Build name
        name_parts = [f"llmc-{template}", param_str]
        if features:
            name_parts.append("-".join(features[:2]))
        
        return "-".join(name_parts).lower()
    
    def _log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(message)

def compile_spec(spec: LLM, 
                 output_dir: Path = Path("out_model"),
                 verbose: bool = False,
                 dataset=None) -> Dict[str, Any]:
    """
    Compile LLM from specification.
    
    Args:
        spec: LLM specification
        output_dir: Output directory path
        verbose: Enable verbose logging
        
    Returns:
        Compilation report
    """
    compiler = LLMCompiler(verbose=verbose)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        # Resolve relative paths relative to the package root to stay inside writable area
        output_dir = Path(__file__).resolve().parent / output_dir
    return compiler.compile(spec, output_dir, dataset=dataset)

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Architecture Compiler")
    parser.add_argument("--spec", type=str, required=True,
                       help="Specification file (JSON)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Load spec
    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Error: Spec file not found: {spec_path}")
        sys.exit(1)
    
    spec_data = json.loads(spec_path.read_text())
    spec = LLM.from_dict(spec_data)
    
    # Compile
    try:
        report = compile_spec(spec, args.output, args.verbose)
        
        if report["success"]:
            print(f"\n✅ Compilation successful!")
            print(f"📁 Output: {report['output_dir']}")
            print(f"🤖 Model: {report['model_name']}")
            print(f"🧮 Parameters: {report['solution']['actual_params']:,}")
            sys.exit(0)
        else:
            print(f"\n❌ Compilation failed: {report.get('error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Compilation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
