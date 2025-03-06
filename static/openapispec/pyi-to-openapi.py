#!/usr/bin/env python3
"""
pyi_to_openapi.py - Convert Python Interface (.pyi) files to OpenAPI specifications

This script parses Python Interface (.pyi) files and converts them into OpenAPI 3.0
specifications. It extracts function signatures, docstrings, parameters, return types,
and other metadata to generate a structured API documentation.

Usage:
    python pyi_to_openapi.py input.pyi -o output.yaml

Author: Claude
"""

import argparse
import ast
import json
import os
import re
import sys
import yaml
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class PyiVisitor(ast.NodeVisitor):
    """AST visitor to extract function and class information from .pyi files."""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.imports = {}
        self.current_class = None
        self.module_docstring = None
        
    def visit_Module(self, node):
        """Extract module docstring."""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            self.module_docstring = node.body[0].value.s
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Handle import statements."""
        for name in node.names:
            self.imports[name.name] = name.name
    
    def visit_ImportFrom(self, node):
        """Handle from-import statements."""
        module = node.module or ""
        for name in node.names:
            if name.asname:
                self.imports[name.asname] = f"{module}.{name.name}"
            else:
                self.imports[name.name] = f"{module}.{name.name}"
    
    def visit_ClassDef(self, node):
        """Extract class definitions."""
        old_class = self.current_class
        self.current_class = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'methods': [],
            'bases': [self._extract_name(base) for base in node.bases],
            'decorators': [self._extract_name(d) for d in node.decorator_list]
        }
        self.classes.append(self.current_class)
        
        # Visit class body
        for child in node.body:
            self.visit(child)
            
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        """Extract function definitions."""
        docstring = ast.get_docstring(node)
        func_info = {
            'name': node.name,
            'docstring': docstring,
            'params': self._extract_params(node.args),
            'returns': self._extract_returns(node),
            'decorators': [self._extract_name(d) for d in node.decorator_list],
        }
        
        # Parse docstring for more detailed information
        if docstring:
            func_info.update(self._parse_docstring(docstring))
        
        if self.current_class:
            self.current_class['methods'].append(func_info)
        else:
            self.functions.append(func_info)
    
    def _extract_name(self, node):
        """Extract name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._extract_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._extract_name(node.func)
        elif isinstance(node, ast.Subscript):
            value = self._extract_name(node.value)
            if isinstance(node.slice, ast.Index):
                slice_value = self._extract_name(node.slice.value)
            else:
                slice_value = self._extract_name(node.slice)
            return f"{value}[{slice_value}]"
        elif isinstance(node, ast.Tuple):
            elts = [self._extract_name(e) for e in node.elts]
            return f"({', '.join(elts)})"
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            return str(node.value)
        else:
            return str(node)
    
    def _extract_params(self, args):
        """Extract function parameters."""
        params = []
        
        # Add positional-only parameters
        if hasattr(args, 'posonlyargs'):  # Python 3.8+
            for arg in args.posonlyargs:
                params.append({
                    'name': arg.arg,
                    'type': self._extract_name(arg.annotation) if arg.annotation else None,
                    'kind': 'POSITIONAL_ONLY'
                })
        
        # Add regular parameters
        for arg in args.args:
            params.append({
                'name': arg.arg,
                'type': self._extract_name(arg.annotation) if arg.annotation else None,
                'kind': 'POSITIONAL_OR_KEYWORD'
            })
        
        # Add *args
        if args.vararg:
            params.append({
                'name': args.vararg.arg,
                'type': self._extract_name(args.vararg.annotation) if args.vararg.annotation else None,
                'kind': 'VAR_POSITIONAL'
            })
        
        # Add keyword-only parameters
        for arg in args.kwonlyargs:
            params.append({
                'name': arg.arg,
                'type': self._extract_name(arg.annotation) if arg.annotation else None,
                'kind': 'KEYWORD_ONLY'
            })
        
        # Add **kwargs
        if args.kwarg:
            params.append({
                'name': args.kwarg.arg,
                'type': self._extract_name(args.kwarg.annotation) if args.kwarg.annotation else None,
                'kind': 'VAR_KEYWORD'
            })
        
        return params
    
    def _extract_returns(self, node):
        """Extract function return type."""
        if node.returns:
            return self._extract_name(node.returns)
        return None
    
    def _parse_docstring(self, docstring):
        """Parse docstring to extract parameter descriptions and return info."""
        result = {
            'param_descriptions': {},
            'return_description': None,
            'examples': [],
            'raises': []
        }
        
        lines = docstring.split('\n')
        current_section = None
        current_param = None
        buffer = []
        
        # Regex patterns for docstring sections
        param_pattern = re.compile(r'^\s*(?:Args?|Parameters?):')
        returns_pattern = re.compile(r'^\s*(?:Returns?|Return Value):')
        raises_pattern = re.compile(r'^\s*(?:Raises?|Exceptions?):')
        examples_pattern = re.compile(r'^\s*Examples?:')
        param_item_pattern = re.compile(r'^\s*(\w+)(?:\s*\(([^)]+)\))?\s*:(.*)$')
        
        for line in lines:
            # Check if this line starts a new section
            if param_pattern.match(line):
                current_section = 'params'
                continue
            elif returns_pattern.match(line):
                current_section = 'returns'
                buffer = []
                continue
            elif raises_pattern.match(line):
                current_section = 'raises'
                buffer = []
                continue
            elif examples_pattern.match(line):
                current_section = 'examples'
                buffer = []
                continue
            
            # Process the line based on current section
            if current_section == 'params':
                param_match = param_item_pattern.match(line)
                if param_match:
                    current_param = param_match.group(1)
                    param_type = param_match.group(2)  # This might be None
                    description = param_match.group(3).strip()
                    result['param_descriptions'][current_param] = {
                        'description': description,
                        'type': param_type
                    }
                elif line.strip() and current_param:
                    # Continue previous parameter description
                    result['param_descriptions'][current_param]['description'] += ' ' + line.strip()
            elif current_section == 'returns':
                if line.strip():
                    buffer.append(line.strip())
                result['return_description'] = ' '.join(buffer)
            elif current_section == 'raises':
                if line.strip():
                    buffer.append(line.strip())
                result['raises'] = self._parse_exception_section(' '.join(buffer))
            elif current_section == 'examples':
                if line.strip():
                    buffer.append(line)
                result['examples'] = buffer
        
        return result
    
    def _parse_exception_section(self, section_text):
        """Parse the exceptions section to extract raised exceptions."""
        exceptions = []
        # Simple regex to find exception names (might need improvement)
        exception_matches = re.finditer(r'(\w+Error|ValueError|TypeError|Exception)(?:\s*-\s*|\s*:\s*)(.*?)(?=\w+Error|\Z)', section_text + ' ')
        
        for match in exception_matches:
            exceptions.append({
                'type': match.group(1),
                'description': match.group(2).strip()
            })
        
        return exceptions


def parse_pyi_file(file_path):
    """Parse a .pyi file and extract its contents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        visitor = PyiVisitor()
        visitor.visit(tree)
        return {
            'module_docstring': visitor.module_docstring,
            'functions': visitor.functions,
            'classes': visitor.classes,
            'imports': visitor.imports
        }
    except SyntaxError as e:
        print(f"Error parsing {file_path}: {e}")
        sys.exit(1)


def python_type_to_openapi(py_type):
    """Convert Python type annotations to OpenAPI schema types."""
    if not py_type:
        return {"type": "object"}
    
    # Basic type mappings
    type_map = {
        'str': {"type": "string"},
        'int': {"type": "integer"},
        'float': {"type": "number"},
        'bool': {"type": "boolean"},
        'None': {"type": "null"},
        'Any': {},  # No type constraints
        'dict': {"type": "object"},
        'Dict': {"type": "object"},
        'list': {"type": "array"},
        'List': {"type": "array"},
        'tuple': {"type": "array"},
        'Tuple': {"type": "array"},
        'Set': {"type": "array", "uniqueItems": True},
        'Optional': {},  # Will be processed below
    }
    
    # Check for Optional[Type]
    optional_match = re.match(r'Optional\[(.*)\]', py_type)
    if optional_match:
        inner_type = optional_match.group(1)
        schema = python_type_to_openapi(inner_type)
        schema['nullable'] = True
        return schema
    
    # Check for Union types
    union_match = re.match(r'Union\[(.*)\]', py_type)
    if union_match:
        types = [t.strip() for t in union_match.group(1).split(',')]
        schemas = [python_type_to_openapi(t) for t in types]
        
        # If one of the types is None, we can use nullable instead
        if any(s.get('type') == 'null' for s in schemas):
            non_null_schemas = [s for s in schemas if s.get('type') != 'null']
            if len(non_null_schemas) == 1:
                schema = non_null_schemas[0]
                schema['nullable'] = True
                return schema
        
        # Otherwise use oneOf
        return {"oneOf": schemas}
    
    # Check for List[Type], Dict[Type, Type], etc.
    generic_match = re.match(r'(\w+)\[(.*)\]', py_type)
    if generic_match:
        container_type = generic_match.group(1)
        inner_types = [t.strip() for t in generic_match.group(2).split(',')]
        
        if container_type in ('List', 'list', 'Sequence', 'Iterable'):
            if inner_types:
                return {
                    "type": "array",
                    "items": python_type_to_openapi(inner_types[0])
                }
            return {"type": "array"}
        
        elif container_type in ('Dict', 'dict'):
            if len(inner_types) >= 2:
                # For simplicity, we'll assume the key is a string in OpenAPI
                return {
                    "type": "object",
                    "additionalProperties": python_type_to_openapi(inner_types[1])
                }
            return {"type": "object"}
        
        elif container_type in ('Set', 'set'):
            if inner_types:
                return {
                    "type": "array",
                    "uniqueItems": True,
                    "items": python_type_to_openapi(inner_types[0])
                }
            return {"type": "array", "uniqueItems": True}
        
        elif container_type == 'Tuple':
            if inner_types:
                return {
                    "type": "array",
                    "items": [python_type_to_openapi(t) for t in inner_types],
                    "minItems": len(inner_types),
                    "maxItems": len(inner_types)
                }
            return {"type": "array"}
    
    # Check for Literal types
    literal_match = re.match(r'Literal\[(.*)\]', py_type)
    if literal_match:
        literals = literal_match.group(1).split(',')
        # Remove quotes from string literals
        enum_values = []
        for lit in literals:
            lit = lit.strip()
            if (lit.startswith('"') and lit.endswith('"')) or (lit.startswith("'") and lit.endswith("'")):
                enum_values.append(lit[1:-1])
            elif lit.lower() == 'true':
                enum_values.append(True)
            elif lit.lower() == 'false':
                enum_values.append(False)
            elif lit.isdigit():
                enum_values.append(int(lit))
            else:
                enum_values.append(lit)
        
        return {"enum": enum_values}
    
    # If it's a basic type, return mapped value
    if py_type in type_map:
        return type_map[py_type]
    
    # For custom/complex types, use string format
    return {"type": "string", "format": py_type}


def generate_openapi_schema(parsed_pyi, title, version="1.0.0"):
    """Generate an OpenAPI schema from parsed .pyi content."""
    # Initialize OpenAPI structure
    openapi = {
        "openapi": "3.0.0",
        "info": {
            "title": title,
            "version": version,
            "description": parsed_pyi.get('module_docstring', '')
        },
        "paths": {},
        "components": {
            "schemas": {},
            "parameters": {},
            "responses": {},
            "securitySchemes": {}
        }
    }
    
    # Create schema components for classes
    for cls in parsed_pyi.get('classes', []):
        properties = {}
        required = []
        
        # Add methods as properties
        for method in cls.get('methods', []):
            if method['name'].startswith('_') and method['name'] != '__init__':
                continue  # Skip private methods except __init__
            
            properties[method['name']] = {
                "type": "object",
                "description": method.get('docstring', ''),
            }
        
        openapi['components']['schemas'][cls['name']] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "description": cls.get('docstring', '')
        }
    
    # Create paths for functions
    for func in parsed_pyi.get('functions', []):
        if func['name'].startswith('_'):
            continue  # Skip private functions
        
        path = f"/{func['name']}"
        parameters = []
        request_body = None
        
        # Process parameters
        for param in func.get('params', []):
            param_name = param['name']
            param_type = param['type']
            
            # Skip self and cls parameters
            if param_name in ('self', 'cls'):
                continue
            
            # Get parameter description from docstring
            param_desc = ""
            if param_name in func.get('param_descriptions', {}):
                param_desc = func['param_descriptions'][param_name].get('description', '')
            
            if param['kind'] in ('POSITIONAL_ONLY', 'POSITIONAL_OR_KEYWORD', 'KEYWORD_ONLY'):
                parameters.append({
                    "name": param_name,
                    "in": "query",  # Default to query - in a real API this could be path or body
                    "description": param_desc,
                    "schema": python_type_to_openapi(param_type),
                    "required": param['kind'] != 'KEYWORD_ONLY'  # Assume keyword-only are optional
                })
            elif param['kind'] == 'VAR_KEYWORD':
                # **kwargs - these would typically be in the request body
                request_body = {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "additionalProperties": True
                            }
                        }
                    }
                }
        
        # Create responses
        responses = {
            "200": {
                "description": func.get('return_description', 'Successful response'),
                "content": {
                    "application/json": {
                        "schema": python_type_to_openapi(func.get('returns'))
                    }
                }
            }
        }
        
        # Add error responses
        for error in func.get('raises', []):
            error_code = "400" if "Value" in error.get('type', '') else "500"
            responses[error_code] = {
                "description": f"{error.get('type', 'Error')}: {error.get('description', '')}"
            }
        
        # Create the path item
        path_item = {
            "post": {  # Default to POST for functions - could be different based on semantics
                "summary": func['name'],
                "description": func.get('docstring', ''),
                "parameters": parameters,
                "responses": responses
            }
        }
        
        if request_body:
            path_item["post"]["requestBody"] = request_body
        
        openapi['paths'][path] = path_item
    
    return openapi


def main():
    """Main function to parse arguments and convert .pyi to OpenAPI."""
    parser = argparse.ArgumentParser(description='Convert Python Interface (.pyi) files to OpenAPI specifications')
    parser.add_argument('input', help='Input .pyi file')
    parser.add_argument('-o', '--output', help='Output file (YAML or JSON)', default='openapi_spec.yaml')
    parser.add_argument('-t', '--title', help='API title', default=None)
    parser.add_argument('-v', '--version', help='API version', default='1.0.0')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    if not args.input.endswith('.pyi'):
        print(f"Warning: Input file '{args.input}' does not have .pyi extension")
    
    # Parse the .pyi file
    parsed_pyi = parse_pyi_file(args.input)
    
    # Determine title from filename if not provided
    title = args.title or os.path.splitext(os.path.basename(args.input))[0]
    
    # Generate OpenAPI schema
    openapi_schema = generate_openapi_schema(parsed_pyi, title, args.version)
    
    # Output format based on file extension
    is_json = args.output.endswith('.json')
    
    with open(args.output, 'w', encoding='utf-8') as f:
        if is_json:
            json.dump(openapi_schema, f, indent=2)
        else:
            yaml.dump(openapi_schema, f, sort_keys=False, default_flow_style=False)
    
    print(f"Successfully converted {args.input} to OpenAPI spec at {args.output}")


if __name__ == "__main__":
    main()
