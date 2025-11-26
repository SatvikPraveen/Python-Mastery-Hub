"""
Python Basics Module - Comprehensive coverage of Python fundamentals.

This module provides interactive learning for Python basics including variables,
data types, control flow, functions, and error handling.
"""

from .variables_concepts import VariablesConcepts
from .data_types_concepts import DataTypesConcepts
from .control_flow_concepts import ControlFlowConcepts
from .functions_concepts import FunctionsConcepts
from .error_handling_concepts import ErrorHandlingConcepts
from .. import LearningModule

# Main module class that aggregates all concepts
class BasicsConcepts(LearningModule):
    """Interactive learning module for Python basics."""
    
    def __init__(self):
        super().__init__(
            name="Python Basics",
            description="Comprehensive coverage of Python fundamentals",
            difficulty="beginner"
        )
        
        # Initialize concept modules
        self.variables = VariablesConcepts()
        self.data_types = DataTypesConcepts()
        self.control_flow = ControlFlowConcepts()
        self.functions = FunctionsConcepts()
        self.error_handling = ErrorHandlingConcepts()
    
    def _setup_module(self) -> None:
        """Setup the learning module."""
        pass  # Topics are already set up in __init__
    
    def get_topics(self):
        """Return list of available topics."""
        return ["variables", "data_types", "control_flow", "functions", "error_handling"]
    
    def demonstrate(self, topic):
        """Demonstrate a specific topic."""
        modules = {
            "variables": self.variables,
            "data_types": self.data_types,
            "control_flow": self.control_flow,
            "functions": self.functions,
            "error_handling": self.error_handling
        }
        
        if topic not in modules:
            raise ValueError(f"Topic '{topic}' not found in basics module")
        
        return modules[topic].demonstrate()

__all__ = [
    "BasicsConcepts",
    "VariablesConcepts", 
    "DataTypesConcepts",
    "ControlFlowConcepts",
    "FunctionsConcepts",
    "ErrorHandlingConcepts"
]