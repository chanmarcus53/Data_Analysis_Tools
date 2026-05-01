"""
Initialization file for data ingestion.
This file sets up the necessary imports and functions for loading, validating, profiling, and reporting on datasets.
Last updated: 2026-04-30
By: Marcus Chan
"""

from .loaders import load
from .validator import validate
from .profiler import profile
from .report import report