"""
This file contains unit tests for the functions in the exploration module.
Last updated: 2026-04-25
By: Marcus Chan
"""

import unittest
import pandas as pd
from preprocessing.exploration import show_feature_types_and_missing_values

class TestExplorationFunctions(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset for testing
        self.data = pd.DataFrame({
            'A': [1, 2, 3, None, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [10.5, 20.5, None, 40.5, 50.5]
        })

    def test_show_feature_types_and_missing_values(self):
        # Capture the output of the function
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output

        show_feature_types_and_missing_values(self.data)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        expected_output = (
            "Feature Type  Missing Values\n"
            "A           float64               1\n"
            "B            object               0\n"
            "C           float64               1\n"
        )

        self.assertEqual(output.strip(), expected_output.strip())