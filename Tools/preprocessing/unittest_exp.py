"""
This file contains unit tests for the functions in the exploration module.
Last updated: 2026-04-25
By: Marcus Chan
"""

import unittest
import pandas as pd
import matplotlib.pyplot as plt
from exploration import show_feature_types_and_missing_values
from exploration import plot_target_distribution
from exploration import plot_correlation_heatmap
from exploration import plot_feature_distributions, fig_close

class TestExplorationFunctions(unittest.TestCase):
    def setUp(self):
        """
        Set up a sample dataset for testing the exploration functions.
        This method is called before each test case is executed to ensure that we have a consistent dataset for testing.

        Parameters:
        - None

        Responses:
        - A sample dataset is created and stored in self.data for use in the test cases.
        """
        # Create a sample dataset for testing
        self.data = pd.DataFrame({
            'A': [1, 2, 3, None, 5],
            'B': ['a', 'b', None, None, 'e'],
            'C': [10.5, 20.5, None, 40.5, 50.5]
        })

        self.data2 = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [23, 24, 25, 26, 27],
            'C': [10.5, 20.5, 30.5, 40.5, 50.5]
        })

    def test_show_feature_types_and_missing_values(self):
        """
        Tests the show_feature_types_and_missing_values function to ensure it correctly identifies 
        feature types and counts missing values.

        Parameters:
        - None

        Responses:
        - The output of the function is captured and compared against the expected output to verify correctness.
        """
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
            "A      float64               1\n"
            "B          str               2\n"
            "C      float64               1\n"
        )
        print(self.assertEqual(output.strip(), expected_output.strip()))

    def test_plot_target_distribution(self):
        """
        Tests the plot_target_distribution function to ensure it runs without errors 
        when given a valid dataset and target column.

        Parameters:
        - None

        Responses:
        - The function is executed, and if it raises an exception, the test will fail with an appropriate message.
        """
        # This test will check if the function runs without errors
        try:
            plot_target_distribution(self.data, 'B')
            # We can also check if the plot is created, but since we are not displaying it in a test environment, 
            # we will just check for exceptions
            self.assertIsNotNone(plt.gcf())
            fig_close()
        except Exception as e:
            self.fail(f"plot_target_distribution raised an exception: {e}")

    def test_plot_correlation_heatmap(self):
        """
        Tests the plot_correlation_heatmap function to ensure it runs without errors 
        when given a valid dataset.

        Parameters:
        - None

        Responses:
        - The function is executed, and if it raises an exception, the test will fail with an appropriate message.
        """
        # This test will check if the function runs without errors
        try:
            plot_correlation_heatmap(self.data2)
            # We can also check if the plot is created, but since we are not displaying it in a test environment, 
            # we will just check for exceptions
            self.assertIsNotNone(plt.gcf())
            fig_close()
        except Exception as e:
            self.fail(f"plot_correlation_heatmap raised an exception: {e}")

    def test_plot_feature_distributions(self):
        """
        Tests the plot_feature_distributions function to ensure it runs without errors 
        when given a valid dataset.

        Parameters:
        - None

        Responses:
        - The function is executed, and if it raises an exception, the test will fail with an appropriate message.
        """
        # This test will check if the function runs without errors
        try:
            plot_feature_distributions(self.data)
            # We can also check if the plot is created, but since we are not displaying it in a test environment, 
            # we will just check for exceptions
            self.assertIsNotNone(plt.gcf())
            fig_close()
        except Exception as e:
            self.fail(f"plot_feature_distributions raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()