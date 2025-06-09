"""
Python Implementation of Reconfigurable Terahertz Holograms

Based on the paper 'Machine Learning Enables Multi-Degree-of-Freedom Reconfigurable Terahertz Holograms 
with Cascaded Diffractive Optical Elements'.

This program implements the DONN (Diffractive Optical Neural Network) model described in the paper, including:
1. Design and optimization of cascaded diffractive layers
2. Rayleigh-Sommerfeld diffraction propagation model
3. Machine learning-based pixel height optimization
4. Multiple reconfigurable scenarios simulation and visualization

Author: Manus AI
Date: 2025-05-23
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from donn_model import DONN
from rayleigh_sommerfeld import RayleighSommerfeldPropagation
from training import DONNTrainer
from scenarios import scenario1_varying_number_of_elements, scenario5_rotating_elements, combined_scenario_translation_rotation
from wavelength_scenario import wavelength_dependent_scenario
from visualization import visualize_all_results

def main():
    """
    Main function, runs the complete reconfigurable terahertz hologram simulation process
    """
    print("Starting reconfigurable terahertz hologram simulation...")
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir = "/home/ubuntu/terahertz_hologram/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Scenario 1: Varying the number of cascaded elements
    print("\nRunning Scenario 1: Varying the number of cascaded elements")
    model1, results1 = scenario1_varying_number_of_elements()
    
    # Run Scenario 5: Rotating diffractive layers
    print("\nRunning Scenario 5: Rotating diffractive layers")
    model5, results5 = scenario5_rotating_elements()
    
    # Run Combined Scenario: Translation and Rotation
    print("\nRunning Combined Scenario: Translation and Rotation")
    model_combined, results_combined = combined_scenario_translation_rotation()
    
    # Run Wavelength-Dependent Scenario
    print("\nRunning Wavelength-Dependent Scenario")
    model_wavelength, results_wavelength = wavelength_dependent_scenario()
    
    # Collect all results
    all_results = {
        'scenario1': {'model': model1, 'results': results1},
        'scenario5': {'model': model5, 'results': results5},
        'combined': {'model': model_combined, 'results': results_combined},
        'wavelength': {'model': model_wavelength, 'results': results_wavelength}
    }
    
    # Visualize all results
    print("\nGenerating visualization results...")
    visualize_all_results(all_results, output_dir)
    
    print("\nSimulation complete! All results saved to", output_dir)

if __name__ == "__main__":
    main()
