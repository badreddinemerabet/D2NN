"""
Multi-Scenario Reconfigurable Simulation Implementation

Based on the paper 'Machine Learning Enables Multi-Degree-of-Freedom Reconfigurable Terahertz Holograms 
with Cascaded Diffractive Optical Elements'.

This module implements various reconfigurable scenarios described in the paper, including 
varying the number of cascaded elements and rotating diffractive layers.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from donn_model import DONN
from rayleigh_sommerfeld import RayleighSommerfeldPropagation, visualize_field
from training import DONNTrainer

def scenario1_varying_number_of_elements():
    """
    Scenario 1: Varying the number of cascaded elements
    
    Implements the first scenario from the paper, generating different holographic images 
    by changing the number of cascaded elements:
    - Using only the first layer generates letter "T"
    - Using only the second layer generates letter "H"
    - Cascading both layers generates letter "Z"
    
    Returns:
        Trained DONN model and evaluation results
    """
    print("Implementing Scenario 1: Varying the number of cascaded elements")
    
    # Create output directory
    output_dir = "/home/ubuntu/terahertz_hologram/results/scenario1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DONN model
    model = DONN(
        wavelength=1e-3,  # 1mm corresponds to 0.3THz
        pixel_size=0.8e-3,  # 0.8mm
        pixel_count=60,  # 60x60 pixels
        refractive_index=1.7,  # Refractive index 1.7
        max_height=1.4e-3  # Maximum height 1.4mm
    )
    
    # Add two diffractive layers
    model.add_layer()  # First layer
    model.add_layer()  # Second layer
    
    # Initialize trainer
    trainer = DONNTrainer(model, learning_rate=0.07)
    
    # Define training scenarios
    scenarios = [
        {
            'target': 'T',
            'layer_indices': [0],  # Use only first layer
            'distances': [40e-3]  # 40mm propagation distance
        },
        {
            'target': 'H',
            'layer_indices': [1],  # Use only second layer
            'distances': [40e-3]  # 40mm propagation distance
        },
        {
            'target': 'Z',
            'layer_indices': [0, 1],  # Cascade both layers
            'distances': [10e-3, 30e-3]  # 10mm between layers, 30mm to imaging plane
        }
    ]
    
    # Train model
    loss_history = trainer.train(scenarios, epochs=400, save_dir=output_dir)
    
    # Evaluate model
    eval_scenarios = [
        {
            'name': 'Scenario1_T',
            'target': 'T',
            'layer_indices': [0],
            'distances': [40e-3]
        },
        {
            'name': 'Scenario1_H',
            'target': 'H',
            'layer_indices': [1],
            'distances': [40e-3]
        },
        {
            'name': 'Scenario1_Z',
            'target': 'Z',
            'layer_indices': [0, 1],
            'distances': [10e-3, 30e-3]
        }
    ]
    
    results = trainer.evaluate(eval_scenarios, save_dir=output_dir)
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Scenario 1: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    return model, results

def scenario5_rotating_elements():
    """
    Scenario 5: Rotating diffractive layers
    
    Implements the fifth scenario from the paper, generating different holographic images 
    by rotating the second diffractive layer:
    - 0° rotation generates letter "U"
    - 90° rotation generates letter "T"
    - 180° rotation generates letter "A"
    - 270° rotation generates letter "H"
    
    Returns:
        Trained DONN model and evaluation results
    """
    print("Implementing Scenario 5: Rotating diffractive layers")
    
    # Create output directory
    output_dir = "/home/ubuntu/terahertz_hologram/results/scenario5"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DONN model
    model = DONN(
        wavelength=1e-3,  # 1mm corresponds to 0.3THz
        pixel_size=0.8e-3,  # 0.8mm
        pixel_count=60,  # 60x60 pixels
        refractive_index=1.7,  # Refractive index 1.7
        max_height=1.4e-3  # Maximum height 1.4mm
    )
    
    # Add two diffractive layers
    model.add_layer()  # First layer
    model.add_layer()  # Second layer
    
    # Initialize trainer
    trainer = DONNTrainer(model, learning_rate=0.07)
    
    # Define training scenarios
    scenarios = [
        {
            'target': 'U',
            'layer_indices': [0, 1],  # Use both layers
            'distances': [10e-3, 40e-3]  # 10mm between layers, 40mm to imaging plane
        },
        {
            'target': 'T',
            'layer_indices': [0, 1],  # Use both layers, second layer rotated 90°
            'distances': [10e-3, 40e-3],
            'rotation': 90
        },
        {
            'target': 'A',
            'layer_indices': [0, 1],  # Use both layers, second layer rotated 180°
            'distances': [10e-3, 40e-3],
            'rotation': 180
        },
        {
            'target': 'H',
            'layer_indices': [0, 1],  # Use both layers, second layer rotated 270°
            'distances': [10e-3, 40e-3],
            'rotation': 270
        }
    ]
    
    # Train model
    loss_history = trainer.train(scenarios, epochs=400, save_dir=output_dir)
    
    # Evaluate model
    eval_scenarios = [
        {
            'name': 'Scenario5_U_0deg',
            'target': 'U',
            'layer_indices': [0, 1],
            'distances': [10e-3, 40e-3]
        },
        {
            'name': 'Scenario5_T_90deg',
            'target': 'T',
            'layer_indices': [0, 1],
            'distances': [10e-3, 40e-3],
            'rotation': 90
        },
        {
            'name': 'Scenario5_A_180deg',
            'target': 'A',
            'layer_indices': [0, 1],
            'distances': [10e-3, 40e-3],
            'rotation': 180
        },
        {
            'name': 'Scenario5_H_270deg',
            'target': 'H',
            'layer_indices': [0, 1],
            'distances': [10e-3, 40e-3],
            'rotation': 270
        }
    ]
    
    results = trainer.evaluate(eval_scenarios, save_dir=output_dir)
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Scenario 5: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    return model, results

def combined_scenario_translation_rotation():
    """
    Combined Scenario: Translation and Rotation
    
    Implements the combined scenario from the paper, simultaneously changing 
    the translation and rotation of diffractive layers:
    - 10mm distance, 0° rotation generates letter "U"
    - 30mm distance, 0° rotation generates letter "T"
    - 10mm distance, 90° rotation generates letter "A"
    - 30mm distance, 90° rotation generates letter "H"
    
    Returns:
        Trained DONN model and evaluation results
    """
    print("Implementing Combined Scenario: Translation and Rotation")
    
    # Create output directory
    output_dir = "/home/ubuntu/terahertz_hologram/results/combined_scenario"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DONN model
    model = DONN(
        wavelength=1e-3,  # 1mm corresponds to 0.3THz
        pixel_size=0.8e-3,  # 0.8mm
        pixel_count=60,  # 60x60 pixels
        refractive_index=1.7,  # Refractive index 1.7
        max_height=1.4e-3  # Maximum height 1.4mm
    )
    
    # Add two diffractive layers
    model.add_layer()  # First layer
    model.add_layer()  # Second layer
    
    # Initialize trainer
    trainer = DONNTrainer(model, learning_rate=0.07)
    
    # Define training scenarios
    scenarios = [
        {
            'target': 'U',
            'layer_indices': [0, 1],
            'distances': [10e-3, 50e-3]  # 10mm between layers, 50mm to imaging plane
        },
        {
            'target': 'T',
            'layer_indices': [0, 1],
            'distances': [30e-3, 30e-3]  # 30mm between layers, 30mm to imaging plane
        },
        {
            'target': 'A',
            'layer_indices': [0, 1],
            'distances': [10e-3, 50e-3],
            'rotation': 90  # Second layer rotated 90°
        },
        {
            'target': 'H',
            'layer_indices': [0, 1],
            'distances': [30e-3, 30e-3],
            'rotation': 90  # Second layer rotated 90°
        }
    ]
    
    # Train model
    loss_history = trainer.train(scenarios, epochs=400, save_dir=output_dir)
    
    # Evaluate model
    eval_scenarios = [
        {
            'name': 'Combined_U_10mm_0deg',
            'target': 'U',
            'layer_indices': [0, 1],
            'distances': [10e-3, 50e-3]
        },
        {
            'name': 'Combined_T_30mm_0deg',
            'target': 'T',
            'layer_indices': [0, 1],
            'distances': [30e-3, 30e-3]
        },
        {
            'name': 'Combined_A_10mm_90deg',
            'target': 'A',
            'layer_indices': [0, 1],
            'distances': [10e-3, 50e-3],
            'rotation': 90
        },
        {
            'name': 'Combined_H_30mm_90deg',
            'target': 'H',
            'layer_indices': [0, 1],
            'distances': [30e-3, 30e-3],
            'rotation': 90
        }
    ]
    
    results = trainer.evaluate(eval_scenarios, save_dir=output_dir)
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Combined Scenario: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    return model, results

def run_all_scenarios():
    """
    Run simulations for all scenarios
    """
    # Create main output directory
    os.makedirs("/home/ubuntu/terahertz_hologram/results", exist_ok=True)
    
    # Run Scenario 1: Varying the number of cascaded elements
    model1, results1 = scenario1_varying_number_of_elements()
    
    # Run Scenario 5: Rotating diffractive layers
    model5, results5 = scenario5_rotating_elements()
    
    # Run Combined Scenario: Translation and Rotation
    model_combined, results_combined = combined_scenario_translation_rotation()
    
    print("All scenario simulations completed!")
    
    return {
        'scenario1': {'model': model1, 'results': results1},
        'scenario5': {'model': model5, 'results': results5},
        'combined': {'model': model_combined, 'results': results_combined}
    }

if __name__ == "__main__":
    # Run all scenarios
    results = run_all_scenarios()
