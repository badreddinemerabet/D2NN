"""
Wavelength-Dependent Scenario Implementation

This module implements a new scenario where different wavelengths produce different 
holographic images, extending the multi-degree-of-freedom concept from the paper.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from donn_model import DONN
from rayleigh_sommerfeld import RayleighSommerfeldPropagation, visualize_field
from training import DONNTrainer

class WavelengthDependentDONN(DONN):
    """
    Extended DONN model that supports wavelength-dependent operation
    
    This class extends the base DONN model to allow for different wavelengths
    during inference, enabling wavelength-dependent holographic imaging.
    """
    
    def __init__(self, **kwargs):
        """Initialize with the same parameters as the base DONN class"""
        super(WavelengthDependentDONN, self).__init__(**kwargs)
        
    def set_wavelength(self, wavelength):
        """
        Set a new operating wavelength
        
        Args:
            wavelength: New wavelength in meters
        """
        self.wavelength = wavelength
        
    def propagate_with_wavelength(self, input_field, layer_indices, distances, wavelength):
        """
        Propagate optical field through layers with a specific wavelength
        
        Args:
            input_field: Input optical field
            layer_indices: List of diffractive layer indices
            distances: List of propagation distances
            wavelength: Operating wavelength in meters
            
        Returns:
            Output optical field
        """
        # Store original wavelength
        original_wavelength = self.wavelength
        
        # Set new wavelength
        self.set_wavelength(wavelength)
        
        field = input_field
        
        # Iterate through each diffractive layer and propagation distance
        for i, (layer_idx, distance) in enumerate(zip(layer_indices, distances)):
            # Through diffractive layer
            layer = self.get_layer(layer_idx)
            
            # Calculate phase modulation with current wavelength
            actual_heights = layer.heights.numpy() * layer.max_height
            phase_shift = 2 * np.pi / self.wavelength * actual_heights * (layer.refractive_index - 1)
            
            # Create complex modulation factor
            modulation = tf.complex(
                tf.cos(phase_shift),
                tf.sin(phase_shift)
            )
            
            # Upsample to 4x4 grid
            modulation_upsampled = tf.repeat(tf.repeat(modulation, 4, axis=0), 4, axis=1)
            
            # Apply zero padding
            paddings = tf.constant([[120, 120], [120, 120]])
            modulation_padded = tf.pad(modulation_upsampled, paddings)
            
            # Apply modulation to field
            field = field * modulation_padded
            
            # If not the last layer, perform propagation
            if i < len(layer_indices) - 1 or distances[-1] > 0:
                # Create propagation layer with current wavelength
                propagation = RayleighSommerfeldPropagation(
                    distance=distance,
                    wavelength=self.wavelength,
                    pixel_size=self.pixel_size / 4  # Account for 4x4 upsampling
                )
                
                # Through propagation layer
                field = propagation(field)
        
        # Restore original wavelength
        self.set_wavelength(original_wavelength)
        
        return field


def wavelength_dependent_scenario():
    """
    Wavelength-Dependent Scenario
    
    Implements a scenario where different wavelengths produce different holographic images:
    - 0.3 THz (1.0mm) wavelength generates letter "W"
    - 0.25 THz (1.2mm) wavelength generates letter "A"
    - 0.2 THz (1.5mm) wavelength generates letter "V"
    - 0.15 THz (2.0mm) wavelength generates letter "E"
    
    Returns:
        Trained DONN model and evaluation results
    """
    print("Implementing Wavelength-Dependent Scenario")
    
    # Create output directory
    output_dir = "/home/ubuntu/terahertz_hologram/results/wavelength_scenario"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wavelength-dependent DONN model
    # Base wavelength is 1.0mm (0.3 THz)
    model = WavelengthDependentDONN(
        wavelength=1.0e-3,  # 1mm corresponds to 0.3THz
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
    
    # Define training scenarios with different target letters for different wavelengths
    scenarios = [
        {
            'target': 'W',
            'layer_indices': [0, 1],
            'distances': [10e-3, 40e-3],
            'wavelength': 1.0e-3  # 1.0mm (0.3 THz)
        },
        {
            'target': 'A',
            'layer_indices': [0, 1],
            'distances': [10e-3, 40e-3],
            'wavelength': 1.2e-3  # 1.2mm (0.25 THz)
        },
        {
            'target': 'V',
            'layer_indices': [0, 1],
            'distances': [10e-3, 40e-3],
            'wavelength': 1.5e-3  # 1.5mm (0.2 THz)
        },
        {
            'target': 'E',
            'layer_indices': [0, 1],
            'distances': [10e-3, 40e-3],
            'wavelength': 2.0e-3  # 2.0mm (0.15 THz)
        }
    ]
    
    # Custom training function for wavelength-dependent scenario
    def train_wavelength_dependent(model, scenarios, epochs=400, save_dir=None):
        """
        Train model for wavelength-dependent scenario
        
        Args:
            model: WavelengthDependentDONN model instance
            scenarios: List of scenarios with wavelength information
            epochs: Number of training epochs
            save_dir: Save directory
            
        Returns:
            Training history
        """
        # Create save directory
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Initialize loss history
        loss_history = []
        
        # Create input field (plane wave)
        input_field = tf.ones((1, model.padded_size, model.padded_size), dtype=tf.complex64)
        
        # Create target images
        target_images = {}
        for scenario in scenarios:
            if scenario['target'] not in target_images:
                target_images[scenario['target']] = trainer.create_target_image(scenario['target'])
        
        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Train on each scenario
            for scenario in scenarios:
                # Set wavelength for this scenario
                original_wavelength = model.wavelength
                model.set_wavelength(scenario['wavelength'])
                
                target_image = target_images[scenario['target']]
                loss = trainer.train_step(
                    input_field, 
                    target_image, 
                    scenario['layer_indices'], 
                    scenario['distances']
                )
                epoch_loss += loss.numpy()
                
                # Restore original wavelength
                model.set_wavelength(original_wavelength)
            
            # Calculate average loss
            epoch_loss /= len(scenarios)
            loss_history.append(epoch_loss)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, Time: {elapsed_time:.2f}s")
                
                # Save loss curve
                if save_dir:
                    plt.figure(figsize=(10, 6))
                    plt.plot(loss_history)
                    plt.title('Training Loss - Wavelength-Dependent Scenario')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
                    plt.close()
        
        # Save final model
        if save_dir:
            model.save_model(os.path.join(save_dir, 'model.npz'))
            
            # Save pixel height distribution plot
            model.visualize_heights(os.path.join(save_dir, 'pixel_heights.png'))
        
        return loss_history
    
    # Train model with custom wavelength-dependent training function
    loss_history = train_wavelength_dependent(model, scenarios, epochs=400, save_dir=output_dir)
    
    # Evaluate model for different wavelengths
    def evaluate_wavelength_dependent(model, scenarios, save_dir=None):
        """
        Evaluate model for wavelength-dependent scenario
        
        Args:
            model: WavelengthDependentDONN model instance
            scenarios: List of scenarios with wavelength information
            save_dir: Save directory
            
        Returns:
            Evaluation results
        """
        # Create save directory
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Create input field (plane wave)
        input_field = tf.ones((1, model.padded_size, model.padded_size), dtype=tf.complex64)
        
        # Evaluation results
        results = []
        
        # Evaluate each scenario
        for i, scenario in enumerate(scenarios):
            # Set wavelength for this scenario
            wavelength = scenario['wavelength']
            
            # Forward propagation with specific wavelength
            output_field = model.propagate_with_wavelength(
                input_field, 
                scenario['layer_indices'], 
                scenario['distances'],
                wavelength
            )
            
            # Calculate output intensity
            output_intensity = tf.abs(output_field[0])**2
            normalized_intensity = output_intensity / tf.reduce_max(output_intensity)
            
            # Create target image
            target_image = trainer.create_target_image(scenario['target'])
            
            # Calculate efficiency
            target_mask = target_image > 0
            efficiency = tf.reduce_sum(normalized_intensity * target_mask) / tf.reduce_sum(normalized_intensity)
            efficiency = efficiency.numpy() * 100  # Convert to percentage
            
            # Format wavelength for display
            thz_freq = 0.3 / wavelength * 1e-3
            
            # Save result
            result = {
                'name': f"Wavelength_{wavelength*1000:.1f}mm_{thz_freq:.2f}THz",
                'target': scenario['target'],
                'wavelength': wavelength,
                'thz_freq': thz_freq,
                'efficiency': efficiency,
                'output_field': output_field[0].numpy(),
                'normalized_intensity': normalized_intensity.numpy()
            }
            results.append(result)
            
            # Visualize and save result
            if save_dir:
                plt.figure(figsize=(12, 5))
                
                # Plot target image
                plt.subplot(1, 2, 1)
                plt.imshow(target_image, cmap='viridis')
                plt.title(f"Target: {scenario['target']}")
                plt.colorbar()
                
                # Plot output intensity
                plt.subplot(1, 2, 2)
                plt.imshow(normalized_intensity, cmap='viridis')
                plt.title(f"Output at {wavelength*1000:.1f}mm ({thz_freq:.2f} THz)\nEfficiency: {efficiency:.2f}%")
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{result['name']}.png"), dpi=300)
                plt.close()
        
        # Create summary visualization
        plt.figure(figsize=(15, 10))
        
        for i, result in enumerate(results):
            plt.subplot(2, 2, i+1)
            plt.imshow(result['normalized_intensity'], cmap='viridis')
            plt.title(f"{result['target']} at {result['wavelength']*1000:.1f}mm ({result['thz_freq']:.2f} THz)\nEfficiency: {result['efficiency']:.2f}%")
            plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "wavelength_summary.png"), dpi=300)
        plt.close()
        
        return results
    
    # Evaluate model
    eval_scenarios = scenarios
    results = evaluate_wavelength_dependent(model, eval_scenarios, save_dir=output_dir)
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Wavelength-Dependent Scenario: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    return model, results

# Add this function to main.py
def update_main_with_wavelength_scenario():
    """Update main.py to include the wavelength-dependent scenario"""
    main_path = "/home/ubuntu/terahertz_hologram/main.py"
    
    # Read current main.py
    with open(main_path, 'r') as f:
        main_content = f.read()
    
    # Add import for wavelength-dependent scenario
    import_line = "from scenarios import scenario1_varying_number_of_elements, scenario5_rotating_elements, combined_scenario_translation_rotation"
    new_import = "from scenarios import scenario1_varying_number_of_elements, scenario5_rotating_elements, combined_scenario_translation_rotation\nfrom wavelength_scenario import wavelength_dependent_scenario"
    main_content = main_content.replace(import_line, new_import)
    
    # Add wavelength scenario to main function
    combined_scenario_line = "    # Run Combined Scenario: Translation and Rotation\n    print(\"\\nRunning Combined Scenario: Translation and Rotation\")\n    model_combined, results_combined = combined_scenario_translation_rotation()"
    wavelength_scenario_code = """    # Run Combined Scenario: Translation and Rotation
    print("\\nRunning Combined Scenario: Translation and Rotation")
    model_combined, results_combined = combined_scenario_translation_rotation()
    
    # Run Wavelength-Dependent Scenario
    print("\\nRunning Wavelength-Dependent Scenario")
    model_wavelength, results_wavelength = wavelength_dependent_scenario()"""
    
    main_content = main_content.replace(combined_scenario_line, wavelength_scenario_code)
    
    # Update all_results dictionary
    all_results_line = "    all_results = {\n        'scenario1': {'model': model1, 'results': results1},\n        'scenario5': {'model': model5, 'results': results5},\n        'combined': {'model': model_combined, 'results': results_combined}\n    }"
    new_all_results = """    all_results = {
        'scenario1': {'model': model1, 'results': results1},
        'scenario5': {'model': model5, 'results': results5},
        'combined': {'model': model_combined, 'results': results_combined},
        'wavelength': {'model': model_wavelength, 'results': results_wavelength}
    }"""
    
    main_content = main_content.replace(all_results_line, new_all_results)
    
    # Write updated main.py
    with open(main_path, 'w') as f:
        f.write(main_content)
    
    print("Updated main.py to include wavelength-dependent scenario")

if __name__ == "__main__":
    # Test wavelength-dependent scenario
    model, results = wavelength_dependent_scenario()
    
    # Update main.py
    update_main_with_wavelength_scenario()
