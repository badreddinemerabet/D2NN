"""
Training Process and Loss Function Implementation

Based on the paper 'Machine Learning Enables Multi-Degree-of-Freedom Reconfigurable Terahertz Holograms 
with Cascaded Diffractive Optical Elements'.

This module implements the loss function based on the difference between target holographic images 
and generated intensity distributions, as well as the end-to-end training process.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
from donn_model import DONN, DiffractiveLayer
from rayleigh_sommerfeld import RayleighSommerfeldPropagation, visualize_field

class DONNTrainer:
    """
    DONN Trainer
    
    Implements end-to-end training process, including forward propagation, 
    loss calculation, and gradient optimization.
    """
    def __init__(self, donn_model, learning_rate=0.07):
        """
        Initialize trainer
        
        Args:
            donn_model: DONN model instance
            learning_rate: Learning rate, default 0.07 (consistent with paper)
        """
        self.model = donn_model
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_history = []
        
    def create_target_image(self, letter, image_size=480):
        """
        Create target image (letter)
        
        Args:
            letter: Target letter
            image_size: Image size, default 480x480 (consistent with paper)
            
        Returns:
            Target image tensor
        """
        # Create blank image
        target = np.zeros((image_size, image_size), dtype=np.float32)
        
        # Image center position
        center_x, center_y = image_size // 2, image_size // 2
        
        # Letter size (1/4 of the entire image)
        letter_size = image_size // 4
        
        # Letter stroke width
        stroke_width = letter_size // 8
        
        # Draw different shapes based on the letter
        if letter == 'T':
            # Draw T's horizontal line
            target[center_y - letter_size//2:center_y - letter_size//2 + stroke_width, 
                   center_x - letter_size//2:center_x + letter_size//2] = 1.0
            # Draw T's vertical line
            target[center_y - letter_size//2:center_y + letter_size//2, 
                   center_x - stroke_width//2:center_x + stroke_width//2] = 1.0
                   
        elif letter == 'H':
            # Draw H's left vertical line
            target[center_y - letter_size//2:center_y + letter_size//2, 
                   center_x - letter_size//2:center_x - letter_size//2 + stroke_width] = 1.0
            # Draw H's right vertical line
            target[center_y - letter_size//2:center_y + letter_size//2, 
                   center_x + letter_size//2 - stroke_width:center_x + letter_size//2] = 1.0
            # Draw H's horizontal line
            target[center_y - stroke_width//2:center_y + stroke_width//2, 
                   center_x - letter_size//2:center_x + letter_size//2] = 1.0
                   
        elif letter == 'Z':
            # Draw Z's top horizontal line
            target[center_y - letter_size//2:center_y - letter_size//2 + stroke_width, 
                   center_x - letter_size//2:center_x + letter_size//2] = 1.0
            # Draw Z's bottom horizontal line
            target[center_y + letter_size//2 - stroke_width:center_y + letter_size//2, 
                   center_x - letter_size//2:center_x + letter_size//2] = 1.0
            # Draw Z's diagonal line
            for i in range(letter_size):
                pos_y = center_y - letter_size//2 + i
                pos_x = center_x + letter_size//2 - i
                target[pos_y:pos_y + stroke_width, pos_x - stroke_width:pos_x] = 1.0
                
        elif letter == 'D':
            # Draw D's left vertical line
            target[center_y - letter_size//2:center_y + letter_size//2, 
                   center_x - letter_size//2:center_x - letter_size//2 + stroke_width] = 1.0
            # Draw D's arc
            for i in range(letter_size):
                y_offset = i - letter_size//2
                x_offset = int(np.sqrt(max(0, (letter_size//2)**2 - y_offset**2)))
                target[center_y + y_offset, 
                       center_x - letter_size//2 + stroke_width:center_x - letter_size//2 + x_offset] = 1.0
                       
        elif letter == 'O':
            # Draw O (circle)
            for i in range(image_size):
                for j in range(image_size):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    if letter_size//2 - stroke_width <= dist <= letter_size//2:
                        target[i, j] = 1.0
                        
        elif letter == 'E':
            # Draw E's vertical line
            target[center_y - letter_size//2:center_y + letter_size//2, 
                   center_x - letter_size//2:center_x - letter_size//2 + stroke_width] = 1.0
            # Draw E's top horizontal line
            target[center_y - letter_size//2:center_y - letter_size//2 + stroke_width, 
                   center_x - letter_size//2:center_x + letter_size//2] = 1.0
            # Draw E's middle horizontal line
            target[center_y - stroke_width//2:center_y + stroke_width//2, 
                   center_x - letter_size//2:center_x + letter_size//2] = 1.0
            # Draw E's bottom horizontal line
            target[center_y + letter_size//2 - stroke_width:center_y + letter_size//2, 
                   center_x - letter_size//2:center_x + letter_size//2] = 1.0
                   
        elif letter == 'M':
            # Draw M's left vertical line
            target[center_y - letter_size//2:center_y + letter_size//2, 
                   center_x - letter_size//2:center_x - letter_size//2 + stroke_width] = 1.0
            # Draw M's right vertical line
            target[center_y - letter_size//2:center_y + letter_size//2, 
                   center_x + letter_size//2 - stroke_width:center_x + letter_size//2] = 1.0
            # Draw M's left diagonal line
            for i in range(letter_size//2):
                pos_y = center_y - letter_size//2 + i
                pos_x = center_x - letter_size//2 + i
                target[pos_y:pos_y + stroke_width, pos_x:pos_x + stroke_width] = 1.0
            # Draw M's right diagonal line
            for i in range(letter_size//2):
                pos_y = center_y - letter_size//2 + i
                pos_x = center_x + letter_size//2 - i - stroke_width
                target[pos_y:pos_y + stroke_width, pos_x:pos_x + stroke_width] = 1.0
                
        elif letter == 'L':
            # Draw L's vertical line
            target[center_y - letter_size//2:center_y + letter_size//2, 
                   center_x - letter_size//2:center_x - letter_size//2 + stroke_width] = 1.0
            # Draw L's horizontal line
            target[center_y + letter_size//2 - stroke_width:center_y + letter_size//2, 
                   center_x - letter_size//2:center_x + letter_size//2] = 1.0
                   
        elif letter == 'A':
            # Draw A's left diagonal line
            for i in range(letter_size):
                pos_y = center_y + letter_size//2 - i
                pos_x = center_x - letter_size//2 + i//2
                target[pos_y - stroke_width:pos_y, pos_x:pos_x + stroke_width] = 1.0
            # Draw A's right diagonal line
            for i in range(letter_size):
                pos_y = center_y + letter_size//2 - i
                pos_x = center_x + letter_size//2 - i//2
                target[pos_y - stroke_width:pos_y, pos_x - stroke_width:pos_x] = 1.0
            # Draw A's horizontal line
            target[center_y:center_y + stroke_width, 
                   center_x - letter_size//4:center_x + letter_size//4] = 1.0
                   
        elif letter == 'I':
            # Draw I's vertical line
            target[center_y - letter_size//2:center_y + letter_size//2, 
                   center_x - stroke_width//2:center_x + stroke_width//2] = 1.0
            # Draw I's top horizontal line
            target[center_y - letter_size//2:center_y - letter_size//2 + stroke_width, 
                   center_x - letter_size//4:center_x + letter_size//4] = 1.0
            # Draw I's bottom horizontal line
            target[center_y + letter_size//2 - stroke_width:center_y + letter_size//2, 
                   center_x - letter_size//4:center_x + letter_size//4] = 1.0
                   
        elif letter == 'U':
            # Draw U's left vertical line
            target[center_y - letter_size//2:center_y + letter_size//4, 
                   center_x - letter_size//2:center_x - letter_size//2 + stroke_width] = 1.0
            # Draw U's right vertical line
            target[center_y - letter_size//2:center_y + letter_size//4, 
                   center_x + letter_size//2 - stroke_width:center_x + letter_size//2] = 1.0
            # Draw U's bottom arc
            for i in range(letter_size):
                x_offset = i - letter_size//2
                if abs(x_offset) < letter_size//2:
                    y_pos = center_y + letter_size//4 + (letter_size//4) * (abs(x_offset) / (letter_size//2))
                    target[int(y_pos):int(y_pos) + stroke_width, 
                           center_x + x_offset] = 1.0
        
        # Convert to TensorFlow tensor
        return tf.convert_to_tensor(target, dtype=tf.float32)
    
    def compute_loss(self, output_field, target_image):
        """
        Compute loss function
        
        Implements paper equation (4):
        L_loss = (1/MN) * sum((|E_out|/max(|E_out|) - T_target)^2)
        
        Args:
            output_field: Output optical field
            target_image: Target image
            
        Returns:
            Loss value
        """
        # Calculate output field intensity
        output_intensity = tf.abs(output_field)**2
        
        # Normalize output intensity
        max_intensity = tf.reduce_max(output_intensity)
        normalized_intensity = output_intensity / max_intensity
        
        # Calculate mean squared error with target image
        loss = tf.reduce_mean(tf.square(normalized_intensity - target_image))
        
        return loss
    
    def propagate_through_layers(self, input_field, layer_indices, distances):
        """
        Propagate optical field through specified diffractive layers and propagation distances
        
        Args:
            input_field: Input optical field
            layer_indices: List of diffractive layer indices
            distances: List of propagation distances
            
        Returns:
            Output optical field
        """
        field = input_field
        
        # Iterate through each diffractive layer and propagation distance
        for i, (layer_idx, distance) in enumerate(zip(layer_indices, distances)):
            # Through diffractive layer
            layer = self.model.get_layer(layer_idx)
            field = layer(field)
            
            # If not the last layer, perform propagation
            if i < len(layer_indices) - 1 or distances[-1] > 0:
                # Create propagation layer
                propagation = RayleighSommerfeldPropagation(
                    distance=distance,
                    wavelength=self.model.wavelength,
                    pixel_size=self.model.pixel_size / 4  # Account for 4x4 upsampling
                )
                
                # Through propagation layer
                field = propagation(field)
        
        return field
    
    @tf.function
    def train_step(self, input_field, target_image, layer_indices, distances):
        """
        Single training step
        
        Args:
            input_field: Input optical field
            target_image: Target image
            layer_indices: List of diffractive layer indices
            distances: List of propagation distances
            
        Returns:
            Loss value
        """
        # Get trainable variables
        trainable_vars = []
        for idx in layer_indices:
            trainable_vars.extend(self.model.get_layer(idx).trainable_variables)
        
        # Use GradientTape to record operations for gradient calculation
        with tf.GradientTape() as tape:
            # Forward propagation
            output_field = self.propagate_through_layers(input_field, layer_indices, distances)
            
            # Calculate loss
            loss = self.compute_loss(output_field, target_image)
        
        # Calculate gradients
        gradients = tape.gradient(loss, trainable_vars)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss
    
    def train(self, scenarios, epochs=400, save_dir=None):
        """
        Train model
        
        Args:
            scenarios: List of scenarios, each a dictionary containing:
                - target: Target letter
                - layer_indices: List of diffractive layer indices
                - distances: List of propagation distances
            epochs: Number of training epochs, default 400 (consistent with paper)
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
        input_field = tf.ones((1, self.model.padded_size, self.model.padded_size), dtype=tf.complex64)
        
        # Create target images
        target_images = {}
        for scenario in scenarios:
            if scenario['target'] not in target_images:
                target_images[scenario['target']] = self.create_target_image(scenario['target'])
        
        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Train on each scenario
            for scenario in scenarios:
                target_image = target_images[scenario['target']]
                loss = self.train_step(
                    input_field, 
                    target_image, 
                    scenario['layer_indices'], 
                    scenario['distances']
                )
                epoch_loss += loss.numpy()
            
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
                    plt.title('Training Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
                    plt.close()
        
        # Save final model
        if save_dir:
            self.model.save_model(os.path.join(save_dir, 'model.npz'))
            
            # Save pixel height distribution plot
            self.model.visualize_heights(os.path.join(save_dir, 'pixel_heights.png'))
        
        return loss_history
    
    def evaluate(self, scenarios, save_dir=None):
        """
        Evaluate model
        
        Args:
            scenarios: List of scenarios, each a dictionary containing:
                - name: Scenario name
                - target: Target letter
                - layer_indices: List of diffractive layer indices
                - distances: List of propagation distances
            save_dir: Save directory
            
        Returns:
            Evaluation results
        """
        # Create save directory
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Create input field (plane wave)
        input_field = tf.ones((1, self.model.padded_size, self.model.padded_size), dtype=tf.complex64)
        
        # Evaluation results
        results = []
        
        # Evaluate each scenario
        for i, scenario in enumerate(scenarios):
            # Forward propagation
            output_field = self.propagate_through_layers(
                input_field, 
                scenario['layer_indices'], 
                scenario['distances']
            )
            
            # Calculate output intensity
            output_intensity = tf.abs(output_field[0])**2
            normalized_intensity = output_intensity / tf.reduce_max(output_intensity)
            
            # Create target image
            target_image = self.create_target_image(scenario['target'])
            
            # Calculate efficiency (ratio of intensity in target area to total intensity)
            target_mask = target_image > 0
            efficiency = tf.reduce_sum(normalized_intensity * target_mask) / tf.reduce_sum(normalized_intensity)
            efficiency = efficiency.numpy() * 100  # Convert to percentage
            
            # Save result
            result = {
                'name': scenario.get('name', f'Scenario_{i+1}'),
                'target': scenario['target'],
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
                plt.title(f"Output (Efficiency: {efficiency:.2f}%)")
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{result['name']}.png"), dpi=300)
                plt.close()
        
        return results
