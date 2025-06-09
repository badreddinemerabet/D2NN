"""
DONN (Diffractive Optical Neural Network) Model Implementation

Based on the paper 'Machine Learning Enables Multi-Degree-of-Freedom Reconfigurable Terahertz Holograms 
with Cascaded Diffractive Optical Elements'.

This model implements cascaded diffractive layers using TensorFlow, optimizing pixel heights 
through gradient descent and error backpropagation.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import os
import time

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DiffractiveLayer(Layer):
    """
    Diffractive Optical Layer Implementation
    
    Each diffractive layer consists of an array of pixels, where each pixel modulates
    the phase of the incident light through its height. Pixel heights are trainable
    parameters optimized through gradient descent.
    """
    def __init__(self, pixel_size=0.8e-3, pixel_count=60, wavelength=1e-3, 
                 refractive_index=1.7, max_height=1.4e-3, **kwargs):
        """
        Initialize diffractive layer
        
        Args:
            pixel_size: Pixel size in meters, default 0.8mm
            pixel_count: Number of pixels per side, default 60x60
            wavelength: Wavelength in meters, default 1mm (corresponding to 0.3THz)
            refractive_index: Material refractive index, default 1.7
            max_height: Maximum pixel height in meters, default 1.4mm
        """
        super(DiffractiveLayer, self).__init__(**kwargs)
        self.pixel_size = pixel_size
        self.pixel_count = pixel_count
        self.wavelength = wavelength
        self.refractive_index = refractive_index
        self.max_height = max_height
        
        # Calculate layer physical dimensions
        self.layer_size = self.pixel_size * self.pixel_count  # 48mm x 48mm
        
    def build(self, input_shape):
        """
        Build layer, initialize trainable parameters (pixel heights)
        
        Args:
            input_shape: Input tensor shape
        """
        # Initialize pixel heights with random values (between 0 and 1)
        initializer = tf.random_uniform_initializer(0.0, 1.0)
        self.heights = self.add_weight(
            shape=(self.pixel_count, self.pixel_count),
            initializer=initializer,
            trainable=True,
            name='pixel_heights',
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0)  # Constrain heights between 0 and 1
        )
        super(DiffractiveLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass, apply phase modulation
        
        Args:
            inputs: Input optical field [batch_size, height, width]
            
        Returns:
            Modulated optical field
        """
        # Convert normalized heights (0-1) to actual heights (0-max_height)
        actual_heights = self.heights * self.max_height
        
        # Calculate phase modulation (according to paper equation 1)
        phase_shift = 2 * np.pi / self.wavelength * actual_heights * (self.refractive_index - 1)
        
        # Create complex modulation factor (amplitude 1, phase determined by height)
        modulation = tf.complex(
            tf.cos(phase_shift),
            tf.sin(phase_shift)
        )
        
        # Upsample to 4x4 grid (paper mentions each pixel is subdivided into 4x4 grid)
        modulation_upsampled = tf.repeat(tf.repeat(modulation, 4, axis=0), 4, axis=1)
        
        # Apply zero padding (paper mentions 120 zero padding on each side)
        paddings = tf.constant([[120, 120], [120, 120]])
        modulation_padded = tf.pad(modulation_upsampled, paddings)
        
        # Apply modulation factor to input optical field
        modulated_field = inputs * modulation_padded
        
        return modulated_field
    
    def get_config(self):
        """Get layer configuration"""
        config = super(DiffractiveLayer, self).get_config()
        config.update({
            'pixel_size': self.pixel_size,
            'pixel_count': self.pixel_count,
            'wavelength': self.wavelength,
            'refractive_index': self.refractive_index,
            'max_height': self.max_height
        })
        return config
    
    def get_heights(self):
        """Get current pixel height distribution"""
        return self.heights.numpy() * self.max_height
    
    def set_heights(self, heights):
        """Set pixel height distribution (for pre-trained model loading or scenario switching)"""
        normalized_heights = heights / self.max_height
        self.heights.assign(tf.clip_by_value(normalized_heights, 0.0, 1.0))


class DONN:
    """
    Diffractive Optical Neural Network (DONN) Model
    
    Implements a cascaded diffractive layer structure for optimizing pixel heights
    to generate target holographic images.
    """
    def __init__(self, wavelength=1e-3, pixel_size=0.8e-3, pixel_count=60, 
                 refractive_index=1.7, max_height=1.4e-3):
        """
        Initialize DONN model
        
        Args:
            wavelength: Wavelength in meters, default 1mm (corresponding to 0.3THz)
            pixel_size: Pixel size in meters, default 0.8mm
            pixel_count: Number of pixels per side, default 60x60
            refractive_index: Material refractive index, default 1.7
            max_height: Maximum pixel height in meters, default 1.4mm
        """
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.pixel_count = pixel_count
        self.refractive_index = refractive_index
        self.max_height = max_height
        
        # Calculate upsampled and padded dimensions
        self.upsampled_size = self.pixel_count * 4  # 4x4 grid per pixel
        self.padded_size = self.upsampled_size + 240  # 120 zero padding on each side
        
        # List of diffractive layers
        self.diffractive_layers = []
        
        # Create input plane wave
        self.input_field = tf.ones((1, self.padded_size, self.padded_size), dtype=tf.complex64)
        
        # Training history
        self.loss_history = []
        
    def add_layer(self):
        """Add a new diffractive layer to the model"""
        layer = DiffractiveLayer(
            pixel_size=self.pixel_size,
            pixel_count=self.pixel_count,
            wavelength=self.wavelength,
            refractive_index=self.refractive_index,
            max_height=self.max_height
        )
        self.diffractive_layers.append(layer)
        return layer
    
    def get_layer(self, index):
        """Get diffractive layer by index"""
        if 0 <= index < len(self.diffractive_layers):
            return self.diffractive_layers[index]
        else:
            raise IndexError(f"Layer index {index} out of range")
    
    def get_layer_count(self):
        """Get current number of diffractive layers"""
        return len(self.diffractive_layers)
    
    def save_model(self, filepath):
        """
        Save model parameters (pixel heights) to file
        
        Args:
            filepath: Save path
        """
        heights_list = [layer.get_heights() for layer in self.diffractive_layers]
        np.savez(filepath, 
                 heights=heights_list, 
                 wavelength=self.wavelength,
                 pixel_size=self.pixel_size,
                 pixel_count=self.pixel_count,
                 refractive_index=self.refractive_index,
                 max_height=self.max_height)
        
    def load_model(self, filepath):
        """
        Load model parameters from file
        
        Args:
            filepath: Load path
        """
        data = np.load(filepath, allow_pickle=True)
        heights_list = data['heights']
        
        # Update model parameters
        self.wavelength = float(data['wavelength'])
        self.pixel_size = float(data['pixel_size'])
        self.pixel_count = int(data['pixel_count'])
        self.refractive_index = float(data['refractive_index'])
        self.max_height = float(data['max_height'])
        
        # Rebuild diffractive layers
        self.diffractive_layers = []
        for heights in heights_list:
            layer = self.add_layer()
            layer.set_heights(heights)
    
    def visualize_heights(self, save_path=None):
        """
        Visualize pixel height distributions of all diffractive layers
        
        Args:
            save_path: Optional, path to save the image
        """
        n_layers = len(self.diffractive_layers)
        if n_layers == 0:
            print("No diffractive layers to visualize")
            return
        
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 5))
        if n_layers == 1:
            axes = [axes]
            
        for i, layer in enumerate(self.diffractive_layers):
            heights = layer.get_heights() * 1000  # Convert to mm
            im = axes[i].imshow(heights, cmap='viridis')
            axes[i].set_title(f'Layer {i+1} Heights (mm)')
            fig.colorbar(im, ax=axes[i])
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
