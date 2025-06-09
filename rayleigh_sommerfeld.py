"""
Rayleigh-Sommerfeld Diffraction Model Implementation

Based on the paper 'Machine Learning Enables Multi-Degree-of-Freedom Reconfigurable Terahertz Holograms 
with Cascaded Diffractive Optical Elements'.

This module implements the scalar diffraction theory model for calculating light propagation 
between diffractive layers and imaging planes.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class RayleighSommerfeldPropagation(tf.keras.layers.Layer):
    """
    TensorFlow layer implementing Rayleigh-Sommerfeld diffraction propagation
    
    Uses scalar diffraction theory to calculate optical field propagation in free space.
    """
    def __init__(self, distance, wavelength, pixel_size, **kwargs):
        """
        Initialize Rayleigh-Sommerfeld propagation layer
        
        Args:
            distance: Propagation distance in meters
            wavelength: Wavelength in meters
            pixel_size: Pixel size in meters
        """
        super(RayleighSommerfeldPropagation, self).__init__(**kwargs)
        self.distance = distance
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        
    def build(self, input_shape):
        """Build layer, pre-calculate propagation kernel"""
        # Get input field dimensions
        self.height = input_shape[1]
        self.width = input_shape[2]
        
        # Pre-calculate propagation kernel
        self.propagation_kernel = self._calculate_propagation_kernel()
        
        super(RayleighSommerfeldPropagation, self).build(input_shape)
        
    def _calculate_propagation_kernel(self):
        """
        Calculate Rayleigh-Sommerfeld propagation kernel
        
        Implements paper equation (2):
        w(x,y,z) = (z/r²) * (1/2πr + j/λ) * exp(j2πr/λ)
        where r = sqrt((x-xi)² + (y-yi)² + (z-zi)²)
        """
        # Create coordinate grid
        x = tf.range(-self.width//2, self.width//2, dtype=tf.float32) * self.pixel_size
        y = tf.range(-self.height//2, self.height//2, dtype=tf.float32) * self.pixel_size
        X, Y = tf.meshgrid(x, y)
        
        # Calculate r (distance from origin to each point)
        r = tf.sqrt(X**2 + Y**2 + self.distance**2)
        
        # Calculate parts of the propagation kernel
        z_over_r_squared = self.distance / (r**2)
        term1 = 1.0 / (2.0 * np.pi * r)
        term2 = tf.complex(tf.zeros_like(r), 1.0 / self.wavelength)
        phase = 2.0 * np.pi * r / self.wavelength
        
        # Combine parts to form complete propagation kernel
        kernel_real = z_over_r_squared * term1 * tf.cos(phase)
        kernel_imag = z_over_r_squared * (term1 * tf.sin(phase) + 1.0/self.wavelength * tf.cos(phase))
        kernel = tf.complex(kernel_real, kernel_imag)
        
        # Shift kernel to frequency domain center
        kernel = tf.signal.fftshift(kernel)
        
        return kernel
    
    def call(self, inputs):
        """
        Apply Rayleigh-Sommerfeld propagation
        
        Args:
            inputs: Input optical field [batch_size, height, width]
            
        Returns:
            Propagated optical field
        """
        # Implement propagation using convolution theorem
        # 1. Perform FFT on input field
        field_fft = tf.signal.fft2d(inputs)
        
        # 2. Multiply with propagation kernel in frequency domain
        propagated_fft = field_fft * tf.cast(self.propagation_kernel, dtype=field_fft.dtype)
        
        # 3. Perform IFFT to get propagated field
        propagated_field = tf.signal.ifft2d(propagated_fft)
        
        return propagated_field
    
    def get_config(self):
        """Get layer configuration"""
        config = super(RayleighSommerfeldPropagation, self).get_config()
        config.update({
            'distance': self.distance,
            'wavelength': self.wavelength,
            'pixel_size': self.pixel_size
        })
        return config


def angular_spectrum_propagation(field, distance, wavelength, pixel_size):
    """
    Implement Rayleigh-Sommerfeld diffraction propagation using angular spectrum method
    (Non-TensorFlow implementation for validation)
    
    Args:
        field: Input optical field
        distance: Propagation distance in meters
        wavelength: Wavelength in meters
        pixel_size: Pixel size in meters
        
    Returns:
        Propagated optical field
    """
    # Get field dimensions
    height, width = field.shape
    
    # Create frequency grid
    kx = np.fft.fftfreq(width, d=pixel_size)
    ky = np.fft.fftfreq(height, d=pixel_size)
    kX, kY = np.meshgrid(kx, ky)
    
    # Calculate propagation phase
    k = 2 * np.pi / wavelength
    kz = np.sqrt(k**2 - (2*np.pi*kX)**2 - (2*np.pi*kY)**2 + 0j)
    
    # Apply propagation phase
    field_fft = np.fft.fft2(field)
    propagated_fft = field_fft * np.exp(1j * kz * distance)
    propagated_field = np.fft.ifft2(propagated_fft)
    
    return propagated_field


def visualize_field(field, title="Field Intensity", save_path=None):
    """
    Visualize optical field intensity
    
    Args:
        field: Complex optical field
        title: Image title
        save_path: Optional, path to save the image
    """
    # Calculate intensity (square of amplitude)
    if isinstance(field, tf.Tensor):
        intensity = tf.abs(field)**2
        intensity = intensity.numpy()
    else:
        intensity = np.abs(field)**2
    
    # Normalize intensity
    normalized_intensity = intensity / np.max(intensity)
    
    # Plot intensity map
    plt.figure(figsize=(8, 6))
    plt.imshow(normalized_intensity, cmap='viridis')
    plt.colorbar(label='Normalized Intensity')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
