# Python Implementation of Reconfigurable Terahertz Holograms

## Project Overview

This project implements the Python code for reconfigurable terahertz holograms based on the research paper "Machine Learning Enables Multi-Degree-of-Freedom Reconfigurable Terahertz Holograms with Cascaded Diffractive Optical Elements". The project uses TensorFlow to implement a DONN (Diffractive Optical Neural Network) model that optimizes the pixel height distribution of cascaded diffractive layers to achieve multi-degree-of-freedom reconfigurable holograms.

## Core Features

1. **DONN Model**: Implementation of trainable cascaded diffractive optical layers, each consisting of 60×60 pixels that modulate the phase of incident light through optimized pixel heights.

2. **Rayleigh-Sommerfeld Diffraction Model**: Implementation of scalar diffraction theory for calculating light propagation between diffractive layers and imaging planes.

3. **Training Process**: Loss function based on the difference between target holographic images and generated intensity distributions, with pixel height optimization through gradient descent and error backpropagation.

4. **Multiple Reconfigurable Scenarios**: Implementation of various scenarios from the paper, including:
   - Scenario 1: Varying the number of cascaded elements
   - Scenario 5: Rotating diffractive layers
   - Combined Scenario: Simultaneous translation and rotation

5. **Result Visualization**: Visualization of pixel height distributions, training loss curves, and normalized hologram intensity distributions.

## File Structure

- `donn_model.py`: Core implementation of the DONN model, including diffractive layers and pixel height optimization
- `rayleigh_sommerfeld.py`: Implementation of the Rayleigh-Sommerfeld diffraction propagation model
- `training.py`: Implementation of training process and loss function
- `scenarios.py`: Configuration and implementation of multiple reconfigurable scenarios
- `visualization.py`: Result visualization functionality
- `main.py`: Main program that runs the complete simulation process
- `wavelength_scenario.py`: Implementation of wavelength-dependent holographic imaging

## Usage Instructions

1. Ensure required dependencies are installed:
   ```
   pip install tensorflow numpy matplotlib
   ```

2. Run the main program:
   ```
   python main.py
   ```

3. View results:
   All simulation results will be saved in the `/home/ubuntu/terahertz_hologram/results` directory, including:
   - Optimized pixel height distributions
   - Training loss curves
   - Normalized hologram intensity distributions for each scenario

## Implementation Details

### DONN Model

The DONN model consists of multiple cascaded diffractive layers, each containing 60×60 pixels. Each pixel modulates the phase of incident light through its height. Pixel heights range from 0 to 1.4mm, corresponding to a 2π phase modulation at 0.3THz frequency.

### Rayleigh-Sommerfeld Diffraction Model

The Rayleigh-Sommerfeld diffraction formula is used to calculate optical field propagation in free space:

```
w(x,y,z) = (z/r²) * (1/2πr + j/λ) * exp(j2πr/λ)
```

where r = sqrt((x-xi)² + (y-yi)² + (z-zi)²)

### Training Process

The training process uses stochastic gradient descent and error backpropagation to optimize pixel height distributions. The loss function is defined as the mean squared error between the target image and the normalized generated intensity distribution:

```
L_loss = (1/MN) * sum((|E_out|/max(|E_out|) - T_target)^2)
```

### Multiple Reconfigurable Scenarios

The implementation includes various reconfigurable scenarios from the paper:

1. **Scenario 1**: Varying the number of cascaded elements
   - Using only the first layer generates letter "T"
   - Using only the second layer generates letter "H"
   - Cascading both layers generates letter "Z"

2. **Scenario 5**: Rotating diffractive layers
   - 0° rotation generates letter "U"
   - 90° rotation generates letter "T"
   - 180° rotation generates letter "A"
   - 270° rotation generates letter "H"

3. **Combined Scenario**: Translation and rotation
   - 10mm distance, 0° rotation generates letter "U"
   - 30mm distance, 0° rotation generates letter "T"
   - 10mm distance, 90° rotation generates letter "A"
   - 30mm distance, 90° rotation generates letter "H"
4. **Wavelength-Dependent Scenario**: Different wavelengths produce different images
   - 0.3 THz (1.0mm) wavelength generates letter "W"
   - 0.25 THz (1.2mm) wavelength generates letter "A"
   - 0.2 THz (1.5mm) wavelength generates letter "V"
   - 0.15 THz (2.0mm) wavelength generates letter "E"


## Reference

Wei Jia, Dajun Lin, and Berardi Sensale-Rodriguez. "Machine Learning Enables Multi-Degree-of-Freedom Reconfigurable Terahertz Holograms with Cascaded Diffractive Optical Elements." Advanced Optical Materials, 2023.
