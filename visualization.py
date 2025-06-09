"""
Results Visualization Module

Based on the paper 'Machine Learning Enables Multi-Degree-of-Freedom Reconfigurable Terahertz Holograms 
with Cascaded Diffractive Optical Elements'.

This module provides functionality for visualizing pixel height distributions, training loss curves,
and normalized hologram intensity distributions.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_pixel_heights(model, save_path=None, fig_size=(12, 5)):
    """
    Visualize pixel height distributions of diffractive layers
    
    Args:
        model: DONN model instance
        save_path: Save path
        fig_size: Figure size
    """
    n_layers = model.get_layer_count()
    if n_layers == 0:
        print("No diffractive layers to visualize")
        return
    
    fig, axes = plt.subplots(1, n_layers, figsize=fig_size)
    if n_layers == 1:
        axes = [axes]
    
    # Create custom colormap (similar to the paper's color scheme)
    colors = [(0, 0, 0.5), (0, 0, 1), (0, 0.5, 1), (0, 1, 1), 
              (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
    cmap_name = 'custom_cmap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    for i in range(n_layers):
        layer = model.get_layer(i)
        heights = layer.get_heights() * 1000  # Convert to mm
        
        im = axes[i].imshow(heights, cmap=custom_cmap, vmin=0, vmax=1.4*1000)
        axes[i].set_title(f'Diffractive Layer {i+1} Pixel Height Distribution (mm)')
        axes[i].set_xlabel('Pixel X')
        axes[i].set_ylabel('Pixel Y')
        
        # Add colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Height (mm)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_loss_curve(loss_history, title='Training Loss Curve', save_path=None):
    """
    Visualize training loss curve
    
    Args:
        loss_history: Loss history record
        title: Figure title
        save_path: Save path
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add loss decrease annotations
    epochs = len(loss_history)
    plt.annotate(f'Initial Loss: {loss_history[0]:.4f}', 
                 xy=(0, loss_history[0]), 
                 xytext=(epochs*0.05, loss_history[0]*1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    plt.annotate(f'Final Loss: {loss_history[-1]:.4f}', 
                 xy=(epochs-1, loss_history[-1]), 
                 xytext=(epochs*0.7, loss_history[-1]*2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_hologram_intensity(intensity, target=None, title='Normalized Hologram Intensity Distribution', save_path=None):
    """
    Visualize normalized hologram intensity distribution
    
    Args:
        intensity: Intensity distribution array
        target: Target image (optional)
        title: Figure title
        save_path: Save path
    """
    if target is not None:
        # If target image is provided, display side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display target image
        axes[0].imshow(target, cmap='viridis')
        axes[0].set_title('Target Image')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # Display intensity distribution
        im = axes[1].imshow(intensity, cmap='viridis')
        axes[1].set_title(title)
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        
        # Add colorbar
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Normalized Intensity')
    else:
        # If no target image, display only intensity distribution
        plt.figure(figsize=(8, 6))
        im = plt.imshow(intensity, cmap='viridis')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Add colorbar
        plt.colorbar(im, label='Normalized Intensity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_scenario_results(results, scenario_name, save_dir=None):
    """
    Visualize scenario results
    
    Args:
        results: Scenario evaluation results list
        scenario_name: Scenario name
        save_dir: Save directory
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a large figure containing all results
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(5*n_results, 5))
    
    if n_results == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        # Get normalized intensity
        intensity = result['normalized_intensity']
        
        # Display intensity distribution
        im = axes[i].imshow(intensity, cmap='viridis')
        axes[i].set_title(f"{result['target']} (Efficiency: {result['efficiency']:.2f}%)")
        
        # Add colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Normalized Intensity')
    
    plt.suptitle(f'{scenario_name} Results', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the super title
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{scenario_name}_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_comparison_visualization(all_results, save_dir=None):
    """
    Create comparison visualization of all scenarios
    
    Args:
        all_results: Dictionary of all scenario results
        save_dir: Save directory
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a large figure containing representative results from all scenarios
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    
    # Scenario 1: Varying the number of cascaded elements
    scenario1_results = all_results['scenario1']['results']
    ax1 = axes[0]
    ax1.set_title('Scenario 1: Varying the Number of Cascaded Elements', fontsize=14)
    
    # Create subplot grid
    grid_size = len(scenario1_results)
    grid1 = make_axes_locatable(ax1).grid(grid_size, 1)
    
    for i, result in enumerate(scenario1_results):
        # Get normalized intensity
        intensity = result['normalized_intensity']
        
        # Display intensity distribution
        sub_ax = grid1[i, 0].axes
        im = sub_ax.imshow(intensity, cmap='viridis')
        sub_ax.set_title(f"{result['target']} (Efficiency: {result['efficiency']:.2f}%)")
        sub_ax.axis('off')
    
    # Scenario 5: Rotating diffractive layers
    scenario5_results = all_results['scenario5']['results']
    ax2 = axes[1]
    ax2.set_title('Scenario 5: Rotating Diffractive Layers', fontsize=14)
    
    # Create subplot grid
    grid_size = len(scenario5_results)
    grid2 = make_axes_locatable(ax2).grid(grid_size, 1)
    
    for i, result in enumerate(scenario5_results):
        # Get normalized intensity
        intensity = result['normalized_intensity']
        
        # Display intensity distribution
        sub_ax = grid2[i, 0].axes
        im = sub_ax.imshow(intensity, cmap='viridis')
        sub_ax.set_title(f"{result['target']} (Efficiency: {result['efficiency']:.2f}%)")
        sub_ax.axis('off')
    
    # Combined Scenario: Translation and Rotation
    combined_results = all_results['combined']['results']
    ax3 = axes[2]
    ax3.set_title('Combined Scenario: Translation and Rotation', fontsize=14)
    
    # Create subplot grid
    grid_size = len(combined_results)
    grid3 = make_axes_locatable(ax3).grid(grid_size, 1)
    
    for i, result in enumerate(combined_results):
        # Get normalized intensity
        intensity = result['normalized_intensity']
        
        # Display intensity distribution
        sub_ax = grid3[i, 0].axes
        im = sub_ax.imshow(intensity, cmap='viridis')
        sub_ax.set_title(f"{result['target']} (Efficiency: {result['efficiency']:.2f}%)")
        sub_ax.axis('off')
    
    # Hide main axes
    for ax in axes:
        ax.axis('off')
    
    plt.suptitle('Multi-Degree-of-Freedom Reconfigurable Terahertz Hologram Comparison', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for the super title
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'all_scenarios_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_wavelength_scenario_results(results, save_dir=None):
    """
    Visualize wavelength-dependent scenario results
    
    Args:
        results: Wavelength scenario evaluation results list
        save_dir: Save directory
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a large figure containing all results
    n_results = len(results)
    fig, axes = plt.subplots(2, (n_results+1)//2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        # Get normalized intensity
        intensity = result['normalized_intensity']
        
        # Display intensity distribution
        im = axes[i].imshow(intensity, cmap='viridis')
        axes[i].set_title(f"{result['target']} at {result['wavelength']*1000:.1f}mm ({result['thz_freq']:.2f} THz)\nEfficiency: {result['efficiency']:.2f}%")
        
        # Add colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Normalized Intensity')
    
    # Hide any unused subplots
    for i in range(n_results, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Wavelength-Dependent Scenario Results', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the super title
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'wavelength_scenario_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_all_results(all_results, base_dir):
    """
    Visualize all results
    
    Args:
        all_results: Dictionary of all scenario results
        base_dir: Base save directory
    """
    # Create visualization directory
    vis_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize Scenario 1 results
    scenario1_dir = os.path.join(vis_dir, 'scenario1')
    os.makedirs(scenario1_dir, exist_ok=True)
    
    visualize_pixel_heights(
        all_results['scenario1']['model'], 
        save_path=os.path.join(scenario1_dir, 'pixel_heights.png')
    )
    
    visualize_scenario_results(
        all_results['scenario1']['results'],
        'Scenario 1: Varying the Number of Cascaded Elements',
        save_dir=scenario1_dir
    )
    
    # Visualize Scenario 5 results
    scenario5_dir = os.path.join(vis_dir, 'scenario5')
    os.makedirs(scenario5_dir, exist_ok=True)
    
    visualize_pixel_heights(
        all_results['scenario5']['model'], 
        save_path=os.path.join(scenario5_dir, 'pixel_heights.png')
    )
    
    visualize_scenario_results(
        all_results['scenario5']['results'],
        'Scenario 5: Rotating Diffractive Layers',
        save_dir=scenario5_dir
    )
    
    # Visualize Combined Scenario results
    combined_dir = os.path.join(vis_dir, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    
    visualize_pixel_heights(
        all_results['combined']['model'], 
        save_path=os.path.join(combined_dir, 'pixel_heights.png')
    )
    
    visualize_scenario_results(
        all_results['combined']['results'],
        'Combined Scenario: Translation and Rotation',
        save_dir=combined_dir
    )
    
    # Create comparison visualization of all scenarios
    create_comparison_visualization(all_results, save_dir=vis_dir)
    

    # Visualize Wavelength Scenario results
    if 'wavelength' in all_results:
        wavelength_dir = os.path.join(vis_dir, 'wavelength_scenario')
        os.makedirs(wavelength_dir, exist_ok=True)
        
        visualize_pixel_heights(
            all_results['wavelength']['model'], 
            save_path=os.path.join(wavelength_dir, 'pixel_heights.png')
        )
        
        visualize_wavelength_scenario_results(
            all_results['wavelength']['results'],
            save_dir=wavelength_dir
        )


    print(f"All visualization results saved to {vis_dir}")
