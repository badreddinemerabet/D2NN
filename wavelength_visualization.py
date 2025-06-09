"""
Visualization Module for Wavelength-Dependent Scenario

This module extends the visualization functionality to include the wavelength-dependent scenario.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def update_visualization_module():
    """Update the visualization module to include wavelength-dependent scenario"""
    vis_path = "/home/ubuntu/terahertz_hologram/visualization.py"
    
    # Read current visualization.py
    with open(vis_path, 'r') as f:
        vis_content = f.read()
    
    # Add import for wavelength visualization
    if "def visualize_wavelength_scenario_results" not in vis_content:
        # Add the function to the file
        new_function = """
def visualize_wavelength_scenario_results(results, save_dir=None):
    \"\"\"
    Visualize wavelength-dependent scenario results
    
    Args:
        results: Wavelength scenario evaluation results list
        save_dir: Save directory
    \"\"\"
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
        axes[i].set_title(f"{result['target']} at {result['wavelength']*1000:.1f}mm ({result['thz_freq']:.2f} THz)\\nEfficiency: {result['efficiency']:.2f}%")
        
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
"""
        # Find the end of the last function
        last_function_end = vis_content.rfind("def ")
        last_function_end = vis_content.find("\n\n", last_function_end)
        
        if last_function_end == -1:
            # If not found, append to the end
            vis_content += new_function
        else:
            # Insert after the last function
            vis_content = vis_content[:last_function_end+2] + new_function + vis_content[last_function_end+2:]
    
    # Update visualize_all_results function to include wavelength scenario
    all_results_function = "def visualize_all_results(all_results, base_dir):"
    if all_results_function in vis_content:
        # Find the end of the function
        function_start = vis_content.find(all_results_function)
        function_end = vis_content.find("    print(f\"All visualization results saved to {vis_dir}\")", function_start)
        
        if function_end != -1:
            # Add wavelength scenario visualization
            wavelength_vis_code = """
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
"""
            # Insert before the print statement
            vis_content = vis_content[:function_end] + wavelength_vis_code + vis_content[function_end:]
    
    # Write updated visualization.py
    with open(vis_path, 'w') as f:
        f.write(vis_content)
    
    print("Updated visualization.py to include wavelength-dependent scenario")

def update_readme_with_wavelength_scenario():
    """Update README.md to include information about the wavelength-dependent scenario"""
    readme_path = "/home/ubuntu/terahertz_hologram/README.md"
    
    # Read current README.md
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    # Add wavelength scenario to Multiple Reconfigurable Scenarios section
    scenarios_section = "### Multiple Reconfigurable Scenarios"
    if scenarios_section in readme_content:
        scenarios_end = readme_content.find("## Reference", readme_content.find(scenarios_section))
        
        if scenarios_end != -1:
            # Add wavelength scenario description
            wavelength_scenario_desc = """
4. **Wavelength-Dependent Scenario**: Different wavelengths produce different images
   - 0.3 THz (1.0mm) wavelength generates letter "W"
   - 0.25 THz (1.2mm) wavelength generates letter "A"
   - 0.2 THz (1.5mm) wavelength generates letter "V"
   - 0.15 THz (2.0mm) wavelength generates letter "E"

"""
            # Insert before the Reference section
            readme_content = readme_content[:scenarios_end] + wavelength_scenario_desc + readme_content[scenarios_end:]
    
    # Add wavelength_scenario.py to File Structure section
    file_structure_section = "## File Structure"
    if file_structure_section in readme_content:
        file_structure_end = readme_content.find("## Usage Instructions", readme_content.find(file_structure_section))
        
        if file_structure_end != -1:
            # Add wavelength scenario file
            wavelength_file_desc = "- `wavelength_scenario.py`: Implementation of wavelength-dependent holographic imaging\n"
            
            # Find the position to insert (after the last file entry)
            main_py_pos = readme_content.find("- `main.py`:", readme_content.find(file_structure_section))
            if main_py_pos != -1:
                next_line_pos = readme_content.find("\n", main_py_pos)
                if next_line_pos != -1:
                    # Insert after main.py line
                    readme_content = readme_content[:next_line_pos+1] + wavelength_file_desc + readme_content[next_line_pos+1:]
    
    # Write updated README.md
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("Updated README.md to include wavelength-dependent scenario")

if __name__ == "__main__":
    # Update visualization module
    update_visualization_module()
    
    # Update README
    update_readme_with_wavelength_scenario()
