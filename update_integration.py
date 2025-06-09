"""
Integration script for wavelength-dependent scenario

This script updates the necessary files to integrate the wavelength-dependent scenario
into the main codebase and ensures proper visualization.
"""

import os
import sys

def update_visualization_module():
    """Update the visualization.py file to include wavelength-dependent scenario"""
    vis_path = "/home/ubuntu/terahertz_hologram/visualization.py"
    
    # Read current visualization.py
    with open(vis_path, 'r') as f:
        vis_content = f.read()
    
    # Add wavelength visualization function
    wavelength_vis_function = '''
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
'''
    
    # Check if function already exists
    if "def visualize_wavelength_scenario_results" not in vis_content:
        # Find position to insert (before the last function)
        last_function = "def visualize_all_results"
        insert_pos = vis_content.rfind(last_function)
        
        if insert_pos != -1:
            # Insert before the last function
            vis_content = vis_content[:insert_pos] + wavelength_vis_function + "\n\n" + vis_content[insert_pos:]
    
    # Update visualize_all_results function to include wavelength scenario
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
    
    # Check if wavelength visualization is already included
    if "# Visualize Wavelength Scenario results" not in vis_content:
        # Find position to insert (before the final print statement)
        insert_pos = vis_content.find("    print(f\"All visualization results saved to {vis_dir}\")")
        
        if insert_pos != -1:
            # Insert before the print statement
            vis_content = vis_content[:insert_pos] + wavelength_vis_code + "\n\n" + vis_content[insert_pos:]
    
    # Write updated visualization.py
    with open(vis_path, 'w') as f:
        f.write(vis_content)
    
    print("Updated visualization.py to include wavelength-dependent scenario")

def update_main_with_wavelength_scenario():
    """Update main.py to include the wavelength-dependent scenario"""
    main_path = "/home/ubuntu/terahertz_hologram/main.py"
    
    # Read current main.py
    with open(main_path, 'r') as f:
        main_content = f.read()
    
    # Add import for wavelength-dependent scenario
    import_line = "from scenarios import scenario1_varying_number_of_elements, scenario5_rotating_elements, combined_scenario_translation_rotation"
    new_import = "from scenarios import scenario1_varying_number_of_elements, scenario5_rotating_elements, combined_scenario_translation_rotation\nfrom wavelength_scenario import wavelength_dependent_scenario"
    
    if new_import not in main_content:
        main_content = main_content.replace(import_line, new_import)
    
    # Add wavelength scenario to main function
    combined_scenario_line = "    # Run Combined Scenario: Translation and Rotation\n    print(\"\\nRunning Combined Scenario: Translation and Rotation\")\n    model_combined, results_combined = combined_scenario_translation_rotation()"
    wavelength_scenario_code = """    # Run Combined Scenario: Translation and Rotation
    print("\\nRunning Combined Scenario: Translation and Rotation")
    model_combined, results_combined = combined_scenario_translation_rotation()
    
    # Run Wavelength-Dependent Scenario
    print("\\nRunning Wavelength-Dependent Scenario")
    model_wavelength, results_wavelength = wavelength_dependent_scenario()"""
    
    if "# Run Wavelength-Dependent Scenario" not in main_content:
        main_content = main_content.replace(combined_scenario_line, wavelength_scenario_code)
    
    # Update all_results dictionary
    all_results_line = "    all_results = {\n        'scenario1': {'model': model1, 'results': results1},\n        'scenario5': {'model': model5, 'results': results5},\n        'combined': {'model': model_combined, 'results': results_combined}\n    }"
    new_all_results = """    all_results = {
        'scenario1': {'model': model1, 'results': results1},
        'scenario5': {'model': model5, 'results': results5},
        'combined': {'model': model_combined, 'results': results_combined},
        'wavelength': {'model': model_wavelength, 'results': results_wavelength}
    }"""
    
    if "'wavelength': {'model': model_wavelength" not in main_content:
        main_content = main_content.replace(all_results_line, new_all_results)
    
    # Write updated main.py
    with open(main_path, 'w') as f:
        f.write(main_content)
    
    print("Updated main.py to include wavelength-dependent scenario")

def update_readme_with_wavelength_scenario():
    """Update README.md to include information about the wavelength-dependent scenario"""
    readme_path = "/home/ubuntu/terahertz_hologram/README.md"
    
    # Read current README.md
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    # Add wavelength scenario to Multiple Reconfigurable Scenarios section
    wavelength_scenario_desc = """
4. **Wavelength-Dependent Scenario**: Different wavelengths produce different images
   - 0.3 THz (1.0mm) wavelength generates letter "W"
   - 0.25 THz (1.2mm) wavelength generates letter "A"
   - 0.2 THz (1.5mm) wavelength generates letter "V"
   - 0.15 THz (2.0mm) wavelength generates letter "E"
"""
    
    # Check if wavelength scenario is already included
    if "**Wavelength-Dependent Scenario**" not in readme_content:
        # Find position to insert (after the combined scenario)
        combined_scenario_pos = readme_content.find("3. **Combined Scenario**")
        if combined_scenario_pos != -1:
            # Find the end of the combined scenario section
            next_section_pos = readme_content.find("\n\n", combined_scenario_pos)
            if next_section_pos != -1:
                # Insert after combined scenario
                readme_content = readme_content[:next_section_pos] + wavelength_scenario_desc + readme_content[next_section_pos:]
    
    # Add wavelength_scenario.py to File Structure section
    wavelength_file_desc = "- `wavelength_scenario.py`: Implementation of wavelength-dependent holographic imaging\n"
    
    # Check if wavelength_scenario.py is already included
    if "`wavelength_scenario.py`" not in readme_content:
        # Find position to insert (after main.py)
        main_py_pos = readme_content.find("- `main.py`")
        if main_py_pos != -1:
            # Find the end of the main.py line
            next_line_pos = readme_content.find("\n", main_py_pos)
            if next_line_pos != -1:
                # Insert after main.py line
                readme_content = readme_content[:next_line_pos+1] + wavelength_file_desc + readme_content[next_line_pos+1:]
    
    # Write updated README.md
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("Updated README.md to include wavelength-dependent scenario")

def main():
    """Main function to update all necessary files"""
    print("Integrating wavelength-dependent scenario...")
    
    # Update visualization module
    update_visualization_module()
    
    # Update main.py
    update_main_with_wavelength_scenario()
    
    # Update README.md
    update_readme_with_wavelength_scenario()
    
    print("Integration complete!")

if __name__ == "__main__":
    main()
