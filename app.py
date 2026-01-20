import matplotlib
import numpy as np
matplotlib.use('Agg')  # Set non-interactive backend BEFORE other imports

import matplotlib.pyplot as plt
import os
import json
from sectionproperties.pre.library import concrete_rectangular_section

from concreteproperties import (
    Concrete,
    ConcreteLinear,
    ConcreteSection,
    RectangularStressBlock,
    SteelBar,
    SteelElasticPlastic,
)


# ============================================================================
# REUSABLE HELPER FUNCTIONS
# ============================================================================

def make_serializable(obj):
    """Convert object to JSON serializable format"""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {k: make_serializable(v) for k, v in vars(obj).items() if not k.startswith('_')}
    else:
        return str(obj)


def save_to_json(data, save_path):
    """
    Reusable function to save data to JSON file.
    
    Parameters:
    -----------
    data : dict
        Dictionary to save
    save_path : str
        Path where to save the JSON file
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON saved to: {save_path}")


def save_plot(ax, save_path, dpi=150):
    """
    Reusable function to save matplotlib plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object
    save_path : str
        Path where to save the plot
    dpi : int
        DPI for the saved image
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    ax.get_figure().savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(ax.get_figure())
    print(f"Plot saved to: {save_path}")


# ============================================================================
# 1. PLOT SECTION
# ============================================================================

def plot_and_save_concrete_section(
    # Section dimensions
    depth,
    width,
    
    # Top reinforcement
    dia_top,
    area_top,
    n_top,
    cover_top,
    
    # Bottom reinforcement
    dia_bot,
    area_bot,
    n_bot,
    cover_bot,
    
    # Concrete properties
    concrete_name="Concrete",
    concrete_density=2.4e-6,
    elastic_modulus=30.1e3,
    compressive_strength=32,
    alpha=0.802,
    gamma=0.89,
    ultimate_strain=0.003,
    flexural_tensile_strength=3.4,
    concrete_colour="lightgrey",
    
    # Steel properties
    steel_name="Steel",
    steel_density=7.85e-6,
    yield_strength=500,
    steel_elastic_modulus=200e3,
    fracture_strain=0.05,
    steel_colour="grey",
    
    # Save options
    save_path="section_plot.png",
    plot_title="Reinforced Concrete Section",
):
    """
    Plot and save a reinforced concrete section.
    """
    # Create concrete material
    concrete = Concrete(
        name=concrete_name,
        density=concrete_density,
        stress_strain_profile=ConcreteLinear(elastic_modulus=elastic_modulus),
        ultimate_stress_strain_profile=RectangularStressBlock(
            compressive_strength=compressive_strength,
            alpha=alpha,
            gamma=gamma,
            ultimate_strain=ultimate_strain,
        ),
        flexural_tensile_strength=flexural_tensile_strength,
        colour=concrete_colour,
    )
    
    # Create steel material
    steel = SteelBar(
        name=steel_name,
        density=steel_density,
        stress_strain_profile=SteelElasticPlastic(
            yield_strength=yield_strength,
            elastic_modulus=steel_elastic_modulus,
            fracture_strain=fracture_strain,
        ),
        colour=steel_colour,
    )
    
    # Create geometry
    geom = concrete_rectangular_section(
        d=depth,
        b=width,
        dia_top=dia_top,
        area_top=area_top,
        n_top=n_top,
        c_top=cover_top,
        dia_bot=dia_bot,
        area_bot=area_bot,
        n_bot=n_bot,
        c_bot=cover_bot,
        conc_mat=concrete,
        steel_mat=steel,
    )
    
    # Create concrete section
    conc_sec = ConcreteSection(geom)
    
    # Plot the section (render=False prevents display)
    ax = conc_sec.plot_section(title=plot_title, render=False)
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Save the figure
    ax.get_figure().savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Close the figure to free memory
    plt.close(ax.get_figure())
    
    print(f"Section plot saved to: {save_path}")
    
    return conc_sec


# ============================================================================
# 2. GROSS AND TRANSFORMED PROPERTIES
# ============================================================================

def save_properties_to_json(conc_sec, gross_path="gross_properties.json", transformed_path="transformed_properties.json"):
    """Save gross and transformed properties to separate JSON files."""
    gross_props = conc_sec.get_gross_properties()
    transformed_props = conc_sec.get_transformed_gross_properties(elastic_modulus=30.1e3)
    
    # Convert to serializable dictionaries
    gross_dict = make_serializable(gross_props)
    transformed_dict = make_serializable(transformed_props)
    
    # Remove default_units if present
    if 'default_units' in gross_dict:
        del gross_dict['default_units']
    if 'default_units' in transformed_dict:
        del transformed_dict['default_units']
    
    save_to_json(gross_dict, gross_path)
    save_to_json(transformed_dict, transformed_path)


# ============================================================================
# 3. CRACKED PROPERTIES
# ============================================================================

def calculate_and_save_cracked_properties(
    conc_sec,
    theta_sag=0,
    theta_hog=np.pi,
    elastic_modulus=30.1e3,
    save_json_sag="cracked_sag.json",
    save_json_hog="cracked_hog.json",
    save_plot_sag="cracked_sag.png",
    save_plot_hog="cracked_hog.png",
):
    """
    Calculate cracked properties for sagging and hogging, save to JSON and plot.
    """
    # Calculate cracked properties
    cracked_res_sag = conc_sec.calculate_cracked_properties(theta=theta_sag)
    cracked_res_hog = conc_sec.calculate_cracked_properties(theta=theta_hog)
    
    # Calculate transformed properties
    cracked_res_sag.calculate_transformed_properties(elastic_modulus=elastic_modulus)
    cracked_res_hog.calculate_transformed_properties(elastic_modulus=elastic_modulus)
    
    # Save sagging properties to JSON
    sag_dict = make_serializable(cracked_res_sag)
    if 'default_units' in sag_dict:
        del sag_dict['default_units']
    save_to_json(sag_dict, save_json_sag)
    
    # Save hogging properties to JSON
    hog_dict = make_serializable(cracked_res_hog)
    if 'default_units' in hog_dict:
        del hog_dict['default_units']
    save_to_json(hog_dict, save_json_hog)
    
    # Plot sagging cracked geometries
    ax_sag = cracked_res_sag.plot_cracked_geometries(labels=[], cp=False, legend=False, render=False)
    save_plot(ax_sag, save_plot_sag)
    
    # Plot hogging cracked geometries
    ax_hog = cracked_res_hog.plot_cracked_geometries(labels=[], cp=False, legend=False, render=False)
    save_plot(ax_hog, save_plot_hog)
    
    return cracked_res_sag, cracked_res_hog


# ============================================================================
# 8. STRESS ANALYSIS
# ============================================================================

def calculate_uncracked_stress(
    conc_sec,
    n=0,
    m_x=0,
    m_y=0,
    save_json=None,
    save_plot_path=None,
):
    """
    Calculate elastic uncracked stress.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    n : float
        Axial force (default: 0)
    m_x : float
        Bending moment about x-axis (default: 0)
    m_y : float
        Bending moment about y-axis (default: 0)
    save_json : str or None
        Path to save JSON results
    save_plot_path : str or None
        Path to save stress plot
    
    Returns:
    --------
    StressResult object
    """
    # Calculate uncracked stress
    stress_result = conc_sec.calculate_uncracked_stress(n=n, m_x=m_x, m_y=m_y)
    
    # Save to JSON if requested
    if save_json:
        stress_dict = make_serializable(stress_result)
        if 'default_units' in stress_dict:
            del stress_dict['default_units']
        
        # Add summary
        forces = stress_result.sum_forces()
        moments = stress_result.sum_moments()
        stress_dict['summary'] = {
            'applied_n': n,
            'applied_m_x': m_x,
            'applied_m_y': m_y,
            'sum_forces_n': forces,
            'sum_moments_m_x': moments[0],
            'sum_moments_m_y': moments[1],
            'sum_moments_m_xy': moments[2],
        }
        
        save_to_json(stress_dict, save_json)
    
    # Save plot if requested
    if save_plot_path:
        ax = stress_result.plot_stress(render=False)
        save_plot(ax, save_plot_path)
    
    return stress_result


def calculate_cracked_stress(
    conc_sec,
    cracked_results,
    m,
    save_json=None,
    save_plot_path=None,
):
    """
    Calculate elastic cracked stress.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    cracked_results : CrackedResults
        Results from calculate_cracked_properties()
    m : float
        Applied bending moment
    save_json : str or None
        Path to save JSON results
    save_plot_path : str or None
        Path to save stress plot
    
    Returns:
    --------
    StressResult object
    """
    # Calculate cracked stress
    stress_result = conc_sec.calculate_cracked_stress(
        cracked_results=cracked_results,
        m=m,
    )
    
    # Save to JSON if requested
    if save_json:
        stress_dict = make_serializable(stress_result)
        if 'default_units' in stress_dict:
            del stress_dict['default_units']
        
        # Add summary
        forces = stress_result.sum_forces()
        moments = stress_result.sum_moments()
        stress_dict['summary'] = {
            'applied_moment': m,
            'cracking_moment': cracked_results.m_cr,
            'sum_forces_n': forces,
            'sum_moments_m_x': moments[0],
            'sum_moments_m_y': moments[1],
            'sum_moments_m_xy': moments[2],
        }
        
        save_to_json(stress_dict, save_json)
    
    # Save plot if requested
    if save_plot_path:
        ax = stress_result.plot_stress(render=False)
        save_plot(ax, save_plot_path)
    
    return stress_result


def calculate_service_stress(
    conc_sec,
    moment_curvature_results,
    m=None,
    kappa=None,
    save_json=None,
    save_plot_path=None,
):
    """
    Calculate service stress from moment-curvature analysis.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    moment_curvature_results : MomentCurvatureResults
        Results from moment_curvature_analysis()
    m : float or None
        Applied bending moment (provide either m or kappa)
    kappa : float or None
        Applied curvature (provide either m or kappa)
    save_json : str or None
        Path to save JSON results
    save_plot_path : str or None
        Path to save stress plot
    
    Returns:
    --------
    StressResult object
    """
    # Calculate service stress
    stress_result = conc_sec.calculate_service_stress(
        moment_curvature_results=moment_curvature_results,
        m=m,
        kappa=kappa,
    )
    
    # Save to JSON if requested
    if save_json:
        stress_dict = make_serializable(stress_result)
        if 'default_units' in stress_dict:
            del stress_dict['default_units']
        
        # Add summary
        forces = stress_result.sum_forces()
        moments = stress_result.sum_moments()
        stress_dict['summary'] = {
            'applied_moment': m,
            'applied_curvature': kappa,
            'sum_forces_n': forces,
            'sum_moments_m_x': moments[0],
            'sum_moments_m_y': moments[1],
            'sum_moments_m_xy': moments[2],
        }
        
        save_to_json(stress_dict, save_json)
    
    # Save plot if requested
    if save_plot_path:
        ax = stress_result.plot_stress(render=False)
        save_plot(ax, save_plot_path)
    
    return stress_result


def calculate_ultimate_stress(
    conc_sec,
    ultimate_results,
    save_json=None,
    save_plot_path=None,
):
    """
    Calculate ultimate stress.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    ultimate_results : UltimateBendingResults
        Results from ultimate_bending_capacity()
    save_json : str or None
        Path to save JSON results
    save_plot_path : str or None
        Path to save stress plot
    
    Returns:
    --------
    StressResult object
    """
    # Calculate ultimate stress
    stress_result = conc_sec.calculate_ultimate_stress(
        ultimate_results=ultimate_results,
    )
    
    # Save to JSON if requested
    if save_json:
        stress_dict = make_serializable(stress_result)
        if 'default_units' in stress_dict:
            del stress_dict['default_units']
        
        # Add summary
        forces = stress_result.sum_forces()
        moments = stress_result.sum_moments()
        stress_dict['summary'] = {
            'neutral_axis_depth': ultimate_results.d_n,
            'k_u': ultimate_results.k_u,
            'sum_forces_n': forces,
            'sum_moments_m_x': moments[0],
            'sum_moments_m_y': moments[1],
            'sum_moments_m_xy': moments[2],
        }
        
        save_to_json(stress_dict, save_json)
    
    # Save plot if requested
    if save_plot_path:
        ax = stress_result.plot_stress(render=False)
        save_plot(ax, save_plot_path)
    
    return stress_result


def multiple_service_stress_analysis(
    conc_sec,
    moment_curvature_results,
    stress_points,
    save_dir="output/service_stress",
):
    """
    Perform service stress analysis at multiple points.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    moment_curvature_results : MomentCurvatureResults
        Results from moment_curvature_analysis()
    stress_points : list of dict
        List of dicts with 'm_or_k' and 'val' keys
        Example: [{"m_or_k": "m", "val": 50e6}, {"m_or_k": "k", "val": 1e-6}]
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    list of StressResult objects
    """
    stress_results = []
    
    for idx, point in enumerate(stress_points):
        # Determine if input is moment or curvature
        if point["m_or_k"] == "m":
            m_val = point["val"]
            kappa_val = None
        else:
            m_val = None
            kappa_val = point["val"]
        
        # Calculate service stress
        stress_result = conc_sec.calculate_service_stress(
            moment_curvature_results=moment_curvature_results,
            m=m_val,
            kappa=kappa_val,
        )
        
        # Get actual moment value
        if kappa_val is not None:
            m_actual = stress_result.sum_moments()[2]
        else:
            m_actual = m_val
        
        # Save results
        save_json_path = f"{save_dir}/service_stress_{idx+1}.json"
        save_plot_path = f"{save_dir}/service_stress_{idx+1}.png"
        
        stress_dict = make_serializable(stress_result)
        if 'default_units' in stress_dict:
            del stress_dict['default_units']
        
        forces = stress_result.sum_forces()
        moments = stress_result.sum_moments()
        stress_dict['summary'] = {
            'point_number': idx + 1,
            'applied_moment': m_actual,
            'applied_curvature': kappa_val,
            'sum_forces_n': forces,
            'sum_moments_m_xy': moments[2],
        }
        
        save_to_json(stress_dict, save_json_path)
        
        ax = stress_result.plot_stress(
            title=f"M = {m_actual / 1e6:.0f} kN.m",
            render=False,
        )
        save_plot(ax, save_plot_path)
        
        stress_results.append(stress_result)
        print(f"Service stress point {idx+1}: M = {m_actual / 1e6:.2f} kN.m")
    
    return stress_results


def multiple_ultimate_stress_analysis(
    conc_sec,
    ultimate_cases,
    labels=None,
    save_dir="output/ultimate_stress",
):
    """
    Perform ultimate stress analysis at multiple points.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    ultimate_cases : list of dict
        List of dicts with 'theta' and 'n' keys
        Example: [{"theta": 0, "n": 0}, {"theta": 0, "n": 1000e3}]
    labels : list of str or None
        Labels for each case
    save_dir : str
        Directory to save results
    
    Returns:
    --------
    list of StressResult objects
    """
    stress_results = []
    
    for idx, case in enumerate(ultimate_cases):
        label = labels[idx] if labels else f"case_{idx+1}"
        
        # Calculate ultimate bending capacity
        ult_result = conc_sec.ultimate_bending_capacity(
            theta=case.get("theta", 0),
            n=case.get("n", 0),
        )
        
        # Calculate ultimate stress
        stress_result = conc_sec.calculate_ultimate_stress(
            ultimate_results=ult_result,
        )
        
        # Save results
        save_json_path = f"{save_dir}/ultimate_stress_{label}.json"
        save_plot_path = f"{save_dir}/ultimate_stress_{label}.png"
        
        stress_dict = make_serializable(stress_result)
        if 'default_units' in stress_dict:
            del stress_dict['default_units']
        
        forces = stress_result.sum_forces()
        moments = stress_result.sum_moments()
        stress_dict['summary'] = {
            'label': label,
            'axial_force': case.get("n", 0),
            'bending_angle_deg': np.degrees(case.get("theta", 0)),
            'neutral_axis_depth': ult_result.d_n,
            'sum_forces_n': forces,
            'sum_moments_m_xy': moments[2],
        }
        
        save_to_json(stress_dict, save_json_path)
        
        ax = stress_result.plot_stress(title=label, render=False)
        save_plot(ax, save_plot_path)
        
        stress_results.append(stress_result)
        print(f"Ultimate stress - {label}: N = {case.get('n', 0) / 1e3:.0f} kN, M = {ult_result.m_xy / 1e6:.2f} kN.m")
    
    return stress_results


# ============================================================================
# 7. BIAXIAL BENDING DIAGRAM
# ============================================================================

def biaxial_bending_diagram(
    conc_sec,
    n=0,
    n_points=48,
    save_json=None,
    save_plot_path=None,
):
    """
    Generate biaxial bending diagram.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    n : float
        Axial force (default: 0)
    n_points : int
        Number of calculation points (default: 48)
    save_json : str or None
        Path to save JSON results
    save_plot_path : str or None
        Path to save plot
    
    Returns:
    --------
    BiaxialBendingResults object
    """
    # Perform biaxial bending analysis
    bb_results = conc_sec.biaxial_bending_diagram(
        n=n,
        n_points=n_points,
        progress_bar=False,
    )
    
    # Save to JSON if requested
    if save_json:
        bb_dict = make_serializable(bb_results)
        if 'default_units' in bb_dict:
            del bb_dict['default_units']
        
        # Add summary
        bb_dict['summary'] = {
            'axial_force': n,
            'number_of_points': len(bb_results.results),
            'max_m_x': max([abs(r.m_x) for r in bb_results.results]) if bb_results.results else None,
            'max_m_y': max([abs(r.m_y) for r in bb_results.results]) if bb_results.results else None,
        }
        
        save_to_json(bb_dict, save_json)
    
    # Save plot if requested
    if save_plot_path:
        ax = bb_results.plot_diagram(render=False)
        save_plot(ax, save_plot_path)
    
    return bb_results


def multiple_biaxial_bending_diagrams(
    conc_sec,
    axial_forces,
    n_points=24,
    save_json=None,
    save_plot_2d_path=None,
    save_plot_3d_path=None,
    labels=None,
):
    """
    Generate multiple biaxial bending diagrams with different axial forces.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    axial_forces : list of float
        List of axial forces to analyze
    n_points : int
        Number of calculation points per diagram (default: 24)
    save_json : str or None
        Path to save combined JSON results
    save_plot_2d_path : str or None
        Path to save 2D plot
    save_plot_3d_path : str or None
        Path to save 3D plot
    labels : list of str or None
        Labels for each diagram
    
    Returns:
    --------
    list of BiaxialBendingResults objects
    
    Example:
    --------
    >>> axial_forces = [0, 1000e3, 2000e3, 3000e3]
    >>> results = multiple_biaxial_bending_diagrams(
    ...     conc_sec,
    ...     axial_forces,
    ...     labels=[f"N={n/1e3:.0f}kN" for n in axial_forces]
    ... )
    """
    from concreteproperties.results import BiaxialBendingResults
    
    bb_results = []
    
    # Generate diagram for each axial force
    for n in axial_forces:
        bb_res = conc_sec.biaxial_bending_diagram(
            n=n,
            n_points=n_points,
            progress_bar=False,
        )
        bb_results.append(bb_res)
    
    # Save combined JSON if requested
    if save_json:
        combined_dict = {}
        for idx, result in enumerate(bb_results):
            n_val = axial_forces[idx]
            label = labels[idx] if labels else f"N_{int(n_val/1e3)}_kN"
            result_dict = make_serializable(result)
            if 'default_units' in result_dict:
                del result_dict['default_units']
            
            # Add summary
            result_dict['summary'] = {
                'axial_force': n_val,
                'number_of_points': len(result.results),
                'max_m_x': max([abs(r.m_x) for r in result.results]) if result.results else None,
                'max_m_y': max([abs(r.m_y) for r in result.results]) if result.results else None,
            }
            
            combined_dict[label] = result_dict
        save_to_json(combined_dict, save_json)
    
    # Save 2D plot if requested
    if save_plot_2d_path:
        ax = BiaxialBendingResults.plot_multiple_diagrams_2d(
            bb_results,  # Pass as positional argument
            fmt="o-",
            labels=labels,
            render=False,
        )
        save_plot(ax, save_plot_2d_path)
    
    # Save 3D plot if requested
    if save_plot_3d_path:
        ax = BiaxialBendingResults.plot_multiple_diagrams_3d(
            bb_results,  # Pass as positional argument
        )
        save_plot(ax, save_plot_3d_path)
    
    return bb_results


def find_decompression_point(conc_sec):
    """
    Find the decompression point for M_x and M_y bending.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    
    Returns:
    --------
    tuple of (n_decomp_mx, n_decomp_my)
    """
    # Generate moment interaction diagrams
    mi_x = conc_sec.moment_interaction_diagram(
        theta=0,
        progress_bar=False,
    )
    mi_y = conc_sec.moment_interaction_diagram(
        theta=np.pi / 2,
        progress_bar=False,
    )
    
    # Decompression point is typically the second point
    n_decomp_mx = mi_x.results[1].n if len(mi_x.results) > 1 else None
    n_decomp_my = mi_y.results[1].n if len(mi_y.results) > 1 else None
    
    return n_decomp_mx, n_decomp_my


def biaxial_bending_analysis_suite(
    conc_sec,
    axial_forces=None,
    n_points=24,
    save_dir="output/biaxial",
):
    """
    Comprehensive biaxial bending analysis suite.
    
    Generates:
    - Decompression point analysis
    - Multiple biaxial bending diagrams
    - 2D and 3D plots
    - JSON results
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    axial_forces : list of float or None
        List of axial forces (if None, auto-generates from 0 to decompression)
    n_points : int
        Number of calculation points per diagram
    save_dir : str
        Directory to save all results
    
    Returns:
    --------
    dict containing all results and decompression points
    """
    # Find decompression points
    print("Finding decompression points...")
    n_decomp_mx, n_decomp_my = find_decompression_point(conc_sec)
    
    print(f"Decompression point for M_x: N = {n_decomp_mx / 1e3:.2f} kN")
    print(f"Decompression point for M_y: N = {n_decomp_my / 1e3:.2f} kN")
    
    # Generate axial forces if not provided
    if axial_forces is None:
        n_max = min(n_decomp_mx, n_decomp_my) * 0.99  # 99% of decompression
        axial_forces = np.linspace(0, n_max, 5).tolist()
    
    # Generate labels
    labels = [f"N = {n / 1e3:.0f} kN" for n in axial_forces]
    
    # Generate multiple biaxial bending diagrams
    print(f"\nGenerating {len(axial_forces)} biaxial bending diagrams...")
    bb_results = multiple_biaxial_bending_diagrams(
        conc_sec=conc_sec,
        axial_forces=axial_forces,
        n_points=n_points,
        save_json=f"{save_dir}/biaxial_multiple.json",
        save_plot_2d_path=f"{save_dir}/biaxial_2d.png",
        save_plot_3d_path=f"{save_dir}/biaxial_3d.png",
        labels=labels,
    )
    
    # Generate individual diagrams for key cases
    print("\nGenerating individual biaxial diagrams...")
    
    # Pure bending
    bb_pure = biaxial_bending_diagram(
        conc_sec=conc_sec,
        n=0,
        n_points=n_points,
        save_json=f"{save_dir}/biaxial_n0.json",
        save_plot_path=f"{save_dir}/biaxial_n0.png",
    )
    
    # With axial load (if provided)
    if len(axial_forces) > 1:
        bb_axial = biaxial_bending_diagram(
            conc_sec=conc_sec,
            n=axial_forces[len(axial_forces)//2],  # middle value
            n_points=n_points,
            save_json=f"{save_dir}/biaxial_with_axial.json",
            save_plot_path=f"{save_dir}/biaxial_with_axial.png",
        )
    else:
        bb_axial = None
    
    return {
        'decompression_mx': n_decomp_mx,
        'decompression_my': n_decomp_my,
        'axial_forces': axial_forces,
        'labels': labels,
        'multiple_results': bb_results,
        'pure_bending': bb_pure,
        'with_axial': bb_axial,
    }


# ============================================================================
# 6. MOMENT INTERACTION DIAGRAM
# ============================================================================

def moment_interaction_diagram(
    conc_sec,
    theta=0,
    limits=None,
    control_points=None,
    labels=None,
    n_points=24,
    n_spacing=None,
    max_comp=None,
    max_comp_labels=None,
    save_json=None,
    save_plot_path=None,
    plot_moment="m_xy",
    plot_labels=False,
    label_offset=False,
):
    """
    Generate moment interaction diagram.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    theta : float
        Angle (in radians) the neutral axis makes with horizontal (default: 0)
    limits : list of tuples or None
        Control points defining start and end of diagram
    control_points : list of tuples or None
        Additional control points to analyze
    labels : list of str or None
        Labels for limits and control points
    n_points : int
        Number of points to compute (default: 24)
    n_spacing : int or None
        Override n_points with equally spaced axial loads
    max_comp : float or None
        Maximum compressive force limit
    max_comp_labels : list of str or None
        Labels for max_comp intersection points
    save_json : str or None
        Path to save JSON results
    save_plot_path : str or None
        Path to save plot
    plot_moment : str
        Which moment to plot ("m_xy", "m_x", or "m_y")
    plot_labels : bool
        Show labels on plot
    label_offset : bool
        Offset labels from diagram
    
    Returns:
    --------
    MomentInteractionResults object
    """
    # Perform moment interaction analysis
    mi_results = conc_sec.moment_interaction_diagram(
        theta=theta,
        limits=limits,
        control_points=control_points,
        labels=labels,
        n_points=n_points,
        n_spacing=n_spacing,
        max_comp=max_comp,
        max_comp_labels=max_comp_labels,
        progress_bar=False,
    )
    
    # Save to JSON if requested
    if save_json:
        mi_dict = make_serializable(mi_results)
        if 'default_units' in mi_dict:
            del mi_dict['default_units']
        
        # Add summary
        mi_dict['summary'] = {
            'number_of_points': len(mi_results.results),
            'bending_angle_deg': np.degrees(theta),
            'max_axial_force': max([r.n for r in mi_results.results]) if mi_results.results else None,
            'min_axial_force': min([r.n for r in mi_results.results]) if mi_results.results else None,
            'max_moment': max([r.m_xy for r in mi_results.results]) if mi_results.results else None,
        }
        
        save_to_json(mi_dict, save_json)
    
    # Save plot if requested
    if save_plot_path:
        ax = mi_results.plot_diagram(
            moment=plot_moment,
            labels=plot_labels,
            label_offset=label_offset,
            render=False,
        )
        save_plot(ax, save_plot_path)
    
    return mi_results


def multiple_moment_interaction_diagrams(
    sections_or_results,
    labels=None,
    save_json=None,
    save_plot_path=None,
    plot_moment="m_xy",
    theta=0,
):
    """
    Generate and plot multiple moment interaction diagrams.
    
    Parameters:
    -----------
    sections_or_results : list
        List of ConcreteSection objects or MomentInteractionResults objects
    labels : list of str or None
        Labels for each diagram
    save_json : str or None
        Path to save combined JSON results
    save_plot_path : str or None
        Path to save combined plot
    plot_moment : str
        Which moment to plot ("m_xy", "m_x", or "m_y")
    theta : float
        Angle for analysis (only used if sections provided)
    
    Returns:
    --------
    list of MomentInteractionResults objects
    
    Example:
    --------
    >>> sections = [section1, section2, section3]
    >>> results = multiple_moment_interaction_diagrams(
    ...     sections,
    ...     labels=["Section 1", "Section 2", "Section 3"]
    ... )
    """
    from concreteproperties.results import MomentInteractionResults
    
    # Check if we have sections or results
    if hasattr(sections_or_results[0], 'moment_interaction_diagram'):
        # We have sections, need to analyze
        mi_results = []
        for section in sections_or_results:
            mi_res = section.moment_interaction_diagram(
                theta=theta,
                progress_bar=False,
            )
            mi_results.append(mi_res)
    else:
        # We already have results
        mi_results = sections_or_results
    
    # Save combined JSON if requested
    if save_json:
        combined_dict = {}
        for idx, result in enumerate(mi_results):
            label = labels[idx] if labels else f"diagram_{idx}"
            result_dict = make_serializable(result)
            if 'default_units' in result_dict:
                del result_dict['default_units']
            
            # Add summary
            result_dict['summary'] = {
                'number_of_points': len(result.results),
                'max_axial_force': max([r.n for r in result.results]) if result.results else None,
                'min_axial_force': min([r.n for r in result.results]) if result.results else None,
                'max_moment': max([r.m_xy for r in result.results]) if result.results else None,
            }
            
            combined_dict[label] = result_dict
        save_to_json(combined_dict, save_json)
    
    # Save combined plot if requested
    if save_plot_path:
        ax = MomentInteractionResults.plot_multiple_diagrams(
            moment_interaction_results=mi_results,
            labels=labels,
            fmt="-",
            moment=plot_moment,
            render=False,
        )
        save_plot(ax, save_plot_path)
    
    return mi_results


def positive_negative_interaction_diagrams(
    conc_sec,
    save_json=None,
    save_plot_path=None,
):
    """
    Generate positive and negative moment interaction diagrams.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    save_json : str or None
        Path to save combined JSON results
    save_plot_path : str or None
        Path to save combined plot
    
    Returns:
    --------
    tuple of (positive_results, negative_results)
    """
    # Generate positive bending diagram
    mi_res_pos = conc_sec.moment_interaction_diagram(
        theta=0,
        progress_bar=False,
    )
    
    # Generate negative bending diagram
    mi_res_neg = conc_sec.moment_interaction_diagram(
        theta=np.pi,
        progress_bar=False,
    )
    
    # Save combined JSON if requested
    if save_json:
        combined_dict = {
            'positive_bending': make_serializable(mi_res_pos),
            'negative_bending': make_serializable(mi_res_neg),
        }
        
        # Remove default_units
        for key in combined_dict:
            if 'default_units' in combined_dict[key]:
                del combined_dict[key]['default_units']
            
            # Add summary
            result = mi_res_pos if key == 'positive_bending' else mi_res_neg
            combined_dict[key]['summary'] = {
                'number_of_points': len(result.results),
                'max_axial_force': max([r.n for r in result.results]) if result.results else None,
                'min_axial_force': min([r.n for r in result.results]) if result.results else None,
                'max_moment': max([r.m_xy for r in result.results]) if result.results else None,
            }
        
        save_to_json(combined_dict, save_json)
    
    # Save combined plot if requested
    if save_plot_path:
        from concreteproperties.results import MomentInteractionResults
        
        ax = MomentInteractionResults.plot_multiple_diagrams(
            moment_interaction_results=[mi_res_pos, mi_res_neg],
            labels=["Positive", "Negative"],
            fmt="-",
            render=False,
        )
        save_plot(ax, save_plot_path)
    
    return mi_res_pos, mi_res_neg


def advanced_moment_interaction_diagram(
    conc_sec,
    theta=0,
    save_json=None,
    save_plot_path=None,
):
    """
    Generate moment interaction diagram with advanced control points.
    
    This creates a diagram with multiple control points:
    - Decompression point (D, 1.0)
    - Steel decompression (fy, 0.0)
    - 50% yield strain (fy, 0.5)
    - 100% yield strain (fy, 1.0)
    - Custom neutral axis depth
    - Pure bending (N, 0.0)
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    theta : float
        Angle (in radians) the neutral axis makes with horizontal
    save_json : str or None
        Path to save JSON results
    save_plot_path : str or None
        Path to save plot
    
    Returns:
    --------
    MomentInteractionResults object
    """
    # Generate advanced diagram
    mi_results = conc_sec.moment_interaction_diagram(
        theta=theta,
        limits=[
            ("kappa0", 0.0),
            ("d_n", 1e-6),
        ],
        control_points=[
            ("D", 1.0),
            ("fy", 0.0),
            ("fy", 0.5),
            ("fy", 1.0),
            ("d_n", 200.0),
            ("N", 0.0),
        ],
        labels=["NA", "I", "C", "D", "E", "F", "G", "H"],
        n_spacing=36,
        progress_bar=False,
    )
    
    # Save to JSON if requested
    if save_json:
        mi_dict = make_serializable(mi_results)
        if 'default_units' in mi_dict:
            del mi_dict['default_units']
        
        # Add summary
        mi_dict['summary'] = {
            'number_of_points': len(mi_results.results),
            'bending_angle_deg': np.degrees(theta),
            'control_points': ['Decompression', 'Steel Decompression', '50% Yield', '100% Yield', 'd_n=200mm', 'Pure Bending'],
        }
        
        save_to_json(mi_dict, save_json)
    
    # Save plot if requested
    if save_plot_path:
        ax = mi_results.plot_diagram(
            fmt="-kx",
            labels=True,
            label_offset=True,
            render=False,
        )
        # Reset axis limits to ensure labels are within plot
        ax.set_xlim(-20, 850)
        ax.set_ylim(-3000, 9000)
        save_plot(ax, save_plot_path)
    
    return mi_results


# ============================================================================
# 5. ULTIMATE BENDING CAPACITY
# ============================================================================

def ultimate_bending_capacity(
    conc_sec,
    theta=0,
    n=0,
    save_json=None,
):
    """
    Calculate ultimate bending capacity.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    theta : float
        Angle (in radians) the neutral axis makes with horizontal (default: 0)
    n : float
        Axial force (default: 0)
    save_json : str or None
        Path to save JSON results (None = don't save)
    
    Returns:
    --------
    UltimateBendingResults object
    """
    # Perform ultimate bending capacity analysis
    ult_results = conc_sec.ultimate_bending_capacity(theta=theta, n=n)
    
    # Save to JSON if requested
    if save_json:
        ult_dict = make_serializable(ult_results)
        if 'default_units' in ult_dict:
            del ult_dict['default_units']
        
        # Add summary
        ult_dict['summary'] = {
            'bending_angle_deg': np.degrees(theta),
            'axial_force': n,
            'resultant_moment': ult_results.m_xy,
            'neutral_axis_depth': ult_results.d_n,
            'k_u': ult_results.k_u,
        }
        
        save_to_json(ult_dict, save_json)
    
    return ult_results


def multiple_ultimate_bending_capacities(
    conc_sec,
    analysis_cases,
    save_json=None,
    labels=None,
):
    """
    Calculate multiple ultimate bending capacities.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    analysis_cases : list of dict
        List of dictionaries containing analysis parameters
        Each dict can have: theta, n
    save_json : str or None
        Path to save combined JSON results
    labels : list of str or None
        Labels for each analysis case
    
    Returns:
    --------
    list of UltimateBendingResults objects
    
    Example:
    --------
    >>> cases = [
    ...     {"theta": 0, "n": 0},
    ...     {"theta": np.pi, "n": 0},
    ...     {"theta": np.pi/2, "n": 0},
    ... ]
    >>> results = multiple_ultimate_bending_capacities(
    ...     conc_sec, cases, labels=["Sagging", "Hogging", "Weak Axis"]
    ... )
    """
    results = []
    
    # Default parameters
    defaults = {"theta": 0, "n": 0}
    
    # Perform each analysis
    for case in analysis_cases:
        params = {**defaults, **case}
        
        ult_results = conc_sec.ultimate_bending_capacity(
            theta=params["theta"],
            n=params["n"],
        )
        results.append(ult_results)
    
    # Save combined JSON if requested
    if save_json:
        combined_dict = {}
        for idx, result in enumerate(results):
            label = labels[idx] if labels else f"case_{idx}"
            result_dict = make_serializable(result)
            if 'default_units' in result_dict:
                del result_dict['default_units']
            
            # Add summary
            theta_val = analysis_cases[idx].get("theta", 0)
            result_dict['summary'] = {
                'bending_angle_deg': np.degrees(theta_val),
                'axial_force': analysis_cases[idx].get("n", 0),
                'resultant_moment': result.m_xy,
                'neutral_axis_depth': result.d_n,
                'k_u': result.k_u,
            }
            
            combined_dict[label] = result_dict
        save_to_json(combined_dict, save_json)
    
    return results


def compare_ultimate_capacities_with_axial_load(
    conc_sec,
    bending_angles,
    axial_loads,
    save_json=None,
    angle_labels=None,
):
    """
    Compare ultimate bending capacities at different bending angles and axial loads.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    bending_angles : list of float
        List of bending angles (in radians) to analyze
    axial_loads : list of float
        List of axial loads to analyze
    save_json : str or None
        Path to save comparison JSON
    angle_labels : list of str or None
        Labels for each bending angle
    
    Returns:
    --------
    dict of results organized by axial load and bending angle
    
    Example:
    --------
    >>> compare_ultimate_capacities_with_axial_load(
    ...     conc_sec,
    ...     bending_angles=[0, np.pi, np.pi/2],
    ...     axial_loads=[0, 5000e3],
    ...     angle_labels=["Sagging", "Hogging", "Weak Axis"]
    ... )
    """
    results = {}
    
    for n in axial_loads:
        n_label = f"N_{int(n/1e3)}_kN" if n != 0 else "N_0_kN"
        results[n_label] = {}
        
        for idx, theta in enumerate(bending_angles):
            angle_label = angle_labels[idx] if angle_labels else f"theta_{np.degrees(theta):.1f}_deg"
            
            ult_result = conc_sec.ultimate_bending_capacity(theta=theta, n=n)
            
            results[n_label][angle_label] = {
                'theta_rad': theta,
                'theta_deg': np.degrees(theta),
                'n': ult_result.n,
                'd_n': ult_result.d_n,
                'k_u': ult_result.k_u,
                'm_x': ult_result.m_x,
                'm_y': ult_result.m_y,
                'm_xy': ult_result.m_xy,
            }
    
    # Save to JSON if requested
    if save_json:
        save_to_json(results, save_json)
    
    return results


# ============================================================================
# 4. MOMENT CURVATURE ANALYSIS
# ============================================================================

def plot_stress_strain_profiles(
    materials,
    material_names=None,
    save_dir="output/stress_strain",
):
    """
    Plot stress-strain profiles for multiple materials.
    
    Parameters:
    -----------
    materials : list
        List of material objects (Concrete or SteelBar)
    material_names : list of str or None
        Names for each material (if None, uses material.name)
    save_dir : str
        Directory to save plots
    
    Returns:
    --------
    list of file paths where plots were saved
    """
    saved_files = []
    
    for idx, material in enumerate(materials):
        name = material_names[idx] if material_names else material.name
        safe_name = name.replace(" ", "_").replace("/", "_")
        save_path = f"{save_dir}/{safe_name}.png"
        
        ax = material.stress_strain_profile.plot_stress_strain(
            title=name,
            eng=True,
            render=False,
        )
        save_plot(ax, save_path)
        saved_files.append(save_path)
    
    return saved_files


def moment_curvature_analysis(
    conc_sec,
    theta=0,
    n=0,
    kappa_inc=1e-6,
    kappa_mult=2,
    kappa_inc_max=5e-6,
    delta_m_min=0.15,
    delta_m_max=0.3,
    save_json=None,
    save_plot_path=None,
):
    """
    Perform moment curvature analysis and optionally save results.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    theta : float
        Angle (in radians) the neutral axis makes with horizontal (default: 0)
    n : float
        Axial force (default: 0)
    kappa_inc : float
        Initial curvature increment (default: 1e-6)
    kappa_mult : float
        Multiplier for curvature increment (default: 2)
    kappa_inc_max : float
        Maximum curvature increment (default: 5e-6)
    delta_m_min : float
        Min relative change in moment (default: 0.15)
    delta_m_max : float
        Max relative change in moment (default: 0.3)
    save_json : str or None
        Path to save JSON results (None = don't save)
    save_plot_path : str or None
        Path to save plot (None = don't save)
    
    Returns:
    --------
    MomentCurvatureResults object
    """
    # Perform moment curvature analysis
    mk_results = conc_sec.moment_curvature_analysis(
        theta=theta,
        n=n,
        kappa_inc=kappa_inc,
        kappa_mult=kappa_mult,
        kappa_inc_max=kappa_inc_max,
        delta_m_min=delta_m_min,
        delta_m_max=delta_m_max,
        progress_bar=False,
    )
    
    # Save to JSON if requested
    if save_json:
        mk_dict = make_serializable(mk_results)
        if 'default_units' in mk_dict:
            del mk_dict['default_units']
        
        # Add summary statistics
        mk_dict['summary'] = {
            'number_of_calculations': len(mk_results.kappa),
            'failure_curvature': mk_results.kappa[-1] if mk_results.kappa else None,
            'max_moment': max(mk_results.m_xy) if mk_results.m_xy else None,
        }
        
        save_to_json(mk_dict, save_json)
    
    # Save plot if requested
    if save_plot_path:
        ax = mk_results.plot_results(render=False)
        save_plot(ax, save_plot_path)
    
    return mk_results


def multiple_moment_curvature_analysis(
    conc_sec,
    analysis_cases,
    save_json=None,
    save_plot_path=None,
    labels=None,
):
    """
    Perform multiple moment curvature analyses with different parameters.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    analysis_cases : list of dict
        List of dictionaries containing analysis parameters
        Each dict can have: theta, n, kappa_inc, kappa_mult, etc.
    save_json : str or None
        Path to save combined JSON results
    save_plot_path : str or None
        Path to save combined plot
    labels : list of str or None
        Labels for each analysis case
    
    Returns:
    --------
    list of MomentCurvatureResults objects
    
    Example:
    --------
    >>> cases = [
    ...     {"n": 0, "kappa_inc": 1e-6},
    ...     {"n": 1000e3, "kappa_inc": 1e-6},
    ...     {"n": -1000e3, "kappa_inc": 1e-6},
    ... ]
    >>> results = multiple_moment_curvature_analysis(
    ...     conc_sec, cases, labels=["N=0", "N=1000kN", "N=-1000kN"]
    ... )
    """
    from concreteproperties.results import MomentCurvatureResults
    
    results = []
    
    # Default parameters
    defaults = {
        "theta": 0,
        "n": 0,
        "kappa_inc": 1e-6,
        "kappa_mult": 2,
        "kappa_inc_max": 5e-6,
        "delta_m_min": 0.15,
        "delta_m_max": 0.3,
    }
    
    # Perform each analysis
    for case in analysis_cases:
        params = {**defaults, **case}  # Merge defaults with case-specific params
        
        mk_results = conc_sec.moment_curvature_analysis(
            theta=params["theta"],
            n=params["n"],
            kappa_inc=params["kappa_inc"],
            kappa_mult=params["kappa_mult"],
            kappa_inc_max=params["kappa_inc_max"],
            delta_m_min=params["delta_m_min"],
            delta_m_max=params["delta_m_max"],
            progress_bar=False,
        )
        results.append(mk_results)
    
    # Save combined JSON if requested
    if save_json:
        combined_dict = {}
        for idx, result in enumerate(results):
            label = labels[idx] if labels else f"case_{idx}"
            result_dict = make_serializable(result)
            if 'default_units' in result_dict:
                del result_dict['default_units']
            
            # Add summary
            result_dict['summary'] = {
                'number_of_calculations': len(result.kappa),
                'failure_curvature': result.kappa[-1] if result.kappa else None,
                'max_moment': max(result.m_xy) if result.m_xy else None,
            }
            
            combined_dict[label] = result_dict
        save_to_json(combined_dict, save_json)
    
    # Save combined plot if requested
    if save_plot_path:
        ax = MomentCurvatureResults.plot_multiple_results(
            moment_curvature_results=results,
            labels=labels,
            fmt="-",
            render=False,
        )
        save_plot(ax, save_plot_path)
    
    return results


def compare_moment_curvature_parameters(
    conc_sec,
    parameter_sets,
    save_json=None,
    save_plot_path=None,
    labels=None,
):
    """
    Compare moment-curvature results with different analysis parameters.
    
    Parameters:
    -----------
    conc_sec : ConcreteSection
        The concrete section object
    parameter_sets : list of dict
        List of parameter dictionaries to compare
    save_json : str or None
        Path to save comparison JSON
    save_plot_path : str or None
        Path to save comparison plot
    labels : list of str or None
        Labels for each parameter set
    
    Returns:
    --------
    list of MomentCurvatureResults and dict of statistics
    """
    from concreteproperties.results import MomentCurvatureResults
    
    results = []
    stats = {}
    
    for idx, params in enumerate(parameter_sets):
        label = labels[idx] if labels else f"params_{idx}"
        
        mk_result = conc_sec.moment_curvature_analysis(
            theta=params.get("theta", 0),
            n=params.get("n", 0),
            kappa_inc=params.get("kappa_inc", 1e-7),
            kappa_mult=params.get("kappa_mult", 2),
            kappa_inc_max=params.get("kappa_inc_max", 5e-6),
            delta_m_min=params.get("delta_m_min", 0.15),
            delta_m_max=params.get("delta_m_max", 0.3),
            progress_bar=False,
        )
        results.append(mk_result)
        
        stats[label] = {
            'number_of_calculations': len(mk_result.kappa),
            'failure_curvature': mk_result.kappa[-1],
            'max_moment': max(mk_result.m_xy),
            'parameters': params,
        }
    
    # Save statistics to JSON
    if save_json:
        save_to_json(stats, save_json)
    
    # Save comparison plot
    if save_plot_path:
        ax = MomentCurvatureResults.plot_multiple_results(
            moment_curvature_results=results,
            labels=labels,
            fmt="-",
            render=False,
        )
        save_plot(ax, save_plot_path)
    
    return results, stats


def plot_initial_cracking_region(
    mk_result,
    n_points=12,
    save_plot_path=None,
):
    """
    Plot the initial cracking region of moment-curvature diagram.
    
    Parameters:
    -----------
    mk_result : MomentCurvatureResults
        Moment curvature results object
    n_points : int
        Number of initial points to plot
    save_plot_path : str or None
        Path to save plot
    
    Returns:
    --------
    matplotlib axes object
    """
    fig, ax = plt.subplots()
    kappa = np.array(mk_result.kappa)
    moment = np.array(mk_result.m_xy) / 1e6
    
    ax.plot(kappa[:n_points], moment[:n_points], "x-")
    ax.set_xlabel("Curvature [-]")
    ax.set_ylabel("Bending Moment [kN.m]")
    ax.set_title("Initial Cracking Region")
    ax.grid(True, alpha=0.3)
    
    if save_plot_path:
        save_plot(ax, save_plot_path)
    
    return ax


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


import streamlit as st
import json
import os
import shutil
from pathlib import Path
import hashlib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Concrete Section Analyzer",
    page_icon="🏗️",
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'sections' not in st.session_state:
    st.session_state.sections = []
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = None

# File paths
USERS_DB = Path("users.json")
BASE_OUTPUT_DIR = Path("user_outputs")

# Utility functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if USERS_DB.exists():
        with open(USERS_DB, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_DB, 'w') as f:
        json.dump(users, f, indent=2)

def get_user_directory(username):
    user_dir = BASE_OUTPUT_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def clear_user_directory(username):
    user_dir = get_user_directory(username)
    if user_dir.exists():
        shutil.rmtree(user_dir)
        user_dir.mkdir(parents=True, exist_ok=True)
        return True
    return False

def get_file_icon(filename):
    ext = Path(filename).suffix.lower()
    icons = {
        '.png': '🖼️',
        '.jpg': '🖼️',
        '.jpeg': '🖼️',
        '.json': '📊',
        '.txt': '📄',
        '.pdf': '📑'
    }
    return icons.get(ext, '📁')

# Authentication
def show_auth():
    st.title("🏗️ Concrete Section Analyzer")
    st.write("Advanced structural analysis and visualization platform")
    
    st.divider()
    
    auth_mode = st.radio("Select Mode", ["Sign In", "Sign Up"], horizontal=True)
    
    username = st.text_input("Username", key="auth_username")
    password = st.text_input("Password", type="password", key="auth_password")
    
    if auth_mode == "Sign In":
        if st.button("🔐 Sign In", use_container_width=True):
            users = load_users()
            if username in users and users[username]['password'] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    else:
        confirm_password = st.text_input("Confirm Password", type="password", key="auth_confirm")
        
        if st.button("✨ Create Account", use_container_width=True):
            if not username or not password:
                st.error("❌ Please fill in all fields")
            elif password != confirm_password:
                st.error("❌ Passwords don't match")
            else:
                users = load_users()
                if username in users:
                    st.error("❌ Username already exists")
                else:
                    users[username] = {
                        'password': hash_password(password),
                        'created_at': datetime.now().isoformat()
                    }
                    save_users(users)
                    get_user_directory(username)
                    st.success("✅ Account created! Please sign in.")
    
    st.info("🔒 Your data is securely stored in isolated user directories")

def show_main_app():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"🏗️ Concrete Section Analyzer")
        st.write(f"Welcome, **{st.session_state.username}**")
    
    with col2:
        st.write("")  # Spacer
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.sections = []
            st.rerun()
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Section Configuration", "📊 Results & Downloads", "📚 User Guide"])
    
    with tab1:
        show_configuration_tab()
    
    with tab2:
        show_results_tab()
    
    with tab3:
        show_guide_tab()

def show_configuration_tab():
    st.subheader("🔧 Section Parameters")
    
    st.info("💡 **Quick Start:** Add your concrete section parameters below. You can add multiple sections, modify them, or delete them as needed.")
    
    st.write("#### Enter Section Code")
    

    template_options = {
        "Empty": "",
        "Simple Section": """# Simple section analysis...""",
        "Full Analysis": """# Full comprehensive analysis..."""
    }

    template_choice = st.selectbox("Choose Template", list(template_options.keys()))

    section_code = st.text_area(
        "Python Code",
        value=template_options[template_choice],
        height=400,
    )

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("➕ Add Section", use_container_width=True, type="primary"):
            if section_code.strip():
                section_data = {
                    'code': section_code,
                    'created_at': datetime.now().isoformat(),
                    'id': len(st.session_state.sections)
                }
                st.session_state.sections.append(section_data)
                st.success(f"✅ Section {len(st.session_state.sections)} added!")
                st.session_state.selected_section = len(st.session_state.sections) - 1
            else:
                st.error("❌ Please enter section code")
    
    with col2:
        if st.button("✏️ Modify Selected", use_container_width=True):
            if st.session_state.selected_section is not None and section_code.strip():
                st.session_state.sections[st.session_state.selected_section]['code'] = section_code
                st.session_state.sections[st.session_state.selected_section]['modified_at'] = datetime.now().isoformat()
                st.success(f"✅ Section {st.session_state.selected_section + 1} modified!")
            else:
                st.warning("⚠️ Select a section and enter code to modify")
    
    with col3:
        if st.button("🗑️ Delete Selected", use_container_width=True):
            if st.session_state.selected_section is not None:
                deleted_id = st.session_state.selected_section + 1
                del st.session_state.sections[st.session_state.selected_section]
                st.session_state.selected_section = None
                st.success(f"✅ Section {deleted_id} deleted!")
                st.rerun()
            else:
                st.warning("⚠️ Select a section to delete")
    
    with col4:
        if st.button("▶️ Run Analysis", use_container_width=True, type="primary"):
            if st.session_state.sections:
                run_analysis()
            else:
                st.warning("⚠️ Add at least one section")
    
    st.divider()
    
    # Display sections
    if st.session_state.sections:
        st.write("#### 📋 Existing Sections")
        
        for idx, section in enumerate(st.session_state.sections):
            is_selected = st.session_state.selected_section == idx
            
            with st.expander(
                f"{'✨ ' if is_selected else ''}Section {idx + 1}" + 
                (' [SELECTED]' if is_selected else ''),
                expanded=is_selected
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.code(section['code'], language='python')
                    st.caption(f"Created: {section['created_at'][:19]}")
                    if 'modified_at' in section:
                        st.caption(f"Modified: {section['modified_at'][:19]}")
                
                with col2:
                    if st.button(f"📌 Select", key=f"select_{idx}", use_container_width=True):
                        st.session_state.selected_section = idx
                        st.rerun()
    else:
        st.warning("⚠️ **No sections added yet.** Add your first section using the code editor above.")

def run_analysis():
    user_dir = get_user_directory(st.session_state.username)
    
    # Copy helper functions module to user directory
    import shutil
    shutil.copy('concrete_analysis.py', user_dir / 'concrete_analysis.py')
    
    # Create analysis script
    analysis_file = user_dir / "analysis_script.py"
    
    full_code = """import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np

from concrete_analysis import *  # Import all helper functions

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('output', exist_ok=True)

"""
    
    # Add user sections
    for idx, section in enumerate(st.session_state.sections):
        full_code += f"\n# ========== Section {idx + 1} ==========\n"
        full_code += section['code']
        full_code += "\n"
    
    # Write the script
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write(full_code)
    
    with st.spinner("🔄 Running analysis... This may take several minutes."):
        try:
            import subprocess
            import sys
            
            # Run the analysis script using absolute path
            result = subprocess.run(
                [sys.executable, str(analysis_file.absolute())],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                st.success("✅ Analysis completed successfully!")
                st.balloons()
                
                # Show output if available
                if result.stdout:
                    with st.expander("📋 Analysis Output"):
                        st.text(result.stdout)
                
                st.info("Check the 'Results & Downloads' tab to view and download your results.")
            else:
                st.error("❌ Error during analysis")
                
                if result.stderr:
                    with st.expander("🔍 Error Details", expanded=True):
                        st.code(result.stderr, language="text")
                
                if result.stdout:
                    with st.expander("📋 Partial Output"):
                        st.text(result.stdout)
                        
        except subprocess.TimeoutExpired:
            st.error("❌ Analysis timeout (exceeded 10 minutes)")
            st.info("Consider reducing the number of analysis points or sections.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

def show_results_tab():
    st.subheader("📊 Analysis Results")
    
    user_dir = get_user_directory(st.session_state.username)
    
    # Actions
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.info(f"📁 **Directory:** `{user_dir}`")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Directory", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                clear_user_directory(st.session_state.username)
                st.session_state.confirm_clear = False
                st.success("✅ Directory cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm")
    
    st.divider()
    
    # Get files
    image_files = []
    json_files = []
    
    if user_dir.exists():
        for file in user_dir.rglob('*'):
            if file.is_file():
                ext = file.suffix.lower()
                if ext in ['.png', '.jpg', '.jpeg']:
                    image_files.append(file)
                elif ext == '.json':
                    json_files.append(file)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Images", len(image_files))
    with col2:
        st.metric("JSON Files", len(json_files))
    with col3:
        st.metric("Total Files", len(image_files) + len(json_files))
    
    st.divider()
    
    # Display images
    if image_files:
        st.write("#### 🖼️ Images")
        
        cols = st.columns(3)
        for idx, img_file in enumerate(sorted(image_files)):
            with cols[idx % 3]:
                st.write(f"**{img_file.name}**")
                st.image(str(img_file), use_container_width=True)
                
                with open(img_file, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download",
                        data=f,
                        file_name=img_file.name,
                        mime="image/png",
                        key=f"img_{idx}"
                    )
    
    # Display JSON files
    if json_files:
        st.write("#### 📊 JSON Files")
        
        for idx, json_file in enumerate(sorted(json_files)):
            with st.expander(f"{get_file_icon(json_file.name)} {json_file.name}"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    st.json(data)
                    
                    with open(json_file, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download",
                            data=f,
                            file_name=json_file.name,
                            mime="application/json",
                            key=f"json_{idx}"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if not (image_files or json_files):
        st.info("📭 **No files yet.** Run an analysis to generate results.")

def show_guide_tab():
    st.markdown("### 📚 User Guide")
    
    st.markdown("""
    ## 🎯 Overview
    
    The Concrete Section Analyzer helps you analyze reinforced concrete sections using the `concreteproperties` library.
    
    **Features:**
    - Define custom concrete sections
    - Perform moment-curvature analysis
    - Calculate ultimate bending capacities
    - Generate interaction diagrams
    - Analyze stress distributions
    
    ---
    
    ## 📦 Installation Requirements
    
    This application requires the following Python packages to be installed:
    
    ```bash
    pip install concreteproperties
    pip install sectionproperties
    pip install numpy
    pip install matplotlib
    ```
    
    **Note:** Make sure these packages are installed in the same Python environment where you're running Streamlit.
    
    ---
    
    ## 🚀 Getting Started
    
    ### 1. Verify Installation
    
    Before running analysis, ensure the required packages are installed. You can test by running:
    
    ```python
    import concreteproperties
    import sectionproperties
    print("Libraries installed successfully!")
    ```
    
    ### 2. Define Your Section
    
    Go to **Section Configuration** tab and enter your code. You need to import the analysis functions first:
    
    ```python
    import numpy as np
    from concreteproperties import (
        Concrete,
        ConcreteLinear,
        ConcreteSection,
        RectangularStressBlock,
        SteelBar,
        SteelElasticPlastic,
    )
    from sectionproperties.pre.library import concrete_rectangular_section
    
    # Define materials
    concrete = Concrete(
        name="32 MPa Concrete",
        density=2.4e-6,
        stress_strain_profile=ConcreteLinear(elastic_modulus=30.1e3),
        ultimate_stress_strain_profile=RectangularStressBlock(
            compressive_strength=32,
            alpha=0.802,
            gamma=0.89,
            ultimate_strain=0.003,
        ),
        flexural_tensile_strength=3.4,
        colour="lightgrey",
    )
    
    steel = SteelBar(
        name="500 MPa Steel",
        density=7.85e-6,
        stress_strain_profile=SteelElasticPlastic(
            yield_strength=500,
            elastic_modulus=200e3,
            fracture_strain=0.05,
        ),
        colour="grey",
    )
    
    # Create section geometry
    geom = concrete_rectangular_section(
        d=600,           # depth in mm
        b=400,           # width in mm
        dia_top=20,      # top bar diameter
        area_top=310,    # top bar area in mm²
        n_top=3,         # number of top bars
        c_top=30,        # top cover in mm
        dia_bot=24,      # bottom bar diameter
        area_bot=450,    # bottom bar area in mm²
        n_bot=3,         # number of bottom bars
        c_bot=30,        # bottom cover in mm
        conc_mat=concrete,
        steel_mat=steel,
    )
    
    # Create concrete section
    conc_sec = ConcreteSection(geom)
    
    # Plot section
    ax = conc_sec.plot_section(title="RC Section", render=False)
    ax.get_figure().savefig('output/section.png', bbox_inches='tight', dpi=150)
    import matplotlib.pyplot as plt
    plt.close()
    ```
    
    ### 3. Add Analysis
    
    You can add various analyses to your section:
    
    ```python
    # Moment-curvature analysis
    mk_results = conc_sec.moment_curvature_analysis(
        theta=0,
        n=0,
        kappa_inc=1e-6,
        progress_bar=False,
    )
    
    # Plot results
    ax = mk_results.plot_results(render=False)
    ax.get_figure().savefig('output/moment_curvature.png', bbox_inches='tight')
    plt.close()
    
    # Ultimate bending capacity
    ult_results = conc_sec.ultimate_bending_capacity(theta=0, n=0)
    print(f"Ultimate moment: {ult_results.m_xy / 1e6:.2f} kN.m")
    
    # Moment interaction diagram
    mi_results = conc_sec.moment_interaction_diagram(
        theta=0,
        progress_bar=False,
    )
    
    ax = mi_results.plot_diagram(render=False)
    ax.get_figure().savefig('output/interaction_diagram.png', bbox_inches='tight')
    plt.close()
    ```
    
    ### 4. Run Analysis
    
    Click **"Run Analysis"** to process your sections.
    
    ### 5. View Results
    
    Go to **Results & Downloads** tab to see generated files.
    
    ---
    
    ## 📖 Key Concepts
    
    ### Moment-Curvature Analysis
    Shows the relationship between moment and curvature from elastic through cracking to ultimate capacity.
    
    **Parameters:**
    - `theta`: Bending angle (0 for sagging, π for hogging)
    - `n`: Axial force (positive = compression)
    - `kappa_inc`: Curvature increment (smaller = more accurate)
    - `progress_bar`: Set to False in streamlit
    
    ### Ultimate Bending Capacity
    Maximum moment before failure, considering:
    - Material strengths
    - Reinforcement layout
    - Axial load effects
    
    ### Interaction Diagrams
    Show relationship between axial load and moment capacity for combined loading.
    
    ### Important: Matplotlib Backend
    
    Always set matplotlib to non-interactive mode at the start:
    
    ```python
    import matplotlib
    matplotlib.use('Agg')
    ```
    
    And always use `render=False` when plotting and close figures:
    
    ```python
    ax = conc_sec.plot_section(render=False)
    ax.get_figure().savefig('output/plot.png')
    import matplotlib.pyplot as plt
    plt.close()
    ```
    
    ---
    
    ## 💡 Example: Complete Analysis
    
    ```python
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    from concreteproperties import (
        Concrete, ConcreteLinear, ConcreteSection,
        RectangularStressBlock, SteelBar, SteelElasticPlastic
    )
    from sectionproperties.pre.library import concrete_rectangular_section
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Materials
    concrete = Concrete(
        name="40 MPa Concrete",
        density=2.4e-6,
        stress_strain_profile=ConcreteLinear(elastic_modulus=32.8e3),
        ultimate_stress_strain_profile=RectangularStressBlock(
            compressive_strength=40, alpha=0.79, gamma=0.87, ultimate_strain=0.003
        ),
        flexural_tensile_strength=3.8,
        colour="lightgrey",
    )
    
    steel = SteelBar(
        name="500 MPa Steel",
        density=7.85e-6,
        stress_strain_profile=SteelElasticPlastic(
            yield_strength=500, elastic_modulus=200e3, fracture_strain=0.05
        ),
        colour="grey",
    )
    
    # Create section
    geom = concrete_rectangular_section(
        d=600, b=400,
        dia_top=20, area_top=310, n_top=3, c_top=30,
        dia_bot=24, area_bot=450, n_bot=3, c_bot=30,
        conc_mat=concrete, steel_mat=steel,
    )
    
    conc_sec = ConcreteSection(geom)
    
    # Section plot
    ax = conc_sec.plot_section(title="RC Beam Section", render=False)
    ax.get_figure().savefig('output/section.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Moment-curvature
    mk = conc_sec.moment_curvature_analysis(theta=0, n=0, progress_bar=False)
    ax = mk.plot_results(render=False)
    ax.get_figure().savefig('output/mk_curve.png', bbox_inches='tight')
    plt.close()
    
    # Ultimate capacity
    ult = conc_sec.ultimate_bending_capacity(theta=0, n=0)
    print(f"Ultimate: {ult.m_xy/1e6:.2f} kN.m, d_n={ult.d_n:.2f}mm")
    
    # Interaction diagram
    mi = conc_sec.moment_interaction_diagram(theta=0, progress_bar=False)
    ax = mi.plot_diagram(render=False)
    ax.get_figure().savefig('output/interaction.png', bbox_inches='tight')
    plt.close()
    
    print("Analysis complete!")
    ```
    
    ---
    
    ## 🔧 Common Issues
    
    ### Import Errors
    If you see `ModuleNotFoundError: No module named 'concreteproperties'`:
    ```bash
    pip install concreteproperties sectionproperties
    ```
    
    ### Unicode Errors
    Already handled by the app - UTF-8 encoding is used.
    
    ### Matplotlib Errors
    Always use:
    ```python
    import matplotlib
    matplotlib.use('Agg')
    ```
    at the very beginning of your code.
    
    ### Progress Bar Errors
    Always set `progress_bar=False` in analysis functions when using Streamlit.
    
    ### Plot Not Showing
    Remember to:
    1. Use `render=False` in all plot functions
    2. Save the figure: `ax.get_figure().savefig(...)`
    3. Close the figure: `plt.close()`
    
    ---
    
    ## 📞 Support
    
    For concreteproperties documentation, visit:
    - GitHub: https://github.com/robbievanleeuwen/concrete-properties
    - Docs: https://concrete-properties.readthedocs.io/
    
    For issues with this Streamlit app, check that:
    - All required packages are installed
    - Code uses `render=False` and `progress_bar=False`
    - Matplotlib backend is set to 'Agg'
    - Output directory exists and is writable
    """)
    
    st.divider()
    st.caption("🏗️ Concrete Section Analyzer | Built for Structural Engineers")
    st.caption("Powered by concreteproperties library")

def main():
    if not st.session_state.authenticated:
        show_auth()
    else:
        show_main_app()

if __name__ == "__main__":
    main()





