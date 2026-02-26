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
if 'shell_sections' not in st.session_state:
    st.session_state.shell_sections = []
if 'selected_shell_section' not in st.session_state:
    st.session_state.selected_shell_section = None

# File paths
USERS_DB = Path("users.json")
BASE_OUTPUT_DIR = Path("user_outputs")

# ─── Utility functions ────────────────────────────────────────────────────────

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

def get_shell_user_directory(username):
    user_dir = BASE_OUTPUT_DIR / username / "shell_outputs"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def clear_user_directory(username):
    user_dir = get_user_directory(username)
    if user_dir.exists():
        shutil.rmtree(user_dir)
        user_dir.mkdir(parents=True, exist_ok=True)
        return True
    return False

def clear_shell_user_directory(username):
    user_dir = get_shell_user_directory(username)
    if user_dir.exists():
        shutil.rmtree(user_dir)
        user_dir.mkdir(parents=True, exist_ok=True)
        return True
    return False

def get_file_icon(filename):
    ext = Path(filename).suffix.lower()
    icons = {'.png': '🖼️', '.jpg': '🖼️', '.jpeg': '🖼️', '.json': '📊', '.txt': '📄', '.pdf': '📑', '.html': '🌐'}
    return icons.get(ext, '📁')

# ─── Example templates ────────────────────────────────────────────────────────

MEMBER_TEMPLATE_EMPTY = ""

MEMBER_TEMPLATE_SIMPLE = """\
import numpy as np

# =============================================================================
# MATERIALS
# =============================================================================
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

# =============================================================================
# SECTION ARGS
# =============================================================================
section_args = dict(
    base_geometry_type="rectangular",
    base_geometry_params={"d": 500, "b": 300},
    base_material=concrete,
    reinforcement=[{
        "type": "rectangular_array",
        "material": steel,
        "params": {
            "area": 314,
            "n_x": 3,
            "x_s": 100,
            "n_y": 2,
            "y_s": 420,
            "anchor": (40, 40),
        },
    }],
)

# =============================================================================
# 1. GROSS PROPERTIES
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_gross_properties=True,
    elastic_modulus=30.1e3,
    save_dir="output/01_gross_properties",
)
"""

MEMBER_TEMPLATE_FULL = """\
import numpy as np

# =============================================================================
# MATERIALS
# =============================================================================
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

section_args = dict(
    base_geometry_type="rectangular",
    base_geometry_params={"d": 500, "b": 300},
    base_material=concrete,
    reinforcement=[{
        "type": "rectangular_array",
        "material": steel,
        "params": {
            "area": 314,
            "n_x": 3,
            "x_s": 100,
            "n_y": 2,
            "y_s": 420,
            "anchor": (40, 40),
        },
    }],
)

# =============================================================================
# 1. GROSS PROPERTIES
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_gross_properties=True,
    elastic_modulus=30.1e3,
    save_dir="output/01_gross_properties",
)

# =============================================================================
# 2. CRACKED PROPERTIES
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_cracked_properties=True,
    elastic_modulus=30.1e3,
    theta_sag=0,
    theta_hog=np.pi,
    save_dir="output/02_cracked_properties",
)

# =============================================================================
# 3. MOMENT CURVATURE
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_moment_curvature=True,
    mk_theta=0,
    mk_n=0,
    mk_kappa_inc=1e-6,
    mk_kappa_mult=2,
    mk_kappa_inc_max=5e-6,
    mk_delta_m_min=0.15,
    mk_delta_m_max=0.3,
    save_dir="output/03_moment_curvature",
)

# =============================================================================
# 4. MULTIPLE MOMENT CURVATURE
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_multiple_moment_curvature=True,
    multiple_mk_cases=[
        {"n": 0,       "kappa_inc": 1e-6},
        {"n": 500e3,   "kappa_inc": 1e-6},
        {"n": -500e3,  "kappa_inc": 1e-6},
    ],
    multiple_mk_labels=["N=0", "N=500kN", "N=-500kN"],
    save_dir="output/04_moment_curvature_multiple",
)

# =============================================================================
# 5. ULTIMATE BENDING
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_ultimate_bending=True,
    ult_theta=0,
    ult_n=0,
    save_dir="output/05_ultimate_bending",
)

# =============================================================================
# 6. MULTIPLE ULTIMATE BENDING
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_multiple_ultimate_bending=True,
    multiple_ult_cases=[
        {"theta": 0,        "n": 0},
        {"theta": np.pi,    "n": 0},
        {"theta": np.pi/2,  "n": 0},
    ],
    multiple_ult_labels=["Sagging", "Hogging", "Weak Axis"],
    save_dir="output/06_ultimate_bending_multiple",
)

# =============================================================================
# 7. MOMENT INTERACTION
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_moment_interaction=True,
    mi_theta=0,
    mi_n_points=24,
    save_dir="output/07_moment_interaction",
)

# =============================================================================
# 8. POSITIVE & NEGATIVE INTERACTION
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_positive_negative_interaction=True,
    save_dir="output/08_moment_interaction_pos_neg",
)

# =============================================================================
# 9. ADVANCED INTERACTION
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_advanced_interaction=True,
    mi_theta=0,
    save_dir="output/09_moment_interaction_advanced",
)

# =============================================================================
# 10. MULTIPLE INTERACTION
# =============================================================================
sec1, _ = create_dynamic_concrete_section(
    base_geometry_type="rectangular",
    base_geometry_params={"d": 500, "b": 300},
    base_material=concrete,
    reinforcement=[{
        "type": "rectangular_array",
        "material": steel,
        "params": {"area": 314, "n_x": 3, "x_s": 100, "n_y": 2, "y_s": 420, "anchor": (40, 40)},
    }],
)
sec2, _ = create_dynamic_concrete_section(
    base_geometry_type="rectangular",
    base_geometry_params={"d": 600, "b": 350},
    base_material=concrete,
    reinforcement=[{
        "type": "rectangular_array",
        "material": steel,
        "params": {"area": 314, "n_x": 4, "x_s": 90, "n_y": 2, "y_s": 520, "anchor": (40, 40)},
    }],
)
output = run_concrete_analysis(
    **section_args,
    run_multiple_interaction=True,
    multiple_mi_sections=[sec1, sec2],
    multiple_mi_labels=["300x500", "350x600"],
    save_dir="output/10_moment_interaction_multiple",
)

# =============================================================================
# 11. BIAXIAL BENDING
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_biaxial_bending=True,
    bb_n=0,
    bb_n_points=48,
    save_dir="output/11_biaxial_bending",
)

# =============================================================================
# 12. MULTIPLE BIAXIAL
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_multiple_biaxial=True,
    bb_axial_forces=[0, 500e3, 1000e3, 2000e3],
    bb_multiple_n_points=24,
    bb_labels=["N=0", "N=500kN", "N=1000kN", "N=2000kN"],
    save_dir="output/12_biaxial_multiple",
)

# =============================================================================
# 13. BIAXIAL SUITE
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_biaxial_suite=True,
    bb_suite_axial_forces=[0, 500e3, 1000e3],
    bb_suite_n_points=24,
    save_dir="output/13_biaxial_suite",
)

# =============================================================================
# 14. UNCRACKED STRESS
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_uncracked_stress=True,
    stress_n=0,
    stress_m_x=50e6,
    stress_m_y=0,
    save_dir="output/14_stress_uncracked",
)

# =============================================================================
# 15. CRACKED STRESS
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_cracked_properties=True,
    run_cracked_stress=True,
    elastic_modulus=30.1e3,
    theta_sag=0,
    theta_hog=np.pi,
    cracked_stress_m=100e6,
    save_dir="output/15_stress_cracked",
)

# =============================================================================
# 16. SERVICE STRESS
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_moment_curvature=True,
    run_service_stress=True,
    mk_theta=0,
    mk_n=0,
    mk_kappa_inc=1e-6,
    mk_kappa_mult=2,
    mk_kappa_inc_max=5e-6,
    mk_delta_m_min=0.15,
    mk_delta_m_max=0.3,
    service_stress_m=100e6,
    save_dir="output/16_stress_service",
)

# =============================================================================
# 17. ULTIMATE STRESS
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_ultimate_bending=True,
    run_ultimate_stress=True,
    ult_theta=0,
    ult_n=0,
    save_dir="output/17_stress_ultimate",
)

# =============================================================================
# 18. MULTIPLE SERVICE STRESS
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_moment_curvature=True,
    run_multiple_service_stress=True,
    mk_theta=0,
    mk_n=0,
    mk_kappa_inc=1e-6,
    mk_kappa_mult=2,
    mk_kappa_inc_max=5e-6,
    mk_delta_m_min=0.15,
    mk_delta_m_max=0.3,
    multiple_service_stress_points=[
        {"m_or_k": "m", "val": 50e6},
        {"m_or_k": "m", "val": 100e6},
        {"m_or_k": "k", "val": 1e-5},
    ],
    save_dir="output/18_stress_service_multiple",
)

# =============================================================================
# 19. MULTIPLE ULTIMATE STRESS
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_multiple_ultimate_stress=True,
    multiple_ultimate_stress_cases=[
        {"theta": 0,     "n": 0},
        {"theta": np.pi, "n": 0},
        {"theta": 0,     "n": 1000e3},
    ],
    multiple_ultimate_stress_labels=["Sagging", "Hogging", "Axial+Sagging"],
    save_dir="output/19_stress_ultimate_multiple",
)

# =============================================================================
# 20. STRESS STRAIN PROFILES
# =============================================================================
output = run_concrete_analysis(
    **section_args,
    run_stress_strain_profiles=True,
    stress_strain_materials=[concrete, steel],
    stress_strain_names=["32 MPa Concrete", "500 MPa Steel"],
    save_dir="output/20_stress_strain_profiles",
)
"""

# ─── Shell design templates ───────────────────────────────────────────────────

SHELL_TEMPLATE_EMPTY = ""

SHELL_TEMPLATE_SLAB = """\
import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# USER INPUTS — Slab Example
# =============================================================================
ELEMENT_TYPE = "slab"
OUT_DIR      = "output/shell_slab"

FC_MPA       = 32.0     # concrete f'c [MPa]
FY_MPA       = 500.0    # steel fy [MPa]
H_MM         = 250.0    # section thickness [mm]
COVER_MM     = 30.0     # clear cover [mm]
DB_MM        = 16.0     # bar diameter [mm]
HW_MM        = 3000.0   # storey height for slenderness [mm]
LW_MM        = 6000.0   # wall/slab span [mm]
PHI_F        = 0.90     # flexure phi
PHI_V        = 0.75     # shear phi
PHI_C        = 0.65     # compression phi
RHO_MIN      = None     # None = ACI auto
B_MM         = 1000.0   # strip width [mm]
PREFERRED_DB = None     # None = auto select

# Section forces [kN/m, kN·m/m]
FORCES_JSON = json.dumps({
    "MXX": [50, 45, 30, 20, 35],
    "MYY": [40, 35, 25, 15, 28],
    "MXY": [10, -8,  5, -3,  6],
    "FXX": [20, 18, 12,  8, 14],
    "FYY": [15, 13, 10,  6, 11],
    "FXY": [ 5, -4,  3, -2,  4],
    "VXZ": [30, 25, 20, 12, 22],
})

STRESS_STRAIN_JSON = json.dumps({
    "sigma22": [5.2, 4.8, 3.1, 2.0, 4.0],
    "eps_max": [0.0012, 0.0010, 0.0008, 0.0005, 0.0009],
})

COL_REACTIONS_JSON = json.dumps({
    "1":  {"Vz_kN": 450, "Mx_kNm": 30, "My_kNm": 20},
    "5":  {"Vz_kN": 430, "Mx_kNm": 25, "My_kNm": 18},
    "21": {"Vz_kN": 470, "Mx_kNm": 35, "My_kNm": 22},
    "25": {"Vz_kN": 440, "Mx_kNm": 28, "My_kNm": 19},
})

COL_PUNCHING_JSON = json.dumps({
    "1":  {"col_width_in": 24, "col_depth_in": 24, "condition": "NW", "overhang_x": 6, "overhang_y": 6},
    "5":  {"col_width_in": 24, "col_depth_in": 24, "condition": "NE", "overhang_x": 6, "overhang_y": 6},
    "21": {"col_width_in": 24, "col_depth_in": 24, "condition": "SW", "overhang_x": 6, "overhang_y": 6},
    "25": {"col_width_in": 24, "col_depth_in": 24, "condition": "SE", "overhang_x": 6, "overhang_y": 6},
})

os.makedirs(OUT_DIR, exist_ok=True)

forces        = json.loads(FORCES_JSON)
ss            = json.loads(STRESS_STRAIN_JSON)
col_reactions = json.loads(COL_REACTIONS_JSON)
col_punch     = json.loads(COL_PUNCHING_JSON)
N             = len(forces["MXX"])

report, punch, detailing, sw = run_engine(
    forces=forces, ss=ss, col_reactions=col_reactions, col_punch=col_punch,
    element_type=ELEMENT_TYPE, fc_mpa=FC_MPA, fy_mpa=FY_MPA, h_mm=H_MM,
    cover_mm=COVER_MM, db_mm=DB_MM, hw_mm=HW_MM, lw_mm=LW_MM,
    phi_f=PHI_F, phi_v=PHI_V, phi_c=PHI_C, rho_min_user=RHO_MIN, b_mm=B_MM)

schedule = build_schedule(detailing, H_MM, COVER_MM, PREFERRED_DB)

plot_contours(detailing, N)

schedule_out = {k: {kk: vv for kk, vv in v.items() if kk != "description"}
                for k, v in schedule.items()}
with open(os.path.join(OUT_DIR, "rebar_schedule.json"), "w") as f:
    json.dump(schedule_out, f, indent=2)

write_html(report, punch, detailing, sw, schedule,
           {"element_type": ELEMENT_TYPE, "fc_mpa": FC_MPA, "fy_mpa": FY_MPA,
            "h_mm": H_MM, "cover_mm": COVER_MM, "db_mm": DB_MM,
            "phi_f": PHI_F, "phi_v": PHI_V, "phi_c": PHI_C, "N": N},
           OUT_DIR)

print(f"Shell slab analysis complete -> {OUT_DIR}")
"""

SHELL_TEMPLATE_WALL = """\
import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# USER INPUTS — Shear Wall Example
# =============================================================================
ELEMENT_TYPE = "shear_wall"
OUT_DIR      = "output/shell_wall"

FC_MPA       = 40.0
FY_MPA       = 500.0
H_MM         = 300.0    # wall thickness [mm]
COVER_MM     = 40.0
DB_MM        = 20.0
HW_MM        = 4000.0   # storey height [mm]
LW_MM        = 8000.0   # wall length [mm]
PHI_F        = 0.90
PHI_V        = 0.75
PHI_C        = 0.65
RHO_MIN      = 0.0025   # ACI 18.10.2.1
B_MM         = 1000.0
PREFERRED_DB = 20

# In-plane shear wall forces
FORCES_JSON = json.dumps({
    "MXX": [120, 110, 85, 60, 95],
    "MYY": [80,  70,  55, 40, 65],
    "MXY": [30,  -25, 15, -10, 20],
    "FXX": [40,  35,  25, 15, 30],
    "FYY": [30,  25,  18, 12, 22],
    "FXY": [15,  -12, 8,  -5, 10],
    "VXZ": [80,  70,  55, 35, 60],
})

STRESS_STRAIN_JSON = json.dumps({
    "sigma22": [7.5, 6.8, 4.5, 3.0, 5.5],
    "eps_max": [0.0015, 0.0013, 0.0009, 0.0006, 0.0011],
})

# No punching check for shear walls
COL_REACTIONS_JSON = json.dumps({})
COL_PUNCHING_JSON  = json.dumps({})

os.makedirs(OUT_DIR, exist_ok=True)

forces        = json.loads(FORCES_JSON)
ss            = json.loads(STRESS_STRAIN_JSON)
col_reactions = json.loads(COL_REACTIONS_JSON)
col_punch     = json.loads(COL_PUNCHING_JSON)
N             = len(forces["MXX"])

report, punch, detailing, sw = run_engine(
    forces=forces, ss=ss, col_reactions=col_reactions, col_punch=col_punch,
    element_type=ELEMENT_TYPE, fc_mpa=FC_MPA, fy_mpa=FY_MPA, h_mm=H_MM,
    cover_mm=COVER_MM, db_mm=DB_MM, hw_mm=HW_MM, lw_mm=LW_MM,
    phi_f=PHI_F, phi_v=PHI_V, phi_c=PHI_C, rho_min_user=RHO_MIN, b_mm=B_MM)

schedule = build_schedule(detailing, H_MM, COVER_MM, PREFERRED_DB)

plot_contours(detailing, N)

schedule_out = {k: {kk: vv for kk, vv in v.items() if kk != "description"}
                for k, v in schedule.items()}
with open(os.path.join(OUT_DIR, "rebar_schedule.json"), "w") as f:
    json.dump(schedule_out, f, indent=2)

write_html(report, punch, detailing, sw, schedule,
           {"element_type": ELEMENT_TYPE, "fc_mpa": FC_MPA, "fy_mpa": FY_MPA,
            "h_mm": H_MM, "cover_mm": COVER_MM, "db_mm": DB_MM,
            "phi_f": PHI_F, "phi_v": PHI_V, "phi_c": PHI_C, "N": N},
           OUT_DIR)

print(f"Shear wall analysis complete -> {OUT_DIR}")
"""

SHELL_TEMPLATE_FOOTING = """\
import numpy as np
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# USER INPUTS — Footing Example
# =============================================================================
ELEMENT_TYPE = "footing"
OUT_DIR      = "output/shell_footing"

FC_MPA       = 25.0
FY_MPA       = 500.0
H_MM         = 600.0    # footing thickness [mm]
COVER_MM     = 75.0     # bottom cover [mm]
DB_MM        = 20.0
HW_MM        = 600.0
LW_MM        = 5000.0   # footing plan dimension [mm]
PHI_F        = 0.90
PHI_V        = 0.75
PHI_C        = 0.65
RHO_MIN      = None     # ACI auto → 0.0018 for h < 900mm
B_MM         = 1000.0
PREFERRED_DB = 20

FORCES_JSON = json.dumps({
    "MXX": [180, 160, 120, 80, 140],
    "MYY": [150, 130, 100, 65, 115],
    "MXY": [20,  -15, 10,  -8, 15],
    "FXX": [25,  20,  15,  10, 18],
    "FYY": [20,  16,  12,  8,  14],
    "FXY": [8,   -6,  4,   -3, 6],
    "VXZ": [120, 105, 80,  50, 90],
})

STRESS_STRAIN_JSON = json.dumps({
    "sigma22": [4.5, 4.0, 2.8, 1.8, 3.2],
    "eps_max": [0.0010, 0.0009, 0.0006, 0.0004, 0.0007],
})

COL_REACTIONS_JSON = json.dumps({
    "1":  {"Vz_kN": 1200, "Mx_kNm": 80,  "My_kNm": 60},
    "5":  {"Vz_kN": 1150, "Mx_kNm": 70,  "My_kNm": 55},
    "21": {"Vz_kN": 1250, "Mx_kNm": 90,  "My_kNm": 65},
    "25": {"Vz_kN": 1180, "Mx_kNm": 75,  "My_kNm": 58},
})

COL_PUNCHING_JSON = json.dumps({
    "1":  {"col_width_in": 20, "col_depth_in": 20, "condition": "NW", "overhang_x": 8, "overhang_y": 8},
    "5":  {"col_width_in": 20, "col_depth_in": 20, "condition": "NE", "overhang_x": 8, "overhang_y": 8},
    "21": {"col_width_in": 20, "col_depth_in": 20, "condition": "SW", "overhang_x": 8, "overhang_y": 8},
    "25": {"col_width_in": 20, "col_depth_in": 20, "condition": "SE", "overhang_x": 8, "overhang_y": 8},
})

os.makedirs(OUT_DIR, exist_ok=True)

forces        = json.loads(FORCES_JSON)
ss            = json.loads(STRESS_STRAIN_JSON)
col_reactions = json.loads(COL_REACTIONS_JSON)
col_punch     = json.loads(COL_PUNCHING_JSON)
N             = len(forces["MXX"])

report, punch, detailing, sw = run_engine(
    forces=forces, ss=ss, col_reactions=col_reactions, col_punch=col_punch,
    element_type=ELEMENT_TYPE, fc_mpa=FC_MPA, fy_mpa=FY_MPA, h_mm=H_MM,
    cover_mm=COVER_MM, db_mm=DB_MM, hw_mm=HW_MM, lw_mm=LW_MM,
    phi_f=PHI_F, phi_v=PHI_V, phi_c=PHI_C, rho_min_user=RHO_MIN, b_mm=B_MM)

schedule = build_schedule(detailing, H_MM, COVER_MM, PREFERRED_DB)

plot_contours(detailing, N)

schedule_out = {k: {kk: vv for kk, vv in v.items() if kk != "description"}
                for k, v in schedule.items()}
with open(os.path.join(OUT_DIR, "rebar_schedule.json"), "w") as f:
    json.dump(schedule_out, f, indent=2)

write_html(report, punch, detailing, sw, schedule,
           {"element_type": ELEMENT_TYPE, "fc_mpa": FC_MPA, "fy_mpa": FY_MPA,
            "h_mm": H_MM, "cover_mm": COVER_MM, "db_mm": DB_MM,
            "phi_f": PHI_F, "phi_v": PHI_V, "phi_c": PHI_C, "N": N},
           OUT_DIR)

print(f"Footing analysis complete -> {OUT_DIR}")
"""

# ─── Authentication ───────────────────────────────────────────────────────────

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
                    users[username] = {'password': hash_password(password), 'created_at': datetime.now().isoformat()}
                    save_users(users)
                    get_user_directory(username)
                    st.success("✅ Account created! Please sign in.")
    st.info("🔒 Your data is securely stored in isolated user directories")

# ─── Main app ─────────────────────────────────────────────────────────────────

def show_main_app():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏗️ Concrete Section Analyzer")
        st.write(f"Welcome, **{st.session_state.username}**")
    with col2:
        st.write("")
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.sections = []
            st.session_state.shell_sections = []
            st.rerun()
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Section Configuration",
        "📊 Results & Downloads",
        "🧱 Shell Design",
        "📚 User Guide",
    ])
    with tab1:
        show_configuration_tab()
    with tab2:
        show_results_tab()
    with tab3:
        show_shell_design_tab()
    with tab4:
        show_guide_tab()

# ─── Member design tab ────────────────────────────────────────────────────────

def show_configuration_tab():
    st.subheader("🔧 Section Parameters")
    st.info("💡 **Quick Start:** Add your concrete section parameters below. You can add multiple sections, modify them, or delete them as needed.")
    st.write("#### Enter Section Code")

    template_options = {
        "Empty": MEMBER_TEMPLATE_EMPTY,
        "Simple Section": MEMBER_TEMPLATE_SIMPLE,
        "Full Analysis": MEMBER_TEMPLATE_FULL,
    }
    template_choice = st.selectbox("Choose Template", list(template_options.keys()), key="member_template")
    section_code = st.text_area("Python Code", value=template_options[template_choice], height=400, key="member_code")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("➕ Add Section", use_container_width=True, type="primary"):
            if section_code.strip():
                st.session_state.sections.append({'code': section_code, 'created_at': datetime.now().isoformat(), 'id': len(st.session_state.sections)})
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
                run_member_analysis()
            else:
                st.warning("⚠️ Add at least one section")

    st.divider()

    if st.session_state.sections:
        st.write("#### 📋 Existing Sections")
        for idx, section in enumerate(st.session_state.sections):
            is_selected = st.session_state.selected_section == idx
            with st.expander(f"{'✨ ' if is_selected else ''}Section {idx + 1}" + (' [SELECTED]' if is_selected else ''), expanded=is_selected):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.code(section['code'], language='python')
                    st.caption(f"Created: {section['created_at'][:19]}")
                    if 'modified_at' in section:
                        st.caption(f"Modified: {section['modified_at'][:19]}")
                with col2:
                    if st.button("📌 Select", key=f"select_{idx}", use_container_width=True):
                        st.session_state.selected_section = idx
                        st.rerun()
    else:
        st.warning("⚠️ **No sections added yet.** Add your first section using the code editor above.")


def run_member_analysis():
    user_dir = get_user_directory(st.session_state.username)
    shutil.copy('concrete_analysis.py', user_dir / 'concrete_analysis.py')
    analysis_file = user_dir / "analysis_script.py"
    full_code = "import matplotlib\nmatplotlib.use('Agg')\nimport os\nimport sys\nimport numpy as np\nfrom concrete_analysis import *\n\nos.chdir(os.path.dirname(os.path.abspath(__file__)))\nos.makedirs('output', exist_ok=True)\n\n"
    for idx, section in enumerate(st.session_state.sections):
        full_code += f"\n# ========== Section {idx + 1} ==========\n{section['code']}\n"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write(full_code)
    _execute_script(analysis_file, "member")


def _execute_script(script_path, label=""):
    with st.spinner("🔄 Running analysis... This may take several minutes."):
        try:
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, str(script_path.absolute())],
                capture_output=True, text=True, timeout=600,
                encoding='utf-8', errors='replace',
            )
            if result.returncode == 0:
                st.success("✅ Analysis completed successfully!")
                st.balloons()
                if result.stdout:
                    with st.expander("📋 Analysis Output"):
                        st.text(result.stdout)
                st.info("Check the Results & Downloads tab to view your outputs.")
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
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# ─── Member results tab ───────────────────────────────────────────────────────

def show_results_tab():
    st.subheader("📊 Analysis Results")
    user_dir = get_user_directory(st.session_state.username)
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
    _show_file_results(user_dir, "member")


def _show_file_results(user_dir, prefix=""):
    image_files, json_files, html_files = [], [], []
    if user_dir.exists():
        for file in user_dir.rglob('*'):
            if file.is_file():
                ext = file.suffix.lower()
                if ext in ['.png', '.jpg', '.jpeg']:
                    image_files.append(file)
                elif ext == '.json':
                    json_files.append(file)
                elif ext == '.html':
                    html_files.append(file)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images", len(image_files))
    col2.metric("JSON Files", len(json_files))
    col3.metric("HTML Reports", len(html_files))
    col4.metric("Total Files", len(image_files) + len(json_files) + len(html_files))
    st.divider()

    if html_files:
        st.write("#### 🌐 HTML Reports")
        for idx, html_file in enumerate(sorted(html_files)):
            with open(html_file, 'rb') as f:
                st.download_button(
                    label=f"⬇️ Download {html_file.name}",
                    data=f,
                    file_name=html_file.name,
                    mime="text/html",
                    key=f"{prefix}_html_{idx}",
                )

    if image_files:
        st.write("#### 🖼️ Images")
        cols = st.columns(3)
        for idx, img_file in enumerate(sorted(image_files)):
            with cols[idx % 3]:
                st.write(f"**{img_file.name}**")
                st.image(str(img_file), use_container_width=True)
                with open(img_file, 'rb') as f:
                    st.download_button(label="⬇️ Download", data=f, file_name=img_file.name, mime="image/png", key=f"{prefix}_img_{idx}")

    if json_files:
        st.write("#### 📊 JSON Files")
        for idx, json_file in enumerate(sorted(json_files)):
            with st.expander(f"{get_file_icon(json_file.name)} {json_file.name}"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    st.json(data)
                    with open(json_file, 'rb') as f:
                        st.download_button(label="⬇️ Download", data=f, file_name=json_file.name, mime="application/json", key=f"{prefix}_json_{idx}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if not (image_files or json_files or html_files):
        st.info("📭 **No files yet.** Run an analysis to generate results.")

# ─── Shell design tab ─────────────────────────────────────────────────────────

def show_shell_design_tab():
    st.subheader("🧱 Shell Element Design — ACI 318-19")
    st.write("Sandwich decomposition (raw M) → Clark-Nielsen per layer | Covers: Slabs, Footings, Shear Walls")

    shell_page = st.radio(
        "Select page",
        ["🔧 Shell Analysis", "📊 Shell Results"],
        horizontal=True,
        key="shell_page_radio",
    )
    st.divider()

    if shell_page == "🔧 Shell Analysis":
        show_shell_analysis_page()
    else:
        show_shell_results_page()


def show_shell_analysis_page():
    st.write("#### Enter Shell Design Code")
    st.info("💡 Choose a template or write your own shell element design code. The engine functions (`run_engine`, `build_schedule`, `plot_contours`, `write_html`, `plot_pmm`) are pre-imported.")

    shell_template_options = {
        "Empty": SHELL_TEMPLATE_EMPTY,
        "Slab Example": SHELL_TEMPLATE_SLAB,
        "Shear Wall Example": SHELL_TEMPLATE_WALL,
        "Footing Example": SHELL_TEMPLATE_FOOTING,
    }
    template_choice = st.selectbox("Choose Template", list(shell_template_options.keys()), key="shell_template")
    shell_code = st.text_area("Python Code", value=shell_template_options[template_choice], height=400, key="shell_code")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("➕ Add Case", use_container_width=True, type="primary"):
            if shell_code.strip():
                st.session_state.shell_sections.append({'code': shell_code, 'created_at': datetime.now().isoformat(), 'id': len(st.session_state.shell_sections)})
                st.success(f"✅ Case {len(st.session_state.shell_sections)} added!")
                st.session_state.selected_shell_section = len(st.session_state.shell_sections) - 1
            else:
                st.error("❌ Please enter shell design code")
    with col2:
        if st.button("✏️ Modify Selected", use_container_width=True, key="shell_modify"):
            if st.session_state.selected_shell_section is not None and shell_code.strip():
                st.session_state.shell_sections[st.session_state.selected_shell_section]['code'] = shell_code
                st.session_state.shell_sections[st.session_state.selected_shell_section]['modified_at'] = datetime.now().isoformat()
                st.success(f"✅ Case {st.session_state.selected_shell_section + 1} modified!")
            else:
                st.warning("⚠️ Select a case and enter code to modify")
    with col3:
        if st.button("🗑️ Delete Selected", use_container_width=True, key="shell_delete"):
            if st.session_state.selected_shell_section is not None:
                deleted_id = st.session_state.selected_shell_section + 1
                del st.session_state.shell_sections[st.session_state.selected_shell_section]
                st.session_state.selected_shell_section = None
                st.success(f"✅ Case {deleted_id} deleted!")
                st.rerun()
            else:
                st.warning("⚠️ Select a case to delete")
    with col4:
        if st.button("▶️ Run Shell Analysis", use_container_width=True, type="primary"):
            if st.session_state.shell_sections:
                run_shell_analysis()
            else:
                st.warning("⚠️ Add at least one design case")

    st.divider()

    if st.session_state.shell_sections:
        st.write("#### 📋 Existing Design Cases")
        for idx, section in enumerate(st.session_state.shell_sections):
            is_selected = st.session_state.selected_shell_section == idx
            with st.expander(f"{'✨ ' if is_selected else ''}Case {idx + 1}" + (' [SELECTED]' if is_selected else ''), expanded=is_selected):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.code(section['code'], language='python')
                    st.caption(f"Created: {section['created_at'][:19]}")
                    if 'modified_at' in section:
                        st.caption(f"Modified: {section['modified_at'][:19]}")
                with col2:
                    if st.button("📌 Select", key=f"shell_select_{idx}", use_container_width=True):
                        st.session_state.selected_shell_section = idx
                        st.rerun()
    else:
        st.warning("⚠️ **No design cases added yet.** Choose a template and click 'Add Case'.")


def run_shell_analysis():
    user_dir = get_shell_user_directory(st.session_state.username)
    shutil.copy('shell_analysis.py', user_dir / 'shell_analysis.py')
    analysis_file = user_dir / "shell_analysis_script.py"

    full_code = (
        "import matplotlib\nmatplotlib.use('Agg')\nimport os, sys\nimport numpy as np\nimport json\nimport warnings\n"
        "warnings.filterwarnings('ignore')\nfrom shell_analysis import (\n"
        "    clark_nielsen_membrane, sandwich_clark_nielsen, run_engine,\n"
        "    plot_pmm, plot_contours, build_schedule, write_html, BAR_TABLE, select_bar,\n"
        ")\n\n"
        f"os.chdir(os.path.dirname(os.path.abspath(__file__)))\n"
        f"import matplotlib.tri as tri\nimport matplotlib.pyplot as plt\n\n"
    )

    for idx, section in enumerate(st.session_state.shell_sections):
        full_code += f"\n# ========== Case {idx + 1} ==========\n{section['code']}\n"

    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write(full_code)
    _execute_script(analysis_file, "shell")


def show_shell_results_page():
    st.subheader("📊 Shell Design Results")
    user_dir = get_shell_user_directory(st.session_state.username)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info(f"📁 **Directory:** `{user_dir}`")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="shell_refresh"):
            st.rerun()
    with col3:
        if st.button("🗑️ Clear Shell Results", use_container_width=True, key="shell_clear"):
            if st.session_state.get('confirm_shell_clear', False):
                clear_shell_user_directory(st.session_state.username)
                st.session_state.confirm_shell_clear = False
                st.success("✅ Shell results cleared!")
                st.rerun()
            else:
                st.session_state.confirm_shell_clear = True
                st.warning("⚠️ Click again to confirm")
    st.divider()
    _show_file_results(user_dir, "shell")

# ─── Guide tab ────────────────────────────────────────────────────────────────

def show_guide_tab():
    st.markdown("### 📚 User Guide")
    st.markdown("""
## 🎯 Overview

The Concrete Section Analyzer helps you analyze reinforced concrete members and shell elements.

**Tabs:**
- **Section Configuration** — Define RC members, run concreteproperties analyses
- **Results & Downloads** — View images, JSON data, download files
- **Shell Design** — ACI 318-19 shell element design (Sandwich → Clark-Nielsen)
- **User Guide** — This page

---

## 📦 Installation Requirements

```bash
pip install concreteproperties sectionproperties numpy matplotlib
```

---

## 🚀 Member Design Quick Start

Choose a template in **Section Configuration**, customize parameters, click **Add Section**, then **Run Analysis**.

### Key rules for member code:
- Use `render=False` in all plot functions
- Use `progress_bar=False` in analysis functions
- Save figures: `ax.get_figure().savefig(...)`; close: `plt.close()`

```python
import numpy as np

concrete = Concrete(
    name="40 MPa Concrete", density=2.4e-6,
    stress_strain_profile=ConcreteLinear(elastic_modulus=32.8e3),
    ultimate_stress_strain_profile=RectangularStressBlock(
        compressive_strength=40, alpha=0.79, gamma=0.87, ultimate_strain=0.003),
    flexural_tensile_strength=3.8, colour="lightgrey",
)
steel = SteelBar(
    name="500 MPa Steel", density=7.85e-6,
    stress_strain_profile=SteelElasticPlastic(
        yield_strength=500, elastic_modulus=200e3, fracture_strain=0.05),
    colour="grey",
)
section_args = dict(
    base_geometry_type="rectangular",
    base_geometry_params={"d": 600, "b": 400},
    base_material=concrete,
    reinforcement=[{"type": "rectangular_array", "material": steel,
        "params": {"area": 314, "n_x": 3, "x_s": 120, "n_y": 2, "y_s": 520, "anchor": (40, 40)}}],
)
output = run_concrete_analysis(**section_args, run_gross_properties=True,
    run_moment_curvature=True, mk_theta=0, mk_n=0, mk_kappa_inc=1e-6,
    mk_kappa_mult=2, mk_kappa_inc_max=5e-6, mk_delta_m_min=0.15, mk_delta_m_max=0.3,
    elastic_modulus=32.8e3, save_dir="output/beam_600x400")
```

---

## 🧱 Shell Design Quick Start

Go to **Shell Design → Shell Analysis**, choose a template (Slab / Shear Wall / Footing), click **Add Case**, then **Run Shell Analysis**. Results appear in **Shell Design → Shell Results**.

### Outputs per case:
- `summary_report.html` — all design checks
- `rebar_schedule.json` — bar designation per face
- `pmm_curve.png` — P-M interaction
- `reinforcement_contours.png` — As contour maps

### Pipeline summary:
```
Raw M (Mx, My, Mxy) + N (Nx, Ny, Nxy)
  → Sandwich decomposition  (z = 0.9·d)
  → Clark-Nielsen per layer (top / bottom)
  → ACI 318-19 design checks
  → HTML report + rebar schedule
```

---

## 🔧 Common Issues

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install concreteproperties sectionproperties` |
| Matplotlib errors | Set `matplotlib.use('Agg')` at top |
| Plot not saved | Use `render=False` + `savefig` + `plt.close()` |
| Progress bar error | Pass `progress_bar=False` |

---

## 📞 Resources

- concreteproperties docs: https://concrete-properties.readthedocs.io/
- GitHub: https://github.com/robbievanleeuwen/concrete-properties
""")
    st.divider()
    st.caption("🏗️ Concrete Section Analyzer | Built for Structural Engineers")
    st.caption("Member design powered by concreteproperties | Shell design: ACI 318-19 Sandwich → Clark-Nielsen")

# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    if not st.session_state.authenticated:
        show_auth()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
