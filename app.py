import streamlit as st
import json
import os
import shutil
from pathlib import Path
import hashlib
from datetime import datetime

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Concrete Section Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1e2e 0%, #1a2f45 100%);
        border-right: 1px solid #2a4a6b;
    }
    [data-testid="stSidebar"] * { color: #c8d8e8 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #e8f4fd !important; }

    /* Sidebar nav buttons */
    div[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: 1px solid #2a4a6b;
        color: #a0b8cc !important;
        width: 100%;
        text-align: left;
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 4px;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: #1e3a54 !important;
        border-color: #4a8ab5 !important;
        color: #e8f4fd !important;
    }

    /* Active nav button */
    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e5fa0 0%, #2a7bc0 100%) !important;
        border-color: #4a9fd4 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(30, 95, 160, 0.4);
    }

    /* Main area */
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f0f4f8;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 13px;
        color: #4a6080;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1a3a5c !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    /* Code editor area */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
        font-size: 12.5px !important;
        background: #1a1e2e !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #4a9fd4 !important;
        box-shadow: 0 0 0 2px rgba(74, 159, 212, 0.2) !important;
    }

    /* Section cards */
    .section-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Header bar */
    .app-header {
        background: linear-gradient(135deg, #0f1e2e 0%, #1a3a5c 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f0f7ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 12px;
    }

    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1e5fa0, #2a7bc0);
        color: white !important;
        border: none;
        border-radius: 6px;
        font-size: 12px;
    }

    /* Sidebar brand */
    .sidebar-brand {
        text-align: center;
        padding: 20px 10px 24px;
        border-bottom: 1px solid #2a4a6b;
        margin-bottom: 20px;
    }
    .sidebar-brand .icon { font-size: 42px; margin-bottom: 6px; }
    .sidebar-brand .title { font-size: 16px; font-weight: 700; color: #e8f4fd; }
    .sidebar-brand .sub { font-size: 11px; color: #6a8aaa; margin-top: 2px; }

    /* Nav section label */
    .nav-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #4a6a8a !important;
        text-transform: uppercase;
        margin: 16px 4px 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
defaults = {
    'authenticated': False,
    'username': None,
    'active_design': 'member',   # 'member' | 'shell'
    'member_sections': [],
    'selected_member': None,
    'shell_sections': [],
    'selected_shell': None,
    'confirm_clear_member': False,
    'confirm_clear_shell': False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── File helpers ─────────────────────────────────────────────────────────────
USERS_DB       = Path("users.json")
BASE_OUTPUT_DIR = Path("user_outputs")

def hash_password(p):   return hashlib.sha256(p.encode()).hexdigest()
def load_users():
    return json.load(open(USERS_DB)) if USERS_DB.exists() else {}
def save_users(u):
    json.dump(u, open(USERS_DB, 'w'), indent=2)

def get_dir(username, sub=None):
    d = BASE_OUTPUT_DIR / username / (sub or "member_outputs")
    d.mkdir(parents=True, exist_ok=True)
    return d

def clear_dir(username, sub):
    d = get_dir(username, sub)
    shutil.rmtree(d); d.mkdir(parents=True, exist_ok=True)

def file_icon(fn):
    return {'.png':'🖼️','.jpg':'🖼️','.jpeg':'🖼️','.json':'📊','.html':'🌐','.txt':'📄'}.get(Path(fn).suffix.lower(),'📁')

# ─── DEFAULT CODE TEMPLATES ───────────────────────────────────────────────────

MEMBER_DEFAULT = '''\
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
# COMMON SECTION ARGS
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
        {"n": 0,      "kappa_inc": 1e-6},
        {"n": 500e3,  "kappa_inc": 1e-6},
        {"n": -500e3, "kappa_inc": 1e-6},
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
        {"theta": 0,       "n": 0},
        {"theta": np.pi,   "n": 0},
        {"theta": np.pi/2, "n": 0},
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
        "params": {
            "area": 314, "n_x": 3, "x_s": 100,
            "n_y": 2, "y_s": 420, "anchor": (40, 40),
        },
    }],
)
sec2, _ = create_dynamic_concrete_section(
    base_geometry_type="rectangular",
    base_geometry_params={"d": 600, "b": 350},
    base_material=concrete,
    reinforcement=[{
        "type": "rectangular_array",
        "material": steel,
        "params": {
            "area": 314, "n_x": 4, "x_s": 90,
            "n_y": 2, "y_s": 520, "anchor": (40, 40),
        },
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
'''

SHELL_DEFAULT = '''\
import numpy as np
import json, os
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# EXAMPLE 1 — SLAB
# =============================================================================
ELEMENT_TYPE = "slab"
OUT_DIR      = "output/shell_slab"

FC_MPA       = 32.0
FY_MPA       = 500.0
H_MM         = 250.0
COVER_MM     = 30.0
DB_MM        = 16.0
HW_MM        = 3000.0
LW_MM        = 6000.0
PHI_F        = 0.90
PHI_V        = 0.75
PHI_C        = 0.65
RHO_MIN      = None
B_MM         = 1000.0
PREFERRED_DB = None

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
    "eps_max":  [0.0012, 0.0010, 0.0008, 0.0005, 0.0009],
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
    phi_f=PHI_F, phi_v=PHI_V, phi_c=PHI_C, rho_min_user=RHO_MIN, b_mm=B_MM,
)
schedule = build_schedule(detailing, H_MM, COVER_MM, PREFERRED_DB)
plot_contours(detailing, N)
with open(os.path.join(OUT_DIR, "rebar_schedule.json"), "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "description"}
               for k, v in schedule.items()}, f, indent=2)
write_html(report, punch, detailing, sw, schedule,
           {"element_type": ELEMENT_TYPE, "fc_mpa": FC_MPA, "fy_mpa": FY_MPA,
            "h_mm": H_MM, "cover_mm": COVER_MM, "db_mm": DB_MM,
            "phi_f": PHI_F, "phi_v": PHI_V, "phi_c": PHI_C, "N": N}, OUT_DIR)
print(f"Slab analysis complete -> {OUT_DIR}")

# =============================================================================
# EXAMPLE 2 — SHEAR WALL
# =============================================================================
ELEMENT_TYPE = "shear_wall"
OUT_DIR      = "output/shell_wall"

FC_MPA       = 40.0
FY_MPA       = 500.0
H_MM         = 300.0
COVER_MM     = 40.0
DB_MM        = 20.0
HW_MM        = 4000.0
LW_MM        = 8000.0
PHI_F        = 0.90
PHI_V        = 0.75
PHI_C        = 0.65
RHO_MIN      = 0.0025
B_MM         = 1000.0
PREFERRED_DB = 20

FORCES_JSON = json.dumps({
    "MXX": [120, 110, 85, 60, 95],
    "MYY": [80,   70, 55, 40, 65],
    "MXY": [30,  -25, 15,-10, 20],
    "FXX": [40,   35, 25, 15, 30],
    "FYY": [30,   25, 18, 12, 22],
    "FXY": [15,  -12,  8, -5, 10],
    "VXZ": [80,   70, 55, 35, 60],
})
STRESS_STRAIN_JSON = json.dumps({
    "sigma22": [7.5, 6.8, 4.5, 3.0, 5.5],
    "eps_max":  [0.0015, 0.0013, 0.0009, 0.0006, 0.0011],
})
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
    phi_f=PHI_F, phi_v=PHI_V, phi_c=PHI_C, rho_min_user=RHO_MIN, b_mm=B_MM,
)
schedule = build_schedule(detailing, H_MM, COVER_MM, PREFERRED_DB)
plot_contours(detailing, N)
with open(os.path.join(OUT_DIR, "rebar_schedule.json"), "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "description"}
               for k, v in schedule.items()}, f, indent=2)
write_html(report, punch, detailing, sw, schedule,
           {"element_type": ELEMENT_TYPE, "fc_mpa": FC_MPA, "fy_mpa": FY_MPA,
            "h_mm": H_MM, "cover_mm": COVER_MM, "db_mm": DB_MM,
            "phi_f": PHI_F, "phi_v": PHI_V, "phi_c": PHI_C, "N": N}, OUT_DIR)
print(f"Shear wall analysis complete -> {OUT_DIR}")

# =============================================================================
# EXAMPLE 3 — FOOTING
# =============================================================================
ELEMENT_TYPE = "footing"
OUT_DIR      = "output/shell_footing"

FC_MPA       = 25.0
FY_MPA       = 500.0
H_MM         = 600.0
COVER_MM     = 75.0
DB_MM        = 20.0
HW_MM        = 600.0
LW_MM        = 5000.0
PHI_F        = 0.90
PHI_V        = 0.75
PHI_C        = 0.65
RHO_MIN      = None
B_MM         = 1000.0
PREFERRED_DB = 20

FORCES_JSON = json.dumps({
    "MXX": [180, 160, 120, 80, 140],
    "MYY": [150, 130, 100, 65, 115],
    "MXY": [20,  -15, 10,  -8,  15],
    "FXX": [25,   20, 15,  10,  18],
    "FYY": [20,   16, 12,   8,  14],
    "FXY": [ 8,   -6,  4,  -3,   6],
    "VXZ": [120, 105, 80,  50,  90],
})
STRESS_STRAIN_JSON = json.dumps({
    "sigma22": [4.5, 4.0, 2.8, 1.8, 3.2],
    "eps_max":  [0.0010, 0.0009, 0.0006, 0.0004, 0.0007],
})
COL_REACTIONS_JSON = json.dumps({
    "1":  {"Vz_kN": 1200, "Mx_kNm": 80, "My_kNm": 60},
    "5":  {"Vz_kN": 1150, "Mx_kNm": 70, "My_kNm": 55},
    "21": {"Vz_kN": 1250, "Mx_kNm": 90, "My_kNm": 65},
    "25": {"Vz_kN": 1180, "Mx_kNm": 75, "My_kNm": 58},
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
    phi_f=PHI_F, phi_v=PHI_V, phi_c=PHI_C, rho_min_user=RHO_MIN, b_mm=B_MM,
)
schedule = build_schedule(detailing, H_MM, COVER_MM, PREFERRED_DB)
plot_contours(detailing, N)
with open(os.path.join(OUT_DIR, "rebar_schedule.json"), "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "description"}
               for k, v in schedule.items()}, f, indent=2)
write_html(report, punch, detailing, sw, schedule,
           {"element_type": ELEMENT_TYPE, "fc_mpa": FC_MPA, "fy_mpa": FY_MPA,
            "h_mm": H_MM, "cover_mm": COVER_MM, "db_mm": DB_MM,
            "phi_f": PHI_F, "phi_v": PHI_V, "phi_c": PHI_C, "N": N}, OUT_DIR)
print(f"Footing analysis complete -> {OUT_DIR}")
'''

# ─── Authentication ───────────────────────────────────────────────────────────
def show_auth():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 40px 0 20px;">
            <div style="font-size:56px;">🏗️</div>
            <h1 style="color:#1a3a5c; font-size:28px; margin:8px 0 4px;">Concrete Section Analyzer</h1>
            <p style="color:#6a8aaa; font-size:14px; margin:0;">Advanced structural analysis & visualization</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        auth_mode = st.radio("", ["Sign In", "Sign Up"], horizontal=True, label_visibility="collapsed")

        with st.container():
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter password")

            if auth_mode == "Sign Up":
                confirm = st.text_input("🔑 Confirm Password", type="password", placeholder="Repeat password")

            if auth_mode == "Sign In":
                if st.button("Sign In →", use_container_width=True, type="primary"):
                    users = load_users()
                    if username in users and users[username]['password'] == hash_password(password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            else:
                if st.button("Create Account →", use_container_width=True, type="primary"):
                    if not username or not password:
                        st.error("Please fill in all fields")
                    elif password != confirm:
                        st.error("Passwords don't match")
                    else:
                        users = load_users()
                        if username in users:
                            st.error("Username already exists")
                        else:
                            users[username] = {
                                'password': hash_password(password),
                                'created_at': datetime.now().isoformat(),
                            }
                            save_users(users)
                            get_dir(username, 'member_outputs')
                            get_dir(username, 'shell_outputs')
                            st.success("Account created! Please sign in.")

        st.markdown("""
        <p style="text-align:center; color:#9aaabb; font-size:12px; margin-top:24px;">
            🔒 Your data is stored in isolated user directories
        </p>
        """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sidebar-brand">
            <div class="icon">🏗️</div>
            <div class="title">Concrete Analyzer</div>
            <div class="sub">Structural Analysis Platform</div>
        </div>
        """, unsafe_allow_html=True)

        # User info
        st.markdown(f"""
        <div style="background:#1e3a54; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
            <span style="font-size:12px; color:#6a9aaa;">Logged in as</span><br>
            <span style="font-size:14px; font-weight:700; color:#e8f4fd;">👤 {st.session_state.username}</span>
        </div>
        """, unsafe_allow_html=True)

        # Navigation label
        st.markdown('<div class="nav-label">Design Module</div>', unsafe_allow_html=True)

        # Member Design button
        member_type = "primary" if st.session_state.active_design == "member" else "secondary"
        if st.button("🧱  Member Design", key="nav_member",
                     type=member_type, use_container_width=True):
            st.session_state.active_design = "member"
            st.rerun()

        # Shell Design button
        shell_type = "primary" if st.session_state.active_design == "shell" else "secondary"
        if st.button("🔲  Shell Design", key="nav_shell",
                     type=shell_type, use_container_width=True):
            st.session_state.active_design = "shell"
            st.rerun()

        st.markdown('<div class="nav-label">Account</div>', unsafe_allow_html=True)
        if st.button("🚪  Sign Out", key="nav_signout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        # Active module badge
        st.markdown("<br>", unsafe_allow_html=True)
        badge_color = "#1e5fa0" if st.session_state.active_design == "member" else "#1a5c3a"
        badge_label = "Member Design" if st.session_state.active_design == "member" else "Shell Design"
        badge_icon  = "🧱" if st.session_state.active_design == "member" else "🔲"
        st.markdown(f"""
        <div style="background:{badge_color}22; border:1px solid {badge_color}55;
                    border-radius:8px; padding:10px 12px; text-align:center;">
            <span style="font-size:18px;">{badge_icon}</span><br>
            <span style="font-size:11px; color:#a0c0e0; font-weight:600;">ACTIVE MODULE</span><br>
            <span style="font-size:13px; color:#e8f4fd; font-weight:700;">{badge_label}</span>
        </div>
        """, unsafe_allow_html=True)

# ─── Script executor ──────────────────────────────────────────────────────────
def execute_script(script_path):
    import subprocess, sys
    with st.spinner("🔄 Running analysis… This may take several minutes."):
        try:
            result = subprocess.run(
                [sys.executable, str(script_path.absolute())],
                capture_output=True, text=True, timeout=600,
                encoding='utf-8', errors='replace',
            )
            if result.returncode == 0:
                st.success("✅ Analysis completed successfully!")
                st.balloons()
                if result.stdout:
                    with st.expander("📋 Console Output"):
                        st.code(result.stdout, language="text")
                st.info("Switch to the **Results** page to view and download outputs.")
            else:
                st.error("❌ Analysis failed")
                if result.stderr:
                    with st.expander("🔍 Error Details", expanded=True):
                        st.code(result.stderr, language="text")
                if result.stdout:
                    with st.expander("📋 Partial Output"):
                        st.code(result.stdout, language="text")
        except subprocess.TimeoutExpired:
            st.error("❌ Timeout — exceeded 10 minutes. Reduce analysis points.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ─── Shared results renderer ──────────────────────────────────────────────────
def render_results(user_dir, clear_key):
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.info(f"📁 Output directory: `{user_dir}`")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key=f"refresh_{clear_key}"):
            st.rerun()
    with col3:
        if st.button("🗑️ Clear All", use_container_width=True, key=f"clear_btn_{clear_key}"):
            if st.session_state.get(f'confirm_{clear_key}', False):
                sub = "member_outputs" if clear_key == "member" else "shell_outputs"
                clear_dir(st.session_state.username, sub)
                st.session_state[f'confirm_{clear_key}'] = False
                st.success("✅ Cleared!")
                st.rerun()
            else:
                st.session_state[f'confirm_{clear_key}'] = True
                st.warning("⚠️ Click again to confirm deletion")

    st.divider()

    imgs, jsons, htmls = [], [], []
    if user_dir.exists():
        for f in sorted(user_dir.rglob('*')):
            if not f.is_file(): continue
            ext = f.suffix.lower()
            if ext in ('.png', '.jpg', '.jpeg'): imgs.append(f)
            elif ext == '.json': jsons.append(f)
            elif ext == '.html': htmls.append(f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🖼️ Images",       len(imgs))
    c2.metric("📊 JSON Files",   len(jsons))
    c3.metric("🌐 HTML Reports", len(htmls))
    c4.metric("📦 Total",        len(imgs)+len(jsons)+len(htmls))

    if not (imgs or jsons or htmls):
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#94a3b8;">
            <div style="font-size:48px;">📭</div>
            <h3 style="margin:12px 0 6px; color:#64748b;">No results yet</h3>
            <p style="font-size:14px;">Run an analysis first to generate output files.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # HTML reports
    if htmls:
        st.markdown("#### 🌐 HTML Reports")
        for i, f in enumerate(htmls):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{f.name}** — `{f.parent.relative_to(user_dir) if f.parent != user_dir else '.'}`")
            with col2:
                st.download_button("⬇️ Download", open(f,'rb').read(),
                                   file_name=f.name, mime="text/html",
                                   key=f"html_{clear_key}_{i}")
        st.divider()

    # Images
    if imgs:
        st.markdown("#### 🖼️ Images")
        cols = st.columns(3)
        for i, f in enumerate(imgs):
            with cols[i % 3]:
                st.markdown(f"<small style='color:#64748b;'>{f.relative_to(user_dir)}</small>", unsafe_allow_html=True)
                st.image(str(f), use_container_width=True)
                st.download_button("⬇️ Download", open(f,'rb').read(),
                                   file_name=f.name, mime="image/png",
                                   key=f"img_{clear_key}_{i}")
        st.divider()

    # JSON files
    if jsons:
        st.markdown("#### 📊 JSON Data")
        for i, f in enumerate(jsons):
            with st.expander(f"📊 {f.relative_to(user_dir)}"):
                try:
                    data = json.load(open(f))
                    st.json(data)
                    st.download_button("⬇️ Download", open(f,'rb').read(),
                                       file_name=f.name, mime="application/json",
                                       key=f"json_{clear_key}_{i}")
                except Exception as e:
                    st.error(f"Could not parse: {e}")

# ─── MEMBER DESIGN tab ────────────────────────────────────────────────────────
def show_member_design():
    st.markdown("""
    <div class="app-header">
        <div>
            <h2 style="margin:0; font-size:20px; color:white;">🧱 Member Design</h2>
            <p style="margin:4px 0 0; font-size:13px; color:#a0c4e0;">
                Gross/Cracked Properties · Moment-Curvature · Interaction Diagrams · Stress Analysis
            </p>
        </div>
        <div style="font-size:12px; color:#6a9aaa;">concreteproperties · ACI / AS 3600</div>
    </div>
    """, unsafe_allow_html=True)

    page_analysis, page_results = st.tabs(["⚙️ Analysis", "📊 Results"])

    # ── Analysis page ──────────────────────────────────────────────────────────
    with page_analysis:
        st.markdown("#### 📝 Code Editor")
        st.caption("All 20 default examples are pre-loaded. Edit, add, delete sections as needed. The helper functions from `concrete_analysis.py` are auto-imported.")

        # Initialize with default code if empty
        if not st.session_state.member_sections:
            st.session_state.member_sections = [{
                'name': 'All 20 Default Examples',
                'code': MEMBER_DEFAULT,
                'created_at': datetime.now().isoformat(),
            }]
            st.session_state.selected_member = 0

        # Selector row
        col_sel, col_name = st.columns([2, 3])
        with col_sel:
            section_names = [f"Section {i+1}: {s['name']}" for i, s in enumerate(st.session_state.member_sections)]
            sel = st.selectbox("Select section", section_names,
                               index=st.session_state.selected_member or 0,
                               key="member_selector")
            idx = section_names.index(sel)
            st.session_state.selected_member = idx

        with col_name:
            new_name = st.text_input("Section name", value=st.session_state.member_sections[idx]['name'], key="member_name_input")

        # Single text area
        current_code = st.session_state.member_sections[idx]['code']
        edited_code = st.text_area(
            "Python Code",
            value=current_code,
            height=520,
            key=f"member_code_{idx}",
            label_visibility="collapsed",
        )

        # Action buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("➕ Add New", use_container_width=True):
                st.session_state.member_sections.append({
                    'name': f'New Section {len(st.session_state.member_sections)+1}',
                    'code': '# Enter your analysis code here\n',
                    'created_at': datetime.now().isoformat(),
                })
                st.session_state.selected_member = len(st.session_state.member_sections) - 1
                st.rerun()
        with col2:
            if st.button("💾 Save Changes", use_container_width=True, type="primary"):
                st.session_state.member_sections[idx]['code'] = edited_code
                st.session_state.member_sections[idx]['name'] = new_name
                st.session_state.member_sections[idx]['modified_at'] = datetime.now().isoformat()
                st.success("✅ Saved!")
        with col3:
            if st.button("🔄 Reset Default", use_container_width=True):
                st.session_state.member_sections[idx]['code'] = MEMBER_DEFAULT
                st.session_state.member_sections[idx]['name'] = 'All 20 Default Examples'
                st.rerun()
        with col4:
            if st.button("🗑️ Delete", use_container_width=True):
                if len(st.session_state.member_sections) > 1:
                    st.session_state.member_sections.pop(idx)
                    st.session_state.selected_member = max(0, idx - 1)
                    st.rerun()
                else:
                    st.warning("Cannot delete the only section.")
        with col5:
            if st.button("▶️ Run Analysis", use_container_width=True, type="primary"):
                # Save any unsaved edits first
                st.session_state.member_sections[idx]['code'] = edited_code
                run_member_analysis_all()

        # Preview panel
        with st.expander("📋 All Sections Overview", expanded=False):
            for i, s in enumerate(st.session_state.member_sections):
                marker = "✨ " if i == st.session_state.selected_member else ""
                st.markdown(f"**{marker}Section {i+1}: {s['name']}** — {len(s['code'].splitlines())} lines")

    # ── Results page ───────────────────────────────────────────────────────────
    with page_results:
        user_dir = get_dir(st.session_state.username, "member_outputs")
        render_results(user_dir, "member")


def run_member_analysis_all():
    user_dir = get_dir(st.session_state.username, "member_outputs")
    try:
        shutil.copy('concrete_analysis.py', user_dir / 'concrete_analysis.py')
    except FileNotFoundError:
        st.error("❌ `concrete_analysis.py` not found in the project directory.")
        return

    script = user_dir / "member_script.py"
    header = (
        "import matplotlib; matplotlib.use('Agg')\n"
        "import os, sys, numpy as np\n"
        "from concrete_analysis import *\n\n"
        f"os.chdir(r'{str(user_dir.absolute())}')\n"
        "os.makedirs('output', exist_ok=True)\n\n"
    )
    body = "\n\n".join(
        f"# ===== Section {i+1}: {s['name']} =====\n{s['code']}"
        for i, s in enumerate(st.session_state.member_sections)
    )
    script.write_text(header + body, encoding='utf-8')
    execute_script(script)

# ─── SHELL DESIGN tab ─────────────────────────────────────────────────────────
def show_shell_design():
    st.markdown("""
    <div class="app-header" style="background: linear-gradient(135deg, #0f2e1e 0%, #1a4a2c 100%);">
        <div>
            <h2 style="margin:0; font-size:20px; color:white;">🔲 Shell Design</h2>
            <p style="margin:4px 0 0; font-size:13px; color:#a0e0c4;">
                Sandwich Decomposition → Clark-Nielsen | Slabs · Footings · Shear Walls
            </p>
        </div>
        <div style="font-size:12px; color:#6aaaa0;">ACI 318-19</div>
    </div>
    """, unsafe_allow_html=True)

    page_analysis, page_results = st.tabs(["⚙️ Analysis", "📊 Results"])

    # ── Analysis page ──────────────────────────────────────────────────────────
    with page_analysis:
        st.markdown("#### 📝 Code Editor")
        st.caption("All 3 default shell examples (Slab, Shear Wall, Footing) are pre-loaded. Edit, add, delete cases. Engine functions are auto-imported from `shell_analysis.py`.")

        # Initialize with default code if empty
        if not st.session_state.shell_sections:
            st.session_state.shell_sections = [{
                'name': 'All 3 Default Examples (Slab + Wall + Footing)',
                'code': SHELL_DEFAULT,
                'created_at': datetime.now().isoformat(),
            }]
            st.session_state.selected_shell = 0

        # Selector row
        col_sel, col_name = st.columns([2, 3])
        with col_sel:
            case_names = [f"Case {i+1}: {s['name']}" for i, s in enumerate(st.session_state.shell_sections)]
            sel = st.selectbox("Select case", case_names,
                               index=st.session_state.selected_shell or 0,
                               key="shell_selector")
            idx = case_names.index(sel)
            st.session_state.selected_shell = idx

        with col_name:
            new_name = st.text_input("Case name", value=st.session_state.shell_sections[idx]['name'], key="shell_name_input")

        # Single text area
        current_code = st.session_state.shell_sections[idx]['code']
        edited_code = st.text_area(
            "Python Code",
            value=current_code,
            height=520,
            key=f"shell_code_{idx}",
            label_visibility="collapsed",
        )

        # Action buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("➕ Add New", use_container_width=True, key="shell_add"):
                st.session_state.shell_sections.append({
                    'name': f'New Case {len(st.session_state.shell_sections)+1}',
                    'code': '# Enter your shell design code here\n',
                    'created_at': datetime.now().isoformat(),
                })
                st.session_state.selected_shell = len(st.session_state.shell_sections) - 1
                st.rerun()
        with col2:
            if st.button("💾 Save Changes", use_container_width=True, type="primary", key="shell_save"):
                st.session_state.shell_sections[idx]['code'] = edited_code
                st.session_state.shell_sections[idx]['name'] = new_name
                st.session_state.shell_sections[idx]['modified_at'] = datetime.now().isoformat()
                st.success("✅ Saved!")
        with col3:
            if st.button("🔄 Reset Default", use_container_width=True, key="shell_reset"):
                st.session_state.shell_sections[idx]['code'] = SHELL_DEFAULT
                st.session_state.shell_sections[idx]['name'] = 'All 3 Default Examples (Slab + Wall + Footing)'
                st.rerun()
        with col4:
            if st.button("🗑️ Delete", use_container_width=True, key="shell_del"):
                if len(st.session_state.shell_sections) > 1:
                    st.session_state.shell_sections.pop(idx)
                    st.session_state.selected_shell = max(0, idx - 1)
                    st.rerun()
                else:
                    st.warning("Cannot delete the only case.")
        with col5:
            if st.button("▶️ Run Analysis", use_container_width=True, type="primary", key="shell_run"):
                st.session_state.shell_sections[idx]['code'] = edited_code
                run_shell_analysis_all()

        with st.expander("📋 All Cases Overview", expanded=False):
            for i, s in enumerate(st.session_state.shell_sections):
                marker = "✨ " if i == st.session_state.selected_shell else ""
                st.markdown(f"**{marker}Case {i+1}: {s['name']}** — {len(s['code'].splitlines())} lines")

    # ── Results page ───────────────────────────────────────────────────────────
    with page_results:
        user_dir = get_dir(st.session_state.username, "shell_outputs")
        render_results(user_dir, "shell")


def run_shell_analysis_all():
    user_dir = get_dir(st.session_state.username, "shell_outputs")
    try:
        shutil.copy('shell_analysis.py', user_dir / 'shell_analysis.py')
    except FileNotFoundError:
        st.error("❌ `shell_analysis.py` not found in the project directory.")
        return

    script = user_dir / "shell_script.py"
    header = (
        "import matplotlib; matplotlib.use('Agg')\n"
        "import os, sys, json, numpy as np, warnings\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.tri as tri\n"
        "warnings.filterwarnings('ignore')\n"
        "from shell_analysis import (\n"
        "    clark_nielsen_membrane, sandwich_clark_nielsen, run_engine,\n"
        "    plot_pmm, plot_contours, build_schedule, write_html,\n"
        "    BAR_TABLE, select_bar,\n"
        ")\n\n"
        f"os.chdir(r'{str(user_dir.absolute())}')\n"
        "os.makedirs('output', exist_ok=True)\n\n"
    )
    body = "\n\n".join(
        f"# ===== Case {i+1}: {s['name']} =====\n{s['code']}"
        for i, s in enumerate(st.session_state.shell_sections)
    )
    script.write_text(header + body, encoding='utf-8')
    execute_script(script)

# ─── USER GUIDE tab ───────────────────────────────────────────────────────────
def show_user_guide():
    st.markdown("""
    <div class="app-header">
        <div>
            <h2 style="margin:0; font-size:20px; color:white;">📚 User Guide</h2>
            <p style="margin:4px 0 0; font-size:13px; color:#a0c4e0;">Getting started, examples, and reference</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
## 🗂️ Project Structure

Place all files in the same directory:
```
your_project/
├── app.py                  ← this Streamlit app
├── concrete_analysis.py    ← member design engine
├── shell_analysis.py       ← shell design engine
├── users.json              ← auto-created on first signup
└── user_outputs/           ← auto-created on first run
```
Run with: `streamlit run app.py`

---

## 🧱 Member Design — Quick Reference

The **Member Design** tab pre-loads all 20 standard analyses. The following functions from `concrete_analysis.py` are auto-imported:

| Function | Purpose |
|---|---|
| `run_concrete_analysis(...)` | Master runner — triggers any combination of analyses |
| `create_dynamic_concrete_section(...)` | Create a section object for use in multi-section comparisons |
| `Concrete`, `SteelBar`, `ConcreteLinear`, etc. | Material constructors |

**Key parameters in `run_concrete_analysis`:**

```python
run_concrete_analysis(
    base_geometry_type  = "rectangular",      # or "circular", "i_section"
    base_geometry_params= {"d": 500, "b": 300},
    base_material       = concrete,
    reinforcement       = [...],
    run_gross_properties          = True,
    run_cracked_properties        = True,
    run_moment_curvature          = True,
    run_ultimate_bending          = True,
    run_moment_interaction        = True,
    run_positive_negative_interaction = True,
    run_biaxial_bending           = True,
    run_multiple_biaxial          = True,
    run_uncracked_stress          = True,
    run_cracked_stress            = True,
    run_service_stress            = True,
    run_ultimate_stress           = True,
    run_stress_strain_profiles    = True,
    elastic_modulus = 30.1e3,
    save_dir        = "output/my_section",
)
```

---

## 🔲 Shell Design — Quick Reference

The **Shell Design** tab pre-loads 3 examples (Slab, Shear Wall, Footing). Functions from `shell_analysis.py` are auto-imported:

| Function | Purpose |
|---|---|
| `run_engine(...)` | Master engine — design checks, detailing |
| `build_schedule(...)` | Generate rebar schedule |
| `plot_contours(...)` | Reinforcement contour plots |
| `write_html(...)` | HTML summary report |
| `plot_pmm(...)` | P-M interaction curve |

**Pipeline:**
```
Mx, My, Mxy  (moments, kN·m/m)
Nx, Ny, Nxy  (in-plane forces, kN/m)
    ↓  Sandwich decomposition  (z = 0.9·d)
    ↓  Clark-Nielsen per layer (top / bottom face)
    ↓  ACI 318-19 design checks
    ↓  rebar_schedule.json  |  summary_report.html
       pmm_curve.png        |  reinforcement_contours.png
```

**Required inputs:**
```python
run_engine(
    forces        = {"MXX":[], "MYY":[], "MXY":[], "FXX":[], "FYY":[], "FXY":[], "VXZ":[]},
    ss            = {"sigma22":[], "eps_max":[]},
    col_reactions = {"node_id": {"Vz_kN":..., "Mx_kNm":..., "My_kNm":...}},
    col_punch     = {"node_id": {"col_width_in":..., "col_depth_in":..., "condition":"I|N|S|E|W|NW|NE|SW|SE"}},
    element_type  = "slab" | "footing" | "shear_wall",
    fc_mpa=32, fy_mpa=500, h_mm=250, cover_mm=30, db_mm=16,
    hw_mm=3000, lw_mm=6000, phi_f=0.90, phi_v=0.75, phi_c=0.65,
    rho_min_user=None,   # None = ACI auto
    b_mm=1000,
)
```

---

## ⚠️ Common Issues

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | `pip install concreteproperties sectionproperties matplotlib numpy` |
| Matplotlib display error | Handled automatically — `Agg` backend set in script header |
| `progress_bar` crash | Always pass `progress_bar=False` in analysis functions |
| Plot not saved | Use `render=False` in all `.plot_*()` calls |
| Engine file not found | Ensure `concrete_analysis.py` / `shell_analysis.py` are in the same folder as `app.py` |

---

## 📖 Resources
- concreteproperties docs: https://concrete-properties.readthedocs.io/
- GitHub: https://github.com/robbievanleeuwen/concrete-properties
""")
    st.divider()
    st.caption("🏗️ Concrete Section Analyzer · Built for Structural Engineers · ACI 318-19 / AS 3600")

# ─── Main app ─────────────────────────────────────────────────────────────────
def show_main_app():
    show_sidebar()

    tab_member, tab_shell, tab_guide = st.tabs([
        "🧱 Member Design",
        "🔲 Shell Design",
        "📚 User Guide",
    ])

    with tab_member:
        show_member_design()

    with tab_shell:
        show_shell_design()

    with tab_guide:
        show_user_guide()

# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    st.title("🏗️ Streamlit Concrete Analyzer | Structural Analysis Platform")
    if not st.session_state.authenticated:
        show_auth()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
