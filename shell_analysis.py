"""
=============================================================================
UNIFIED RCC SHELL ELEMENT DESIGN ENGINE — ACI 318-19
Covers  : Slabs, Footings, Shear Walls
Pipeline: Sandwich decomposition (raw M) → Clark-Nielsen per layer
Sign    : +M sagging → compression top (−M/z), tension bottom (+M/z)
Outputs : summary_report.html  |  rebar_schedule.json
          pmm_curve.png        |  reinforcement_contours.png
NO DEFAULT VALUES — every parameter must be explicitly provided.
=============================================================================
"""
import numpy as np, json, os, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.tri as tri
warnings.filterwarnings("ignore")

# =============================================================================
# USER INPUTS
# =============================================================================
ELEMENT_TYPE = ""       # "slab" | "footing" | "shear_wall"
OUT_DIR      = ""       # output folder path
FC_MPA       = None     # concrete f'c [MPa]
FY_MPA       = None     # steel fy [MPa]
H_MM         = None     # section thickness [mm]
COVER_MM     = None     # clear cover [mm]
DB_MM        = None     # bar diameter [mm]
HW_MM        = None     # wall/column height for slenderness [mm]
LW_MM        = None     # wall length [mm]
PHI_F        = None     # flexure phi
PHI_V        = None     # shear phi
PHI_C        = None     # compression phi
RHO_MIN      = None     # min steel ratio — None = ACI auto
B_MM         = None     # strip width [mm], typically 1000
PREFERRED_DB = None     # fix bar dia for schedule [mm] — None = auto

FORCES_JSON = """
{
  "MXX": [50, 45, 30, 20, 35],
  "MYY": [40, 35, 25, 15, 28],
  "MXY": [10, -8,  5, -3,  6],
  "FXX": [20, 18, 12,  8, 14],
  "FYY": [15, 13, 10,  6, 11],
  "FXY": [ 5, -4,  3, -2,  4],
  "VXZ": [30, 25, 20, 12, 22]
}
"""
STRESS_STRAIN_JSON = """
{
  "sigma22": [5.2, 4.8, 3.1, 2.0, 4.0],
  "eps_max": [0.0012, 0.0010, 0.0008, 0.0005, 0.0009]
}
"""
COL_REACTIONS_JSON = """
{
  "1":  {"Vz_kN": 450, "Mx_kNm": 30, "My_kNm": 20},
  "5":  {"Vz_kN": 430, "Mx_kNm": 25, "My_kNm": 18},
  "21": {"Vz_kN": 470, "Mx_kNm": 35, "My_kNm": 22},
  "25": {"Vz_kN": 440, "Mx_kNm": 28, "My_kNm": 19}
}
"""
COL_PUNCHING_JSON = """
{
  "1":  {"col_width_in": 24, "col_depth_in": 24, "condition": "NW", "overhang_x": 6, "overhang_y": 6},
  "5":  {"col_width_in": 24, "col_depth_in": 24, "condition": "NE", "overhang_x": 6, "overhang_y": 6},
  "21": {"col_width_in": 24, "col_depth_in": 24, "condition": "SW", "overhang_x": 6, "overhang_y": 6},
  "25": {"col_width_in": 24, "col_depth_in": 24, "condition": "SE", "overhang_x": 6, "overhang_y": 6}
}
"""

# =============================================================================
# CLARK-NIELSEN  (per sandwich layer)
# Threshold 1.0 kN/m prevents Nxy²/N → inf at near-zero stress points
# =============================================================================
def clark_nielsen_membrane(Nx, Ny, Nxy):
    t, aN = 1.0, np.abs(Nxy)
    Nx_s = np.where(Nx < -aN, 0.0, Nx + aN)
    Ny_s = np.where(Ny < -aN, 0.0, Ny + aN)
    Ny_s = np.where(Nx < -aN, Ny + np.where(np.abs(Nx) > t, Nxy**2/np.abs(Nx), aN), Ny_s)
    Nx_s = np.where(Ny < -aN, Nx + np.where(np.abs(Ny) > t, Nxy**2/np.abs(Ny), aN), Nx_s)
    return np.maximum(Nx_s, 0.0), np.maximum(Ny_s, 0.0)

# =============================================================================
# SANDWICH DECOMPOSITION -> CLARK-NIELSEN
# =============================================================================
def sandwich_clark_nielsen(Mx, My, Mxy, Nx, Ny, Nxy, h_mm, cover_mm, db_mm):
    z = 0.9 * (h_mm - cover_mm - db_mm / 2.0)
    Nxt_s, Nyt_s = clark_nielsen_membrane(Nx/2 - Mx*1000/z, Ny/2 - My*1000/z, Nxy/2 - Mxy*1000/z)
    Nxb_s, Nyb_s = clark_nielsen_membrane(Nx/2 + Mx*1000/z, Ny/2 + My*1000/z, Nxy/2 + Mxy*1000/z)
    return Nxt_s, Nyt_s, Nxb_s, Nyb_s

# =============================================================================
# PMM CURVE -> pmm_curve.png
# =============================================================================
def plot_pmm(fc, fy, h, b, Ast, Pu_d, Mu_d, phi_f, phi_c):
    Pn_l, Mn_l = [], []
    for c in np.linspace(h*0.1, h*1.2, 40):
        beta = np.clip(0.85 - 0.05*(fc-28)/7, 0.65, 0.85)
        Cc   = 0.85*fc*(beta*c)*b
        fs   = np.clip(0.003*((h-50)-c)/c*200000.0, -fy, fy)
        Ts   = (Ast/2)*fs
        Pn_l.append((Cc - Ts)/1000)
        Mn_l.append(abs((Cc*(h/2 - (beta*c)/2) + Ts*((h-50) - h/2))/1e6))
    phi_Mn, phi_Pn = [phi_f*m for m in Mn_l], [phi_c*p for p in Pn_l]
    inside = any(phi_Mn[i] >= Mu_d and phi_Pn[i] >= Pu_d for i in range(len(phi_Pn)))
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(Mn_l, Pn_l, "b--", lw=1.2, label="Nominal Pn-Mn")
    ax.plot(phi_Mn, phi_Pn, "b-", lw=2.2, label="Design phiPn-phiMn")
    ax.scatter([Mu_d], [Pu_d], color="red", s=80, zorder=5, label="Demand")
    ax.set_xlabel("Moment (kNm)"); ax.set_ylabel("Axial (kN)")
    ax.set_title(f"P-M Interaction ACI 318-19 — {'PASS' if inside else 'FAIL'}")
    ax.legend(); ax.grid(alpha=0.4); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pmm_curve.png"), dpi=150); plt.close(fig)
    return inside

# =============================================================================
# REINFORCEMENT CONTOURS -> reinforcement_contours.png
# =============================================================================
def plot_contours(detailing, N):
    n  = int(np.round(np.sqrt(N)))
    x  = np.tile(np.linspace(0, 1, n), n)[:N]
    y  = np.repeat(np.linspace(0, 1, n), n)[:N]
    tr = tri.Triangulation(x, y)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (key, label, cmap) in enumerate([
        ("As_Top_X","Top-X","YlOrRd"),("As_Top_Y","Top-Y","YlOrRd"),
        ("As_Bot_X","Bot-X","Blues"), ("As_Bot_Y","Bot-Y","Blues")]):
        ax = axes[i//2, i%2]
        tcf = ax.tricontourf(tr, detailing[key], levels=12, cmap=cmap)
        fig.colorbar(tcf, ax=ax, label="mm2/m")
        ax.set_title(f"As {label} [mm2/m]"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    fig.suptitle("ACI 318-19 Detailing — Sandwich → Clark-Nielsen", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "reinforcement_contours.png"), dpi=150); plt.close(fig)

# =============================================================================
# REBAR SCHEDULE -> rebar_schedule.json
# =============================================================================
BAR_TABLE = {8:50.3,10:78.5,12:113.1,16:201.1,20:314.2,25:490.9,32:804.2,40:1256.6}

def select_bar(As_req, h_mm, cover_mm, preferred_db=None):
    s_max = min(3*h_mm, 450)
    for db in ([preferred_db] if preferred_db else sorted(BAR_TABLE)):
        Ab    = BAR_TABLE[db]
        s_min = max(3*db, 100)
        s     = max(min(int((Ab*1000/max(As_req, 1e-6))//25)*25, s_max), s_min)
        As_p  = Ab * 1000 / s
        if As_p >= As_req - 0.01:
            return {"db_mm":db,"spacing_mm":s,"As_req_mm2m":round(As_req,1),
                    "As_prov_mm2m":round(As_p,1),"utilisation":round(As_req/As_p,3),
                    "designation":f"T{db}@{s}"}
    db=max(BAR_TABLE); Ab=BAR_TABLE[db]; s=max(3*db,100); As_p=Ab*1000/s
    return {"db_mm":db,"spacing_mm":s,"As_req_mm2m":round(As_req,1),"As_prov_mm2m":round(As_p,1),
            "utilisation":round(As_req/As_p,3),"designation":f"T{db}@{s} [CHECK SECTION]"}

def build_schedule(detailing, h_mm, cover_mm, preferred_db):
    layers = [("As_Top_X","Top Face — X (hogging)"),("As_Top_Y","Top Face — Y (hogging)"),
              ("As_Bot_X","Bot Face — X (sagging)"),("As_Bot_Y","Bot Face — Y (sagging)")]
    return {key:{**select_bar(float(detailing[key].max()),h_mm,cover_mm,preferred_db),"description":desc}
            for key,desc in layers}

# =============================================================================
# MASTER ENGINE
# =============================================================================
def run_engine(forces, ss, col_reactions, col_punch, element_type, fc_mpa, fy_mpa,
               h_mm, cover_mm, db_mm, hw_mm, lw_mm, phi_f, phi_v, phi_c, rho_min_user, b_mm):
    d_mm = h_mm - cover_mm - db_mm / 2.0
    Ag   = b_mm * h_mm
    Ec   = 4700 * np.sqrt(fc_mpa)
    n_r  = 200000.0 / Ec
    is_wall    = element_type.lower() == "shear_wall"
    is_footing = element_type.lower() == "footing"
    rho_min = (rho_min_user if rho_min_user else
               (0.0018 if h_mm < 900 else 0.0026) if is_footing else
               (0.0025 if h_mm > 250 else 0.0015) if is_wall else 0.0018)

    Mx  = np.array(forces["MXX"]); My  = np.array(forces["MYY"])
    Mxy = np.array(forces["MXY"]); Nx  = np.array(forces["FXX"])
    Ny  = np.array(forces["FYY"]); Nxy = np.array(forces["FXY"])
    Vux = np.abs(np.array(forces["VXZ"]))
    sigma22 = np.abs(np.array(ss["sigma22"]))
    eps_max = np.array(ss["eps_max"])

    Nxt_s, Nyt_s, Nxb_s, Nyb_s = sandwich_clark_nielsen(Mx, My, Mxy, Nx, Ny, Nxy, h_mm, cover_mm, db_mm)
    Nx_star, Ny_star = np.maximum(Nxt_s, Nxb_s), np.maximum(Nyt_s, Nyb_s)

    def calc_as(M_d, N_d):
        Mu  = np.abs(M_d) * 1e6; Nu = np.abs(N_d) * 1000
        Rn  = Mu / (phi_f * b_mm * d_mm**2)
        disc = np.clip(1 - (2*Rn)/(0.85*fc_mpa), 0.001, 1.0)
        rho  = np.where(2*Rn < 0.85*fc_mpa, (0.85*fc_mpa/fy_mpa)*(1-np.sqrt(disc)), 0.025)
        return np.maximum(rho*b_mm*d_mm + Nu/(phi_f*fy_mpa), rho_min*Ag)

    As_x = calc_as(np.abs(Mx), Nx_star)
    As_y = calc_as(np.abs(My), Ny_star)

    lam_s = np.minimum(np.sqrt(2/(1 + 0.004*d_mm)), 1.0)
    rho_w = As_x / (b_mm * d_mm)
    phiVc = phi_v * 0.66 * lam_s * (rho_w**(1/3)) * np.sqrt(fc_mpa) * b_mm * d_mm / 1000
    shear_dc = Vux / np.maximum(phiVc, 1e-9)

    fs_s  = 0.67 * fy_mpa
    s_max = min((15*(40000/(fs_s*145.04)) - 2.5*(cover_mm*0.039))*25.4, 3*h_mm, 450)
    Ig    = b_mm * h_mm**3 / 12
    fr    = 0.62 * np.sqrt(fc_mpa)
    Mcr   = fr * Ig / (h_mm/2) / 1e6
    Ma    = 0.7 * float(np.abs(Mx).max())
    rho_e = float(As_x.max()) / (b_mm * d_mm)
    kd    = np.sqrt((rho_e*n_r)**2 + 2*rho_e*n_r) - rho_e*n_r
    Icr   = (1/3)*b_mm*(kd*d_mm)**3 + n_r*float(As_x.max())*(d_mm - kd*d_mm)**2
    Ie    = Icr/(1 - ((2/3)*Mcr/max(Ma,1e-9))**2*(1 - Icr/Ig)) if Ma > (2/3)*Mcr else Ig

    kl_r  = hw_mm / (0.3 * h_mm)
    pmm_ok = plot_pmm(fc_mpa, fy_mpa, h_mm, b_mm, float(As_x.max()),
                      float(np.abs(Ny).max()), float(np.abs(Mx).max()), phi_f, phi_c)
    sbe = ("REQUIRED" if sigma22.max() > 0.2*fc_mpa else "NOT REQ") if is_wall else "N/A"

    punch = {}
    if col_reactions and not is_wall:
        BX_BY = {"I":(2,2),"N":(2,1),"S":(2,1),"E":(1,2),"W":(1,2),
                 "NE":(1,1),"NW":(1,1),"SE":(1,1),"SW":(1,1)}
        for node, rxn in col_reactions.items():
            cp    = col_punch.get(node, {})
            d_in  = d_mm * 0.03937
            Vz_k  = abs(rxn["Vz_kN"]) * 0.2248
            Mx_ki = rxn.get("Mx_kNm", 0.0) * 8.8507
            My_ki = rxn.get("My_kNm", 0.0) * 8.8507
            c1    = cp["col_width_in"] + d_in
            c2    = cp["col_depth_in"] + d_in
            bx, by = BX_BY.get(cp.get("condition","I"), (2,2))
            bo    = bx*c1 + by*c2
            vc    = phi_v * 4 * np.sqrt(fc_mpa * 145.04) / 1000
            Jc    = (d_in/6)*(c1**3 + c2**3 + 2*c1*c2**2 + 2*c2*c1**2)
            gamma = 1 - 1/(1 + (2/3)*np.sqrt(c1/c2))
            ecc   = np.sqrt((Mx_ki/max(Vz_k,0.001))**2 + (My_ki/max(Vz_k,0.001))**2)
            v_avg = Vz_k/(bo*d_in) + gamma*Vz_k*ecc*(c1/2)/max(Jc, 1e-6)
            DC    = v_avg / vc
            punch[node] = {"DC":round(DC,3),"v_max_ksi":round(float(v_avg),4),
                           "phiVc_ksi":round(float(vc),4),"condition":cp.get("condition","I"),
                           "Status":"PASS" if DC <= 1 else "FAIL - THICKEN"}

    sw = {}
    if is_wall:
        z  = 0.9 * d_mm
        rm = 0.0025 if h_mm > 250 else 0.0015
        Nt_s,  Nyt_s2 = clark_nielsen_membrane(Nx/2-Mx*1000/z, Ny/2-My*1000/z, Nxy/2-Mxy*1000/z)
        Nb_s,  Nyb_s2 = clark_nielsen_membrane(Nx/2+Mx*1000/z, Ny/2+My*1000/z, Nxy/2+Mxy*1000/z)
        sw = {
            "As_Vertical_max_mm2m":   round(float(np.maximum((Nyt_s2+Nyb_s2)*1000/(phi_f*fy_mpa), rm*b_mm*h_mm).max()), 1),
            "As_Horizontal_max_mm2m": round(float(np.maximum((Nt_s+Nb_s)*1000/(phi_f*fy_mpa),    rm*b_mm*h_mm).max()), 1),
        }

    As_min_face = rho_min * b_mm * h_mm / 2
    phi_fy = phi_f * fy_mpa
    detailing = {
        "As_Top_X": np.maximum(Nxt_s*1000/phi_fy, As_min_face),
        "As_Top_Y": np.maximum(Nyt_s*1000/phi_fy, As_min_face),
        "As_Bot_X": np.maximum(Nxb_s*1000/phi_fy, As_min_face),
        "As_Bot_Y": np.maximum(Nyb_s*1000/phi_fy, As_min_face),
    }
    report = {
        "One-Way Shear DC":  (round(float(shear_dc.max()),3), "PASS" if shear_dc.max()<1 else "FAIL"),
        "Slenderness kl/r":  (round(kl_r,2), "SLENDER" if kl_r>22 else "PASS"),
        "Boundary Element":  (sbe, "INFO"),
        "Crushing Strain":   (round(float(eps_max.max()),5), "PASS" if eps_max.max()<0.003 else "FAIL"),
        "PMM Check":         ("PASS" if pmm_ok else "FAIL", "PASS" if pmm_ok else "FAIL"),
        "rho_min (%)":       (round(rho_min*100,3), "ACI OK"),
        "Max Crack Spc mm":  (round(s_max,0), "INFO"),
        "Ie / Ig":           (round(Ie/Ig,3), "INFO"),
        **{f"Punching Node {n} (DC)":(r["DC"],r["Status"]) for n,r in punch.items()},
    }
    return report, punch, detailing, sw

# =============================================================================
# HTML SUMMARY REPORT -> summary_report.html
# =============================================================================
def write_html(report, punch, detailing, sw, schedule, meta, out_dir):
    def rows(data, cols, fmt):
        return "".join(fmt(item) for item in data)

    def check_rows():
        out = ""
        for chk,(val,status) in report.items():
            cls = ("pass" if status in("PASS","ACI OK") else "fail" if status in("FAIL","FAIL - THICKEN")
                   else "warn" if status in("SLENDER","REQUIRED") else "info")
            out += f"<tr><td>{chk}</td><td>{val}</td><td><span class='{cls}'>{status}</span></td></tr>\n"
        return out

    def sched_rows():
        out = ""
        for v in schedule.values():
            cls = "pass" if v["utilisation"] <= 1.0 else "fail"
            out += (f"<tr><td>{v['description']}</td><td>{v['As_req_mm2m']}</td>"
                    f"<td><b>{v['designation']}</b></td><td>{v['As_prov_mm2m']}</td>"
                    f"<td><span class='{cls}'>{v['utilisation']:.3f}</span></td></tr>\n")
        return out

    detail_rows = "".join(
        f"<tr><td>{k}</td><td>{v.max():.1f}</td><td>{v.min():.1f}</td><td>{v.mean():.1f}</td></tr>\n"
        for k,v in detailing.items())
    sw_rows = ("".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in sw.items())
               if sw else "<tr><td colspan='2'>N/A — not a shear wall</td></tr>")

    css = ("body{font-family:Arial,sans-serif;font-size:13px;background:#f4f6f8;color:#222;margin:0;padding:24px}"
           "h1{font-size:20px;color:#1a3a5c;border-bottom:3px solid #1a3a5c;padding-bottom:8px}"
           "h2{font-size:14px;color:#1a3a5c;margin-top:28px;border-left:4px solid #1a3a5c;padding-left:8px}"
           ".meta{background:#fff;border:1px solid #dde;border-radius:6px;padding:12px 16px;"
           "margin-bottom:20px;display:flex;flex-wrap:wrap;gap:16px}"
           ".meta span{font-size:12px;color:#555} .meta b{color:#222}"
           "table{width:100%;border-collapse:collapse;background:#fff;border-radius:6px;"
           "overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:20px}"
           "th{background:#1a3a5c;color:#fff;padding:9px 12px;text-align:left;font-size:12px}"
           "td{padding:8px 12px;border-bottom:1px solid #eef}"
           "tr:last-child td{border-bottom:none}"
           ".pass{color:#1a7a40;font-weight:700} .fail{color:#c0392b;font-weight:700}"
           ".warn{color:#d68910;font-weight:700} .info{color:#555}"
           "img{max-width:100%;border:1px solid #dde;border-radius:6px;margin-top:8px}"
           ".grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}"
           "footer{margin-top:32px;font-size:11px;color:#999;border-top:1px solid #dde;padding-top:10px}")

    m = meta
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<title>RCC Shell Design Report</title><style>{css}</style></head><body>
<h1>RCC Shell Element Design Report — ACI 318-19</h1>
<div class="meta">
  <span><b>Element:</b> {m['element_type'].upper()}</span>
  <span><b>f'c:</b> {m['fc_mpa']} MPa</span><span><b>fy:</b> {m['fy_mpa']} MPa</span>
  <span><b>h:</b> {m['h_mm']} mm</span><span><b>Cover:</b> {m['cover_mm']} mm</span>
  <span><b>db:</b> {m['db_mm']} mm</span>
  <span><b>phi_f / phi_v / phi_c:</b> {m['phi_f']} / {m['phi_v']} / {m['phi_c']}</span>
  <span><b>Nodes:</b> {m['N']}</span>
</div>
<h2>Design Checks</h2>
<table><tr><th>Check</th><th>Value</th><th>Status</th></tr>{check_rows()}</table>
<h2>Rebar Schedule</h2>
<table><tr><th>Layer</th><th>As req (mm2/m)</th><th>Bar</th><th>As prov (mm2/m)</th><th>Utilisation</th></tr>
{sched_rows()}</table>
<h2>Detailing Map Statistics</h2>
<table><tr><th>Layer</th><th>Max</th><th>Min</th><th>Mean (mm2/m)</th></tr>{detail_rows}</table>
<h2>Sandwich Wall</h2>
<table><tr><th>Parameter</th><th>Value</th></tr>{sw_rows}</table>
<div class="grid">
  <div><h2>P-M Interaction</h2><img src="pmm_curve.png" alt="PMM"/></div>
  <div><h2>Reinforcement Contours</h2><img src="reinforcement_contours.png" alt="Contours"/></div>
</div>
<footer>Pipeline: Sandwich (raw M) → Clark-Nielsen per layer &nbsp;|&nbsp;
+M sagging: compression top (-M/z), tension bottom (+M/z) &nbsp;|&nbsp; ACI 318-19</footer>
</body></html>"""

    with open(os.path.join(out_dir, "summary_report.html"), "w") as f:
        f.write(html)

# # =============================================================================
# # MAIN
# # =============================================================================
# if __name__ == "__main__":
#     required = {"ELEMENT_TYPE":ELEMENT_TYPE,"OUT_DIR":OUT_DIR,"FC_MPA":FC_MPA,
#                 "FY_MPA":FY_MPA,"H_MM":H_MM,"COVER_MM":COVER_MM,"DB_MM":DB_MM,
#                 "HW_MM":HW_MM,"LW_MM":LW_MM,"PHI_F":PHI_F,"PHI_V":PHI_V,"PHI_C":PHI_C,"B_MM":B_MM}
#     missing = [k for k,v in required.items() if not v]
#     if missing:
#         raise ValueError(f"NOT SET: {missing}")

#     os.makedirs(OUT_DIR, exist_ok=True)
#     forces        = json.loads(FORCES_JSON)
#     ss            = json.loads(STRESS_STRAIN_JSON)
#     col_reactions = json.loads(COL_REACTIONS_JSON)
#     col_punch     = json.loads(COL_PUNCHING_JSON)
#     N             = len(forces["MXX"])

#     report, punch, detailing, sw = run_engine(
#         forces=forces, ss=ss, col_reactions=col_reactions, col_punch=col_punch,
#         element_type=ELEMENT_TYPE, fc_mpa=FC_MPA, fy_mpa=FY_MPA, h_mm=H_MM,
#         cover_mm=COVER_MM, db_mm=DB_MM, hw_mm=HW_MM, lw_mm=LW_MM,
#         phi_f=PHI_F, phi_v=PHI_V, phi_c=PHI_C, rho_min_user=RHO_MIN, b_mm=B_MM)

#     schedule = build_schedule(detailing, H_MM, COVER_MM, PREFERRED_DB)
#     plot_contours(detailing, N)

#     schedule_out = {k:{kk:vv for kk,vv in v.items() if kk!="description"} for k,v in schedule.items()}
#     with open(os.path.join(OUT_DIR, "rebar_schedule.json"), "w") as f:
#         json.dump(schedule_out, f, indent=2)

#     write_html(report, punch, detailing, sw, schedule,
#                {"element_type":ELEMENT_TYPE,"fc_mpa":FC_MPA,"fy_mpa":FY_MPA,"h_mm":H_MM,
#                 "cover_mm":COVER_MM,"db_mm":DB_MM,"phi_f":PHI_F,"phi_v":PHI_V,"phi_c":PHI_C,"N":N}, OUT_DIR)

#     print(f"Done -> {OUT_DIR}")
#     print("  summary_report.html | rebar_schedule.json | pmm_curve.png | reinforcement_contours.png")
