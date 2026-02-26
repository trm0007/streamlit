import matplotlib; matplotlib.use('Agg')
import os, json
import numpy as np
import matplotlib.pyplot as plt
from sectionproperties.pre import Geometry
from sectionproperties.pre.library import rectangular_section, circular_section, circular_section_by_area, i_section, concrete_rectangular_section
from concreteproperties import (Concrete, ConcreteLinear, ConcreteSection, RectangularStressBlock,
    Steel, SteelBar, SteelElasticPlastic, add_bar, add_bar_rectangular_array, add_bar_circular_array)

# ── utilities ─────────────────────────────────────────────────────────────────
def make_serializable(o):
    if isinstance(o,(int,float,str,bool,type(None))): return o
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,(list,tuple)): return [make_serializable(i) for i in o]
    if isinstance(o,dict): return {k:make_serializable(v) for k,v in o.items()}
    if hasattr(o,'__dict__'): return {k:make_serializable(v) for k,v in vars(o).items() if not k.startswith('_')}
    return str(o)

def save_to_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    json.dump(data, open(path,'w'), indent=4); print(f"JSON saved to: {path}"); return path

def save_plot(ax, path, dpi=150):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    ax.get_figure().savefig(path, bbox_inches='tight', dpi=dpi); plt.close(ax.get_figure()); print(f"Plot saved to: {path}"); return path

def _s(obj):  # serialize + strip default_units
    d = make_serializable(obj); d.pop('default_units', None); return d

# ── geometry helpers ──────────────────────────────────────────────────────────
def _create_geometry(t, p, mat):
    if t=="rectangular": return rectangular_section(d=p["d"], b=p["b"], material=mat)
    if t=="circular": return circular_section_by_area(area=p["area"],n=p.get("n",32),material=mat) if "area" in p else circular_section(d=p["d"],n=p.get("n",32),material=mat)
    if t=="i_section": return i_section(d=p["d"],b=p["b"],t_f=p["t_f"],t_w=p["t_w"],r=p["r"],n_r=p.get("n_r",8),material=mat)
    if t=="custom_points": return Geometry.from_points(points=p["points"],facets=p["facets"],control_points=p["control_points"],holes=p.get("holes"),material=mat)
    raise ValueError(f"Unknown geometry type: {t}")

def _align(geom, ref, cfg):
    t=cfg.get("to","center")
    if t=="center": geom=geom.align_center(align_to=ref)
    elif t=="previous": geom=geom.align_to(other=ref,on=cfg.get("on"),inner=cfg.get("inner",False))
    if "offset" in cfg: geom=geom.shift_section(x_offset=cfg["offset"][0],y_offset=cfg["offset"][1])
    return geom

def create_dynamic_concrete_section(base_geometry_type, base_geometry_params, base_material,
        additional_geometries=None, holes=None, reinforcement=None, moment_centroid=None,
        geometric_centroid_override=False, save_plot_path=None, plot_title="Dynamic Concrete Section"):
    g = _create_geometry(base_geometry_type, base_geometry_params, base_material)
    for ag in (additional_geometries or []):
        ng = _create_geometry(ag["type"],ag["params"],ag.get("material",base_material))
        if "align" in ag: ng = _align(ng, g, ag["align"])
        g = g+ng if ag.get("operation","add")=="add" else g-ng
    for h in (holes or []):
        hg = _create_geometry(h["type"],h["params"],base_material)
        if "align" in h: hg = _align(hg, g, h["align"])
        g = g - hg
    _add = {"single_bar":add_bar,"rectangular_array":add_bar_rectangular_array,"circular_array":add_bar_circular_array}
    for rb in (reinforcement or []): g = _add[rb["type"]](geometry=g, material=rb["material"], **rb["params"])
    sec = ConcreteSection(geometry=g, moment_centroid=moment_centroid, geometric_centroid_override=geometric_centroid_override)
    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path) if os.path.dirname(save_plot_path) else '.', exist_ok=True)
        ax=sec.plot_section(title=plot_title,render=False); ax.get_figure().savefig(save_plot_path,bbox_inches='tight',dpi=150); plt.close(ax.get_figure()); print(f"Section plot saved to: {save_plot_path}")
    return sec, [save_plot_path] if save_plot_path else []

# ── analysis functions ────────────────────────────────────────────────────────
def save_properties_to_json(cs, gross_path="gross.json", transformed_path="transformed.json"):
    return [save_to_json(_s(cs.get_gross_properties()), gross_path),
            save_to_json(_s(cs.get_transformed_gross_properties(elastic_modulus=30.1e3)), transformed_path)]

def calculate_and_save_cracked_properties(cs, theta_sag=0, theta_hog=np.pi, elastic_modulus=30.1e3,
        save_json_sag="cs.json", save_json_hog="ch.json", save_plot_sag="cs.png", save_plot_hog="ch.png"):
    rs,rh = cs.calculate_cracked_properties(theta=theta_sag), cs.calculate_cracked_properties(theta=theta_hog)
    rs.calculate_transformed_properties(elastic_modulus=elastic_modulus); rh.calculate_transformed_properties(elastic_modulus=elastic_modulus)
    files=[]
    for r,jp,pp in [(rs,save_json_sag,save_plot_sag),(rh,save_json_hog,save_plot_hog)]:
        save_to_json(_s(r),jp); files.append(jp)
        save_plot(r.plot_cracked_geometries(labels=[],cp=False,legend=False,render=False),pp); files.append(pp)
    return rs, rh, files

def moment_curvature_analysis(cs, theta=0, n=0, kappa_inc=1e-6, kappa_mult=2, kappa_inc_max=5e-6,
        delta_m_min=0.15, delta_m_max=0.3, save_json=None, save_plot_path=None):
    r=cs.moment_curvature_analysis(theta=theta,n=n,kappa_inc=kappa_inc,kappa_mult=kappa_mult,kappa_inc_max=kappa_inc_max,delta_m_min=delta_m_min,delta_m_max=delta_m_max,progress_bar=False); files=[]
    if save_json:
        d=_s(r); d['summary']={'number_of_calculations':len(r.kappa),'failure_curvature':r.kappa[-1] if r.kappa else None,'max_moment':max(r.m_xy) if r.m_xy else None}
        save_to_json(d,save_json); files.append(save_json)
    if save_plot_path: save_plot(r.plot_results(render=False),save_plot_path); files.append(save_plot_path)
    return r, files

def multiple_moment_curvature_analysis(cs, analysis_cases, save_json=None, save_plot_path=None, labels=None):
    from concreteproperties.results import MomentCurvatureResults
    defs={"theta":0,"n":0,"kappa_inc":1e-6,"kappa_mult":2,"kappa_inc_max":5e-6,"delta_m_min":0.15,"delta_m_max":0.3}
    results=[cs.moment_curvature_analysis(**{**defs,**c},progress_bar=False) for c in analysis_cases]; files=[]
    if save_json:
        cd={}
        for i,r in enumerate(results):
            d=_s(r); d['summary']={'number_of_calculations':len(r.kappa),'failure_curvature':r.kappa[-1] if r.kappa else None,'max_moment':max(r.m_xy) if r.m_xy else None}; cd[labels[i] if labels else f"case_{i}"]=d
        save_to_json(cd,save_json); files.append(save_json)
    if save_plot_path: save_plot(MomentCurvatureResults.plot_multiple_results(moment_curvature_results=results,labels=labels,fmt="-",render=False),save_plot_path); files.append(save_plot_path)
    return results, files

def ultimate_bending_capacity(cs, theta=0, n=0, save_json=None):
    r=cs.ultimate_bending_capacity(theta=theta,n=n); files=[]
    if save_json:
        d=_s(r); d['summary']={'bending_angle_deg':np.degrees(theta),'axial_force':n,'resultant_moment':r.m_xy,'neutral_axis_depth':r.d_n,'k_u':r.k_u}
        save_to_json(d,save_json); files.append(save_json)
    return r, files

def multiple_ultimate_bending_capacities(cs, analysis_cases, save_json=None, labels=None):
    results=[cs.ultimate_bending_capacity(**{"theta":0,"n":0,**c}) for c in analysis_cases]
    if save_json:
        cd={}
        for i,r in enumerate(results):
            d=_s(r); d['summary']={'bending_angle_deg':np.degrees(analysis_cases[i].get("theta",0)),'axial_force':analysis_cases[i].get("n",0),'resultant_moment':r.m_xy,'neutral_axis_depth':r.d_n,'k_u':r.k_u}; cd[labels[i] if labels else f"case_{i}"]=d
        save_to_json(cd,save_json)
    return results

def moment_interaction_diagram(cs, theta=0, limits=None, control_points=None, labels=None, n_points=24,
        n_spacing=None, max_comp=None, max_comp_labels=None, save_json=None, save_plot_path=None, plot_moment="m_xy", plot_labels=False, label_offset=False):
    r=cs.moment_interaction_diagram(theta=theta,limits=limits,control_points=control_points,labels=labels,n_points=n_points,n_spacing=n_spacing,max_comp=max_comp,max_comp_labels=max_comp_labels,progress_bar=False); files=[]
    if save_json:
        d=_s(r); d['summary']={'number_of_points':len(r.results),'bending_angle_deg':np.degrees(theta),'max_axial_force':max(x.n for x in r.results) if r.results else None,'min_axial_force':min(x.n for x in r.results) if r.results else None,'max_moment':max(x.m_xy for x in r.results) if r.results else None}
        save_to_json(d,save_json); files.append(save_json)
    if save_plot_path: save_plot(r.plot_diagram(moment=plot_moment,labels=plot_labels,label_offset=label_offset,render=False),save_plot_path); files.append(save_plot_path)
    return r, files

def multiple_moment_interaction_diagrams(sections_or_results, labels=None, save_json=None, save_plot_path=None, plot_moment="m_xy", theta=0):
    from concreteproperties.results import MomentInteractionResults
    mi=[s.moment_interaction_diagram(theta=theta,progress_bar=False) for s in sections_or_results] if hasattr(sections_or_results[0],'moment_interaction_diagram') else sections_or_results; files=[]
    if save_json:
        cd={}
        for i,r in enumerate(mi):
            d=_s(r); d['summary']={'number_of_points':len(r.results),'max_axial_force':max(x.n for x in r.results) if r.results else None,'min_axial_force':min(x.n for x in r.results) if r.results else None,'max_moment':max(x.m_xy for x in r.results) if r.results else None}; cd[labels[i] if labels else f"diagram_{i}"]=d
        save_to_json(cd,save_json); files.append(save_json)
    if save_plot_path: save_plot(MomentInteractionResults.plot_multiple_diagrams(moment_interaction_results=mi,labels=labels,fmt="-",moment=plot_moment,render=False),save_plot_path); files.append(save_plot_path)
    return mi, files

def positive_negative_interaction_diagrams(cs, save_json=None, save_plot_path=None):
    from concreteproperties.results import MomentInteractionResults
    rp=cs.moment_interaction_diagram(theta=0,progress_bar=False); rn=cs.moment_interaction_diagram(theta=np.pi,progress_bar=False); files=[]
    if save_json:
        cd={'positive_bending':_s(rp),'negative_bending':_s(rn)}
        for k,r in [('positive_bending',rp),('negative_bending',rn)]: cd[k]['summary']={'number_of_points':len(r.results),'max_axial_force':max(x.n for x in r.results) if r.results else None,'min_axial_force':min(x.n for x in r.results) if r.results else None,'max_moment':max(x.m_xy for x in r.results) if r.results else None}
        save_to_json(cd,save_json); files.append(save_json)
    if save_plot_path: save_plot(MomentInteractionResults.plot_multiple_diagrams(moment_interaction_results=[rp,rn],labels=["Positive","Negative"],fmt="-",render=False),save_plot_path); files.append(save_plot_path)
    return rp, rn, files

def advanced_moment_interaction_diagram(cs, theta=0, save_json=None, save_plot_path=None):
    r=cs.moment_interaction_diagram(theta=theta,limits=[("kappa0",0.0),("d_n",1e-6)],control_points=[("D",1.0),("fy",0.0),("fy",0.5),("fy",1.0),("d_n",200.0),("N",0.0)],labels=["NA","I","C","D","E","F","G","H"],n_spacing=36,progress_bar=False); files=[]
    if save_json: d=_s(r); d['summary']={'number_of_points':len(r.results),'bending_angle_deg':np.degrees(theta),'control_points':['Decompression','Steel Decompression','50% Yield','100% Yield','d_n=200mm','Pure Bending']}; save_to_json(d,save_json); files.append(save_json)
    if save_plot_path: ax=r.plot_diagram(fmt="-kx",labels=True,label_offset=True,render=False); ax.set_xlim(-20,850); ax.set_ylim(-3000,9000); save_plot(ax,save_plot_path); files.append(save_plot_path)
    return r, files

def biaxial_bending_diagram(cs, n=0, n_points=48, save_json=None, save_plot_path=None):
    r=cs.biaxial_bending_diagram(n=n,n_points=n_points,progress_bar=False); files=[]
    if save_json: d=_s(r); d['summary']={'axial_force':n,'number_of_points':len(r.results),'max_m_x':max(abs(x.m_x) for x in r.results) if r.results else None,'max_m_y':max(abs(x.m_y) for x in r.results) if r.results else None}; save_to_json(d,save_json); files.append(save_json)
    if save_plot_path: save_plot(r.plot_diagram(render=False),save_plot_path); files.append(save_plot_path)
    return r, files

def multiple_biaxial_bending_diagrams(cs, axial_forces, n_points=24, save_json=None, save_plot_2d_path=None, save_plot_3d_path=None, labels=None):
    from concreteproperties.results import BiaxialBendingResults
    results=[cs.biaxial_bending_diagram(n=n,n_points=n_points,progress_bar=False) for n in axial_forces]; files=[]
    if save_json:
        cd={}
        for i,r in enumerate(results): d=_s(r); d['summary']={'axial_force':axial_forces[i],'number_of_points':len(r.results),'max_m_x':max(abs(x.m_x) for x in r.results) if r.results else None,'max_m_y':max(abs(x.m_y) for x in r.results) if r.results else None}; cd[labels[i] if labels else f"N_{int(axial_forces[i]/1e3)}_kN"]=d
        save_to_json(cd,save_json); files.append(save_json)
    if save_plot_2d_path: save_plot(BiaxialBendingResults.plot_multiple_diagrams_2d(results,fmt="o-",labels=labels,render=False),save_plot_2d_path); files.append(save_plot_2d_path)
    if save_plot_3d_path: save_plot(BiaxialBendingResults.plot_multiple_diagrams_3d(results),save_plot_3d_path); files.append(save_plot_3d_path)
    return results, files

def find_decompression_point(cs):
    mx=cs.moment_interaction_diagram(theta=0,progress_bar=False); my=cs.moment_interaction_diagram(theta=np.pi/2,progress_bar=False)
    return (mx.results[1].n if len(mx.results)>1 else None),(my.results[1].n if len(my.results)>1 else None)

def biaxial_bending_analysis_suite(cs, axial_forces=None, n_points=24, save_dir="output/biaxial"):
    print("Finding decompression points..."); ndx,ndy=find_decompression_point(cs)
    print(f"Decompression point for M_x: N = {ndx/1e3:.2f} kN\nDecompression point for M_y: N = {ndy/1e3:.2f} kN")
    if axial_forces is None: axial_forces=np.linspace(0,min(ndx,ndy)*0.99,5).tolist()
    labels=[f"N = {n/1e3:.0f} kN" for n in axial_forces]
    print(f"\nGenerating {len(axial_forces)} biaxial bending diagrams...")
    bb=multiple_biaxial_bending_diagrams(cs,axial_forces=axial_forces,n_points=n_points,save_json=f"{save_dir}/biaxial_multiple.json",save_plot_2d_path=f"{save_dir}/biaxial_2d.png",save_plot_3d_path=f"{save_dir}/biaxial_3d.png",labels=labels)
    print("\nGenerating individual biaxial diagrams...")
    bp=biaxial_bending_diagram(cs,n=0,n_points=n_points,save_json=f"{save_dir}/biaxial_n0.json",save_plot_path=f"{save_dir}/biaxial_n0.png")
    ba=biaxial_bending_diagram(cs,n=axial_forces[len(axial_forces)//2],n_points=n_points,save_json=f"{save_dir}/biaxial_with_axial.json",save_plot_path=f"{save_dir}/biaxial_with_axial.png") if len(axial_forces)>1 else None
    return {'decompression_mx':ndx,'decompression_my':ndy,'axial_forces':axial_forces,'labels':labels,'multiple_results':bb,'pure_bending':bp,'with_axial':ba,
            'saved_files':[f"{save_dir}/biaxial_multiple.json",f"{save_dir}/biaxial_2d.png",f"{save_dir}/biaxial_3d.png",f"{save_dir}/biaxial_n0.json",f"{save_dir}/biaxial_n0.png"]}

def _stress_save(r, save_json, save_plot_path, extra=None):
    files=[]; fo=r.sum_forces(); mo=r.sum_moments()
    if save_json:
        d=_s(r); d['summary']={**(extra or {}),'sum_forces_n':fo,'sum_moments_m_x':mo[0],'sum_moments_m_y':mo[1],'sum_moments_m_xy':mo[2]}; save_to_json(d,save_json); files.append(save_json)
    if save_plot_path: save_plot(r.plot_stress(render=False),save_plot_path); files.append(save_plot_path)
    return files

def calculate_uncracked_stress(cs, n=0, m_x=0, m_y=0, save_json=None, save_plot_path=None):
    r=cs.calculate_uncracked_stress(n=n,m_x=m_x,m_y=m_y); return r, _stress_save(r,save_json,save_plot_path,{'applied_n':n,'applied_m_x':m_x,'applied_m_y':m_y})

def calculate_cracked_stress(cs, cracked_results, m, save_json=None, save_plot_path=None):
    r=cs.calculate_cracked_stress(cracked_results=cracked_results,m=m); return r, _stress_save(r,save_json,save_plot_path,{'applied_moment':m,'cracking_moment':cracked_results.m_cr})

def calculate_service_stress(cs, moment_curvature_results, m=None, kappa=None, save_json=None, save_plot_path=None):
    r=cs.calculate_service_stress(moment_curvature_results=moment_curvature_results,m=m,kappa=kappa); return r, _stress_save(r,save_json,save_plot_path,{'applied_moment':m,'applied_curvature':kappa})

def calculate_ultimate_stress(cs, ultimate_results, save_json=None, save_plot_path=None):
    r=cs.calculate_ultimate_stress(ultimate_results=ultimate_results); return r, _stress_save(r,save_json,save_plot_path,{'neutral_axis_depth':ultimate_results.d_n,'k_u':ultimate_results.k_u})

def multiple_service_stress_analysis(cs, moment_curvature_results, stress_points, save_dir="output/service_stress"):
    results,files=[],[]
    for i,pt in enumerate(stress_points):
        mv,kv=(pt["val"],None) if pt["m_or_k"]=="m" else (None,pt["val"])
        r=cs.calculate_service_stress(moment_curvature_results=moment_curvature_results,m=mv,kappa=kv)
        ma=r.sum_moments()[2] if kv is not None else mv; jp,pp=f"{save_dir}/service_stress_{i+1}.json",f"{save_dir}/service_stress_{i+1}.png"
        d=_s(r); fo=r.sum_forces(); mo=r.sum_moments(); d['summary']={'point_number':i+1,'applied_moment':ma,'applied_curvature':kv,'sum_forces_n':fo,'sum_moments_m_xy':mo[2]}
        save_to_json(d,jp); save_plot(r.plot_stress(title=f"M = {ma/1e6:.0f} kN.m",render=False),pp); print(f"Service stress point {i+1}: M = {ma/1e6:.2f} kN.m"); files+=[jp,pp]; results.append(r)
    return results, files

def multiple_ultimate_stress_analysis(cs, ultimate_cases, labels=None, save_dir="output/ultimate_stress"):
    results,files=[],[]
    for i,case in enumerate(ultimate_cases):
        lb=labels[i] if labels else f"case_{i+1}"; ur=cs.ultimate_bending_capacity(theta=case.get("theta",0),n=case.get("n",0)); r=cs.calculate_ultimate_stress(ultimate_results=ur)
        jp,pp=f"{save_dir}/ultimate_stress_{lb}.json",f"{save_dir}/ultimate_stress_{lb}.png"
        d=_s(r); fo=r.sum_forces(); mo=r.sum_moments(); d['summary']={'label':lb,'axial_force':case.get("n",0),'bending_angle_deg':np.degrees(case.get("theta",0)),'neutral_axis_depth':ur.d_n,'sum_forces_n':fo,'sum_moments_m_xy':mo[2]}
        save_to_json(d,jp); save_plot(r.plot_stress(title=lb,render=False),pp); print(f"Ultimate stress - {lb}: N = {case.get('n',0)/1e3:.0f} kN, M = {ur.m_xy/1e6:.2f} kN.m"); files+=[jp,pp]; results.append(r)
    return results, files

def plot_stress_strain_profiles(materials, material_names=None, save_dir="output/stress_strain"):
    files=[]
    for i,m in enumerate(materials):
        nm=material_names[i] if material_names else m.name; p=f"{save_dir}/{nm.replace(' ','_').replace('/','_')}.png"
        save_plot(m.stress_strain_profile.plot_stress_strain(title=nm,eng=True,render=False),p); files.append(p)
    return files

def compare_moment_curvature_parameters(cs, parameter_sets, save_json=None, save_plot_path=None, labels=None):
    from concreteproperties.results import MomentCurvatureResults
    results,stats,files=[],{},[]
    for i,p in enumerate(parameter_sets):
        lb=labels[i] if labels else f"params_{i}"
        r=cs.moment_curvature_analysis(theta=p.get("theta",0),n=p.get("n",0),kappa_inc=p.get("kappa_inc",1e-7),kappa_mult=p.get("kappa_mult",2),kappa_inc_max=p.get("kappa_inc_max",5e-6),delta_m_min=p.get("delta_m_min",0.15),delta_m_max=p.get("delta_m_max",0.3),progress_bar=False)
        results.append(r); stats[lb]={'number_of_calculations':len(r.kappa),'failure_curvature':r.kappa[-1],'max_moment':max(r.m_xy),'parameters':p}
    if save_json: save_to_json(stats,save_json); files.append(save_json)
    if save_plot_path: save_plot(MomentCurvatureResults.plot_multiple_results(moment_curvature_results=results,labels=labels,fmt="-",render=False),save_plot_path); files.append(save_plot_path)
    return results, stats, files

def compare_ultimate_capacities_with_axial_load(cs, bending_angles, axial_loads, save_json=None, angle_labels=None):
    results={}
    for n in axial_loads:
        nk=f"N_{int(n/1e3)}_kN" if n!=0 else "N_0_kN"; results[nk]={}
        for i,theta in enumerate(bending_angles):
            ak=angle_labels[i] if angle_labels else f"theta_{np.degrees(theta):.1f}_deg"; r=cs.ultimate_bending_capacity(theta=theta,n=n)
            results[nk][ak]={'theta_rad':theta,'theta_deg':np.degrees(theta),'n':r.n,'d_n':r.d_n,'k_u':r.k_u,'m_x':r.m_x,'m_y':r.m_y,'m_xy':r.m_xy}
    if save_json: save_to_json(results,save_json)
    return results

def plot_initial_cracking_region(mk_result, n_points=12, save_plot_path=None):
    fig,ax=plt.subplots(); ax.plot(np.array(mk_result.kappa)[:n_points],np.array(mk_result.m_xy)[:n_points]/1e6,"x-")
    ax.set_xlabel("Curvature [-]"); ax.set_ylabel("Bending Moment [kN.m]"); ax.set_title("Initial Cracking Region"); ax.grid(True,alpha=0.3)
    if save_plot_path: save_plot(ax,save_plot_path)
    return ax

def plot_and_save_concrete_section(depth,width,dia_top,area_top,n_top,cover_top,dia_bot,area_bot,n_bot,cover_bot,
        concrete_name="Concrete",concrete_density=2.4e-6,elastic_modulus=30.1e3,compressive_strength=32,alpha=0.802,gamma=0.89,ultimate_strain=0.003,flexural_tensile_strength=3.4,concrete_colour="lightgrey",
        steel_name="Steel",steel_density=7.85e-6,yield_strength=500,steel_elastic_modulus=200e3,fracture_strain=0.05,steel_colour="grey",save_path="section_plot.png",plot_title="Reinforced Concrete Section"):
    conc=Concrete(name=concrete_name,density=concrete_density,stress_strain_profile=ConcreteLinear(elastic_modulus=elastic_modulus),ultimate_stress_strain_profile=RectangularStressBlock(compressive_strength=compressive_strength,alpha=alpha,gamma=gamma,ultimate_strain=ultimate_strain),flexural_tensile_strength=flexural_tensile_strength,colour=concrete_colour)
    stl=SteelBar(name=steel_name,density=steel_density,stress_strain_profile=SteelElasticPlastic(yield_strength=yield_strength,elastic_modulus=steel_elastic_modulus,fracture_strain=fracture_strain),colour=steel_colour)
    sec=ConcreteSection(concrete_rectangular_section(d=depth,b=width,dia_top=dia_top,area_top=area_top,n_top=n_top,c_top=cover_top,dia_bot=dia_bot,area_bot=area_bot,n_bot=n_bot,c_bot=cover_bot,conc_mat=conc,steel_mat=stl))
    if os.path.dirname(save_path): os.makedirs(os.path.dirname(save_path),exist_ok=True)
    ax=sec.plot_section(title=plot_title,render=False); ax.get_figure().savefig(save_path,bbox_inches='tight',dpi=150); plt.close(ax.get_figure()); print(f"Section plot saved to: {save_path}")
    return sec, [save_path]

# ── master runner ─────────────────────────────────────────────────────────────
def run_concrete_analysis(base_geometry_type, base_geometry_params, base_material,
        additional_geometries=None, holes=None, reinforcement=None, moment_centroid=None, geometric_centroid_override=False,
        run_gross_properties=False, run_cracked_properties=False, run_moment_curvature=False, run_ultimate_bending=False,
        run_moment_interaction=False, run_positive_negative_interaction=False, run_advanced_interaction=False, run_multiple_interaction=False,
        run_biaxial_bending=False, run_multiple_biaxial=False, run_biaxial_suite=False,
        run_uncracked_stress=False, run_cracked_stress=False, run_service_stress=False, run_ultimate_stress=False,
        run_multiple_service_stress=False, run_multiple_ultimate_stress=False, run_multiple_moment_curvature=False,
        run_multiple_ultimate_bending=False, run_stress_strain_profiles=False,
        elastic_modulus=None, theta_sag=None, theta_hog=None,
        mk_theta=None, mk_n=None, mk_kappa_inc=None, mk_kappa_mult=None, mk_kappa_inc_max=None, mk_delta_m_min=None, mk_delta_m_max=None,
        multiple_mk_cases=None, multiple_mk_labels=None, ult_theta=None, ult_n=None, multiple_ult_cases=None, multiple_ult_labels=None,
        mi_theta=None, mi_n_points=None, multiple_mi_sections=None, multiple_mi_labels=None,
        bb_n=None, bb_n_points=None, bb_axial_forces=None, bb_multiple_n_points=None, bb_labels=None, bb_suite_axial_forces=None, bb_suite_n_points=None,
        stress_n=None, stress_m_x=None, stress_m_y=None, cracked_stress_m=None, service_stress_m=None, service_stress_kappa=None,
        multiple_service_stress_points=None, multiple_ultimate_stress_cases=None, multiple_ultimate_stress_labels=None,
        stress_strain_materials=None, stress_strain_names=None, save_dir="output"):
    af, res = [], {}
    sec,files=create_dynamic_concrete_section(base_geometry_type=base_geometry_type,base_geometry_params=base_geometry_params,base_material=base_material,additional_geometries=additional_geometries,holes=holes,reinforcement=reinforcement,moment_centroid=moment_centroid,geometric_centroid_override=geometric_centroid_override,save_plot_path=f"{save_dir}/section_plot.png")
    af.extend(files); res['section']=sec
    if run_gross_properties: af.extend(save_properties_to_json(sec,gross_path=f"{save_dir}/gross_properties.json",transformed_path=f"{save_dir}/transformed_properties.json"))
    crs=crh=None
    if run_cracked_properties:
        crs,crh,files=calculate_and_save_cracked_properties(sec,theta_sag=theta_sag,theta_hog=theta_hog,elastic_modulus=elastic_modulus,save_json_sag=f"{save_dir}/cracked_sag.json",save_json_hog=f"{save_dir}/cracked_hog.json",save_plot_sag=f"{save_dir}/cracked_sag.png",save_plot_hog=f"{save_dir}/cracked_hog.png")
        af.extend(files); res['cracked_sag']=crs; res['cracked_hog']=crh
    if run_stress_strain_profiles: af.extend(plot_stress_strain_profiles(materials=stress_strain_materials,material_names=stress_strain_names,save_dir=f"{save_dir}/stress_strain"))
    mkr=None
    if run_moment_curvature:
        mkr,files=moment_curvature_analysis(sec,theta=mk_theta,n=mk_n,kappa_inc=mk_kappa_inc,kappa_mult=mk_kappa_mult,kappa_inc_max=mk_kappa_inc_max,delta_m_min=mk_delta_m_min,delta_m_max=mk_delta_m_max,save_json=f"{save_dir}/moment_curvature.json",save_plot_path=f"{save_dir}/moment_curvature.png")
        af.extend(files); res['moment_curvature']=mkr
    if run_multiple_moment_curvature:
        r,files=multiple_moment_curvature_analysis(sec,analysis_cases=multiple_mk_cases,labels=multiple_mk_labels,save_json=f"{save_dir}/moment_curvature_multiple.json",save_plot_path=f"{save_dir}/moment_curvature_multiple.png"); af.extend(files); res['moment_curvature_multiple']=r
    ultr=None
    if run_ultimate_bending:
        ultr,files=ultimate_bending_capacity(sec,theta=ult_theta,n=ult_n,save_json=f"{save_dir}/ultimate_bending.json"); af.extend(files); res['ultimate_bending']=ultr
    if run_multiple_ultimate_bending:
        r=multiple_ultimate_bending_capacities(sec,analysis_cases=multiple_ult_cases,labels=multiple_ult_labels,save_json=f"{save_dir}/ultimate_bending_multiple.json"); af.append(f"{save_dir}/ultimate_bending_multiple.json"); res['ultimate_bending_multiple']=r
    if run_moment_interaction:
        r,files=moment_interaction_diagram(sec,theta=mi_theta,n_points=mi_n_points,save_json=f"{save_dir}/moment_interaction.json",save_plot_path=f"{save_dir}/moment_interaction.png"); af.extend(files); res['moment_interaction']=r
    if run_positive_negative_interaction:
        rp,rn,files=positive_negative_interaction_diagrams(sec,save_json=f"{save_dir}/moment_interaction_pos_neg.json",save_plot_path=f"{save_dir}/moment_interaction_pos_neg.png"); af.extend(files); res['moment_interaction_pos']=rp; res['moment_interaction_neg']=rn
    if run_advanced_interaction:
        r,files=advanced_moment_interaction_diagram(sec,theta=mi_theta,save_json=f"{save_dir}/moment_interaction_advanced.json",save_plot_path=f"{save_dir}/moment_interaction_advanced.png"); af.extend(files); res['moment_interaction_advanced']=r
    if run_multiple_interaction:
        r,files=multiple_moment_interaction_diagrams(sections_or_results=multiple_mi_sections,labels=multiple_mi_labels,save_json=f"{save_dir}/moment_interaction_multiple.json",save_plot_path=f"{save_dir}/moment_interaction_multiple.png"); af.extend(files); res['moment_interaction_multiple']=r
    if run_biaxial_bending:
        r,files=biaxial_bending_diagram(sec,n=bb_n,n_points=bb_n_points,save_json=f"{save_dir}/biaxial_bending.json",save_plot_path=f"{save_dir}/biaxial_bending.png"); af.extend(files); res['biaxial_bending']=r
    if run_multiple_biaxial:
        r,files=multiple_biaxial_bending_diagrams(sec,axial_forces=bb_axial_forces,n_points=bb_multiple_n_points,labels=bb_labels,save_json=f"{save_dir}/biaxial_multiple.json",save_plot_2d_path=f"{save_dir}/biaxial_multiple_2d.png",save_plot_3d_path=f"{save_dir}/biaxial_multiple_3d.png"); af.extend(files); res['biaxial_multiple']=r
    if run_biaxial_suite:
        r=biaxial_bending_analysis_suite(sec,axial_forces=bb_suite_axial_forces,n_points=bb_suite_n_points,save_dir=f"{save_dir}/biaxial_suite"); af.extend(r.get('saved_files',[])); res['biaxial_suite']=r
    if run_uncracked_stress:
        r,files=calculate_uncracked_stress(sec,n=stress_n,m_x=stress_m_x,m_y=stress_m_y,save_json=f"{save_dir}/stress_uncracked.json",save_plot_path=f"{save_dir}/stress_uncracked.png"); af.extend(files); res['stress_uncracked']=r
    if run_cracked_stress:
        r,files=calculate_cracked_stress(sec,cracked_results=crs,m=cracked_stress_m,save_json=f"{save_dir}/stress_cracked.json",save_plot_path=f"{save_dir}/stress_cracked.png"); af.extend(files); res['stress_cracked']=r
    if run_service_stress:
        r,files=calculate_service_stress(sec,moment_curvature_results=mkr,m=service_stress_m,kappa=service_stress_kappa,save_json=f"{save_dir}/stress_service.json",save_plot_path=f"{save_dir}/stress_service.png"); af.extend(files); res['stress_service']=r
    if run_ultimate_stress:
        r,files=calculate_ultimate_stress(sec,ultimate_results=ultr,save_json=f"{save_dir}/stress_ultimate.json",save_plot_path=f"{save_dir}/stress_ultimate.png"); af.extend(files); res['stress_ultimate']=r
    if run_multiple_service_stress:
        r,files=multiple_service_stress_analysis(sec,moment_curvature_results=mkr,stress_points=multiple_service_stress_points,save_dir=f"{save_dir}/stress_service_multiple"); af.extend(files); res['stress_service_multiple']=r
    if run_multiple_ultimate_stress:
        r,files=multiple_ultimate_stress_analysis(sec,ultimate_cases=multiple_ultimate_stress_cases,labels=multiple_ultimate_stress_labels,save_dir=f"{save_dir}/stress_ultimate_multiple"); af.extend(files); res['stress_ultimate_multiple']=r
    print(f"\n{'='*60}\nANALYSIS COMPLETE — {len(af)} files saved\n{'='*60}")
    for f in af: print(f"  {f}")
    return {'section':sec,'results':res,'saved_files':af}


# # =============================================================================
# # MATERIALS
# # =============================================================================
# concrete = Concrete(name="32 MPa Concrete", density=2.4e-6, stress_strain_profile=ConcreteLinear(elastic_modulus=30.1e3),
#     ultimate_stress_strain_profile=RectangularStressBlock(compressive_strength=32, alpha=0.802, gamma=0.89, ultimate_strain=0.003),
#     flexural_tensile_strength=3.4, colour="lightgrey")
# steel = SteelBar(name="500 MPa Steel", density=7.85e-6, stress_strain_profile=SteelElasticPlastic(yield_strength=500, elastic_modulus=200e3, fracture_strain=0.05), colour="grey")

# # =============================================================================
# # COMMON SECTION ARGS
# # =============================================================================
# section_args = dict(base_geometry_type="rectangular", base_geometry_params={"d": 500, "b": 300}, base_material=concrete,
#     reinforcement=[{"type": "rectangular_array", "material": steel, "params": {"area": 314, "n_x": 3, "x_s": 100, "n_y": 2, "y_s": 420, "anchor": (40, 40)}}])

# # =============================================================================
# # 1. GROSS PROPERTIES
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_gross_properties=True, elastic_modulus=30.1e3, save_dir="output/01_gross_properties")

# # =============================================================================
# # 2. CRACKED PROPERTIES
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_cracked_properties=True, elastic_modulus=30.1e3, theta_sag=0, theta_hog=np.pi, save_dir="output/02_cracked_properties")

# # =============================================================================
# # 3. MOMENT CURVATURE
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_moment_curvature=True, mk_theta=0, mk_n=0, mk_kappa_inc=1e-6, mk_kappa_mult=2, mk_kappa_inc_max=5e-6, mk_delta_m_min=0.15, mk_delta_m_max=0.3, save_dir="output/03_moment_curvature")

# # =============================================================================
# # 4. MULTIPLE MOMENT CURVATURE
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_multiple_moment_curvature=True,
#     multiple_mk_cases=[{"n": 0, "kappa_inc": 1e-6}, {"n": 500e3, "kappa_inc": 1e-6}, {"n": -500e3, "kappa_inc": 1e-6}],
#     multiple_mk_labels=["N=0", "N=500kN", "N=-500kN"], save_dir="output/04_moment_curvature_multiple")

# # =============================================================================
# # 5. ULTIMATE BENDING
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_ultimate_bending=True, ult_theta=0, ult_n=0, save_dir="output/05_ultimate_bending")

# # =============================================================================
# # 6. MULTIPLE ULTIMATE BENDING
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_multiple_ultimate_bending=True,
#     multiple_ult_cases=[{"theta": 0, "n": 0}, {"theta": np.pi, "n": 0}, {"theta": np.pi/2, "n": 0}],
#     multiple_ult_labels=["Sagging", "Hogging", "Weak Axis"], save_dir="output/06_ultimate_bending_multiple")

# # =============================================================================
# # 7. MOMENT INTERACTION
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_moment_interaction=True, mi_theta=0, mi_n_points=24, save_dir="output/07_moment_interaction")

# # =============================================================================
# # 8. POSITIVE & NEGATIVE INTERACTION
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_positive_negative_interaction=True, save_dir="output/08_moment_interaction_pos_neg")

# # =============================================================================
# # 9. ADVANCED INTERACTION
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_advanced_interaction=True, mi_theta=0, save_dir="output/09_moment_interaction_advanced")

# # =============================================================================
# # 10. MULTIPLE INTERACTION
# # =============================================================================
# sec1, _ = create_dynamic_concrete_section(base_geometry_type="rectangular", base_geometry_params={"d": 500, "b": 300}, base_material=concrete,
#     reinforcement=[{"type": "rectangular_array", "material": steel, "params": {"area": 314, "n_x": 3, "x_s": 100, "n_y": 2, "y_s": 420, "anchor": (40, 40)}}])
# sec2, _ = create_dynamic_concrete_section(base_geometry_type="rectangular", base_geometry_params={"d": 600, "b": 350}, base_material=concrete,
#     reinforcement=[{"type": "rectangular_array", "material": steel, "params": {"area": 314, "n_x": 4, "x_s": 90, "n_y": 2, "y_s": 520, "anchor": (40, 40)}}])
# output = run_concrete_analysis(**section_args, run_multiple_interaction=True, multiple_mi_sections=[sec1, sec2], multiple_mi_labels=["300x500", "350x600"], save_dir="output/10_moment_interaction_multiple")

# # =============================================================================
# # 11. BIAXIAL BENDING
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_biaxial_bending=True, bb_n=0, bb_n_points=48, save_dir="output/11_biaxial_bending")

# # =============================================================================
# # 12. MULTIPLE BIAXIAL
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_multiple_biaxial=True, bb_axial_forces=[0, 500e3, 1000e3, 2000e3], bb_multiple_n_points=24,
#     bb_labels=["N=0", "N=500kN", "N=1000kN", "N=2000kN"], save_dir="output/12_biaxial_multiple")

# # =============================================================================
# # 13. BIAXIAL SUITE
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_biaxial_suite=True, bb_suite_axial_forces=[0, 500e3, 1000e3], bb_suite_n_points=24, save_dir="output/13_biaxial_suite")

# # =============================================================================
# # 14. UNCRACKED STRESS
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_uncracked_stress=True, stress_n=0, stress_m_x=50e6, stress_m_y=0, save_dir="output/14_stress_uncracked")

# # =============================================================================
# # 15. CRACKED STRESS
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_cracked_properties=True, run_cracked_stress=True, elastic_modulus=30.1e3, theta_sag=0, theta_hog=np.pi, cracked_stress_m=100e6, save_dir="output/15_stress_cracked")

# # =============================================================================
# # 16. SERVICE STRESS
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_moment_curvature=True, run_service_stress=True, mk_theta=0, mk_n=0, mk_kappa_inc=1e-6, mk_kappa_mult=2, mk_kappa_inc_max=5e-6, mk_delta_m_min=0.15, mk_delta_m_max=0.3, service_stress_m=100e6, save_dir="output/16_stress_service")

# # =============================================================================
# # 17. ULTIMATE STRESS
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_ultimate_bending=True, run_ultimate_stress=True, ult_theta=0, ult_n=0, save_dir="output/17_stress_ultimate")

# # =============================================================================
# # 18. MULTIPLE SERVICE STRESS
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_moment_curvature=True, run_multiple_service_stress=True, mk_theta=0, mk_n=0, mk_kappa_inc=1e-6, mk_kappa_mult=2, mk_kappa_inc_max=5e-6, mk_delta_m_min=0.15, mk_delta_m_max=0.3,
#     multiple_service_stress_points=[{"m_or_k": "m", "val": 50e6}, {"m_or_k": "m", "val": 100e6}, {"m_or_k": "k", "val": 1e-5}],
#     save_dir="output/18_stress_service_multiple")

# # =============================================================================
# # 19. MULTIPLE ULTIMATE STRESS
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_multiple_ultimate_stress=True,
#     multiple_ultimate_stress_cases=[{"theta": 0, "n": 0}, {"theta": np.pi, "n": 0}, {"theta": 0, "n": 1000e3}],
#     multiple_ultimate_stress_labels=["Sagging", "Hogging", "Axial+Sagging"], save_dir="output/19_stress_ultimate_multiple")

# # =============================================================================
# # 20. STRESS STRAIN PROFILES
# # =============================================================================
# output = run_concrete_analysis(**section_args, run_stress_strain_profiles=True, stress_strain_materials=[concrete, steel],
#     stress_strain_names=["32 MPa Concrete", "500 MPa Steel"], save_dir="output/20_stress_strain_profiles")
