"""Entirely fictional room objects; no real floor plan is loaded."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from airflow.geometry import rasterize_walls_mask
from airflow.simulation import simulate_field,plot_heatmap_with_suggestions

def fictional_inputs():
    objects=[{'type':'rect','left':290,'top':80,'width':20,'height':250,'fill':'#444444','stroke':'#444444','label':'wall'}]
    sources=[{'x':180.,'y':200.,'load':500.}]
    supply=[{'x':0.,'y':150.,'strength':1.}]
    exhaust=[{'x':600.,'y':250.,'strength':1.}]
    return objects,sources,supply,exhaust

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('results'));args=ap.parse_args()
    objects,sources,supply,exhaust=fictional_inputs()
    wall=rasterize_walls_mask(objects,600,400,40,30)
    T,u,v=simulate_field(sources,supply,exhaust,600,400,40,30,wall_mask=wall,flow_iters=10)
    report={'fictional':True,'shape':list(T.shape),'finite':bool(np.isfinite(T).all() and np.isfinite(u).all() and np.isfinite(v).all()),
        'solid_velocity_zero':bool(np.all(u[wall]==0) and np.all(v[wall]==0)),
        'relative_field_min':float(T.min()),'relative_field_max':float(T.max())}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'room.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
