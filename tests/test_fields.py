import io
import numpy as np
import pytest
from PIL import Image,ImageDraw
from airflow.geometry import rasterize_walls_mask,wall_mask_from_floorplan_image_bytes
from airflow.simulation import simulate_field,_impenetrable_faces

def test_object_and_image_wall_rasterisation():
    wall=rasterize_walls_mask([{'type':'rect','left':40,'top':20,'width':10,'height':60,'stroke':'#444444','label':'wall'}],100,100,30,30)
    assert wall.dtype==bool and wall.any()
    image=Image.new('RGB',(100,100),'white');ImageDraw.Draw(image).rectangle((40,20,50,80),fill='black')
    stream=io.BytesIO();image.save(stream,format='PNG')
    mask=wall_mask_from_floorplan_image_bytes(stream.getvalue(),100,100,30,30)
    assert mask.shape==(30,30) and mask.any()

def test_wall_velocity_and_finite_field():
    wall=np.zeros((15,20),bool);wall[4:11,10]=True
    u,v=_impenetrable_faces(np.ones(wall.shape),np.ones(wall.shape),wall)
    assert np.all(u[wall]==0) and np.all(v[wall]==0)
    T,u,v=simulate_field([{'x':30.,'y':50.,'load':100.}],[{'x':0.,'y':50.,'strength':1.}],
        [{'x':100.,'y':50.,'strength':1.}],100,100,20,15,wall_mask=wall,flow_iters=3)
    assert all(np.isfinite(field).all() for field in (T,u,v))
    assert T.max()>0
    assert np.all(T[wall]==0) and np.all(u[wall]==0) and np.all(v[wall]==0)

def test_unimplemented_backend_rejected():
    with pytest.raises(ValueError):simulate_field([],[],[],100,100,20,15,backend='openfoam')
