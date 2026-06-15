'''
@FileName    : single_inference.py
@Description : End-to-end single frame inference pipeline: 
               Load demo.pkl -> Batch Crop (Cliff) -> Group Prediction -> 3D Reconstruction -> Visualization
'''
import os
import cv2
import numpy as np
from utils.renderer_pyrd import Renderer

def save_demo_results(self, results, output_dir):
    """
    demo结果保存函数
    """
    img_path = results['imgs'][0]
    img_name = os.path.basename(img_path)
    img_id = os.path.splitext(img_name)[0]

    img_output_dir = os.path.join(output_dir, img_id)
    os.makedirs(img_output_dir, exist_ok=True)
    meshes_dir = os.path.join(img_output_dir, 'meshes')
    os.makedirs(meshes_dir, exist_ok=True)
    print(f"\n Starting Visualization, saving to: {img_output_dir}")

    pred_verts = results['pred_verts']
    pred_trans = results['pred_trans']
    focal = results['focal_length'][0]

    pred_verts = pred_verts + pred_trans[:,np.newaxis,:]

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0],
                        faces=self.smpl.faces,
                        same_mesh_color=False)
    pred_smpl = renderer.render_front_view(pred_verts, bg_img_rgb=img.copy())
        
    render_name = f"{img_id}_front.jpg"
    cv2.imwrite(os.path.join(output_dir, render_name), pred_smpl)

    renderer.delete()

    num_people = pred_verts.shape[0]
    for person_idx in range(num_people):
        person_verts = pred_verts[person_idx]
        mesh_name = os.path.join(meshes_dir, f"{person_idx:03d}.obj")
        self.smpl.write_obj(person_verts, mesh_name)