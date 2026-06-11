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
    健壮的可视化渲染函数：处理多人员、维度对齐及原图坐标系关键点绘制
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)
    print(f"\n Starting Visualization, saving to: {output_dir}")

    img_path = results['imgs'][0]
    pred_verts = results['pred_verts']
    pred_trans = results['pred_trans']
    focal = results['focal_length'][0]

    pred_verts = pred_verts + pred_trans[:,np.newaxis,:]

    print(img_path)

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0],
                        faces=self.smpl.faces,
                        same_mesh_color=True)
    pred_smpl = renderer.render_front_view(pred_verts, bg_img_rgb=img.copy())

    img_name = os.path.basename(img_path)
    img_id = os.path.splitext(img_name)[0]
        
    render_name = f"{img_id}_front.jpg"
    cv2.imwrite(os.path.join(output_dir, render_name), pred_smpl)

    renderer.delete()
    print(f"Saved visualization for frame {img_id})")