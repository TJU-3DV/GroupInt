'''
 @FileName    : demo.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''
import torch
from torch.utils.data import DataLoader
from cmd_parser import parse_config
from utils.module_utils import seed_worker, set_seed
from modules import init, ModelLoader, DatasetLoader
    
###########Load config file in debug mode#########
import sys
sys.argv = ['','--config=cfg_files/demo.yaml'] #MoCap End2End HMAE_conv Lifting

def main(**args):

    # Global setting
    dtype = torch.float32
    device = torch.device(index=args.get('gpu_index'), type='cuda')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # Load model
    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)

    # create data loader
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    eval_dataset = dataset.load_demo_data()
    print("len(eval_dataset):", len(eval_dataset))
    # print("dataset[0]:", eval_dataset[0])

    # Load handle function with the task name
    task = args.get('task')
    # print("1")
    exec('from process import %s_demo' %task)
    # print("11")
    eval('%s_demo' %task)(model, eval_dataset, device=device)
    # print("111")


if __name__ == "__main__":
    args = parse_config()
    main(**args)





