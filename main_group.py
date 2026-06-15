'''
 @FileName    : main_group.py
 @Author      : Yanchi Jiang
 @Email       : jiangyc160@163.com
 @Description : 
'''
import torch
from torch.utils.data import DataLoader
from utils.cmd_parser import parse_config
from utils.module_utils import seed_worker, set_seed
from modules import init, LossLoader, ModelLoader, DatasetLoader
# import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
###########Load config file in debug mode#########
import sys
# sys.argv = ['','--config=cfg_files/config_group.yaml'] #MoCap End2End HMAE_conv Lifting

def main(**args):
    seed = 7
    g = set_seed(seed)

    # Global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'), type='cuda')
    mode = args.get('mode')
    task = args.get('task')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # Load loss function
    loss = LossLoader(device=device, **args)

    # Load model
    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)

    # create data loader
    print(">>> start DatasetLoader")
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    print(">>> DatasetLoader done")
    if mode == 'train':
        print(">>> start load_trainset")
        train_dataset = dataset.load_trainset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=workers, pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        print(">>> train_loader created")
        if args.get('use_sch'):
            model.load_scheduler(train_dataset.cumulative_sizes[-1])
    print(">>> start load_testset")
    test_dataset = dataset.load_testset()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=workers, pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    print(">>> test_loader created")

    # Load handle function with the task name
    task = args.get('task')
    exec('from process import %s_train' %task)
    exec('from process import %s_test' %task)

    for epoch in range(num_epoch):
        # training mode
        if mode == 'train':
            if task == 'group':
                training_loss = eval('%s_train' %task)(model, loss, train_loader, epoch, num_epoch, device=device)
                if (epoch) % 1 == 0:
                    edge_acc, precision, recall, f1 = eval('%s_test' %task)(model, loss, test_loader, device=device)
                    lr = model.optimizer.state_dict()['param_groups'][0]['lr']
                    logger.append([int(epoch + 1), lr, training_loss, edge_acc, precision, recall, f1])
                else:
                    edge_acc, precision, recall, f1 = -1., -1., -1.
                
                model.save_group_model(precision, recall, f1, epoch, task)

            else:
                training_loss = eval('%s_train' %task)(model, loss, train_loader, epoch, num_epoch, device=device)

                if (epoch) % 1 == 0:
                    testing_loss = eval('%s_test' %task)(model, loss, test_loader, device=device)
                    lr = model.optimizer.state_dict()['param_groups'][0]['lr']
                    logger.append([int(epoch + 1), lr, training_loss, testing_loss])
                else:
                    testing_loss = -1.

                # save trained model
                if args['save_best']:
                    model.save_best_model(testing_loss, epoch, task)
                else:
                    model.save_model(testing_loss, epoch, task)

        # testing mode
        elif epoch == 0 and mode == 'test':
            if task == 'group':
                training_loss = -1.
                edge_acc, precision, recall, f1 = eval('%s_test' %task)(model, loss, test_loader, device=device)
                lr = model.optimizer.state_dict()['param_groups'][0]['lr']
                logger.append([int(epoch + 1), lr, training_loss, edge_acc, precision, recall, f1])
            else:
                training_loss = -1.
                testing_loss = eval('%s_test' %task)(model, loss, test_loader, device=device)

                lr = model.optimizer.state_dict()['param_groups'][0]['lr']
                logger.append([int(epoch + 1), lr, training_loss, testing_loss])

    logger.close()


if __name__ == "__main__":
    args = parse_config()
    main(**args)





