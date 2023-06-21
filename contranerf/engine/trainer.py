import os
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter

from .test import test
from contranerf.utils import *
from contranerf.modeling import Model
from contranerf.data.create_training_dataset import create_training_dataset


def train(args, cfg):
    rank = args.local_rank
    out_folder = os.path.join(cfg.output_base, cfg.output, cfg.expname)
    logger = setup_logger(output=out_folder, distributed_rank=rank, name='lib')

    logger.info('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # log and save the args and config files
    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config") and args.config != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config,
                open(args.config, "r").read(),
            )
        )

    path = os.path.join(out_folder, "config.yaml")
    logger.info("Running with full config:\n{}".format(cfg.dump(), ".yaml"))
    with open(path, 'w') as f:
        f.write(cfg.dump())
    logger.info("Full config saved to {}".format(path))

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args, cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               worker_init_fn=lambda _: np.random.seed(),
                                               num_workers=cfg.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)

    # Create model
    model = Model(args, cfg, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)

    # Create criterion
    criterion = Criterion()
    tb_dir = os.path.join(cfg.output_base, cfg.output, cfg.expname, 'logs/')
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        logger.info('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}

    global_step = model.start_step + 1
    end_step = cfg.solver.iterations
    epoch = 0
    while global_step < end_step + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop
            ret, batch = model(train_data)

            # compute loss
            model.optimizer.zero_grad()
            loss = 0
            if cfg.model.loss.rgb:
                loss += criterion(ret['rgb'], batch)
            if cfg.model.loss.rgb_coarse:
                loss += criterion(ret['rgb_coarse'], batch)

            loss.backward()
            scalars_to_log['loss'] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # logging
            if args.local_rank == 0:
                if global_step % cfg.logging.print_iter == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret['rgb']['rgb'], batch['rgb']).item()
                    scalars_to_log['train/loss'] = mse_error
                    scalars_to_log['train/psnr-training-batch'] = mse2psnr(mse_error)
                    mse_error = img2mse(ret['rgb_coarse']['rgb'], batch['rgb']).item()
                    scalars_to_log['train/coarse-loss'] = mse_error
                    scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)

                    logstr = '{} Epoch: {}  step: {} '.format(cfg.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    logger.info(logstr)
                    logger.info('each iter time {:.05f} seconds'.format(dt))

                if global_step % cfg.logging.weights_iter == 0:
                    logger.info('Saving checkpoints at {} to {}...'.format(global_step, out_folder))
                    fpath = os.path.join(out_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)
            
            # Evaluation
            if global_step % cfg.test.test_iter == 0:
                model.switch_to_eval()
                with torch.no_grad():
                    if "scannet_test" in cfg.test.datasets:
                        test(args, cfg, model, cfg.test.scannet_scenes, "scannet_test", logger, "eval_scannet", global_step)
                        if args.local_rank == 0:
                            merge_results(cfg, "eval_scannet", global_step)
                model.switch_to_train()

            global_step += 1
            if global_step > model.start_step + cfg.solver.iterations + 1:
                break
        epoch += 1
