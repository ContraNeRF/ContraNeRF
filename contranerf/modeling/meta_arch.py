import os
import logging
import torch

from contranerf.solver import build_scheduler, build_optimizer
from .build import build_model

logger = logging.getLogger(__name__)


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


class Model(object):
    def __init__(self, args, cfg, load_opt=True, load_scheduler=True):
        self.args = args
        self.cfg = cfg
        # build model
        self.model = self.build_model(args, cfg)
        # optimizer and learning rate scheduler
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.scheduler = self.build_scheduler(cfg, self.optimizer)

        out_folder = os.path.join(cfg.output, cfg.expname)
        self.start_step = self.load_from_ckpt(out_folder,
                                              load_opt=load_opt,
                                              load_scheduler=load_scheduler)

    def __call__(self, data):
        return self.model(data)

    @classmethod
    def build_model(cls, args, cfg):
        device = torch.device('cuda:{}'.format(args.local_rank))
        model = build_model(cfg).to(device)
        logger.info("Model:\n{}".format(model))
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)
    
    @classmethod
    def build_scheduler(cls, cfg, optim):
        return build_scheduler(cfg, optim)

    def switch_to_eval(self):
        self.model.eval()

    def switch_to_train(self):
        self.model.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'model': de_parallel(self.model).state_dict(),
                   }
        torch.save(to_save, filename)
    
    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
            to_load['model'] = {'module.'+k: v for k, v in to_load['model'].items()}
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        try:
            self.model.load_state_dict(to_load['model'])
        except:
            self.model.net_coarse.load_state_dict({
                k[11:]:v for k, v in to_load['model'].items() if k.startswith('net_coarse')})
            self.model.net_fine.load_state_dict({
                k[9:]:v for k, v in to_load['model'].items() if k.startswith('net_fine')})
            self.model.feature_net.load_state_dict({
                k[12:]:v for k, v in to_load['model'].items() if k.startswith('feature_net')})

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            logger.info('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            logger.info('No ckpts found, training from scratch...')
            step = 0

        return step
