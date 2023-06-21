import logging
import torch

from . import dataset_dict


def create_training_dataset(args, cfg):
    logger = logging.getLogger(__name__)
    logger.info('training dataset: {}'.format(cfg.dataset.train))

    train_dataset = dataset_dict[cfg.dataset.train[0]](cfg, 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    return train_dataset, train_sampler
