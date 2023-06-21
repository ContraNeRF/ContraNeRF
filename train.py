import torch

from contranerf import arg_parser, setup_cfg, train
from contranerf.utils import synchronize


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    cfg = setup_cfg(args)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args, cfg)
