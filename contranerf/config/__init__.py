import argparse

__all__ = ["arg_parser", "setup_cfg"]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument("--local_rank", type=int, default=0, help='rank for distributed training')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument("--no_load_opt", action='store_true',
                        help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true',
                        help='do not load scheduler when reloading')
    parser.add_argument("--ckpt_path", type=str, default="",
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    return parser


def get_cfg():
    from .default import _C
    return _C.clone()


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    return cfg
