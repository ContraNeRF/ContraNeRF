from .config import CfgNode as CN

_C = CN()

_C.output_base = ""
_C.output = ""
_C.expname = "pretraining"
_C.workers = 6

# Solver
_C.solver = CN()
_C.solver.iterations = 100000
_C.solver.batch_size = 512
_C.solver.base_lr = 0.001
_C.solver.weight_decay = 0.01
_C.solver.custom_list = [["epipolar_layer", "view_layer"], [['lr', 0.0001], ['lr', 0.0001]]]
_C.solver.lr_decay_step = 50000
_C.solver.lr_decay_factor = 0.5
_C.solver.milestones = [50000, 100000]
_C.solver.warmup_factor = 0.00025
_C.solver.warmup_iters = 4000
_C.solver.warmup_method = 'linear'
_C.solver.optimizer = 'AdamW'
_C.solver.scheduler = 'WarmupMultiStepLR'
_C.solver.t_max = 10000

# Data
_C.data = CN()
_C.data.rootdir_scannet = ""
_C.data.rootdir_front3d = ""
_C.data.num_source_views = 10
_C.data.num_source_views_train = 10
_C.data.sample_mode = 'uniform'
_C.data.center_ratio = 0.8
_C.data.num_samples = 64
_C.data.num_importance = 64
_C.data.inv_uniform = True
_C.data.deterministic = False
_C.data.rectify_inplane_rotation = False
_C.data.testskip = 8
_C.data.camera_std = 0.0005

# Logging
_C.logging = CN()
_C.logging.print_iter = 100
_C.logging.weights_iter = 10000

# Dataset
_C.dataset = CN()
_C.dataset.train = ["front3d"]

# Test
_C.test = CN()
_C.test.chunk_size = 1024
_C.test.test_iter = 100000
_C.test.datasets = ["scannet_test"]
_C.test.scannet_scenes = ["scene0289_00", "scene0204_00", "scene0205_00", "scene0587_02", 
                          "scene0611_01", "scene0269_01", "scene0549_01", "scene0456_00"]

# Model
_C.model = CN()
_C.model.name = 'ContraNeRF'
_C.model.loss = CN()
_C.model.loss.rgb = True
_C.model.loss.rgb_coarse = True

# MLPNet
_C.mlpnet = CN()
_C.mlpnet.name = 'MLPNet'
_C.mlpnet.coarse_feat_dim = 32
_C.mlpnet.fine_feat_dim = 32
_C.mlpnet.image_feat_dim = 32
_C.mlpnet.white_bkgd = False
_C.mlpnet.anti_alias_pooling = True

# CrossView
_C.crossview = CN()
_C.crossview.n_sample = 16
_C.crossview.deterministic = True
_C.crossview.n_layers = 1
_C.crossview.skip_connect = True
_C.crossview.epipolar = CN()
_C.crossview.epipolar.dropout = 0.0
_C.crossview.inv_uniform = True
