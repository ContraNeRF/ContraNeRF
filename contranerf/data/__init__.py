from .scannet import *
from .front3d import *


dataset_dict = {
    'scannet_test': ScanNetTestDataset,
    'front3d': Front3DDataset,
}
