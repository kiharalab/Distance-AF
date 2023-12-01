import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # Set CuDNN to use deterministic algorithms
    cudnn.deterministic = True
    cudnn.benchmark = False