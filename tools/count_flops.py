import datetime
import torch
from copy import deepcopy
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_str
from thop import profile

from yolox.exp import get_exp
from tools.train import make_parser
from yolox.models import transformers


def main(exp, args):
    if exp is not None:
        model = exp.get_model()
        img_size = 640
        mac_or_flops = 2
    else:
        model = args.name
        img_size = 224
        if '-' in args.name:
            model, img_size = args.name.split('-')
            img_size = int(img_size)
        model = getattr(transformers, model)()
        mac_or_flops = 1
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, requires_grad=False)

    flops = FlopCountAnalysis(model, x)
    print('[fvcore] FLOPs: {:.2f} G'.format(flops.total() / 1e9 *
                                            mac_or_flops))
    # print(flop_count_str(flops))

    flops, params = profile(deepcopy(model), inputs=(x, ), verbose=False)
    print('[thop] Params: {:.2f} M, FLOPs: {:.2f} G'.format(
        params / 1e6, flops / 1e9 * mac_or_flops))

    model.cuda()
    x = x.cuda()
    n_iters = 100
    start = datetime.datetime.now()
    for _ in tqdm(range(n_iters)):
        model(x)
    end = datetime.datetime.now()

    dt = end - start
    dt = dt.seconds * 1e6 + dt.microseconds
    print(f'time: {dt / n_iters / 1e3:.2f} ms')
    print(f'Tensor GPU Mem: {torch.cuda.max_memory_allocated() / 1024**2:.2f} M')


if __name__ == '__main__':
    """
    If `args.exp_file` is assigned by `-f`, build the complete detector
    model and test under image size of 640. The actual FLOPs is reported.

    Else if `args.name` is assigned by `-n`, build the backbone from
    `yolox.models.transformers` only and test under image size of 224 or
    the specified after '-'. The actual MACs is reported.
    """
    args = make_parser().parse_args()
    if args.exp_file:
        exp = get_exp(args.exp_file, args.name)
        exp.merge(args.opts)
    elif args.name:
        exp = None
    main(exp, args)
