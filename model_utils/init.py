import paddle
import paddle.distributed as dist

from ppocr.modeling.architectures import build_model
from ppocr.utils.save_load import init_model

from tools.program import load_config, check_gpu


def init():
    config = load_config('configs/e2e/pet_receipts.yml')
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    alg = config['Architecture']['algorithm']
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN',
        'CLS', 'PGNet', 'Distillation', 'TableAttn'
    ]

    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id) if use_gpu else 'cpu'
    device = paddle.set_device(device)
    config['Global']['distributed'] = dist.get_world_size() != 1

    global_config = config['Global']
    targets = global_config['targets']
    # build model
    model = build_model(config['Architecture'])

    init_model(config, model)

    return model, config
