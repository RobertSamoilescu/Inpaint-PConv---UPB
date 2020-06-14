import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from util.evaluation import evaluate
from network.loss import InpaintingLoss
from network.net import PConvUNet, VGG16FeatureExtractor, NLayerDiscriminator
from util.io import load_ckpt
from util.io import save_ckpt
from util.data import *


parser = argparse.ArgumentParser()

# training options
# parser.add_argument('--root', type=str, default='/mnt/storage/workspace/roberts/nuscene/dataset')
# parser.add_argument('--save_dir', type=str, default='/mnt/storage/workspace/roberts/inpainting/snapshots')
# parser.add_argument('--log_dir', type=str, default='/mnt/storage/workspace/roberts/inpainting/logs')

parser.add_argument('--root', type=str, default='./dataset')
parser.add_argument('--save_dir', type=str, default='./snapshots')
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--n_threads', type=int, default=5)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images_{:.2f}'.format(args.save_dir))
    os.makedirs('{:s}/ckpt_{:.2f}'.format(args.save_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

# define datasets and dataloader
dataset_train = UPBDataset(root_dir=args.root, train=True)
dataset_val = UPBDataset(root_dir=args.root, train=False)

iterator_train = iter(data.DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads
))

# define generator and discriminator
modelG = PConvUNet().to(device)

if args.finetune:
    lr = args.lr_finetune
    modelG.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0

# define optimizers and criterion
optimizerG = torch.optim.Adam(filter(lambda p: p.requires_grad, modelG.parameters()), lr=lr)
criterionG = InpaintingLoss(VGG16FeatureExtractor()).to(device)

# resume from checkpoint
if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', modelG)],
        [('optimizer', optimizerG)]
    )
    for param_group in optimizerG.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)


# training loop
for i in tqdm(range(start_iter, args.max_iter)):
    modelG.train()

    sample = next(iterator_train)
    image, mask, gt = sample["img"], sample["mask"], sample["gt"]
    image = image.float().to(device)
    mask = mask.float().to(device)
    gt = gt.float().to(device)

    # train generator
    output, _ = modelG(image, mask)
    loss_dict = criterionG(image, mask, output, gt)

    # compute and log losses
    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    if (i + 1) % args.log_interval == 0:
        writer.add_scalar('loss', loss.item(), i + 1)

    # optimization step
    optimizerG.zero_grad()
    loss.backward()
    optimizerG.step()

    # checkpoints
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                   [('model', modelG)], [('optimizer', optimizerG)], i + 1)

    # outputs snapshots
    if (i + 1) % args.vis_interval == 0:
        modelG.eval()
        evaluate(modelG, dataset_val, device,
                  '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1))

writer.close()
