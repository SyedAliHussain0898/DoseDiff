import os
from dataset import Dataset_PSDM_train, Dataset_PSDM_val
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
import argparse
import shutil
from guided_diffusion.unet import UNetModel_MS_Former_MultiStage
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import MultiStageSpacedDiffusion, space_timesteps
from guided_diffusion.resample import create_named_schedule_sampler
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--T', type=int, default=1000, help='T')
parser.add_argument('--epoch', type=int, default=100, help='all_epochs')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
args = parser.parse_args()

device = torch.device("cuda")

train_bs = args.bs
val_bs = 2 * args.bs
lr_max = 0.0001
img_size = (128, 128)
all_epochs = args.epoch
data_root_train = 'preprocessed_data/NPY/train'
data_root_val = 'preprocessed_data/NPY/validation'
L2 = 0.0001

save_name = f'T{args.T}_bs{train_bs}_epoch{args.epoch}'

if os.path.exists(os.path.join('trained_models', save_name)):
    shutil.rmtree(os.path.join('trained_models', save_name))
os.makedirs(os.path.join('trained_models', save_name), exist_ok=True)
print(save_name)

train_writer = SummaryWriter(os.path.join('trained_models', save_name, 'log/train'), flush_secs=2)

train_data = Dataset_PSDM_train(data_root=data_root_train)
val_data = Dataset_PSDM_val(data_root=data_root_val)
train_dataloader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=val_bs, shuffle=False, num_workers=2, pin_memory=True)

print(f'train_length: {train_data.len}   val_length: {val_data.len}')

dis_channels = 20

model = UNetModel_MS_Former_MultiStage(
    image_size=img_size,
    in_channels=1,
    ct_channels=1,
    dis_channels=dis_channels,
    model_channels=128,
    out_channels=1,
    num_res_blocks=2,
    attention_resolutions=(16, 32),
    dropout=0,
    channel_mult=(1, 1, 2, 3, 4),
    conv_resample=True,
    dims=2,
    num_classes=None,
    use_checkpoint=False,
    use_fp16=False,
    num_heads=4,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_new_attention_order=False,
    num_stages=3
)

model.to(device)

schedule_sampler = create_named_schedule_sampler("loss-second-moment", diffusion)
diffusion = MultiStageSpacedDiffusion(
    use_timesteps=space_timesteps(args.T, 'ddim4'),
    betas=gd.get_named_beta_schedule("linear", args.T),
    model_mean_type=gd.ModelMeanType.EPSILON,
    model_var_type=gd.ModelVarType.FIXED_LARGE,
    loss_type=gd.LossType.MSE,
    rescale_timesteps=False,
    num_stages=3,
   stage_distribution='geometric'
)

diffusion_test = MultiStageSpacedDiffusion(
    use_timesteps=space_timesteps(args.T, 'ddim4'),
    betas=gd.get_named_beta_schedule("linear", args.T),
    model_mean_type=gd.ModelMeanType.EPSILON,
    model_var_type=gd.ModelVarType.FIXED_LARGE,
    loss_type=gd.LossType.MSE,
    rescale_timesteps=False,
    num_stages=3,
    stage_distribution='geometric'
)

optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=L2)
lr_scheduler = MultiStepLR(optimizer, milestones=[int((7 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)

best_MAE = 1000
epoch_start = 0

if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
    print(f"Loading checkpoint from {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'epoch' in ckpt:
        epoch_start = ckpt['epoch']
    if 'best_MAE' in ckpt:
        best_MAE = ckpt['best_MAE']
    print(f"Resumed from epoch {epoch_start} with best MAE={best_MAE:.4f}")

for epoch in range(epoch_start, all_epochs):
    lr = optimizer.param_groups[0]['lr']
    model.train()
    train_epoch_loss = []
    for i, (ct, dis, rtdose) in enumerate(train_dataloader):
        ct, dis, rtdose = ct.to(device).float(), dis.to(device).float(), rtdose.to(device).float()
        optimizer.zero_grad()

        t, weights = schedule_sampler.sample(rtdose.shape[0], rtdose.device)
        stage_indices = model.get_stage_from_timestep(t, args.T)

        losses = diffusion.training_losses(
            model=model,
            x_start=rtdose,
            t=t,
            model_kwargs={'ct': ct, 'dis': dis, 'stage_indices': stage_indices},
            noise=None
        )

        loss = (losses["loss"] * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_epoch_loss.append(loss.item())
        print(f'[{epoch + 1}/{all_epochs}, {i + 1}/{len(train_dataloader)}] train_loss: {loss.item():.3f}')

    lr_scheduler.step()
    train_writer.add_scalar('lr', lr, epoch + 1)
    train_writer.add_scalar('train_loss', np.mean(train_epoch_loss), epoch + 1)

    if (epoch == 0) or (((epoch + 1) % 10) == 0):
        model.eval()
        val_epoch_MAE, image_CT, ture_rtdose, pred_rtdose = [], [], [], []
        with torch.no_grad():
            for i, (ct, dis, rtdose) in enumerate(val_dataloader):
                ct, dis, rtdose = ct.to(device).float(), dis.to(device).float(), rtdose.to(device).float()
                stage_indices = torch.full((ct.size(0),), 2, device=device, dtype=torch.long)

                pred = diffusion_test.ddim_sample_loop(
                    model=model,
                    shape=(ct.size(0), 1, img_size[0], img_size[1]),
                    noise=None,
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs={'ct': ct, 'dis': dis, 'stage_indices': stage_indices},
                    device=None,
                    progress=False,
                    eta=0.0
                )

                rtdose = (rtdose + 1) * 40
                pred = (pred + 1) * 40
                body_mask = dis[:, 6: 7]
                MAE = (torch.abs(rtdose - pred) * body_mask).sum() / body_mask.sum()

                val_epoch_MAE.append(MAE.item())
                if i in [0, 2, 4, 8]:
                    image_CT.append(ct[0:1].cpu())
                    ture_rtdose.append(rtdose[0:1].cpu())
                    pred_rtdose.append(pred[0:1].cpu())

        train_writer.add_scalar('val_MAE', np.mean(val_epoch_MAE), epoch + 1)
        train_writer.add_image('image_CT', make_grid(torch.cat(image_CT), 2, normalize=True), epoch + 1)
        train_writer.add_image('ture_rtdose', make_grid(torch.cat(ture_rtdose), 2, normalize=True), epoch + 1)
        train_writer.add_image('pred_rtdose', make_grid(torch.cat(pred_rtdose), 2, normalize=True), epoch + 1)

        torch.save(model.state_dict(), os.path.join('trained_models', save_name, f'model_epoch{epoch + 1}.pth'))
        if np.mean(val_epoch_MAE) < best_MAE:
            best_MAE = np.mean(val_epoch_MAE)
            torch.save(model.state_dict(), os.path.join('trained_models', save_name, 'model_best_mae.pth'))

        ckpt_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_MAE': best_MAE
        }
        torch.save(ckpt_dict, os.path.join('trained_models', save_name, f'checkpoint_epoch{epoch + 1}.pth'))

train_writer.close()
print(f'saved_model_name: {save_name}')
