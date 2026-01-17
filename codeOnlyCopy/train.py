# train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader
from dataset import PoseSequenceDataset
from models import PoseTransformer
import torch.optim as optim
from tqdm import tqdm


def collate_batch(batch):
    # batch: list of tuples (pose_tensor, img_tensor)
    poses = [b[0] for b in batch]
    imgs = [b[1] for b in batch]
    xs = torch.stack(poses, dim=0)  # (B, T, D)
    # imgs may contain None; replace with zeros
    if any(i is None for i in imgs):
        # determine size from first non-None or default to (3,512,512)
        first = next((i for i in imgs if i is not None), None)
        if first is None:
            img_shape = (3,512,512)
        else:
            img_shape = tuple(first.shape)
        new_imgs = []
        for i in imgs:
            if i is None:
                new_imgs.append(torch.zeros(img_shape, dtype=torch.float32))
            else:
                new_imgs.append(i)
        imgs = torch.stack(new_imgs, dim=0)
    else:
        imgs = torch.stack(imgs, dim=0)
    return xs, imgs

def train_loop(data_root, save_dir="checkpoints", seq_len=64, batch_size=8, epochs=30, device="cuda"):
    ds = PoseSequenceDataset(data_root, seq_len=seq_len, stride=8)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch)
    sample = ds[0]
    # sample is (pose_tensor, img_tensor)
    input_dim = sample[0].shape[1]
    # create image encoder and transformer with conditioning
    from models import PoseTransformer, SimpleImageEncoder
    cond_dim = 256
    img_encoder = SimpleImageEncoder(out_dim=cond_dim).to(device)
    model = PoseTransformer(input_dim=input_dim, cond_dim=cond_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    # optionally freeze img_encoder (use separate optimizer if fine-tuning)
    opt_img = optim.AdamW(img_encoder.parameters(), lr=3e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    os.makedirs(save_dir, exist_ok=True)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {ep}"):
            poses, imgs = batch
            poses = poses.to(device)
            imgs = imgs.to(device)
            # Simple training objective: given first K frames predict remaining frames
            K = max(4, seq_len // 4)
            context = poses[:, :K, :]   # conditioning
            target = poses[:, K:, :]
            # encode images to conditioning vectors
            with torch.no_grad():
                # You can set requires_grad True and train the encoder if desired
                img_feats = img_encoder(imgs)
            # build decoder input: feed whole sequence with zeros placeholders and pass cond
            input_seq = torch.cat([context, torch.zeros(poses.shape[0], seq_len-K, poses.shape[2], device=device)], dim=1)
            preds = model((input_seq, img_feats))  # (B,T,D)
            pred_target = preds[:, K:, :]
            loss = torch.nn.functional.mse_loss(pred_target, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # no img encoder update by default; if fine-tuning, step opt_img here
            total_loss += float(loss.item())
        scheduler.step()
        print(f"Epoch {ep} avg loss: {total_loss/len(loader):.6f}")
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_ep{ep}.pth"))

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_root", required=True)
    # parser.add_argument("--save_dir", default="checkpoints")
    # parser.add_argument("--epochs", type=int, default=30)
    # parser.add_argument("--batch", type=int, default=8)
    # args = parser.parse_args()
    # train_loop(args.data_root, args.save_dir, epochs=args.epochs, batch_size=args.batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataDir = os.path.dirname(os.path.abspath(__file__))
    dataRoot = os.path.join(dataDir, "data")

    train_loop(dataRoot, save_dir="checkpoints", epochs=1, batch_size=8)