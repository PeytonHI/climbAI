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
    # batch: list of tensors (T, D)
    xs = torch.stack(batch, dim=0)  # (B, T, D)
    return xs

def train_loop(data_root, save_dir="checkpoints", seq_len=64, batch_size=8, epochs=30, device="cuda"):
    ds = PoseSequenceDataset(data_root, seq_len=seq_len, stride=8)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch)
    sample = ds[0]
    input_dim = sample.shape[1]
    model = PoseTransformer(input_dim=input_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    os.makedirs(save_dir, exist_ok=True)
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {ep}"):
            batch = batch.to(device)  # (B, T, D)
            # Simple training objective: given first K frames predict remaining frames
            K = max(4, seq_len // 4)
            context = batch[:, :K, :]   # conditioning
            target = batch[:, K:, :]
            # build decoder input: repeat context + zeros placeholders
            # For simplicity, feed whole sequence and compute loss only on target frames
            input_seq = torch.cat([context, torch.zeros(batch.shape[0], seq_len-K, batch.shape[2], device=device)], dim=1)
            preds = model(input_seq)  # (B,T,D)
            pred_target = preds[:, K:, :]
            loss = torch.nn.functional.mse_loss(pred_target, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
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