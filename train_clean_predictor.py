
import os
import json
import random
from time import time, sleep
from datetime import datetime
import warnings

import numpy as np
import torch
from torch.nn.functional import l1_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from egnn_predictor.models import EGNN_predictor

from utils.utils_edm import (
    remove_mean_with_mask,
    assert_correctly_masked,
    assert_mean_zero_with_mask,
    normalize,
    MyDataParallel
)

from data.aromatic_dataloader import create_data_loaders, AromaticDataset
from prediction_args import PredictionArgs


#from utils.args_edm import Args_EDM
#from models_edm import get_model

warnings.simplefilter(action="ignore", category=FutureWarning)


def check_mask_correct(variables, node_mask):
    """
    Check that tensors are properly masked.
    """
    for variable in variables:
        if variable.shape[-1] != 0:
            assert_correctly_masked(variable, node_mask)



def compute_clean_loss(
    model,
    x,
    h,
    node_mask,
    edge_mask,
    target,
):
    """
    Clean Case 2 loss:
    - NO diffusion noise
    - NO random t
    - Uses EDM normalization only (t = 0)
    """

    # 1) Normalize coordinates and node features with EDM
    # x_norm, h_norm, _ = edm_model.normalize(
    #     x,
    #     {"categorical": h, "integer": torch.zeros(0, device=x.device)},
    #     node_mask,
    # )

    x_norm, h_norm, _ = normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0, device=x.device)},
        node_mask,
    )

    # 2) Build [x_norm | h_norm] input
    xh = torch.cat([x_norm, h_norm["categorical"]], dim=-1)  # [bs, n_nodes, d+in_nf]

    bs, n_nodes, _ = x.shape
    edge_mask_flat = edge_mask.view(bs, n_nodes * n_nodes)   # [bs, n_nodes^2]

    # 3) Use fixed t = 0 (no diffusion, just a constant conditioning value)
    t = torch.zeros(bs, 1, device=x.device)

    # 4) Forward pass
    preds = model(xh, node_mask, edge_mask_flat, t)  # [bs, num_targets]

    # 5) L1 loss in normalized target space
    loss = l1_loss(preds, target)
    error = (preds - target).abs().detach()  # per-sample, per-target
    return loss, error


def train_epoch_clean(
    epoch,
    cond_predictor,
    dataloader,
    optimizer,
    args,
    writer,
):
    cond_predictor.train()
    start_time = time()
    loss_list = []
    rl_loss = []

    with tqdm(dataloader, unit="batch", desc=f"Train (clean) {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(tepoch):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            loss, _ = compute_clean_loss(
                cond_predictor,
                x,
                h,
                node_mask,
                edge_mask,
                y,
            )

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            # Rescaled loss in original target units
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            tepoch.set_postfix(loss=np.mean(loss_list).item() )

    print(
        f"[{epoch}|train] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
        f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
        f" in {int(time()-start_time)} secs"
    )
    sleep(0.01)
    if writer is not None:
        writer.add_scalar("Train loss", np.mean(loss_list), epoch)
        writer.add_scalar("Train L1 (rescaled)", np.mean(rl_loss), epoch)


def val_epoch_clean(
    tag,
    epoch,
    cond_predictor,
    dataloader,
    args,
    writer,
):
    cond_predictor.eval()
    with torch.no_grad():
        start_time = time()
        loss_list = []
        rl_loss = []

        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(dataloader):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            loss, _  = compute_clean_loss(
                cond_predictor,
                x,
                h,
                node_mask,
                edge_mask,
                y,
            )

            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())
            

        print(
            f"[{epoch}|{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )
        sleep(0.01)
        if writer is not None:
            writer.add_scalar(f"{tag} loss", np.mean(loss_list), epoch)
            writer.add_scalar(f"{tag} L1 (rescaled)", np.mean(rl_loss), epoch)

    return np.mean(loss_list)


def get_cond_predictor_model(args, dataset: AromaticDataset):
    cond_predictor = EGNN_predictor(
        in_nf=dataset.num_node_features,
        device=args.device,
        hidden_nf=args.nf,
        out_nf=dataset.num_targets,
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        recurrent=True,
        tanh=args.tanh,
        attention=args.attention,
        condition_time=False, 
        coords_range=args.coords_range,
    )

    if args.dp:  # and torch.cuda.device_count() > 1:
        cond_predictor = MyDataParallel(cond_predictor)
    if args.restore is not None:
        model_state_dict = torch.load(args.exp_dir + "/model.pt", map_location=args.device)
        cond_predictor.load_state_dict(model_state_dict)
    return cond_predictor



def main(pred_args, device):
    # ---------------------------
    # Data
    # ---------------------------
    train_loader, val_loader, test_loader = create_data_loaders(pred_args)

    # ---------------------------
    # EDM model ONLY for normalize()
    # ---------------------------
    # edm_args = Args_EDM().parse_args([])
    # # Make EDM dataset match predictor dataset if possible
    # if hasattr(pred_args, "dataset"):
    #     edm_args.dataset = pred_args.dataset
    # edm_args.device = device

    # edm_model, _, _ = get_model(edm_args, train_loader)
    # edm_model.to(device)
    # edm_model.eval()

    # ---------------------------
    # Predictor model
    # ---------------------------
    cond_predictor = get_cond_predictor_model(pred_args, train_loader.dataset)
    cond_predictor.to(device)

    if pred_args.dp and torch.cuda.device_count() > 1:
        cond_predictor = torch.nn.DataParallel(cond_predictor)

    # ---------------------------
    # Optimizer
    # ---------------------------
    optimizer = torch.optim.Adam(
        cond_predictor.parameters(),
        lr=pred_args.lr,
        amsgrad=True, 
        weight_decay=1e-12,
    )

    # ---------------------------
    # Logging & experiment dir
    # ---------------------------
    if not os.path.isdir(pred_args.exp_dir):
        os.makedirs(pred_args.exp_dir, exist_ok=True)

    with open(os.path.join(pred_args.exp_dir, "args_clean.txt"), "w") as f:
        json.dump(pred_args.__dict__, f, indent=2, default=str)


    writer = None
    if getattr(pred_args, "log_tensorboard", False):
        writer = SummaryWriter(log_dir=os.path.join(pred_args.exp_dir, "tb_clean"))

    print("Clean predictor training args:")
    print(pred_args)

    # ---------------------------
    # Training loop
    # ---------------------------
    best_val_mae = 1e9
    best_epoch = 0

    print("Begin CLEAN training (no diffusion noise, t = 0)")
    for epoch in range(pred_args.num_epochs):
        train_epoch_clean(
            epoch,
            cond_predictor,
            train_loader,
            optimizer,
            pred_args,
            writer,
        )
        val_mae = val_epoch_clean(
            "val",
            epoch,
            cond_predictor,
            val_loader,
            pred_args,
            writer,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(
                cond_predictor.state_dict(),
                os.path.join(pred_args.exp_dir, "model_clean.pt"),
            )
            print(f"Saved new best model at epoch {epoch} with MAE={val_mae:.4f}")

    print(f"Best val MAE: {best_val_mae:.4f} at epoch {best_epoch}")

    # ---------------------------
    # Final test evaluation
    # ---------------------------
    print("Testing best CLEAN model...")
    # reload best weights
    state = torch.load(
        os.path.join(pred_args.exp_dir, "model_clean.pt"),
        map_location=device,
    )
    cond_predictor.load_state_dict(state)
    cond_predictor.to(device)
    test_mae = val_epoch_clean(
        "test",
        best_epoch,
        cond_predictor,
        test_loader,
        pred_args,
        writer,
    )
    print(f"Final test MAE (orig units): {test_mae:.4f}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use same argument structure as cond_prediction
    pred_args = PredictionArgs().parse_args()

    # You can override some defaults here if you want:
    # pred_args.num_epochs = 100
    # pred_args.lr = 1e-4
    # pred_args.exp_dir = "prediction_summary/cond_predictor_clean/..."

    pred_args.device = device

    pred_args.exp_dir = f"{pred_args.save_dir}/{pred_args.name}"

    if not os.path.isdir(pred_args.exp_dir):
        os.makedirs(pred_args.exp_dir)

    #with open(os.path.join(pred_args.exp_dir, "args_clean.txt"), "w") as f:
        #json.dump(pred_args.__dict__, f, indent=2)
    with open(os.path.join(pred_args.exp_dir, "args_clean.txt"), "w") as f:
        args_dict = dict(pred_args.__dict__)
        if "device" in args_dict:
            args_dict["device"] = str(args_dict["device"])
        json.dump(args_dict, f, indent=2)

    main(pred_args, device)
