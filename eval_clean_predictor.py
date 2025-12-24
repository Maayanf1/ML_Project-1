# eval_clean_predictor.py
#
# Evaluate the CLEAN Case 2 predictor (model_clean.pt) on train/val/test.
# - No diffusion noise
# - t = 0
# - Uses EDM normalization only
#
# Assumes you already ran train_clean_predictor.py and have:
#   <exp_dir>/model_clean.pt
#   <exp_dir>/args_clean.txt  (optional)


import json
import random
from time import time, sleep
import warnings
import os

import matplotlib.pyplot as plt

from utils.utils_edm import (
    remove_mean_with_mask,
    assert_mean_zero_with_mask,
    normalize,
)

from data.aromatic_dataloader import create_data_loaders, AromaticDataset
from prediction_args import PredictionArgs
from train_clean_predictor import (
    get_cond_predictor_model,
    check_mask_correct,
    compute_clean_loss,
)
from utils.helpers import get_cond_predictor_args

# from utils.args_edm import Args_EDM
# from models_edm import get_model

warnings.simplefilter(action="ignore", category=FutureWarning)
import numpy as np
import torch
import os

from torch import linspace


def val_epoch(tag, cond_predictor, dataloader, args, t_fix=None):
    cond_predictor.eval()
    with torch.no_grad():
        start_time = time()
        loss_list = []
        rl_loss = []
        error_list = []
        # with tqdm(dataloader, unit="batch", desc=f"{tag} {epoch}") as tepoch:
        for i, (x, node_mask, edge_mask, node_features, y) in enumerate(dataloader):
            x = x.to(args.device)
            y = y.to(args.device)
            node_mask = node_mask.to(args.device).unsqueeze(2)
            edge_mask = edge_mask.to(args.device)
            h = node_features.to(args.device)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            loss, err = compute_clean_loss(
                cond_predictor,
                x,
                h,
                node_mask,
                edge_mask,
                y,
            )
            error_list.append(err)
            loss_list.append(loss.item())
            rl_loss.append(dataloader.dataset.rescale_loss(loss).item())

            # tepoch.set_postfix(loss=np.mean(loss_list).item())
        print(
            f"[{tag}] loss: {np.mean(loss_list):.4f}+-{np.std(loss_list):.4f}, "
            f"L1 (rescaled): {np.mean(rl_loss):.4f}, "
            f" in {int(time() - start_time)} secs"
        )
        std = dataloader.dataset.std
        # print(f"Error: {torch.cat(error_list).mean(0)}")
        # print(f"Error: {torch.cat(error_list).mean(0).cpu()*std}")
        print()
        sleep(0.01)

    return (torch.cat(error_list).cpu() * std[None, :]).mean().item()


def main(pred_args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(pred_args)

    cond_predictor = get_cond_predictor_model(pred_args, train_loader.dataset)

    ##No need to pass over all the diffusion times 

    ##TODO : remove the test part 
    print("Test (clean, t=0 only):")
    test_mae = val_epoch("test", cond_predictor, test_loader, pred_args)
    print(f"Test MAE (clean): {test_mae:.4f}")

    print("Val (clean, t=0 only):")
    val_mae = val_epoch("val", cond_predictor, val_loader, pred_args)
    print(f"Val MAE (clean): {val_mae:.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    ##TODO: change the path and add the files
    pred_args = get_cond_predictor_args(
        f"C:\Users\Maayan Farkash\Documents\project ML\our files\PBHs-design"
    )

    print("\n\nArgs:", pred_args)

    # Where the magic is
    main(pred_args)