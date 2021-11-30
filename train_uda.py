import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_pretrained_bert import BertAdam

from utils.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import sys

def get_args(parser):
    #UDA config
    parser.add_argument("--sup_size", type=int, default=-1, help="Number of supervised pairs to use. -1: all training samples. 4000: 4000 supervised examples")
    parser.add_argument("--aug_copy", type=int, default=0, help="Number of different augmented data generated.")
    parser.add_argument("--unsup_ratio",type=float, default=0,help="The ratio between batch size of unlabeled data and labeled data,i.e., unsup_ratio * train_batch_size is the batch_size for unlabeled data.Do not use the unsupervised objective if set to 0.")
    parser.add_argument("--tsa",type=str, default='', choices=['', "linear_schedule", "log_schedule", "exp_schedule"], help="anneal schedule of training signal annealing. tsa='' means not using TSA. See the paper for other schedules")
    parser.add_argument("--uda_confidence_thresh", type=float,default=-1, help="The threshold on predicted probability on unsupervised data. If set UDA loss will only be calculated on unlabeled examples whose largest probability is larger than the threshold")
    parser.add_argument("--uda_confidence_thresh",type=float, default=-1,
    help="The threshold on predicted probability on unsupervised data. If set,"
    "UDA loss will only be calculated on unlabeled examples whose largest"
    "probability is larger than the threshold")
    parser.add_argument("--uda_confidence_thresh",type=float, default=-1,
    help="The threshold on predicted probability on unsupervised data. If set,"
    "UDA loss will only be calculated on unlabeled examples whose largest"
    "probability is larger than the threshold")
    
    parser.add_argument("--uda_softmax_temp", -1, type=float,
    help="The temperature of the Softmax when making prediction on unlabeled"
    "examples. -1 means to use normal Softmax")
    parser.add_argument( "ent_min_coeff", type=float, default=0,help="")
    parser.add_argument("--unsup_coeff", type=int, default=1,
    help="The coefficient on the UDA loss. setting unsup_coeff to 1 works for most settings.When you have extermely few samples, consider increasing unsup_coeff")
    parser.add_argument('--moving_average_decay', type=float, default=0.9999,
    help='Moving average decay rate.')
    
def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion

def get_optimizer(model, args):
    if args.model in ["bert", "bert_img","mmbt","mult","mmbt_uda","mmbt2","imgbert"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer

def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch, "eval")
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics

def _kl_divergence_with_logits(p_logits, q_logits):
    p = F.softmax(p_logits,dim=-1)
    log_p = F.log_softmax(p_logits,dim=-1)
    log_q = F.log_softmax(q_logits,dim=-1)

    kl = torch.sum(p * (log_p - log_q),-1)
    return kl

def model_forward(i_epoch, model, args, criterion, batch, partition="train"):
    txt, segment, mask, img, img_aug, tgt= batch
    
    if args.model =="FastImage":
        img=img.cuda()
        out =model(img)
    if args.model =="FastText":
        txt=txt.cuda()
        out =model(txt)
    if args.model =="MultFusion":
        txt=txt.cuda()
        img=img.cuda()
        out =model(txt,img)
    
    if args.model in ["bert","bert_img"]:
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        out = model(txt, mask, segment)
    if args.model in ["imgbert"]:
        img=img.cuda()
        out = model(img)
    if args.model == "mult":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        img= img.cuda()
        audio=audio.cuda()
        out =model(txt,mask,segment,img,audio)
    if args.model in ["mmbt","mult2","mmbt2"]:
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        out = model(txt, mask, segment, img)
    if args.model == "mmbt_uda":
        if partition=="train":
            txt, img, img_aug = txt.cuda(), img.cuda(), img_aug.cuda()
            mask, segment = mask.cuda(), segment.cuda()
            out = model(txt, mask, segment, img)
            q_logits = model(txt, mask, segment, img_aug)
            model.eval()
            p_logits = model(txt, mask, segment, img)
            model.train()
            loss_aug = torch.sum(_kl_divergence_with_logits(p_logits, q_logits))
        else:
            txt, img = txt.cuda(), img.cuda()
            mask, segment = mask.cuda(), segment.cuda()
            out = model(txt, mask, segment, img)
            
        

    tgt = tgt.cuda()
    loss = criterion(out, tgt)
    if args.model == "mmbt_uda" and partition=="train":
        loss = loss+loss_aug
    return loss, out, tgt

def train(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)

    '''
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)
    '''

    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")
    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        args.partition="train"
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _ = model_forward(i_epoch, model, args, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        args.partition="eval"
        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    args.partition="eval"
    test_metrics = model_eval(
        np.inf, test_loader, model, args, criterion, store_preds=True
    )
    log_metrics(f"Test", test_metrics, args, logger)
    
    
def test(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
    model.cuda()

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = model_eval(
        np.inf, test_loader, model, args, criterion, store_preds=True
    )
    log_metrics(f"Test", test_metrics, args, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    
    if args.type=="train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
