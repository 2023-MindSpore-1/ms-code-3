# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

from tqdm import tqdm

from models.model_crossattn import VisionTransformer, CONFIGS

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule, ConstantLRSchedule
from utils.data_utils import get_loader
import math
import itertools
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import x2ms_adapter
import x2ms_adapter.util_api as util_api

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logger = logging.getLogger(__name__)

class triplet_loss(nn.Cell):
    def __init__(self):
        super().__init__()
    
    def construct(self, grd_global, sat_global, args):
        dist_array = 2.0 - 2.0 * x2ms_adapter.matmul(sat_global, grd_global.T)
        matrix_diag = nn.MatrixDiag()
        pos_dist = matrix_diag(dist_array)
        pair_n = args.train_batch_size * (args.train_batch_size - 1.0)

        triplet_dist_g2s = pos_dist - dist_array
        log = ops.Log()
        exp = ops.Exp()
        loss_g2s = x2ms_adapter.sum(log(1.0 + exp(triplet_dist_g2s * args.loss_weight)))/pair_n
        expand_dims = ops.ExpandDims()
        triplet_dist_s2g = expand_dims(pos_dist, 1) - dist_array
        loss_s2g = x2ms_adapter.sum(log(1.0 + exp(triplet_dist_s2g * args.loss_weight)))/pair_n
        loss = (loss_g2s + loss_s2g) / 2.0

        return loss



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return x2ms_adapter.tensor_api.mean((preds == labels))


def save_model(args, model_grd, model_sat,optimizer):

    model_checkpoint = os.path.join(args.output_dir, "model_checkpoint.pth")
    checkpoint = {
        'model_grd':x2ms_adapter.state_dict(model_grd),
        'model_sat':x2ms_adapter.state_dict(model_sat),
        'optimizer': x2ms_adapter.state_dict(optimizer)
    }
    x2ms_adapter.save(checkpoint, model_checkpoint)
    
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    model_grd = VisionTransformer(config, args.img_size)
    model_sat = VisionTransformer(config, args.img_size_sat)

    # load pretrained model
    model_grd.load_from(np.load(args.pretrained_dir))
    model_sat.load_from(np.load(args.pretrained_dir))

    model_grd.to(args.device)
    model_sat.to(args.device)

    num_params = count_parameters(model_grd) + count_parameters(model_sat)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)

    logger.info("Total Parameter: \t%2.1fM" % num_params)

    return args, model_grd, model_sat


def count_parameters(model):
    params = sum(x2ms_adapter.tensor_api.numel(p) for p in x2ms_adapter.get_params(model) if p.requires_grad)
    return params/1000000


def valid(args, model_grd, model_sat, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model_grd.set_train(False)
    model_sat.set_train(False)
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    loss_fct = triplet_loss()

    sat_global_descriptor = x2ms_adapter.zeros([8884, 768]).to(args.device)
    grd_global_descriptor = x2ms_adapter.zeros([8884, 768]).to(args.device)
    val_i =0
    for step, (x_grd, x_sat) in enumerate(epoch_iterator):
        
        x_grd=x_grd.to(args.device)
        x_sat=x_sat.to(args.device)
        
        grd_global = model_grd(x_grd)
        sat_global = model_sat(x_sat)


        eval_loss = loss_fct(grd_global, sat_global, args)
        eval_losses.update(x2ms_adapter.tensor_api.item(eval_loss))

        sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global
        grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global
        val_i += sat_global.shape[0]

        
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    print('   compute accuracy')
    accuracy_1 = 0.0
    accuracy_5 = 0.0

    data_amount = 0.0
    dist_array = 2.0 - 2.0 * x2ms_adapter.matmul(sat_global_descriptor, grd_global_descriptor.T)
    print('start')
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = x2ms_adapter.sum(dist_array[:, i] < gt_dist)
        if prediction < 1:
            accuracy_1 += 1.0
        if prediction < 5:
            accuracy_5 += 1.0
        data_amount += 1.0
    accuracy_1 /= data_amount
    accuracy_5 /= data_amount
    print("Valid Accuracy: %f" % (accuracy_1*100.0))


    # save eval result
    file = './Result/'+ args.dataset + '/' + str(args.model_type) + '_accuracy.txt'
    if not os.path.exists('./Result/'+ args.dataset):
        os.makedirs('./Result/'+ args.dataset)
    with open(file, 'a') as file:
        file.write(str(global_step) + ' ' + ' : ' + str(accuracy_1*100.0) + '  '+ str(accuracy_5*100.0) + '\n')

    # print the valid information
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy_1)

    writer.add_scalar("test/accuracy", scalar_value=accuracy_1, global_step=global_step)

    return accuracy_1, accuracy_5


def train(args, model_grd, model_sat):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = util_api.SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler

    optimizer = nn.AdamWeightDecay(itertools.chain(x2ms_adapter.get_params(model_grd), x2ms_adapter.get_params(model_sat)),
                                learning_rate=args.learning_rate,
                                eps=1e-6,
                                weight_decay=args.weight_decay)
                    
    t_total = args.num_steps
    
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    if args.fp16:
        [model_grd, model_sat], optimizer = util_api.amp_initialize(models=[model_grd,model_sat],
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    x2ms_adapter.cuda_device_count() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    # loss function
    criterion = triplet_loss()

    model_grd.zero_grad()
    model_sat.zero_grad()

    losses = AverageMeter()
    global_step, best_acc = 0, 0


    while True:
        model_grd.set_train()
        "WITH_LOSS_CELL_PLACEHOLDER"
        "TRAIN_ONE_STEP_CELL_PLACEHOLDER"
        model_sat.set_train()
        "WITH_LOSS_CELL_PLACEHOLDER"
        "TRAIN_ONE_STEP_CELL_PLACEHOLDER"

        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, (x_grd, x_sat) in enumerate(epoch_iterator):
            
            x_grd, x_sat=x_grd.to(args.device), x_sat.to(args.device)

            grd_global = model_grd(x_grd)
            sat_global = model_sat(x_sat)


            loss = criterion(grd_global, sat_global, args)
        

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # if args.fp16:
            #     scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(x2ms_adapter.tensor_api.item(loss)*args.gradient_accumulation_steps)
                if args.fp16:
                    util_api.clip_grad_norm(util_api.amp_master_params(optimizer), args.max_grad_norm)
                else:
                    util_api.clip_grad_norm(list(x2ms_adapter.get_params(model_grd))+list(x2ms_adapter.get_params(model_sat)), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                if global_step % args.eval_every == 0:

                    accuracy, accuracy_5 = valid(args, model_grd, model_sat, writer, test_loader, global_step)
                    
                    if best_acc < accuracy:
                        save_model(args, model_grd, model_sat,optimizer)
                        best_acc = accuracy

                    model_grd.set_train()
                    "WITH_LOSS_CELL_PLACEHOLDER"
                    "TRAIN_ONE_STEP_CELL_PLACEHOLDER"
                    model_sat.set_train()
                    "WITH_LOSS_CELL_PLACEHOLDER"
                    "TRAIN_ONE_STEP_CELL_PLACEHOLDER"

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CVUSA", "CVACT"], default="CVUSA",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "R50-ViT-L_16", "R50-ViT-L_32","R50-ViT-B_32"],
                        default="R50-ViT-B_16",
                        help="Which variant to use.")

    parser.add_argument("--polar", type=int,choices=[1,0],
                        default=1,
                        help="polar transform or not")

    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--dataset_dir", default="./CVUSA/", type=str,
                    help="The dataset path.")


    parser.add_argument("--img_size", default=(128, 512), type=int,
                        help="Ground Resolution size")
    parser.add_argument("--img_size_sat", default=(128, 512), type=int,
                        help="Sat Resolution size")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1110, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0.03, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=222000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--loss_weight", default=10, type=float,
                        help="loss_weight")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = x2ms_adapter.Device("cuda" if x2ms_adapter.is_cuda_available() else "cpu")
        args.n_gpu = x2ms_adapter.cuda_device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        x2ms_adapter.cuda_set_device(args.local_rank)
        device = x2ms_adapter.Device("cuda", args.local_rank)
        x2ms_adapter.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))


    # Model & Tokenizer Setup
    args, model_grd, model_sat = setup(args)

    # Training
    train(args, model_grd, model_sat)


if __name__ == "__main__":
    main()
