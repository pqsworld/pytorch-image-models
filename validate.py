#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import imp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import csv
import glob
import json
import time
import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from collections import OrderedDict
from contextlib import suppress

from timm.models import (
    create_model,
    apply_test_time_pool,
    load_checkpoint,
    is_model,
    list_models,
)
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    RealLabelsImagenet,
)
from timm.utils import (
    accuracy_pos_neg,
    accuracy_pos_neg_class,
    AverageMeter,
    natural_key,
    setup_default_logging,
    set_jit_fuser,
    decay_batch_step,
    check_batch_size_retry,
)

from timm.data.dataset import (
    IterableImageDataset,
    ImageDataset,
    ImageFolder_LF,
    ImageFolder_LFID,
)

has_apex = False
try:
    from apex import amp

    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument(
    "--checkpoint",
    # default=r"/home/panq/vendor/pytorch-image-models/mn_adamw/20220718-113954-mobilenetv3_small_100/model_best.pth.tar",
    # default=r"/home/panq/vendor/out/train/lif/pqcfg_4321_1/model_best.pth.tar",
    # default=r"/home/panq/vendor/out/train/lif/pqcfg_4321/model_best.pth.tar",
    default=r"/home/panq/vendor/out/train/pq_lif_3/4321/last.pth.tar",
    # default=r"/home/panq/vendor/pytorch-image-models/out/mn_conv_para_plat/20220718-233251-mobilenetv3_small_100/model_best.pth.tar",
    # default=r"/home/panq/vendor/pytorch-image-models/output/train/20220708-151123-mobilenetv3_small_100-32/model_best.pth.tar",
    # default=r"/home/panq/vendor/pytorch-image-models/out/300_cos/20220719-144319-mobilenetv3_small_100/model_best.pth.tar",
    # default=r"/home/panq/vendor/out/train/spoof/class4/4/model_best.pth.tar",
    # default=r"/home/panq/vendor/pytorch-image-models/out/300_plat/20220719-193332-mobilenetv3_small_100/model_best.pth.tar",
    # default="out/mobilevit2_3/20220718-080118-mobilevitv2_050",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--split-pos-neg",
    action="store_false",
    default=True,
    help="split accuracy to positive and negitive spaces",
)
parser.add_argument(
    "--channels-1to3",
    action="store_false",
    default=True,
    help="transfrom 1 channels image to 3image in Transforms.Compose",
)
parser.add_argument(
    "data",
    default="/hdd/file-input/panq/dataset/noid_6159_newmaterial",
    metavar="DIR",
    help="path to dataset",
)
parser.add_argument(
    "--dataset",
    "-d",
    metavar="NAME",
    default="torch/id",
    help="dataset type (default: ImageFolder/ImageTar if empty)",
)
parser.add_argument(
    "--split",
    metavar="NAME",
    default="test",
    help="dataset split (default: validation)",
)
parser.add_argument(
    "--dataset-download",
    action="store_true",
    default=False,
    help="Allow download of dataset for torch/ and tfds/ datasets that support it.",
)
parser.add_argument(
    "--model",
    "-m",
    metavar="NAME",
    # default="mobilevitv2_050",
    # default="mobilevit_s",
    default="mobilenetv3_small_100",
    help="model architecture (default: dpn92)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 2)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--input-size",
    default=[3, 160, 32],
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
parser.add_argument(
    '--use-train-size',
    action='store_true',
    default=False,
    help='force use of train input size, even when test size is specified in pretrained cfg',
)
parser.add_argument(
    '--crop-pct',
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop pct",
)
parser.add_argument(
    "--mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override mean pixel value of dataset",
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default=None,
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument(
    "--num-classes", type=int, default=2, help="Number classes in dataset"
)
parser.add_argument(
    "--class-map",
    default="",
    type=str,
    metavar="FILENAME",
    help='path to class to idx mapping file (default: "")',
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--log-freq",
    default=1,
    type=int,
    metavar="N",
    help="batch logging frequency (default: 10)",
)

parser.add_argument(
    "--pretrained",
    dest="pretrained",
    default=False,
    action="store_true",
    help="use pre-trained model",
)
parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
parser.add_argument(
    "--test-pool", dest="test_pool", action="store_true", help="enable test time pool"
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=True,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=True,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument(
    "--amp",
    action="store_true",
    default=False,
    help="Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.",
)
parser.add_argument(
    "--apex-amp",
    action="store_true",
    default=False,
    help="Use NVIDIA Apex AMP mixed precision",
)
parser.add_argument(
    "--native-amp",
    action="store_true",
    default=False,
    help="Use Native Torch AMP mixed precision",
)
parser.add_argument(
    "--tf-preprocessing",
    action="store_true",
    default=False,
    help="Use Tensorflow preprocessing pipeline (require CPU TF installed",
)
parser.add_argument(
    "--use-ema",
    dest="use_ema",
    action="store_true",
    default=False,
    help="use ema version of weights if present",
)
parser.add_argument(
    "--torchscript",
    dest="torchscript",
    action="store_true",
    help="convert model torchscript for inference",
)
scripting_group.add_argument(
    '--aot-autograd',
    default=False,
    action='store_true',
    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)",
)
parser.add_argument(
    "--fuser",
    default="",
    type=str,
    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')",
)
parser.add_argument(
    "--results-file",
    default="/home/panq/vendor/out/valid",
    type=str,
    help="Output FOLDER csv file for validation results (summary)",
)
parser.add_argument(
    "--real-labels",
    default="",
    type=str,
    metavar="FILENAME",
    help="Real labels JSON file for imagenet evaluation",
)
parser.add_argument(
    "--valid-labels",
    default="",
    type=str,
    metavar="FILENAME",
    help="Valid label indices txt file for validation of partial label space",
)
parser.add_argument(
    '--retry',
    default=False,
    action='store_true',
    help='Enable batch size decay & retry for single model validation',
)


def accuracy_class(output, target, path):
    """Computes the accuracy over the {all,positive and negative classes} predictions for the specified values"""
    accu = {}

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    # TARGET calued to shape of PRED.saved for pos_nev.
    targ = target.reshape(1, -1).expand_as(pred)
    correct = pred.eq(targ)
    corr_class = torch.squeeze(correct)
    for i in range(len(target)):
        log = ''
        if '/1neg/假手指按物料整理第1和2类' in path[i]:
            log = 'class12'
        elif '/1neg/假手指按物料整理第3类' in path[i]:
            log = 'class3'
        elif '/1neg/假手指按物料整理第4类' in path[i]:
            log = 'class4'
        elif '/0fp/' in path[i]:
            log = 'class0'
        else:
            print("data error")
            input()

        if corr_class[i] == True:
            log = log + ' pred1'
        else:
            log = log + ' pred0'
        _logger.info(path[i] + log)
    accu["all"] = correct[:1].reshape(-1).float().sum(0) * 100.0 / target.size(0)
    assert accu["all"] == accu["all"]

    correct_pos = (pred == 1) * (targ == 1)
    correct_neg = (pred == 0) * (targ == 0)
    sum0 = (target == 0).sum(0)
    sum1 = (target == 1).sum(0)

    if sum1 == 0:
        accu["pos"] = torch.Tensor([100.0]).to(target.data.device)
    else:
        accu["pos"] = (
            correct_pos[:1].reshape(-1).float().sum(0) * 100.0 / (target == 1).sum(0)
        )

    if sum0 == 0:
        accu["neg"] = torch.Tensor([100.0]).to(target.data.device)
    else:
        accu["neg"] = (
            correct_neg[:1].reshape(-1).float().sum(0) * 100.0 / (target == 0).sum(0)
        )

    assert accu["pos"] == accu["pos"]
    assert accu["neg"] == accu["neg"]

    return accu


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info("Validating in mixed precision with native PyTorch AMP.")
    elif args.apex_amp:
        _logger.info("Validating in mixed precision with NVIDIA APEX AMP.")
    else:
        _logger.info("Validating in float32. AMP not enabled.")

    if args.fuser:
        set_jit_fuser(args.fuser)

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript,
    )
    if args.num_classes is None:
        assert hasattr(
            model, "num_classes"
        ), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info("Model %s created, param count: %d" % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args), model=model, use_test_size=True, verbose=True
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(
            model, data_config, use_test_size=True
        )

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)
    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level="O1")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ]
    )
    dataset = create_dataset(
        root=args.data,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
        transforms=test_transform,
    )
    test_set = ImageFolder_LF(args.data, test_transform)
    if args.valid_labels:
        with open(args.valid_labels, "r") as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(
            dataset.filenames(basename=True), real_json=args.real_labels
        )
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config["crop_pct"]
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        channels_1to3=args.channels_1to3,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=16
    )
    batch_time = AverageMeter()
    losses = AverageMeter()
    v_all = AverageMeter()
    v_pos = AverageMeter()
    v_neg = AverageMeter()

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn(
            (args.batch_size,) + tuple(data_config["input_size"])
        ).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        end = time.time()
        for batch_idx, (input, target, path) in enumerate(loader):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)

            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)
            accuracy_class(output.detach(), target, path)

            # measure accuracy and record loss
            accu = accuracy_pos_neg(output.detach(), target, split_pos_neg=True)
            losses.update(loss.item(), input.size(0))
            v_all.update(accu["all"].item(), target.size(0))
            v_pos.update(accu["pos"].item(), (target == 1).sum(0).item())
            v_neg.update(accu["neg"].item(), (target == 0).sum(0).item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    "Test: [{0:>4d}/{1}]  "
                    "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Acc@All: {va.val:>7.4f} ({va.avg:>7.4f})  "
                    "Acc@Pos: {vp.val:>7.4f} ({vp.avg:>7.4f})  "
                    "Acc@Neg: {vn.val:>7.4f} ({vn.avg:>7.4f})  ".format(
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses,
                        va=v_all,
                        vp=v_pos,
                        vn=v_neg,
                    )
                )

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        # FIXME # pqsworld : this is not Now
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        v_all, v_pos, v_neg = v_all.avg, v_pos.avg, v_neg.avg

    results = OrderedDict(
        model=args.model,
        acc_all=round(v_all, 4),
        acc_all_err=round(100 - v_all, 4),
        acc_pos=round(v_pos, 4),
        acc_pos_err=round(100 - v_pos, 4),
        acc_neg=round(v_neg, 4),
        acc_neg_err=round(100 - v_neg, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config["input_size"][-1],
        crop_pct=crop_pct,
        interpolation=data_config["interpolation"],
    )

    _logger.info(
        " * Acc@All {:.3f} ({:.3f}) Acc@Pos {:.3f} ({:.3f}) Acc@Neg {:.3f} ({:.3f})".format(
            results["acc_all"],
            results["acc_all_err"],
            results["acc_pos"],
            results["acc_pos_err"],
            results["acc_neg"],
            results["acc_neg_err"],
        )
    )

    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = "Unknown"
    while batch_size >= 1:
        args.batch_size = batch_size
        torch.cuda.empty_cache()
        try:
            torch.cuda.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            if "channels_last" in error_str:
                break
            _logger.warning(
                f'"{error_str}" while running validation. Reducing batch size to {batch_size} for retry.'
            )
        batch_size = batch_size // 2
    results["error"] = error_str
    _logger.error(f"{args.model} failed to validate ({error_str}).")
    return results


def main():
    args = parser.parse_args()
    experiment = args.checkpoint.split('/')[-2] + '_last.log'
    setup_default_logging(log_path="out/logger/" + experiment)
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + "/*.f.tar")
        checkpoints += glob.glob(args.checkpoint + "/*.pth")
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == "all":
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(
                pretrained=True, exclude_filters=["*_in21k", "*_in22k", "*_dino"]
            )
            model_cfgs = [(n, "") for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, "") for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        time_d = datetime.datetime.now().strftime("_%Y_%m_%d_%H")
        results_file = os.path.join(args.results_file, time_d) or "./results-all.csv"
        _logger.info(
            "Running bulk validation on these pretrained models: {}".format(
                ", ".join(model_names)
            )
        )
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if "error" in r:
                    continue
                if args.checkpoint:
                    r["checkpoint"] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x["acc_all"], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            results = validate(args)
    # output results in JSON to stdout w/ delimiter for runner script
    print(f"--result\n{json.dumps(results, indent=4)}")


def write_results(results_file, results):
    with open(results_file, mode="w") as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == "__main__":
    main()
