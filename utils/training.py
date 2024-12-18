# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import copy
import math
import os
import sys
from argparse import Namespace
from time import time
from typing import Iterable, Tuple
import logging
import torch
from tqdm import tqdm

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel
from models.utils.future_model import FutureModel

from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import log_extra_metrics, log_accs, Logger
from utils.schedulers import get_scheduler
from utils.stats import track_system_stats

try:
    import wandb
except ImportError:
    wandb = None


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs: the output tensor
        dataset: the continual dataset
        k: the task index
    """
    num_classes = dataset.N_CLASSES
    start_c, end_c = dataset.get_offsets(k)
    outputs[:, :start_c] = -float('inf')
    outputs[:, end_c:num_classes] = -float('inf')


@torch.no_grad()
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, return_loss=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        last: a boolean indicating whether to evaluate only the last task
        return_loss: a boolean indicating whether to return the loss in addition to the accuracy

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task. If return_loss is True, the loss is also returned as a third element.
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    loss_fn = dataset.get_loss()
    avg_loss = 0
    total_len = sum(len(x) for x in dataset.test_loaders) if hasattr(dataset.test_loaders[0], '__len__') else None

    pbar = tqdm(dataset.test_loaders, total=total_len, desc='Evaluating', disable=model.args.non_verbose)
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                if model.args.eval_future and k >= model.current_task:
                    outputs = model.future_forward(inputs)
                else:
                    outputs = model(inputs)

            if return_loss:
                loss = loss_fn(outputs, labels)
                avg_loss += loss.item()

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1
            pbar.set_postfix({f'acc_task_{k+1}': max(0, correct / total * 100)}, refresh=False)
            pbar.set_description(f"Evaluating Task {k+1}", refresh=False)
            pbar.update(1)

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    pbar.close()

    model.net.train(status)
    if return_loss:
        return accs, accs_mask_classes, avg_loss / total
    return accs, accs_mask_classes


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = args.conf_jobnum.split('-')[0]
    name = f'{run_name}_{run_id}'
    mode = 'disabled' if os.getenv('MAMMOTH_TEST', '0') == '1' else os.getenv('WANDB_MODE', 'online')
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name, mode=mode)
    args.wandb_url = wandb.run.get_url()


def _to_device(name: str, x, device):
    if isinstance(x, torch.Tensor):
        if 'label' in name.lower() or 'target' in name.lower():
            return x.to(device, dtype=torch.long)
        return x.to(device)
    return x




def train_single_epoch(model: ContinualModel,
                       train_loader: Iterable,
                       args: Namespace,
                       epoch: int,
                       current_task: int,
                       system_tracker=None,
                       data_len=None,
                       scheduler=None) -> int:
    """
    Trains the model for a single epoch.

    Args:
        model: the model to be trained
        train_loader: the data loader for the training set
        args: the arguments from the command line
        epoch: the current epoch
        current_task: the current task index
        system_tracker: the system tracker to monitor the system stats
        data_len: the length of the training data loader. If None, the progress bar will not show the training percentage
        scheduler: the scheduler for the current epoch

    Returns:
        the number of iterations performed in the current epoch
    """
    train_iter = iter(train_loader)

    i = 0
    previous_time = time()

    mininterval = 0.5 if data_len is not None and data_len > 1000 else 0.1
    pbar = tqdm(train_iter, total=data_len,
                desc=f"Task {current_task + 1} - Epoch {epoch + 1}",
                disable=args.non_verbose, mininterval=mininterval)
    while True:
        try:
            data = next(train_iter)
        except StopIteration:
            break
        if args.debug_mode and i > model.get_debug_iters():
            break
        if args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
            break

        inputs, labels, not_aug_inputs = data[0], data[1], data[2]
        inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
        not_aug_inputs = not_aug_inputs.to(model.device)

        extra_fields = {
            train_loader.dataset.extra_return_fields[k]: _to_device(train_loader.dataset.extra_return_fields[k], data[3 + k], model.device)
            for k in range(len(data) - 3)
        }

        loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch, **extra_fields)

        assert not math.isnan(loss)

        if scheduler is not None and args.scheduler_mode == 'iter':
            scheduler.step()

        if args.code_optimization == 0 and 'cuda' in str(args.device):
            torch.cuda.synchronize()
        system_tracker()
        i += 1

        time_diff = time() - previous_time
        previous_time = time()
        bar_log = {'loss': loss, 'lr': model.opt.param_groups[0]['lr']}
        if data_len:
            ep_h = 3600 / (data_len * time_diff)
            bar_log['ep/h'] = ep_h
        pbar.set_postfix(bar_log, refresh=False)
        pbar.update()

    if scheduler is not None and args.scheduler_mode == 'epoch':
        scheduler.step()


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    is_fwd_enabled = True
    can_compute_fwd_beforetask = True
    random_results_class, random_results_task = [], []

    if not args.nowand:
        initialize_wandb(args)

    if not args.disable_log:
        logger = Logger(args, dataset.SETTING, dataset.NAME, model.NAME)

    model.net.to(model.device)
    torch.cuda.empty_cache()

    with track_system_stats(logger) as system_tracker:
        results, results_mask_classes = [], []

        if args.eval_future:
            results_transf, results_mask_classes_transf = [], []

        if args.start_from is not None:
            for i in range(args.start_from):
                train_loader, _ = dataset.get_data_loaders()
                model.meta_begin_task(dataset)
                model.meta_end_task(dataset)

        if args.loadcheck is not None:
            model, past_res = mammoth_load_checkpoint(args, model)

            if not args.disable_log and past_res is not None:
                (results, results_mask_classes, csvdump) = past_res
                logger.load(csvdump)

            print('Checkpoint Loaded!')

        print(file=sys.stderr)
        start_task = 0 if args.start_from is None else args.start_from
        end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

        if args.eval_future:
            assert isinstance(model, FutureModel), "Model must be an instance of FutureModel to evaluate on future tasks"
            eval_dataset = get_dataset(args)
            for _ in range(dataset.N_TASKS):
                eval_dataset.get_data_loaders()
                model.change_transform(eval_dataset)
                del eval_dataset.train_loader
        else:
            eval_dataset = dataset

        torch.cuda.empty_cache()
        for t in range(start_task, end_task):
            model.net.train()
            train_loader, _ = dataset.get_data_loaders()

            if not issubclass(dataset.__class__, GCLDataset):
                assert issubclass(train_loader.dataset.__class__, MammothDatasetWrapper), "Dataset must be an instance of MammothDatasetWrapper (did you forget to call the `store_masked_loaders`?)"

            if can_compute_fwd_beforetask and is_fwd_enabled and args.enable_other_metrics:
                # try to compute accuracy at the beginning of the task
                try:
                    logging.info("Evaluating model before task (for Forward Transfer metric)...")
                    random_res_class, random_res_task = evaluate(model, dataset, last=True)
                    random_results_class.append(random_res_class)
                    random_results_task.append(random_res_task)
                except Exception as e:
                    logging.info(f"Could not evaluate before `begin_task`, will try after")
                    # will try after the begin_task in case the model needs to setup something
                    can_compute_fwd_beforetask = False

            model.meta_begin_task(dataset)

            if not can_compute_fwd_beforetask and is_fwd_enabled and args.enable_other_metrics:
                if train_loader.dataset.num_times_iterated == 0:  # compute only if the model has not been trained yet
                    try:
                        logging.info("Evaluating model before task (for Forward Transfer metric)...")
                        random_res_class, random_res_task = evaluate(model, dataset, last=True)
                        random_results_class.append(random_res_class)
                        random_results_task.append(random_res_task)
                    except Exception as e:
                        logging.error(f"Model `{model.NAME}` does not support pre-evaluation, will not compute Forward Transfer metric\n{e}")
                        is_fwd_enabled = False
                else:
                    logging.info("Model used the training data, skipping Forward Transfer metric compute")
                    is_fwd_enabled = False

            if not args.inference_only and args.n_epochs > 0:
                if t and args.enable_other_metrics:
                    accs = evaluate(model, eval_dataset, last=True)
                    results[t - 1] = results[t - 1] + accs[0]
                    if dataset.SETTING == 'class-il':
                        results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

                scheduler = get_scheduler(model, args, reload_optim=True) if not hasattr(model, 'scheduler') else model.scheduler

                epoch = 0
                best_ea_metric = None
                best_ea_model = None
                cur_stopping_patience = args.early_stopping_patience
                while True:
                    data_len = None
                    if not isinstance(dataset, GCLDataset):
                        data_len = len(train_loader)

                    model.begin_epoch(epoch, dataset)


                    train_single_epoch(model, train_loader, args, current_task=t, epoch=epoch,
                                    system_tracker=system_tracker, data_len=data_len, scheduler=scheduler)

                    model.end_epoch(epoch, dataset)

                    epoch += 1
                    if args.fitting_mode == 'epochs' and epoch >= model.args.n_epochs:
                        break
                    elif args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
                        break
                    elif args.fitting_mode == 'early_stopping' and epoch % args.early_stopping_freq == 0 and epoch > 0:
                        epoch_accs, _, epoch_loss = evaluate(model, eval_dataset, return_loss=True, last=True)

                        if args.early_stopping_metric == 'accuracy':
                            ea_metric = np.mean(epoch_accs)  # Higher accuracy is better
                        elif args.early_stopping_metric == 'loss':
                            ea_metric = -epoch_loss  # Lower loss is better
                        else:
                            raise ValueError(f'Unknown early stopping metric {args.early_stopping_metric}')

                        # Higher accuracy is better
                        if best_ea_metric is not None and ea_metric - best_ea_metric < args.early_stopping_epsilon:
                            cur_stopping_patience -= args.early_stopping_freq
                            if cur_stopping_patience <= 0:
                                print(f"\nEarly stopping at epoch {epoch} with metric {abs(ea_metric)}", file=sys.stderr)
                                model.load_state_dict({k: v.to(model.device) for k, v in best_ea_model.items()})
                                break
                            print(f"\nNo improvement at epoch {epoch} (best {abs(best_ea_metric)} | current {abs(ea_metric)}). "
                                  f"Waiting for {cur_stopping_patience} epochs to stop.", file=sys.stderr)
                        else:
                            print(f"\nFound better model with metric {abs(ea_metric)} at epoch {epoch}. "
                                  f"Previous value was {abs(best_ea_metric) if best_ea_metric is not None else 'None'}", file=sys.stderr)
                            best_ea_metric = ea_metric
                            best_ea_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                            cur_stopping_patience = args.early_stopping_patience

                    if args.eval_epochs is not None and (epoch > 0 or args.eval_epochs) and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs:
                        epoch_accs = evaluate(model, eval_dataset)

                        log_accs(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)

            model.meta_end_task(dataset)

            accs = evaluate(model, eval_dataset)

            if args.eval_future and t < dataset.N_TASKS - 1:
                transf_accs = accs[0][t + 1:], accs[1][t + 1:]
                accs = accs[0][:t + 1], accs[1][:t + 1]
                results_transf.append(transf_accs[0])
                results_mask_classes_transf.append(transf_accs[1])

            results.append(accs[0])
            results_mask_classes.append(accs[1])

            log_accs(args, logger, accs, t, dataset.SETTING)

            if args.eval_future:
                avg_transf = np.mean([np.mean(task_) for task_ in results_transf])
                print(f"Transfer Metrics  -  AVG Transfer {avg_transf:.2f}")
                if t < dataset.N_TASKS - 1:
                    log_accs(args, logger, transf_accs, t, dataset.SETTING, future=True)

            if args.savecheck:
                save_obj = {
                    'model': model.state_dict(),
                    'args': args,
                    'results': [results, results_mask_classes, logger.dump()],
                    'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                }
                if 'buffer_size' in model.args:
                    save_obj['buffer'] = copy.deepcopy(model.buffer).to('cpu')

                # Saving model checkpoint for the current task
                checkpoint_name = None
                if args.savecheck == 'task':
                    checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_{t}.pt'
                elif args.savecheck == 'last' and t == end_task - 1:
                    checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_last.pt'
                if checkpoint_name is not None:
                    torch.save(save_obj, checkpoint_name)

        if args.validation:
            # Final evaluation on the real test set
            print("Starting final evaluation on the real test set...", file=sys.stderr)
            del dataset
            args.validation = None
            args.validation_mode = 'current'

            final_dataset = get_dataset(args)
            for _ in range(final_dataset.N_TASKS):
                final_dataset.get_data_loaders()
            accs = evaluate(model, final_dataset)

            log_accs(args, logger, accs, 'final', final_dataset.SETTING, prefix="FINAL")

        if args.enable_other_metrics:
            bwt, bwt_mask_class = logger.add_bwt(results, results_mask_classes)
            log_extra_metrics(args, bwt, bwt_mask_class, 'Backward Transfer', t)
            forgetting, forgetting_mask_class = logger.add_forgetting(results, results_mask_classes)
            log_extra_metrics(args, forgetting, forgetting_mask_class, 'Forgetting', t)
            if is_fwd_enabled:
                fwt, fwt_mask_class = logger.add_fwt(results, random_results_class,
                                                     results_mask_classes, random_results_task)
                log_extra_metrics(args, fwt, fwt_mask_class, 'Forward Transfer', t)
            else:
                logging.warning("Forward Transfer metric incompatible with the current model, skipped.")

        system_tracker.print_stats()

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
