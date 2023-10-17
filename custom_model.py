from adapters.load_adapters import AdapterModel
from heads.heads import RegressionHead
from train_utils import (
    load_optimizer,
    evaluate_model,
    train_epoch,
    get_hparams,
    get_model,
)
import torch.nn as nn
import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    get_scheduler,
)
from utils import (
    setup_basics,
    EarlyStopper,
    neptune_log,
    set_seeds,
)
import copy
from task import (
    get_task,
)
from metrics import Metric
import pdb
import time
import numpy as np

logger = get_logger(__name__)

LOG_TRAIN = True


class custom_model:
    def __init__(self, args, task, run, accelerator):
        self.cache = []
        self.task_name = args.task_name
        self.seed = args.seed
        self.target = args.target
        self.args = get_hparams(args, self.task_name)
        self.test = task.data["test_dataloader"]
        self.train_dataloader = task.data["train_dataloader"]
        self.eval_dataloader = task.data["eval_dataloader"]
        self.run = run
        self.seed = args.seed
        self.accelerator = accelerator
        self.iteration = 0
        self.save_checkpoint = args.save_checkpoint
        self.metric = Metric(self.args)
        self.metric_test = Metric(self.args)
        self.dic_classes = None

    def init_model(self):
        set_seeds(self.seed)
        model = get_model(self.args)
        self.model = AdapterModel(
            model,
            freeze=True,
            ac_kwargs={
                "r": self.args.r,
                "lora_scaling": self.args.lora_scaling,
                "seed": self.seed,
            },
        )
        head = RegressionHead(self.model.model.config.hidden_size)
        self.model.model.lm_head = head
        for name, param in self.model.model.lm_head.named_parameters():
            param.requires_grad = True
        return

    def init_checkpoint(self, PATH):
        self.model.load_state_dict(torch.load(PATH))
        self.model.cuda()
        return

    def evaluate(self):
        self.metric_test.reset()
        test_metric_gold = evaluate_model(
            model=self.model,
            accelerator=self.accelerator,
            eval_dataloader=self.test,
            metric=self.metric_test,
            args=self.args,
        )

        if self.run is not None:
            stats = {
                "test_gold_acc": test_metric_gold[0],
            }

            for suffix in self.suffixes:
                neptune_log(
                    run=self.run,
                    pref=f"test/" + suffix,
                    stats=stats,
                    epoch=self.iteration,
                )

    def train(self):
        torch.cuda.empty_cache()
        self.early_stopper = EarlyStopper(self.args.early_stop)
        self.iteration += 1
        if self.seed is not None:
            set_seed(self.args.seed)
        self.metric.reset()

        # we train from scratch
        self.init_model()

        logger.info(f"  Running task {self.task_name}")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")

        # Re-initialise lr_scheduler + optimized
        optimizer = load_optimizer(self.model, self.args)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(
                self.args.warmup * self.args.num_train_epochs * len(self.train_dataloader.dataset) #aixo esta be? no hauria de ser len(train_dataloader.dataset)?
            ),
            num_training_steps=self.args.num_train_epochs * len(self.train_dataloader.dataset), #aixo esta be? no hauria de ser len(train_dataloader.dataset)?
        )

        # Move to the device
        self.model, optimizer, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, lr_scheduler
        )

        for epoch in range(0, self.args.num_train_epochs):
            total_loss = train_epoch(
                model=self.model,
                train_dataloader=self.train_dataloader,
                accelerator=self.accelerator,
                lr_scheduler=lr_scheduler,
                optimizer=optimizer,
                args=self.args,
                dic_classes=self.dic_classes,
            )

            if (
                epoch % self.args.eval_every_epochs == 0
                or epoch == self.args.num_train_epochs - 1
            ):
                eval_metrics = evaluate_model(
                    model=self.model,
                    accelerator=self.accelerator,
                    eval_dataloader=self.eval_dataloader,
                    metric=self.metric,
                    args=self.args,
                    dic_classes=self.dic_classes,
                    target=self.target,
                )
                self.model.cpu()

                # early stopper requires an increasingly increasing metric
                # we can use epochs instead of eval_metrics[0]
                self.early_stopper.update(eval_metrics[0], self.model)
                self.model.cuda()

                log_msg = f"Epoch {epoch} -----> Average_Train_loss: {total_loss / len(self.train_dataloader.dataset)} ===== Eval_metric: {eval_metrics[0]}"
                logger.info(log_msg)

                if self.run is not None and LOG_TRAIN:
                    self.run[f"{self.iteration}-eval"].log(eval_metrics[0], step=epoch)

            # log metrics are desactivated
            if self.run is not None and LOG_TRAIN:
                stats = {
                    "loss": total_loss / len(self.train_dataloader.dataset),
                    "main_lr": optimizer.param_groups[0]["lr"],
                }

                neptune_log(
                    run=self.run,
                    pref=f"{self.iteration}-train/",
                    stats=stats,
                    epoch=epoch,
                 )

            if self.early_stopper.should_finish():
                break

        # copying from a cpu
        self.model.cpu()
        self.model = copy.deepcopy(self.early_stopper.get_best())
        self.model = self.early_stopper.get_best().cuda()
        del self.early_stopper.best_model

        self.evaluate()
        if self.save_checkpoint != 'no':
            PATH_DEST = 'checkpoints/'+self.task_name+'/'+str(self.seed)+'_'+str(len(self.train_dataloader.dataset)+len(eval_dataloader.dataset))+'.pt'
            torch.save(self.model.state_dict(), PATH_DEST)