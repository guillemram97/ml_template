import os
from transformers import DataCollatorForSeq2Seq, T5Tokenizer
from datasets import arrow_dataset
from torch.utils.data import DataLoader
import datasets
from torch.utils.data import Subset
import torch
from functools import partial
import pandas as pd
import ast
import json
import pdb


class Task:
    def __init__(self, task_name, tokenizer, soft_labels):
        self.task_name = task_name
        DATA_DIR = os.getenv("DATA_PATH")
        self.path = os.path.join(DATA_DIR, task_name)
        self.tokenizer = tokenizer
        self.load_config()
        self.load_data()
        self.soft_labels = soft_labels

    def load_config(self):
        with open(os.path.join(self.path, "config.json")) as f:
            config = json.load(f)
        self.is_classification = config["is_classification"]
        self.is_regression = config["is_regression"]
        if self.is_classification:
            self.classes = config["classes"]
            self.soft_classes = config["soft_classes"]
        self.data_path = config["data"]

    def load_data(self):
        data = {}
        fin = pd.read_csv(
            os.path.join(self.path, self.data_path),
        )
        input = list(fin["input"].values.astype(str))
        output = list(fin["output"].values.astype(str))
        self.raw_data= arrow_dataset.Dataset.from_dict(
            {"input": input, "output": output}
        )

    def load_classes(self):
        self.classes_dict = {}
        self.classes_dict_gold = {}
        for idx, class_name in enumerate(self.classes):
            target = self.tokenizer.encode(class_name, add_special_tokens=False)[0]
            self.classes_dict[self.soft_classes[idx]] = target
            self.classes_dict_gold[class_name] = target
        return

    def preprocess(self, accelerator, args, model=None):
        def process_data_to_model_inputs(is_eval: bool, batch):
            out = {}
            # Tokenizer will automatically set [BOS] <text> [EOS]
            out["input_ids"] = self.tokenizer(
                batch["input"],
                padding=False,
                max_length=args.max_length,
                truncation=True,
            ).input_ids

            if self.is_classification:
                out["gold_soft"] = make_soft(batch["gold_hard"], target="gold")
                if not self.soft_labels:
                    out["llm_soft"] = make_soft(batch["llm_hard"], target="llm")
                else:
                    out["llm_soft"] = select_classes(batch["llm_soft"])
            if self.is_regression:
                out["output"] = [float(num) for num in batch["output"]]
  
            return out

        def collate_for_eval(default_collate, batch):
            inputs = [{"input_ids": x["input_ids"]} for x in batch]
            out = default_collate(inputs)
            out["output"] = [x["output"] for x in batch]
            if self.is_classification:
                out["llm_soft"] = [x["llm_soft"] for x in batch]
                out["gold_soft"] = [x["gold_soft"] for x in batch]
                out["llm_hard"] = [x["llm_hard"] for x in batch]
                out["gold_hard"] = [x["gold_hard"] for x in batch]   
            return out

        def select_classes(batch_soft_labels):
            new_batch = []
            for soft_labels in batch_soft_labels:
                soft_labels = ast.literal_eval(soft_labels)
                # in classification we only have one token
                soft_labels = soft_labels[0]
                new_soft_labels = []
                for key in self.soft_classes:
                    if key in soft_labels:
                        new_soft_labels.append(soft_labels[key])
                    else:
                        new_soft_labels.append(-100)
                new_batch.append(new_soft_labels)
            return new_batch

        def make_soft(batch_hard_labels, target):
            if target == "gold":
                classes_dict = self.classes_dict_gold
            else:
                classes_dict = self.classes_dict
            new_batch = []
            for hard_label in batch_hard_labels:
                new_soft_labels = []
                for label in classes_dict.keys():
                    if label == hard_label:
                        new_soft_labels.append(0)
                    else:
                        new_soft_labels.append(-100)
                new_batch.append(new_soft_labels)
            return new_batch

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=model, padding="longest"
        )
        eval_collator = partial(collate_for_eval, data_collator)

        #processed_data = {}
        #max_samples = getattr(args, f"{split}_samples")
        #self.raw_data = random_subset(
        #    dataset=self.raw_data,
        #    max_samples=max_samples,
        #    seed=args.seed,
        #)

        self.raw_data = arrow_dataset.Dataset.from_list(
            list(self.raw_data)
        )
        aux = self.raw_data.train_test_split(test_size=0.1)
        aux_2 = aux['train'].train_test_split(test_size=0.1)


        train_data = aux_2['train'].map(
            partial(process_data_to_model_inputs, False),
            batched=True,
            batch_size=args.per_device_eval_batch_size,
            remove_columns=self.raw_data.column_names,
        )

        validation_data = aux_2['test'].map(
            partial(process_data_to_model_inputs, True),
            batched=True,
            batch_size=args.per_device_eval_batch_size,
            remove_columns=self.raw_data.column_names,
        )

        test_data = aux['test'].map(
            partial(process_data_to_model_inputs, True),
            batched=True,
            batch_size=args.per_device_eval_batch_size,
            remove_columns=self.raw_data.column_names,
        )




        train_dataloader = DataLoader(
            train_data,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
        )

        eval_dataloader = DataLoader(
            validation_data,
            collate_fn=eval_collator,
            batch_size=args.per_device_eval_batch_size,
        )

        test_dataloader = DataLoader(
            test_data,
            collate_fn=eval_collator,
            batch_size=args.per_device_eval_batch_size,
        )

        train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
            train_dataloader,
            eval_dataloader,
            test_dataloader,
        )

        self.data = {
            "train_dataloader": train_dataloader,
            "eval_dataloader": eval_dataloader,
            "test_dataloader": test_dataloader,
        }

        return


def random_subset(dataset, max_samples: int, seed: int = 42):
    if max_samples >= len(dataset) or max_samples == -1:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, perm[:max_samples].tolist())


def get_task(accelerator, args, model=None):
    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=args.max_length
    )

    # load config, data, and preprocess
    task = Task(args.task_name, tokenizer, args.soft_labels)
    if task.is_classification:
        task.load_classes()
    task.preprocess(accelerator, args, model=None)
    return task
