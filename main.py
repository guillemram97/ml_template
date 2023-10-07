from utils import (
    parse_args,
    setup_basics,
    neptune_log,
    set_seeds,
)
from utils.online_logs import (
    update_online_metrics,
    reset_avg_online_metrics,
    get_online_metrics_mult,
    log_avg_online,
    log_test,
    log_final,
)
import numpy as np
from metrics import Metric
from handler import handler_LLM
from custom_model import custom_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from task import (
    get_task,
    make_datacollator,
)
import pdb
import copy
import gc

logger = get_logger(__name__)


def main():
    args = parse_args()
    accelerator = Accelerator()
    run = setup_basics(accelerator, logger, args)

    # Pre-Logging
    run["args"] = vars(args)
    set_seeds(args.seed)

    task = get_task(
        accelerator=accelerator,
        args=args,
        model=None,
    )
    model = custom_model(args, task, run, accelerator)
    model.train()

if __name__ == "__main__":
    main()
