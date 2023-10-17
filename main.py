from utils import (
    parse_args,
    setup_basics,
    neptune_log,
    set_seeds,
)
import numpy as np
from metrics import Metric
from custom_model import custom_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from task import (
    get_task,
)
import pdb
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
