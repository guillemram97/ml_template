import logging
from torch import nn

from .heads import RegressionHead


logger = logging.getLogger(__name__)

HEAD_T5 = ['lm_head']
HEAD = HEAD_T5


def replace_head(model, head_class, ac_kwargs):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_head(module, head_class, ac_kwargs)

        if isinstance(module, nn.Linear):
            if name in HEAD_T5:
                new_linear = head_class(module.weight, module.bias, **ac_kwargs)
                setattr(model, name, new_linear)


def update_weights(model, adapter_class):
    for module in model.children():
        if len(list(module.children())) > 0:
            update_weights(module, adapter_class)
        if isinstance(module, adapter_class):
            module.update()


class HeadLayer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        freeze: bool,
        ac_kwargs: dict,
    ):
        super().__init__()
        self.model = model
        self.head_class = RegressionHead
        self.ac_kwargs = ac_kwargs

        replace_head(
            model=self.model,
            head_class=RegressionHead,
            ac_kwargs=ac_kwargs,
        )

    def update_weights(self):
        update_weights(self.model, self.adapter_class)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)
        return outputs
