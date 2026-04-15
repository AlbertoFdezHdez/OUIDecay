from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))

from oui_decay_core import (  # noqa: E402
    OUICollector,
    OUIDecayScheduler,
    build_optimizer_with_module_groups,
    list_dynamic_modules,
)


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def main() -> None:
    torch.manual_seed(7)

    model = TinyCNN(num_classes=10)
    optimizer, group_info = build_optimizer_with_module_groups(
        model,
        optimizer_name='adamw',
        lr=3e-4,
        base_wd=5e-4,
        dynamic_module_names=list(list_dynamic_modules(model).keys()),
        betas=(0.9, 0.999),
        adamw_fused_mode='auto',
        device_type='cpu',
    )
    collector = OUICollector(list_dynamic_modules(model), sample_mode='random', seed=7)
    scheduler = OUIDecayScheduler(
        optimizer,
        group_info.module_group_indices,
        base_wd=5e-4,
        update_gap=4,
        s1=0.6666,
        s2=5.0,
        window=5,
    )

    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step, (xb, yb) in enumerate(loader, start=1):
        collector.reset_step()
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        scheduler.step(step, collector.step_values())
        if step % 2 == 0:
            print(f'step={step} loss={float(loss):.4f} wd={scheduler.current_wd()}')

    collector.close()


if __name__ == '__main__':
    main()
