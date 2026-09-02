"""Chemprop 1.7 entry point for trainable transfer from the LiON checkpoint.

The published LiON checkpoint has a regression readout. Chemprop's standard
``--checkpoint_path`` behavior restores that regression architecture verbatim,
which is incompatible with LNPHub's three-class task. This entry point rebuilds
the model from the current multiclass arguments, transfers every shape-compatible
LiON parameter, and leaves all transferred parameters trainable.
"""

import importlib
import inspect
import re
from pathlib import Path

import torch
from chemprop.args import TrainArgs
from chemprop.models import MoleculeModel
from chemprop.train import cross_validate


def _resolved(path):
    return str(Path(path).expanduser().resolve())


def _transfer_checkpoint(path, current_args, device=None, logger=None):
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    state = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state = state["state_dict"]
    model = MoleculeModel(current_args)
    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for loaded_name, value in loaded_state.items():
        if re.match(r"(encoder\.encoder\.)([Wc])", loaded_name) and not current_args.reaction_solvent:
            name = loaded_name.replace("encoder.encoder", "encoder.encoder.0")
        elif re.match(r"(^ffn)", loaded_name):
            name = loaded_name.replace("ffn", "readout")
        else:
            name = loaded_name

        if name in model_state and model_state[name].shape == value.shape:
            compatible[name] = value
        else:
            skipped.append(loaded_name)

    if not compatible:
        raise RuntimeError("No shape-compatible LiON parameters were found in the pretrained checkpoint.")

    model_state.update(compatible)
    model.load_state_dict(model_state)
    target_device = device if device is not None else current_args.device
    model = model.to(target_device)

    encoder_loaded = sum(name.startswith("encoder.") for name in compatible)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    info(
        "LiON transfer initialization: "
        f"loaded {len(compatible):,} compatible tensors "
        f"({encoder_loaded:,} encoder tensors); skipped {len(skipped):,} incompatible tensors."
    )
    info(f"LiON transfer model trainable parameters: {trainable:,}")
    debug("Skipped LiON tensors: " + (", ".join(skipped) if skipped else "none"))
    return model


def main():
    run_module = importlib.import_module("chemprop.train.run_training")
    standard_loader = run_module.load_checkpoint
    args = TrainArgs().parse_args()
    transfer_paths = {_resolved(path) for path in (args.checkpoint_paths or [])}

    if len(transfer_paths) != 1:
        raise ValueError(
            "LiON transfer training requires exactly one --checkpoint_path and --ensemble_size 1."
        )

    def load_checkpoint(path, device=None, logger=None):
        if _resolved(path) not in transfer_paths:
            return standard_loader(path, device=device, logger=logger)

        caller = inspect.currentframe().f_back
        current_args = caller.f_locals.get("args") if caller is not None else None
        if current_args is None:
            raise RuntimeError("Could not obtain Chemprop's active multiclass training arguments.")
        return _transfer_checkpoint(path, current_args, device=device, logger=logger)

    run_module.load_checkpoint = load_checkpoint
    print(
        "LiON transfer mode: regression checkpoint -> trainable three-class model; "
        f"internal Chemprop folds={args.num_folds}, ensemble={args.ensemble_size}",
        flush=True,
    )
    cross_validate(args=args, train_func=run_module.run_training)


if __name__ == "__main__":
    main()
