try:
    import wandb
except ImportError:
    wandb = None


def _wandb_is_active():
    return wandb is not None and getattr(wandb, "run", None) is not None


def _to_python_number(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value

def logging(metrics, step, writer: None,):
    
    if writer is not None:
        for key, value in metrics.items():
            if key == "step":
                continue
            writer.add_scalar(key, _to_python_number(value), step)
        writer.flush()

    if _wandb_is_active():
        wandb.log(
            {key: _to_python_number(value) for key, value in metrics.items()},
            step=step,
        )

