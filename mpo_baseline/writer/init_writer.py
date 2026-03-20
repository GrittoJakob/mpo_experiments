import wandb
from torch.utils.tensorboard import SummaryWriter
    
    
def init_writer(args):
    
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=args.run_name,
        save_code=True,
    )
    wandb.define_metric("grad_updates")
    wandb.define_metric("video", step_metric="grad_upates")


    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    return writer