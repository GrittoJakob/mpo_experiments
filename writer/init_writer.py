from torch.utils.tensorboard import SummaryWriter

def init_writer(args):
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    writer.flush()

    wandb_run = None

    if getattr(args, "wandb_track", False):
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name=args.run_name,
                save_code=True,
            )
            wandb.define_metric("grad_updates")

        except ImportError:
            print("wandb could not be initialized")

    return writer, wandb_run