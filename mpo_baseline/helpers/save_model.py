import torch
import os
import time

def save_actor_critic(mpo, args, num_steps: int, grad_updates: int, out_dir: str = "checkpoints"):
    """
    Speichert Actor + Critic gemeinsam (atomar) in einer .pt-Datei.
    Erwartet mpo.actor und mpo.critic.
    """
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "actor_state_dict": mpo.actor.state_dict(),
        "critic_state_dict": mpo.critic.state_dict(),
        "num_steps": int(num_steps),
        "grad_updates": int(grad_updates),
        "run_name": getattr(args, "run_name", None),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "args": vars(args) if hasattr(args, "__dict__") else None,
    }

    filename = f"{payload['run_name'] or 'run'}_ac_steps{num_steps}_gu{grad_updates}.pt"
    final_path = os.path.join(out_dir, filename)
    tmp_path = final_path + ".tmp"

    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
    print(f"[SAVE] Actor+Critic saved to: {final_path}")

