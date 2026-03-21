import wandb
from torch.utils.tensorboard import SummaryWriter

def logging_wandb(writer, replaybuffer,  stats_m_step, stats_e_step, critic_update_stats, grad_updates, num_steps):
    
    
    # Compute mean reward/return in replay buffer
    mean_reward_buffer = replaybuffer.mean_reward() 
   
    #  Buffer
    writer.add_scalar("buffer/size", len(replaybuffer) ,grad_updates)
    writer.add_scalar("buffer/mean_reward_per_step", mean_reward_buffer ,grad_updates)
    writer.add_scalar("buffer/total_num_steps", num_steps, grad_updates)

    # M-Step Loggingd
    writer.add_scalar("m-step/loss_p", stats_m_step["loss_p"], grad_updates)
    writer.add_scalar("m-step/loss_l", stats_m_step["loss_l"], grad_updates)
    writer.add_scalar("m-step/C_mu_mean", stats_m_step["C_mu_mean"], grad_updates)
    writer.add_scalar("m-step/C_sigma_mean", stats_m_step["C_sigma_mean"], grad_updates)
    writer.add_scalar("m-step/mu_mean", stats_m_step["mu_mean"], grad_updates)
    writer.add_scalar("m-step/std_mean", stats_m_step["std_mean"], grad_updates)
        
    # E-Step Logging
    writer.add_scalar("e-step/eta_dual", stats_e_step["eta_dual"].item(), grad_updates)
    writer.add_scalar("e-step/loss_dual",          stats_e_step["loss_dual"].item(), grad_updates)
    # optional: only if action penalty is enabled / present in stats
    if "eta_penalty" in stats_e_step:
        writer.add_scalar("e-step/eta_penalty", stats_e_step["eta_penalty"].item(), grad_updates)
        writer.add_scalar("e-step/loss_penalty", stats_e_step["loss_penalty"].item(), grad_updates)


    # Retrace Logging
    writer.add_scalar("critic_update/q_loss", critic_update_stats["critic_loss"], grad_updates)
    writer.add_scalar("critic_update/q_current_mean", critic_update_stats["q_current_mean"], grad_updates)
    writer.add_scalar("critic_update/q_target_mean", critic_update_stats["q_target_mean"], grad_updates)
