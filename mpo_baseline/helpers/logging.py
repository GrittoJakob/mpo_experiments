import wandb
from torch.utils.tensorboard import SummaryWriter

def logging_wandb(writer,args, replaybuffer, Runtime,  stats_m_step, stats_e_step, critic_update_stats, grad_updates, num_steps):
    
    
    # Compute mean reward/return in replay buffer
    mean_reward_buffer = replaybuffer.mean_reward()
    mean_return_buffer = replaybuffer.mean_return() 
    mean_episode_len = mean_return_buffer/(mean_reward_buffer+ 1e-8)                  
    
    # Timing
    writer.add_scalar("time/rollout_time_sec", Runtime.rollout, grad_updates)
    writer.add_scalar("time/E-Step", Runtime.E_step, grad_updates)
    writer.add_scalar("time/M-Step",Runtime.M_step, grad_updates)
    writer.add_scalar("time/critic_update", Runtime.Critic_update, grad_updates)
    writer.add_scalar("time/evaluation", Runtime.Eval, grad_updates)
    writer.add_scalar("time/sample_from_buffer", Runtime.Sample_from_buffer, grad_updates)
    writer.add_scalar("time/t_critic_foward_pass", Runtime.target_critic_forward, grad_updates)
    writer.add_scalar("time/sample_actions", Runtime.Sample_actions, grad_updates)

    #  Buffer
    writer.add_scalar("buffer/size", len(replaybuffer) ,grad_updates)
    writer.add_scalar("buffer/mean_return", mean_return_buffer , grad_updates)
    writer.add_scalar("buffer/mean_reward_per_step", mean_reward_buffer ,grad_updates)
    writer.add_scalar("buffer/mean_episode_len", mean_episode_len, grad_updates)
    writer.add_scalar("buffer/total_num_steps", num_steps, grad_updates)
    if args.task_mode in ("inverted", "inverted_multi_task"):
        writer.add_scalar("buffer/mean_vel_return", replaybuffer.mean_vel_ret(), grad_updates)
        writer.add_scalar("buffer/mean_vel_reward", replaybuffer.mean_vel_rew(), grad_updates)
        # mean_pos_ret, mean_neg_return = replaybuffer.mean_vel_pos_neg_ret()
        # writer.add_scalar("buffer/mean_vel_pos_return", mean_pos_ret, grad_updates)
        # writer.add_scalar("buffer/mean_vel_neg_return", mean_neg_return, grad_updates)
    if args.task_mode == "target_goal":
        writer.add_scalar("buffer/mean_position_return", replaybuffer.mean_pos_ret(), grad_updates)
        writer.add_scalar("buffer/mean_position_reward", replaybuffer.mean_pos_rew(), grad_updates)
        writer.add_scalar("buffer/mean_progress", replaybuffer.mean_progress(), grad_updates)
        writer.add_scalar("buffer/mean_vel_return", replaybuffer.mean_vel_ret(), grad_updates)
        writer.add_scalar("buffer/mean_vel_reward", replaybuffer.mean_vel_rew(), grad_updates)

    # M-Step Loggingd
    writer.add_scalar("m-step/loss_p", stats_m_step["loss_p"], grad_updates)
    writer.add_scalar("m-step/loss_l", stats_m_step["loss_l"], grad_updates)
    writer.add_scalar("m-step/C_mu_mean", stats_m_step["C_mu_mean"], grad_updates)
    writer.add_scalar("m-step/C_sigma_mean", stats_m_step["C_sigma_mean"], grad_updates)
    writer.add_scalar("m-step/eta_mu_mean", stats_m_step["eta_mu_mean"], grad_updates)
    writer.add_scalar("m-step/eta_mu_max", stats_m_step["eta_mu_max"], grad_updates)
    writer.add_scalar("m-step/eta_mu_min", stats_m_step["eta_mu_min"], grad_updates)
    writer.add_scalar("m-step/eta_sigma_mean", stats_m_step["eta_sigma_mean"], grad_updates)
    writer.add_scalar("m-step/eta_sigma_min", stats_m_step["eta_sigma_min"], grad_updates)
    writer.add_scalar("m-step/eta_sigma_max", stats_m_step["eta_sigma_max"], grad_updates)
    writer.add_scalar("m-step/mu_mean", stats_m_step["mu_mean"], grad_updates)
    writer.add_scalar("m-step/std_mean", stats_m_step["std_mean"], grad_updates)
    writer.add_scalar("m-step/std_min", stats_m_step["std_min"], grad_updates)
    writer.add_scalar("m-step/std_max", stats_m_step["std_max"], grad_updates)
    writer.add_scalar("m-step/mu_min", stats_m_step["mu_min"], grad_updates)
    writer.add_scalar("m-step/mu_max", stats_m_step["mu_max"], grad_updates)
    
    # E-Step Logging
    writer.add_scalar("e-step/eta_dual", stats_e_step["eta_dual"].item(), grad_updates)
    # weights diagnostics (always present)
    writer.add_scalar("e-step/norm_target_q_mean", stats_e_step["norm_target_q_mean"].item(), grad_updates)
    writer.add_scalar("e-step/norm_target_q_min",  stats_e_step["norm_target_q_min"].item(),  grad_updates)
    writer.add_scalar("e-step/norm_target_q_max",  stats_e_step["norm_target_q_max"].item(),  grad_updates)
    writer.add_scalar("e-step/loss_dual",          stats_e_step["loss_dual"].item(),          grad_updates)
    # optional: only if action penalty is enabled / present in stats
    if "eta_penalty" in stats_e_step:
        writer.add_scalar("e-step/eta_penalty", stats_e_step["eta_penalty"].item(), grad_updates)

        writer.add_scalar("e-step/diff_out_abs_mean", stats_e_step["diff_out_abs_mean"].item(), grad_updates)
        writer.add_scalar("e-step/diff_out_abs_max",  stats_e_step["diff_out_abs_max"].item(),  grad_updates)

        writer.add_scalar("e-step/norm_weights_mean", stats_e_step["norm_weights_mean"].item(), grad_updates)
        writer.add_scalar("e-step/norm_weights_min",  stats_e_step["norm_weights_min"].item(),  grad_updates)
        writer.add_scalar("e-step/norm_weights_max",  stats_e_step["norm_weights_max"].item(),  grad_updates)

        writer.add_scalar("e-step/penalty_weights_mean", stats_e_step["penalty_weights_mean"].item(), grad_updates)
        writer.add_scalar("e-step/penalty_weights_min",  stats_e_step["penalty_weights_min"].item(),  grad_updates)
        writer.add_scalar("e-step/penalty_weights_max",  stats_e_step["penalty_weights_max"].item(),  grad_updates)

        writer.add_scalar("e-step/loss_penalty", stats_e_step["loss_penalty"].item(), grad_updates)


    # Retrace Logging
    writer.add_scalar("critic_update/q_loss", critic_update_stats["critic_loss"], grad_updates)
    writer.add_scalar("critic_update/q_current_mean", critic_update_stats["q_current_mean"], grad_updates)
    writer.add_scalar("critic_update/q_target_mean", critic_update_stats["q_target_mean"], grad_updates)
    writer.add_scalar("critic_update/terminated_rate", critic_update_stats["terminated_rate"], grad_updates)
    writer.add_scalar("critic_update/truncated_rate", critic_update_stats["truncated_rate"], grad_updates)
