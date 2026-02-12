import gymnasium as gym
import torch
import numpy as np


def collect_rollout(env, args, actor, replaybuffer, device, buffer_gpu):
    """
    Collect rollouts with the current policy and store them in the replay buffer.

    Stores per transition:
      (state, action, next_state, reward, terminated, truncated)

    Returns:
      total_steps_collected (int)
    """
    episodes_cpu = []  # only used when buffer_gpu == False
    total_steps_collected = 0
    actor.eval()

    with torch.no_grad():
        while total_steps_collected < args.sample_steps_per_iter:
            # Per-episode storage (always lists -> no branching in the step loop)
            states_list, acts_list, next_states_list = [], [], []
            rews_list, terminated_list, truncated_list = [], [], []
            task_invert_list, vel_rew_list, pos_rew_list, progress_list = [], [], [],[]


            state, info = env.reset()

            for _ in range(args.sample_episode_maxstep):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)

                action = actor.action(state_tensor)  # numpy
                if args.use_action_clipping:
                    action = np.clip(action, args.action_space_low, args.action_space_high)

                next_state, reward, terminated, truncated, info = env.step(action)
                total_steps_collected += 1
                
                # velocity rewards
                task_invert = 0.0
                vel_rew = 0.0
                pos_rew = 0.0
                progress = 0.0
                if args.task_mode in ("inverted", "inverted_multi_task"):
                    vel_rew = info['velocity_reward']
                    task_invert = info['task_direction']
                elif args.task_mode == "target_goal":
                    progress = info["goal_progress"]
                    vel_rew = info['velocity_reward']
                    pos_rew = info["position_reward"] 
               

                # Store transition parts
                states_list.append(state)
                acts_list.append(action)
                next_states_list.append(next_state)
                rews_list.append(reward)
                terminated_list.append(terminated)
                truncated_list.append(truncated)
                task_invert_list.append(task_invert)
                vel_rew_list.append(vel_rew)
                pos_rew_list.append(pos_rew)
                progress_list.append(progress)

              

                done = terminated or truncated
                if done:
                    break

                state = next_state

            # Episode finished -> store once, depending on buffer type
            if buffer_gpu:
                states_np           = np.asarray(states_list,      dtype=np.float32)
                actions_np          = np.asarray(acts_list,        dtype=np.float32)
                next_states_np      = np.asarray(next_states_list, dtype=np.float32)
                rewards_np          = np.asarray(rews_list,        dtype=np.float32)
                terminated_np       = np.asarray(terminated_list,  dtype=np.float32)
                truncated_np        = np.asarray(truncated_list,   dtype=np.float32)
                task_invert_np      = np.asarray(task_invert_list, dtype=np.float32)
                vel_rew_np          = np.asarray(vel_rew_list,     dtype=np.float32) 
                pos_rew_np          = np.asarray(pos_rew_list,     dtype=np.float32)
                progress_np         = np.asarray(progress_list,    dtype = np.float32)
                    
                replaybuffer.store_episode_stacked(
                states_np, actions_np, next_states_np, rewards_np, terminated_np, truncated_np, task_invert_np, vel_rew_np, pos_rew_np, progress_np
                )
                
            else:
                # Build episode as list of tuples (CPU buffer)
            
                episode = list(zip(
                    states_list, acts_list, next_states_list, rews_list, terminated_list, truncated_list, task_invert_list, vel_rew_list, pos_rew_list, progress_list
                    ))
                
                episodes_cpu.append(episode)

    actor.train()

    if not buffer_gpu:
        replaybuffer.store_episodes(episodes_cpu)

    return total_steps_collected



# def _to_np(x):
#     if torch.is_tensor(x):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)


# def _info_get(info, key: str, i: int, default: float = 0.0) -> float:
#     """
#     Robust: info kann bei VectorEnv entweder
#     - dict of arrays/lists/scalars
#     - list[dict]
#     sein.
#     """
#     try:
#         if isinstance(info, dict):
#             if key not in info:
#                 return float(default)
#             v = info[key]
#             if isinstance(v, (list, tuple, np.ndarray)):
#                 return float(v[i])
#             return float(v)
#         elif isinstance(info, (list, tuple)):
#             return float(info[i].get(key, default))
#     except Exception:
#         pass
#     return float(default)


# def collect_rollout(env, args, actor, replaybuffer, device, buffer_gpu):
#     """
#     Single-Env and SyncVectorEnv are compatible.

#     It stores the following per transition: (state, action, next_state, reward, terminated, truncated, task_invert, vel_rew, pos_rew, progress).

#     For VectorEnv:
#       - Pro Env genau eine Episode bis done (danach werden Steps dieser Env ignoriert).
#       - stops the "Wave" rollout as soon as all environments are finished or the maximum step is reached.
#       - Repeats Waves until sample_steps_per_iter is reached.

#     Returns:
#     total_steps_collected (int): The number of stored transitions (not "calls to env.step").
#     """
#     episodes_cpu = []
#     total_steps_collected = 0
#     actor.eval()

#     # VectorEnv erkennen
#     num_envs = int(getattr(env, "num_envs", 1))
#     is_vector = num_envs > 1

#     with torch.no_grad():
#         while total_steps_collected < args.sample_steps_per_iter:

#             # -----------------------
#             # If SINGLE ENV
#             # -----------------------
#             if not is_vector:
#                 states_list, acts_list, next_states_list = [], [], []
#                 rews_list, terminated_list, truncated_list = [], [], []
#                 task_invert_list, vel_rew_list, pos_rew_list, progress_list = [], [], [], []

#                 state, info = env.reset()

#                 for _ in range(args.sample_episode_maxstep):
#                     st = torch.as_tensor(state, dtype=torch.float32, device=device)
#                     action = actor.action(st)
#                     action = _to_np(action)

#                     if args.use_action_clipping:
#                         action = np.clip(action, args.action_space_low, args.action_space_high)

#                     next_state, reward, terminated, truncated, info = env.step(action)
#                     total_steps_collected += 1

#                     task_invert = 0.0
#                     vel_rew = 0.0
#                     pos_rew = 0.0
#                     progress = 0.0

#                     if args.task_mode in ("inverted", "inverted_multi_task"):
#                         vel_rew = float(info.get("velocity_reward", 0.0))
#                         task_invert = float(info.get("task_direction", 0.0))
#                     elif args.task_mode == "target_goal":
#                         progress = float(info.get("goal_progress", 0.0))
#                         vel_rew = float(info.get("velocity_reward", 0.0))
#                         pos_rew = float(info.get("position_reward", 0.0))

#                     states_list.append(state)
#                     acts_list.append(action)
#                     next_states_list.append(next_state)
#                     rews_list.append(float(reward))
#                     terminated_list.append(bool(terminated))
#                     truncated_list.append(bool(truncated))
#                     task_invert_list.append(task_invert)
#                     vel_rew_list.append(vel_rew)
#                     pos_rew_list.append(pos_rew)
#                     progress_list.append(progress)

#                     if terminated or truncated:
#                         break
#                     state = next_state

#                 if buffer_gpu:
#                     replaybuffer.store_episode_stacked(
#                         np.asarray(states_list, dtype=np.float32),
#                         np.asarray(acts_list, dtype=np.float32),
#                         np.asarray(next_states_list, dtype=np.float32),
#                         np.asarray(rews_list, dtype=np.float32),
#                         np.asarray(terminated_list, dtype=np.float32),
#                         np.asarray(truncated_list, dtype=np.float32),
#                         np.asarray(task_invert_list, dtype=np.float32),
#                         np.asarray(vel_rew_list, dtype=np.float32),
#                         np.asarray(pos_rew_list, dtype=np.float32),
#                         np.asarray(progress_list, dtype=np.float32),
#                     )
#                 else:
#                     episode = list(
#                         zip(
#                             states_list, acts_list, next_states_list, rews_list,
#                             terminated_list, truncated_list,
#                             task_invert_list, vel_rew_list, pos_rew_list, progress_list
#                         )
#                     )
#                     episodes_cpu.append(episode)

#                 continue

#             # -----------------------
#             # VECTOR ENV (SyncVectorEnv)
#             # -----------------------
#             # For each env sample one episode
#             states = [[] for _ in range(num_envs)]
#             acts = [[] for _ in range(num_envs)]
#             next_states = [[] for _ in range(num_envs)]
#             rews = [[] for _ in range(num_envs)]
#             terms = [[] for _ in range(num_envs)]
#             truncs = [[] for _ in range(num_envs)]
#             task_inverts = [[] for _ in range(num_envs)]
#             vel_rews = [[] for _ in range(num_envs)]
#             pos_rews = [[] for _ in range(num_envs)]
#             progresses = [[] for _ in range(num_envs)]

#             obs, info = env.reset()
#             finished = np.zeros((num_envs,), dtype=bool)

#             for _ in range(args.sample_episode_maxstep):
#                 active = ~finished
#                 if not active.any():
#                     break

#                 obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

               
#                 actions_list = [None] * num_envs
#                 sample_action = None

#                 for i in range(num_envs):
#                     if not active[i]:
#                         continue
#                     a = actor.action(obs_t[i])
#                     a = _to_np(a).astype(np.float32, copy=False)
#                     actions_list[i] = a
#                     if sample_action is None:
#                         sample_action = a

#                 if sample_action is None:
#                     break

#                 zero_action = np.zeros_like(sample_action, dtype=np.float32)
#                 for i in range(num_envs):
#                     if actions_list[i] is None:
#                         actions_list[i] = zero_action

#                 actions_arr = np.stack(actions_list, axis=0)
#                 if args.use_action_clipping:
#                     actions_arr = np.clip(actions_arr, args.action_space_low, args.action_space_high)

#                 next_obs, reward, terminated, truncated, info = env.step(actions_arr)

#                 reward = np.asarray(reward)
#                 terminated = np.asarray(terminated, dtype=bool)
#                 truncated = np.asarray(truncated, dtype=bool)
#                 done = terminated | truncated

                
#                 total_steps_collected += int(active.sum())

      
#                 for i in range(num_envs):
#                     if not active[i]:
#                         continue

#                     task_invert = 0.0
#                     vel_rew = 0.0
#                     pos_rew = 0.0
#                     progress = 0.0

#                     if args.task_mode in ("inverted", "inverted_multi_task"):
#                         vel_rew = _info_get(info, "velocity_reward", i, 0.0)
#                         task_invert = _info_get(info, "task_direction", i, 0.0)
#                     elif args.task_mode == "target_goal":
#                         progress = _info_get(info, "goal_progress", i, 0.0)
#                         vel_rew = _info_get(info, "velocity_reward", i, 0.0)
#                         pos_rew = _info_get(info, "position_reward", i, 0.0)

#                     states[i].append(obs[i])
#                     acts[i].append(actions_arr[i])
#                     next_states[i].append(next_obs[i])
#                     rews[i].append(float(reward[i]))
#                     terms[i].append(bool(terminated[i]))
#                     truncs[i].append(bool(truncated[i]))
#                     task_inverts[i].append(float(task_invert))
#                     vel_rews[i].append(float(vel_rew))
#                     pos_rews[i].append(float(pos_rew))
#                     progresses[i].append(float(progress))

#                     if done[i]:
#                         finished[i] = True

#                 obs = next_obs

            
#             for i in range(num_envs):
#                 if len(states[i]) == 0:
#                     continue

#                 if buffer_gpu:
#                     replaybuffer.store_episode_stacked(
#                         np.asarray(states[i], dtype=np.float32),
#                         np.asarray(acts[i], dtype=np.float32),
#                         np.asarray(next_states[i], dtype=np.float32),
#                         np.asarray(rews[i], dtype=np.float32),
#                         np.asarray(terms[i], dtype=np.float32),
#                         np.asarray(truncs[i], dtype=np.float32),
#                         np.asarray(task_inverts[i], dtype=np.float32),
#                         np.asarray(vel_rews[i], dtype=np.float32),
#                         np.asarray(pos_rews[i], dtype=np.float32),
#                         np.asarray(progresses[i], dtype=np.float32),
#                     )
#                 else:
#                     episode = list(
#                         zip(
#                             states[i], acts[i], next_states[i], rews[i],
#                             terms[i], truncs[i],
#                             task_inverts[i], vel_rews[i], pos_rews[i], progresses[i]
#                         )
#                     )
#                     episodes_cpu.append(episode)

#     actor.train()

#     if (not buffer_gpu) and episodes_cpu:
#         replaybuffer.store_episodes(episodes_cpu)

#     return total_steps_collected
