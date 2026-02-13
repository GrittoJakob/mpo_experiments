import torch
import gymnasium as gym
import numpy as np

def _info_vec(info, key: str, num_envs: int, default: float = 0.0) -> np.ndarray:
    """VectorEnv info: dict -> values are arrays/lists per env OR scalars."""
    if not isinstance(info, dict) or key not in info:
        return np.full((num_envs,), float(default), dtype=np.float32)
    v = info[key]
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v, dtype=np.float32)
        # safety: if scalar slipped through
        if arr.shape == ():
            return np.full((num_envs,), float(arr), dtype=np.float32)
        return arr.astype(np.float32, copy=False)
    return np.full((num_envs,), float(v), dtype=np.float32)




def collect_rollout_vec_env(env: gym.vector.AsyncVectorEnv, args, actor, replaybuffer, device, buffer_gpu):
    """
    Collect exactly ONE episode per env, store it, then stop when ALL envs have finished (done once).
    Done envs may be reset to allow stepping remaining envs, but are no longer recorded.
    """
    num_envs = int(env.num_envs)
    print("Num envs:", num_envs)

    episodes_cpu = []
    total_steps_collected = 0
    actor.eval()

    # per-env episode buffers
    states      = [[] for _ in range(num_envs)]
    actions     = [[] for _ in range(num_envs)]
    next_states = [[] for _ in range(num_envs)]
    rews        = [[] for _ in range(num_envs)]
    terms       = [[] for _ in range(num_envs)]
    truncs      = [[] for _ in range(num_envs)]
    task_inv    = [[] for _ in range(num_envs)]
    vel_rew     = [[] for _ in range(num_envs)]
    pos_rew     = [[] for _ in range(num_envs)]
    progress    = [[] for _ in range(num_envs)]

    # record exactly the first episode per env
    recording = np.ones((num_envs,), dtype=bool)

    obs, info = env.reset()

    with torch.no_grad():
        while recording.any():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            action = np.asarray(actor.action(obs_t), dtype=np.float32)
            # optional sanity check:
            # assert action.ndim == 2 and action.shape[0] == num_envs, (action.shape, num_envs)

            if args.use_action_clipping:
                action = np.clip(action, args.action_space_low, args.action_space_high)

            next_obs, reward, terminated, truncated, info = env.step(action)

            reward     = np.asarray(reward, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated  = np.asarray(truncated, dtype=bool)
            done       = terminated | truncated

            next_obs_store = np.asarray(next_obs).copy()
            next_obs_store = _apply_final_observation(next_obs_store, info, done)

            # task-specific info vectors
            if args.task_mode in ("inverted", "inverted_multi_task"):
                v_rew = _info_vec(info, "velocity_reward", num_envs, 0.0)
                t_inv = _info_vec(info, "task_direction",  num_envs, 0.0)
                p_rew = np.zeros((num_envs,), dtype=np.float32)
                prog  = np.zeros((num_envs,), dtype=np.float32)
            elif args.task_mode == "target_goal":
                prog  = _info_vec(info, "goal_progress",   num_envs, 0.0)
                v_rew = _info_vec(info, "velocity_reward", num_envs, 0.0)
                p_rew = _info_vec(info, "position_reward", num_envs, 0.0)
                t_inv = np.zeros((num_envs,), dtype=np.float32)
            else:
                v_rew = p_rew = prog = t_inv = np.zeros((num_envs,), dtype=np.float32)

            # store transitions ONLY for envs still recording their first episode
            for i in range(num_envs):
                if not recording[i]:
                    continue

                obs_i      = np.asarray(obs[i], dtype=np.float32).reshape(-1)
                act_i      = np.asarray(action[i], dtype=np.float32).reshape(-1)
                next_obs_i = np.asarray(next_obs_store[i], dtype=np.float32).reshape(-1)

                states[i].append(obs_i)
                actions[i].append(act_i)
                next_states[i].append(next_obs_i)

                rews[i].append(float(reward[i]))
                terms[i].append(bool(terminated[i]))
                truncs[i].append(bool(truncated[i]))

                task_inv[i].append(float(t_inv[i]))
                vel_rew[i].append(float(v_rew[i]))
                pos_rew[i].append(float(p_rew[i]))
                progress[i].append(float(prog[i]))

            total_steps_collected += int(recording.sum())

            # flush episodes that ended this step (only those still recording)
            flush = done & recording
            if flush.any():
                for i in np.where(flush)[0]:
                    if len(states[i]) > 0:
                        if buffer_gpu:
                            replaybuffer.store_episode_stacked(
                                np.asarray(states[i],      dtype=np.float32),
                                np.asarray(actions[i],     dtype=np.float32),
                                np.asarray(next_states[i], dtype=np.float32),
                                np.asarray(rews[i],        dtype=np.float32),
                                np.asarray(terms[i],       dtype=np.float32),
                                np.asarray(truncs[i],      dtype=np.float32),
                                np.asarray(task_inv[i],    dtype=np.float32),
                                np.asarray(vel_rew[i],     dtype=np.float32),
                                np.asarray(pos_rew[i],     dtype=np.float32),
                                np.asarray(progress[i],    dtype=np.float32),
                            )
                        else:
                            episodes_cpu.append(list(zip(
                                states[i], actions[i], next_states[i], rews[i],
                                terms[i], truncs[i], task_inv[i], vel_rew[i], pos_rew[i], progress[i]
                            )))

                    # clear buffers and stop recording this env forever
                    states[i].clear(); actions[i].clear(); next_states[i].clear()
                    rews[i].clear(); terms[i].clear(); truncs[i].clear()
                    task_inv[i].clear(); vel_rew[i].clear(); pos_rew[i].clear(); progress[i].clear()

                    recording[i] = False

            # if env does NOT autoreset, we must reset done envs so we can keep stepping others
            autoreset = isinstance(info, dict) and ("final_observation" in info)
            if (not autoreset) and done.any():
                reset_obs, _ = env.reset(options={"reset_mask": done})
                next_obs = np.asarray(next_obs)
                next_obs[done] = reset_obs[done]

            obs = next_obs

    actor.train()
    if (not buffer_gpu) and episodes_cpu:
        replaybuffer.store_episodes(episodes_cpu)

    return total_steps_collected


def _apply_final_observation(next_obs: np.ndarray, info: dict, done: np.ndarray) -> np.ndarray:
    """
    If VectorEnv uses autoreset, Gymnasium may put terminal obs into info['final_observation']
    and next_obs is already reset-obs. We want terminal next_state for done transitions.
    """
    if not isinstance(info, dict) or "final_observation" not in info:
        return next_obs

    final_obs = info["final_observation"]

    # final_obs can be object array with None for non-done envs
    if isinstance(final_obs, np.ndarray) and final_obs.dtype == object:
        idx = np.where(done)[0]
        for i in idx:
            if final_obs[i] is not None:
                next_obs[i] = np.asarray(final_obs[i])
        return next_obs

    try:
        final_obs_arr = np.asarray(final_obs)
        next_obs[done] = final_obs_arr[done]
    except Exception:
        pass
    return next_obs