import numpy as np
from tqdm import tqdm
import time
from GymAirQuality import SensorTransmissionEnv


def QLearning(env, beta, Nepisodes, alpha):
    Q = np.zeros((51, 11, 51, 51, 3))
    ep_rewards = []

    pbar = tqdm(range(Nepisodes), desc='Q-Learning', unit='ep', dynamic_ncols=True)
    for ep in pbar:
        state, _ = env.reset()
        total_r = 0.0
        done = False
        eps = max(0.01, 1.0 - ep / (0.8 * Nepisodes))

        while not done:
            theta, b, theta_hat, max_val = state
            acts = env.valid_actions(b)

            if np.random.rand() < eps:
                a = np.random.choice(acts)
            else:
                a = acts[np.argmax(Q[theta, b, theta_hat, max_val][acts])]

            next_state, r, term, trunc, _ = env.step(a)
            done = term or trunc

            nt, nb, nth, nmv = next_state
            next_acts = env.valid_actions(nb)
            best_next = np.max(Q[nt, nb, nth, nmv][next_acts])

            Q[theta, b, theta_hat, max_val, a] += alpha * (
                r + beta * best_next - Q[theta, b, theta_hat, max_val, a]
            )
            total_r += r
            state = next_state

        ep_rewards.append(total_r)
        if ep % 100 == 0:
            pbar.set_postfix(eps=f'{eps:.3f}', avg_r=f'{np.mean(ep_rewards[-100:]):.2f}')

    np.save('rewards1.npy', np.array(ep_rewards))

    Q_masked = Q.copy()
    Q_masked[:, :env.eta, :, :, 1] = -np.inf
    Q_masked[:, :env.eta, :, :, 2] = -np.inf
    return np.argmax(Q_masked, axis=4).astype(int)


def QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha):
    Q = np.zeros((51, 11, 51, 51, 3))
    lam = env.lam
    eta = env.eta
    B = env.B
    P_sol = env.P_solar
    ep_rewards = []

    pbar = tqdm(range(Nepisodes), desc='Structural-SK', unit='ep', dynamic_ncols=True)
    for ep in pbar:
        state, _ = env.reset()
        total_r = 0.0
        done = False
        eps = max(0.01, 1.0 - ep / (0.8 * Nepisodes))

        while not done:
            theta, b, theta_hat, max_val = state
            acts = env.valid_actions(b)

            if np.random.rand() < eps:
                a = np.random.choice(acts)
            else:
                a = acts[np.argmax(Q[theta, b, theta_hat, max_val][acts])]

            next_state, r_obs, term, trunc, _ = env.step(a)
            done = term or trunc
            nt = next_state[0]

            t_r, th_r = theta * 0.02, theta_hat * 0.02
            if a == 0:
                diff = abs(t_r - th_r)
                exp_r = -(diff if t_r <= th_r else 1.5 * diff)
            elif a == 1:
                diff = abs(t_r - th_r)
                loss_nack = diff if t_r <= th_r else 1.5 * diff
                exp_r = -(1 - lam) * loss_nack
            else:
                mv_r = max_val * 0.02
                d_ack = abs(t_r - mv_r)
                loss_ack = d_ack if t_r <= mv_r else 1.5 * d_ack
                d_nack = abs(t_r - th_r)
                loss_nack = d_nack if t_r <= th_r else 1.5 * d_nack
                exp_r = -(lam * loss_ack + (1 - lam) * loss_nack)

            exp_nv = 0.0
            nmv_nack = max(max_val, nt)

            for d, pd in enumerate(P_sol):
                if a == 0:
                    nb = min(B, b + d)
                    na = env.valid_actions(nb)
                    exp_nv += pd * np.max(Q[nt, nb, theta_hat, nmv_nack][na])
                else:
                    nb = min(B, b - eta + d)
                    na = env.valid_actions(nb)
                    nth_ack = theta if a == 1 else max_val
                    v_ack = np.max(Q[nt, nb, nth_ack, nt][na])
                    v_nack = np.max(Q[nt, nb, theta_hat, nmv_nack][na])
                    exp_nv += pd * (lam * v_ack + (1 - lam) * v_nack)

            Q[theta, b, theta_hat, max_val, a] += alpha * (
                exp_r + beta * exp_nv - Q[theta, b, theta_hat, max_val, a]
            )
            total_r += r_obs
            state = next_state

        ep_rewards.append(total_r)
        if ep % 100 == 0:
            pbar.set_postfix(eps=f'{eps:.3f}', avg_r=f'{np.mean(ep_rewards[-100:]):.2f}')

    np.save('rewards2.npy', np.array(ep_rewards))

    Q_masked = Q.copy()
    Q_masked[:, :env.eta, :, :, 1] = -np.inf
    Q_masked[:, :env.eta, :, :, 2] = -np.inf
    return np.argmax(Q_masked, axis=4).astype(int)


if __name__ == '__main__':
    env = SensorTransmissionEnv() # Create and instance of the environment.

    Nepisodes = 10000   # Number of episodes to train
    alpha = 0.1         # Learning rate
    beta = 0.98         # Discount factor

    t0 = time.time()

    # Learn the optimal policies using two different TD learning approaches
    policy1 = QLearning(env, beta, Nepisodes, alpha)
    policy2 = QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha)

    # Save the policies
    np.save('policy1.npy', policy1)
    np.save('policy2.npy', policy2)

    env.close() # Close the environment

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed / 60:.1f} min')
