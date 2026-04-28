import numpy as np
from tqdm import tqdm
import time
from GymAirQuality import SensorTransmissionEnv


def _best_next(Q, theta_idx, eta):
    Qs = Q[theta_idx]                              # (11, 51, 51, 3)
    best = Qs[:, :, :, 0].copy()
    can_tx = (np.arange(Qs.shape[0]) >= eta)[:, None, None]
    best = np.where(can_tx,
                    np.maximum(best, np.maximum(Qs[:, :, :, 1], Qs[:, :, :, 2])),
                    best)
    return best                                    # (11, 51, 51)

def test_policy(Q, eta, N_test=50):
    """Run N_test greedy (eps=0) episodes in a fresh env; return mean episode reward."""
    env = SensorTransmissionEnv()
    total = 0.0
    for _ in range(N_test):
        state, _ = env.reset()
        done = False
        while not done:
            t, b, th, mv = state
            acts = env.valid_actions(b)
            a = acts[int(np.argmax(Q[t, b, th, mv][acts]))]
            state, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
    env.close()
    return total / N_test

def QLearning(env, beta, Nepisodes, alpha, M=10, N_test=50):
    Q = np.zeros((51, 11, 51, 51, 3))
    ep_rewards = []

    pbar = tqdm(range(Nepisodes), desc='Q-Learning', unit='ep', dynamic_ncols=True)
    for ep in pbar:
        state, _ = env.reset()
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
            best_q = np.max(Q[nt, nb, nth, nmv][next_acts])
            Q[theta, b, theta_hat, max_val, a] += alpha * (
                r + beta * best_q - Q[theta, b, theta_hat, max_val, a]
            )
            state = next_state
        if (ep + 1) % M == 0:
            test_r = test_policy(Q, env.eta, N_test)
            ep_rewards.append(test_r)
            pbar.set_postfix(eps=f'{eps:.3f}', test_r=f'{test_r:.2f}')

    np.save('rewards1.npy', np.array(ep_rewards))
    Q_masked = Q.copy()
    Q_masked[:, :env.eta, :, :, 1] = -np.inf
    Q_masked[:, :env.eta, :, :, 2] = -np.inf
    return np.argmax(Q_masked, axis=4).astype(int)


def QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha, M=10, N_test=50):
    """
    From each sample, infers the solar delta and ACK outcome using the known
    deterministic structure of transitions, then updates Q-values for all states
    sharing the same pollution level theta (vectorised over battery, estimate, max).
    """
    Q = np.zeros((51, 11, 51, 51, 3))
    eta = env.eta
    B = env.B
    ep_rewards = []

    b_arr = np.arange(B + 1)   # (11,)
    th_arr = np.arange(51)      # (51,)
    m_arr  = np.arange(51)      # (51,)

    pbar = tqdm(range(Nepisodes), desc='SK-Q-Learning', unit='ep', dynamic_ncols=True)
    for ep in pbar:
        state, _ = env.reset()
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

            # Standard single-state Q update
            next_acts = env.valid_actions(nb)
            best_q = np.max(Q[nt, nb, nth, nmv][next_acts])
            Q[theta, b, theta_hat, max_val, a] += alpha * (
                r + beta * best_q - Q[theta, b, theta_hat, max_val, a]
            )

            # Infer delta: battery at cap makes delta ambiguous — skip multi-update
            if nb < B:
                delta = (nb - b) if a == 0 else (nb - b + eta)
            else:
                state = next_state
                continue

            # Infer ACK from how the estimate changed
            if a == 0:
                ack = False
            elif a == 1:
                if nth == theta and theta != theta_hat:
                    ack = True
                elif nth == theta_hat and theta != theta_hat:
                    ack = False
                else:                  # theta == theta_hat: indistinguishable
                    state = next_state
                    continue
            else:                      # a == 2
                if nth == max_val and max_val != theta_hat:
                    ack = True
                elif nth == theta_hat and max_val != theta_hat:
                    ack = False
                else:                  # max_val == theta_hat: indistinguishable
                    state = next_state
                    continue

            # Best next-Q for all (b2, th2, m2) at next pollution level nt
            bq = _best_next(Q, nt, eta)            # (11, 51, 51)

            # Reward for no-ACK / action-0: only depends on theta_hat2
            th_r   = theta * 0.02
            diff   = np.abs(th_r - th_arr * 0.02)
            r_nack = -np.where(th_r <= th_arr * 0.02, diff, 1.5 * diff)  # (51,)

            m_new_nack = np.maximum(m_arr, nt)     # (51,)

            # ── a2 = 0: valid regardless of current action ──────────────────
            b2_new_a0 = np.minimum(B, b_arr + delta)
            best_a0 = bq[b2_new_a0[:, None, None],
                         th_arr    [None, :, None],
                         m_new_nack[None, None, :]]        # (11, 51, 51)
            td_a0 = r_nack[None, :, None] + beta * best_a0
            Q[theta, :, :, :, 0] += alpha * (td_a0 - Q[theta, :, :, :, 0])

            # ── a2 = 1, 2: only predictable when current action was transmit ─
            if a in (1, 2):
                b2_new_tx = np.minimum(B, np.maximum(0, b_arr - eta) + delta)
                tx_mask   = (b_arr >= eta)[:, None, None]

                if ack:
                    # a2=1 ACK: reward=0, th_new=theta, m_new=nt
                    bst_a1 = bq[b2_new_tx, theta, nt]     # (11,)
                    td_a1  = (beta * bst_a1)[:, None, None]
                    Q[theta, :, :, :, 1] += alpha * tx_mask * (
                        td_a1 - Q[theta, :, :, :, 1])

                    # a2=2 ACK: reward=-E(theta,m2), th_new=m2, m_new=nt
                    diff_m = np.abs(th_r - m_arr * 0.02)
                    r_a2   = -np.where(th_r <= m_arr * 0.02, diff_m, 1.5 * diff_m)
                    bst_a2 = bq[b2_new_tx[:, None], m_arr[None, :], nt]  # (11,51)
                    td_a2  = r_a2[None, :] + beta * bst_a2
                    Q[theta, :, :, :, 2] += alpha * tx_mask * (
                        td_a2[:, None, :] - Q[theta, :, :, :, 2])
                else:
                    # a2=1 and a2=2 no-ACK share the same TD target
                    best_nack = bq[b2_new_tx [:, None, None],
                                   th_arr    [None, :, None],
                                   m_new_nack[None, None, :]]  # (11,51,51)
                    td_nack = r_nack[None, :, None] + beta * best_nack
                    Q[theta, :, :, :, 1] += alpha * tx_mask * (
                        td_nack - Q[theta, :, :, :, 1])
                    Q[theta, :, :, :, 2] += alpha * tx_mask * (
                        td_nack - Q[theta, :, :, :, 2])
            state = next_state
        if (ep + 1) % M == 0:
            test_r = test_policy(Q, eta, N_test)
            ep_rewards.append(test_r)
            pbar.set_postfix(eps=f'{eps:.3f}', test_r=f'{test_r:.2f}')

    np.save('rewards2.npy', np.array(ep_rewards))
    Q_masked = Q.copy()
    Q_masked[:, :env.eta, :, :, 1] = -np.inf
    Q_masked[:, :env.eta, :, :, 2] = -np.inf
    return np.argmax(Q_masked, axis=4).astype(int)


if __name__ == '__main__':
    env = SensorTransmissionEnv()

    Nepisodes = 10000
    alpha = 0.1
    beta = 0.98
    t0 = time.time()
    policy1 = QLearning(env, beta, Nepisodes, alpha)
    policy2 = QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha)
    np.save('policy1.npy', policy1)
    np.save('policy2.npy', policy2)
    env.close()
    elapsed = time.time() - t0
    print(f'\nDone in {elapsed / 60:.1f} min')
