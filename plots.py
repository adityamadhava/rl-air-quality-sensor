import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

r1 = np.load('rewards1.npy')
r2 = np.load('rewards2.npy')
p1 = np.load('policy1.npy')  # shape (51, 11, 51, 51) -> theta, b, theta_hat, max_val


def smooth(x, w=200):
    return np.convolve(x, np.ones(w) / w, mode='valid')


# ── Plot 1: Reward curves ──────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))
eps = np.arange(len(smooth(r1)))
ax1.plot(eps, smooth(r1), label='Q-Learning', color='steelblue')
ax1.plot(eps, smooth(r2), label='Structural Knowledge Q-Learning', color='darkorange')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Sum of Reward (smoothed, w=200)')
ax1.set_title('TD Learning — Training Reward Comparison')
ax1.legend()
fig1.tight_layout()
fig1.savefig('reward_plot.png', dpi=150)


# ── Plot 2: Action heatmap — theta vs theta_hat (b=5, max_val=theta) ──────────
# shows action 0 near the diagonal (theta ≈ theta_hat)
action_grid = np.array([[p1[t, 5, th, t] for th in range(51)] for t in range(51)])

cmap3 = mcolors.ListedColormap(['steelblue', 'seagreen', 'tomato'])
norm3 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap3.N)

fig2, ax2 = plt.subplots(figsize=(7, 6))
im = ax2.imshow(action_grid, origin='lower', aspect='auto',
                cmap=cmap3, norm=norm3, extent=[0, 1, 0, 1])
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.6, label='θ = θ̂')
ax2.set_xlabel('θ̂  (central station estimate)')
ax2.set_ylabel('θ  (true pollution)')
ax2.set_title('Optimal Action — θ vs θ̂\n(battery = 5, max_val = θ)')
cbar = fig2.colorbar(im, ax=ax2, ticks=[0, 1, 2])
cbar.set_ticklabels(["0: Don't transmit", '1: Transmit θ', '2: Transmit max'])
ax2.legend()
fig2.tight_layout()
fig2.savefig('action_heatmap.png', dpi=150)


# ── Plot 3: Action 0 rate vs estimation gap (θ̂ − θ) ─────────────────────────
# for states where b >= 2 (transmission is feasible)
# hypothesis: action 0 dominates when θ̂ ≈ θ or θ̂ > θ (already overestimating)
diffs, a0_rate = [], []
for diff in range(-50, 51):
    count, total = 0, 0
    for t in range(51):
        th = t + diff          # theta_hat = theta + diff  →  gap = theta_hat - theta
        if 0 <= th < 51:
            subset = p1[t, 2:, th, :]   # b=2..10, all max_val
            total += subset.size
            count += (subset == 0).sum()
    if total > 0:
        diffs.append(diff * 0.02)
        a0_rate.append(count / total)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(diffs, a0_rate, color='steelblue', linewidth=2)
ax3.axvline(0, color='gray', linestyle='--', alpha=0.6)
ax3.set_xlabel('θ̂ − θ   (positive = central station overestimates)')
ax3.set_ylabel("Fraction of states → action 0")
ax3.set_title("Action 0 (Don't Transmit) — Pattern vs Estimation Gap")
fig3.tight_layout()
fig3.savefig('action0_analysis.png', dpi=150)


# ── Plot 4: Action 2 rate vs (θ − θ̂) and (max_val − θ) ─────────────────────
# hypothesis: action 2 dominates when θ > θ̂ (underestimate) and max_val >> θ
diffs2, a2_rate = [], []
for diff in range(-50, 51):
    count, total = 0, 0
    for t in range(51):
        th = t - diff          # theta - theta_hat = diff
        if 0 <= th < 51:
            subset = p1[t, 2:, th, :]
            total += subset.size
            count += (subset == 2).sum()
    if total > 0:
        diffs2.append(diff * 0.02)
        a2_rate.append(count / total)

# also: action 2 rate vs (max_val - theta), for b >= 2, theta_hat < theta
mv_gaps, a2_mv_rate = [], []
for mv_diff in range(0, 51):  # max_val >= theta always
    count, total = 0, 0
    for t in range(51):
        mv = t + mv_diff
        if mv < 51:
            # only states where theta_hat < theta (underestimate)
            for th in range(0, t):
                subset = p1[t, 2:, th, mv]
                total += subset.size
                count += (subset == 2).sum()
    if total > 0:
        mv_gaps.append(mv_diff * 0.02)
        a2_mv_rate.append(count / total)

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 4))

ax4a.plot(diffs2, a2_rate, color='tomato', linewidth=2)
ax4a.axvline(0, color='gray', linestyle='--', alpha=0.6)
ax4a.set_xlabel('θ − θ̂   (positive = central station underestimates)')
ax4a.set_ylabel('Fraction of states → action 2')
ax4a.set_title('Action 2 (Transmit Max) — vs Estimation Gap')

ax4b.plot(mv_gaps, a2_mv_rate, color='tomato', linewidth=2)
ax4b.set_xlabel('max_val − θ   (gap between max tracked and true)')
ax4b.set_ylabel('Fraction of states → action 2')
ax4b.set_title('Action 2 (Transmit Max) — vs Max Buffer Gap\n(given underestimate)')

fig4.tight_layout()
fig4.savefig('action2_analysis.png', dpi=150)

plt.show()
print('Saved: reward_plot.png, action_heatmap.png, action0_analysis.png, action2_analysis.png')
