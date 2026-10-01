import argparse

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Monte Carlo simulation of shop earnings")
parser.add_argument("--r0", type=float, default=20_000, help="starting monthly revenue (EUR)")
parser.add_argument("--mu", type=float, default=0.05, help="expected annual revenue growth")
parser.add_argument("--sigma", type=float, default=0.25, help="annual revenue volatility")
parser.add_argument("--months", type=int, default=36, help="number of months to simulate")
parser.add_argument("--n-sims", type=int, default=10_000, help="number of simulated paths")
parser.add_argument("--fixed-costs", type=float, default=8_000, help="fixed costs per month (EUR)")
parser.add_argument("--cogs-ratio", type=float, default=0.45, help="cost of goods sold as a share of revenue")
parser.add_argument("--seed", type=int, default=42, help="random seed")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

r0 = args.r0
mu = args.mu
sigma = args.sigma
months = args.months
n_sims = args.n_sims

fixed_costs = args.fixed_costs
cogs_ratio = args.cogs_ratio

mu_m = mu / 12
sigma_m = sigma / np.sqrt(12)

# the -0.5*sigma^2 keeps average growth at mu instead of drifting above it
shocks = rng.normal(mu_m - 0.5 * sigma_m**2, sigma_m, (n_sims, months))
revenue = r0 * np.exp(np.cumsum(shocks, axis=1))

profit = revenue * (1 - cogs_ratio) - fixed_costs
cum_profit = profit.cumsum(axis=1)

final = cum_profit[:, -1]
p05, p50, p95 = np.percentile(final, [5, 50, 95])

print(f"cumulative profit after {months} months, {n_sims:,} simulations")
print(f"  median           {p50:>10,.0f}")
print(f"  5th percentile   {p05:>10,.0f}")
print(f"  95th percentile  {p95:>10,.0f}")
print(f"  ends in the red          {(final < 0).mean():.1%}")
print(f"  has a loss-making month  {(profit < 0).any(axis=1).mean():.1%}")

n_plot = min(300, n_sims)
t = np.arange(1, months + 1)

plt.figure(figsize=(9, 5))
plt.plot(t, cum_profit[:n_plot].T, color="steelblue", lw=0.5, alpha=0.25)
plt.axhline(0, color="black", lw=0.8)
plt.xlabel("month")
plt.ylabel("cumulative profit (EUR)")
plt.title(f"{n_plot} simulated futures for the shop")
plt.tight_layout()
plt.savefig("shop_earnings_paths.png", dpi=150)

plt.show()