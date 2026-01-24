import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# x軸の範囲
x = np.linspace(-5, 5, 1000)

# 自由度のリストとラベル
nu_list = [1, 4, 10, np.inf]
labels = [
    r'$\nu = 1$ : Cauchy distribution',
    r'$\nu = 4$ : This study',
    r'$\nu = 10$ : Close to normal, heavy tail',
    r'$\nu \to \infty$ : Converges to normal'
]

# プロット
plt.figure(figsize=(10, 6))

for nu, label in zip(nu_list, labels):
    if nu == np.inf:
        y = stats.norm.pdf(x)
    else:
        y = stats.t.pdf(x, df=nu)
    
    plt.plot(x, y, linewidth=2, label=label)

plt.xlabel('x', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Student-t Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()