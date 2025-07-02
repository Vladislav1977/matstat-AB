from scipy.special import gdtr
from scipy.stats import gamma as gamma_sc

from matplotlib.widgets import Slider

import numpy as np
from scipy import integrate
from scipy.special import erf

import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



def gamma(lamb):
    func = lambda x: np.exp(-x)*x**(lamb-1)
    return integrate.quad(func, 0, np.inf)[0]

def pdf_gamma(x, a, lamb):
    return (a ** lamb) / gamma(lamb) * (x**(lamb-1))*np.exp(-a*x)

def cdf_gamma(x, a, lamb):
    return np.array([integrate.quad(pdf_gamma, 0, x_0, args=(a, lamb))[0] for x_0 in x])



import matplotlib
print(matplotlib.get_backend())


x = np.linspace(0.0001, 6, 200)

# Initial values
a_init = 1
lamb_init = 1


pdf_dist = pdf_gamma(x, a_init, lamb_init)

cdf_dist = gdtr(a_init, lamb_init, x)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 10))

plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.25)

manager = plt.get_current_fig_manager()
manager.window.setGeometry(100, 100, 1000, 800)  # x, y, width, height


pdf_line, = axes[0].plot(x, pdf_dist, label='PDF', lw=3)
axes[0].set_title(f'Gamms distribution. $\\alpha$ = {a_init}, $\\lambda$ = {lamb_init}', fontsize=15, fontweight='bold')
axes[0].tick_params(axis='x', which='both', bottom=False, top=False)
axes[0].grid(axis='x')
axes[0].grid(axis='y', alpha=0.5, linestyle='--')
axes[0].legend()


cdf_line, = axes[1].plot(x, cdf_dist, label='CDF', lw=3)
#axes[1].set_xticks(np.arange(0, 5, 0.2), minor=True)
#axes[1].set_xticks(np.arange(0, 6, 1))
#axes[1].set_xticklabels([''] + [str(i) for i in range(1, int(max(x)) + 1)])
ticks = axes[1].get_xticks()
axes[1].set_xticks(ticks)
axes[1].set_xticklabels(['' if tick == 0 else str(tick) for tick in ticks])

axes[1].set_xlim(left=0, right=max(x))
axes[1].set_ylim(bottom=0)
axes[1].grid(axis='x')
axes[1].grid(axis='y', alpha=0.5, linestyle='--')
axes[1].legend(loc='lower right')
axes[1].text(5.1, 0.2, r'$\frac{\alpha^\lambda}{\Gamma(\lambda)} \, \int_{0}^{x}t^{\lambda - 1} e^{-\alpha t} dt$')


# Define sliders
ax_a = plt.axes([0.15, 0.15, 0.65, 0.03])    # was 0.10
ax_lamb = plt.axes([0.15, 0.10, 0.65, 0.03]) # was 0.05

slider_a = Slider(ax_a, 'α (shape)', 0.1, 5.0, valinit=a_init, valstep=0.1)
slider_lamb = Slider(ax_lamb, 'λ (rate)', 0.1, 5.0, valinit=lamb_init, valstep=0.1)

# Update function
def update(val):
    a = slider_a.val
    lamb = slider_lamb.val
    new_pdf = pdf_gamma(x, a, lamb)
    new_cdf = gdtr(a, lamb, x)

    pdf_line.set_ydata(new_pdf)
    cdf_line.set_ydata(new_cdf)

    axes[0].set_title(f'Gamma distribution. $\\alpha$ = {a:.1f}, $\\lambda$ = {lamb:.1f}', fontsize=15, fontweight='bold')
    axes[0].set_ylim([0, max(new_pdf)+0.01])
    

    fig.canvas.draw_idle()

# Connect sliders to update function
slider_a.on_changed(update)
slider_lamb.on_changed(update)

plt.show()