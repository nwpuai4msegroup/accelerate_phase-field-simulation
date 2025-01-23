import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#load data from npy file
trues_frame_11th = np.load('trues_11th_frame.npy')
preds_frame_11th = np.load('preds_11th_frame.npy')
trues_frame_13th = np.load('trues_13th_frame.npy')
preds_frame_13th = np.load('preds_13th_frame.npy')
trues_frame_15th = np.load('trues_15th_frame.npy')
preds_frame_15th = np.load('preds_15th_frame.npy')
trues_frame_17th = np.load('trues_17th_frame.npy')
preds_frame_17th = np.load('preds_17th_frame.npy')
trues_frame_19th = np.load('trues_19th_frame.npy')
preds_frame_19th = np.load('preds_19th_frame.npy')


#extract M_1_60_layer_114 as an enample
trues_frame_11th_layer114 = trues_frame_11th[2445]
preds_frame_11th_layer114 = preds_frame_11th[2445]
trues_frame_13th_layer114 = trues_frame_13th[2445]
preds_frame_13th_layer114 = preds_frame_13th[2445]
trues_frame_15th_layer114 = trues_frame_15th[2445]
preds_frame_15th_layer114 = preds_frame_15th[2445]
trues_frame_17th_layer114 = trues_frame_17th[2445]
preds_frame_17th_layer114 = preds_frame_17th[2445]
trues_frame_19th_layer114 = trues_frame_19th[2445]
preds_frame_19th_layer114 = preds_frame_19th[2445]

#transform to one dimension
trues_frame_11th_layer114_flatten = trues_frame_11th_layer114.flatten()
preds_frame_11th_layer114_flatten = preds_frame_11th_layer114.flatten()
trues_frame_13th_layer114_flatten = trues_frame_13th_layer114.flatten()
preds_frame_13th_layer114_flatten = preds_frame_13th_layer114.flatten()
trues_frame_15th_layer114_flatten = trues_frame_15th_layer114.flatten()
preds_frame_15th_layer114_flatten = preds_frame_15th_layer114.flatten()
trues_frame_17th_layer114_flatten = trues_frame_17th_layer114.flatten()
preds_frame_17th_layer114_flatten = preds_frame_17th_layer114.flatten()
trues_frame_19th_layer114_flatten = trues_frame_19th_layer114.flatten()
preds_frame_19th_layer114_flatten = preds_frame_19th_layer114.flatten()

frames = [
    (trues_frame_11th_layer114_flatten,preds_frame_11th_layer114_flatten,"Step 6100"),
    (trues_frame_13th_layer114_flatten,preds_frame_13th_layer114_flatten,"Step 6900"),
    (trues_frame_15th_layer114_flatten,preds_frame_15th_layer114_flatten,"Step 7700"),
    (trues_frame_17th_layer114_flatten,preds_frame_17th_layer114_flatten,"Step 8500"),
    (trues_frame_19th_layer114_flatten,preds_frame_19th_layer114_flatten,"Step 9300")
]
z_values = []
for trues, preds, _ in frames:
    xy = np.vstack([trues, preds])
    z = gaussian_kde(xy)(xy)
    z_values.extend(z)
vmin = min(z_values)
vmax = max(z_values)
print(vmin)
print(vmax)
##plot scatter truth vs prediction
xy = np.vstack([trues_frame_19th_layer114_flatten, preds_frame_19th_layer114_flatten])
kde = gaussian_kde(xy)
z = kde(xy)
print(kde.factor)
plt.figure(figsize=(9, 8), dpi=150)
matplotlib.rcParams['font.family']=['Arial']
matplotlib.rcParams['font.sans-serif']=['Arial']
sc = plt.scatter(trues_frame_19th_layer114_flatten, preds_frame_19th_layer114_flatten, c=z, s=10, alpha=0.9, cmap='viridis', marker='*',vmin = vmin, vmax=vmax)
plt.plot([-0.1, 1], [-0.1, 1], color='red', linestyle='--')
plt.xlabel('True Values', fontproperties='Arial', fontsize=31)
plt.ylabel('Predicted Values', fontproperties='Arial', fontsize=31)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(axis='both', which='major', direction = 'in', width = 1.5, length = 7, labelsize=28)
ticks = np.arange(0,1.10,0.2)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

cbar = plt.colorbar(sc)
cbar.set_label('Density', fontproperties='Arial', fontsize=31)
cbar.ax.tick_params(labelsize = 28)
# plt.savefig('./plot_scatter/cbar.png')
plt.show()


