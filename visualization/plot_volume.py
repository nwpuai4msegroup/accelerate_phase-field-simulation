import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(9,8),dpi=150)
# x_axis_data = [r'$t_{6100}$', r'$t_{6500}$', r'$t_{6900}$',
#                 r'$t_{7300}$', r'$t_{7700}$', r'$t_{8100}$',
#                 r'$t_{8500}$', r'$t_{8900}$', r'$t_{9300}$',
#                 r'$t_{9700}$']
italic_t = {'fontstyle': 'italic'}
x_axis_data = ['t$_{6100}$','t$_{6500}$','t$_{6900}$','t$_{7300}$','t$_{7700}$','t$_{8100}$','t$_{8500}$','t$_{8900}$','t$_{9300}$','t$_{9700}$']
#variant 1
original_axis_data_1 = [15.14,15.61,16.08,16.38,16.35,16.24,16.15,15.81,15.49,15.11]
prediciton_axis_data_1 = [14.64,14.46,14.3,14.11,13.83,13.54,13.22,12.88,12.55,12.17]
#variant 2
original_axis_data_2 = [4.11,3.94,3.41,3.12,3.04,2.82,2.81,2.85,2.56,2.54]
prediciton_axis_data_2 = [4.16,4.12,4.1,4.13,4.15,4.12,4.07,4.02,3.99,3.9]
#variant 3
original_axis_data_3 = [15.91,17.11,18.15,18.87,19.65,20.08,20.57,21.06,21.44,21.76]
prediciton_axis_data_3 = [14.95,15.11,15.18,15.33,15.52,15.62,15.71,15.74,15.74,15.6]
#variant 4
original_axis_data_4 = [8.17,8,7.54,7.54,7.34,7.08,7,6.84,6.79,6.81]
prediciton_axis_data_4 = [8.32,8.44,8.47,8.5,8.47,8.44,8.34,8.2,8.04,7.87]
#variant 5
original_axis_data_5 = [12.59,12.46,12.34,12.03,12.03,11.96,11.83,11.72,11.25,10.97]
prediciton_axis_data_5 = [12.56,12.54,12.5,12.54,12.51,12.47,12.43,12.39,12.42,12.38]
#variant 6
original_axis_data_6 = [15.05,15.91,16.54,17.05,17.68,18.16,18.68,19.25,20.21,20.7]
prediciton_axis_data_6 = [13.42,13.45,13.27,13.22,13.22,13.01,12.85,12.61,12.39,12.15]
matplotlib.rcParams['font.family']=['Arial']
matplotlib.rcParams['font.sans-serif']=['Arial']
# plt.plot(x_axis_data, original_axis_data_1, 'green',marker='o',linestyle='-', alpha=1, linewidth=2, label='Truth',markersize = 10)
# plt.plot(x_axis_data, prediciton_axis_data_1, 'red', marker='o',linestyle='-',alpha=1, linewidth=2, label='Predicted', markersize = 10)
plt.plot(x_axis_data, original_axis_data_6, 'green',marker='o', linestyle='-', alpha=1, linewidth=2,markersize = 10)
plt.plot(x_axis_data, prediciton_axis_data_6, 'red', marker='o',linestyle='-',alpha=1, linewidth=2,markersize = 10)
plt.legend(frameon=False,fontsize=12, prop={'family': 'Arial', 'size': 28})
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(axis='both', which='both', direction='in',width = 1.5, length = 9)
ax.set_xticklabels(x_axis_data, fontdict=italic_t)
ax.tick_params(axis='both', which='both', labelsize=28)
plt.xlabel('Predicted step',fontproperties='Arial',fontsize=31, labelpad=10)
plt.ylabel('Volume fraction of variant 6 (%)',fontproperties='Arial',fontsize=31)
plt.ylim(8,24)
y_min, y_max = 8, 24
y_ticks = np.arange(y_min, y_max + 1, 2)
plt.yticks(y_ticks)
ticks_to_show = [0, 2, 4, 6, 8]
plt.xticks(ticks_to_show, [x_axis_data[i] for i in ticks_to_show])
plt.xticks(ticks_to_show)
plt.subplots_adjust(top=0.9, bottom=0.15)
plt.savefig('./extrapolate_training/variant6.png')
plt.show()