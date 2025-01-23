import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.ticker import FuncFormatter  
def x_major_formatter(x, pos):  
    if x == int(x):  
        return f'{int(x)}'  
    else:  
        return f'{x:.1f}'  
   
def y_major_formatter(y, pos):  
    if y == 0:  
        return '0'  
    elif y == 1:  
        return '1.0'  
    else:  
        return f'{y:.1f}'  
#load data from npy file shape (128,128)
trues_frame_11th = np.load('trues_11th_frame.npy',mmap_mode='r')
preds_frame_11th = np.load('preds_11th_frame.npy',mmap_mode='r')
trues_frame_13th = np.load('trues_13th_frame.npy',mmap_mode='r')
preds_frame_13th = np.load('preds_13th_frame.npy',mmap_mode='r')
trues_frame_15th = np.load('trues_15th_frame.npy',mmap_mode='r')
preds_frame_15th = np.load('preds_15th_frame.npy',mmap_mode='r')
trues_frame_17th = np.load('trues_17th_frame.npy',mmap_mode='r')
preds_frame_17th = np.load('preds_17th_frame.npy',mmap_mode='r')
trues_frame_19th = np.load('trues_19th_frame.npy',mmap_mode='r')
preds_frame_19th = np.load('preds_19th_frame.npy',mmap_mode='r')

##transform to one dimension
trues_frame_11th_flatten = trues_frame_11th.flatten()
preds_frame_11th_flatten = preds_frame_11th.flatten()
trues_frame_13th_flatten = trues_frame_13th.flatten()
preds_frame_13th_flatten = preds_frame_13th.flatten()
trues_frame_15th_flatten = trues_frame_15th.flatten()
preds_frame_15th_flatten = preds_frame_15th.flatten()
trues_frame_17th_flatten = trues_frame_17th.flatten()
preds_frame_17th_flatten = preds_frame_17th.flatten()
trues_frame_19th_flatten = trues_frame_19th.flatten()
preds_frame_19th_flatten = preds_frame_19th.flatten()
#difference between perdicted data and true data
difference_11th = trues_frame_11th_flatten - preds_frame_11th_flatten
difference_13th = trues_frame_13th_flatten - preds_frame_13th_flatten
difference_15th = trues_frame_15th_flatten - preds_frame_15th_flatten
difference_17th = trues_frame_17th_flatten - preds_frame_17th_flatten
difference_19th = trues_frame_19th_flatten - preds_frame_19th_flatten
#abs
difference_11th_abs = np.abs(difference_11th)
difference_13th_abs = np.abs(difference_13th)
difference_15th_abs = np.abs(difference_15th)
difference_17th_abs = np.abs(difference_17th)
difference_19th_abs = np.abs(difference_19th)



def plot_cdf(data, label, fontsize = 20):
    abs_diff = np.abs(data)
    sorted_diff = np.sort(abs_diff)
    cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
    plt.plot(sorted_diff, cdf, linestyle='-', linewidth=2, label=label)
plt.figure(figsize=(9, 8), dpi = 150)
matplotlib.rcParams['font.family']=['Arial']
matplotlib.rcParams['font.sans-serif']=['Arial'
plot_cdf(difference_11th_abs, 't$_{6100}$')
plot_cdf(difference_13th_abs, 't$_{6900}$')
plot_cdf(difference_15th_abs, 't$_{7700}$')
plot_cdf(difference_17th_abs, 't$_{8500}$')
plot_cdf(difference_19th_abs, 't$_{9300}$')
plt.xlabel('Prediction error',fontproperties='Arial',fontsize=31)
plt.ylabel('CDF',fontproperties='Arial',fontsize=31)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(3)
ax.tick_params(axis='both', which='major', direction='in', length=7, width =1.5, labelsize=28)
ax.xaxis.set_major_formatter(FuncFormatter(x_major_formatter))  
ax.yaxis.set_major_formatter(FuncFormatter(y_major_formatter))
plt.legend(loc='lower right', fontsize=12, frameon = False, prop={'family': 'Arial', 'size': 24, 'style':'italic'})
plt.ylim(0,1.05)
plt.xlim(0,1.05)
plt.savefig('./plot_CDF/CDF.png')
plt.show()

