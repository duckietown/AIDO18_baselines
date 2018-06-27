import seaborn as sns
import glob
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np


model3 = pd.read_csv('predictions_opt=GDS,lr=1E-05,fc=2,drop=0.5,img=48x96,batch=100.csv')
model_only_omega = pd.read_csv('predictions_opt=GDS,lr=1E-05,fc=2,drop=0.5,img=48x96,batch=100_only_omega_only_omega.csv')


with sns.axes_style("darkgrid"):
    ####### plot model 3
    fig3 = plt.figure()

    # hide automatic x and y ticks
    plt.xticks([])
    plt.yticks([])

    plt.title('Predicted vs True Velocities')
    ax1 = fig3.add_subplot(211)
    # ax1.set_xlabel('Number of test images')
    ax1.set_ylabel('Velocity V')
    ax1.plot(model3[u'Unnamed: 0'], model3['pred_v'], linestyle="-", lw=2, c='red')
    ax1.plot(model3[u'Unnamed: 0'], model3['true_v'], linestyle="-", lw=2, c='green')
    ax1.legend(['Predicted V', 'True V'], loc="upper right")

    ax2 = fig3.add_subplot(212)
    ax2.set_xlabel('Number of test images')
    ax2.set_ylabel('Velocity Omega')
    ax2.plot(model3[u'Unnamed: 0'], model3['pred_omega'], linestyle="-", lw=2, c='red')
    ax2.plot(model3[u'Unnamed: 0'], model3['true_omega'], linestyle="-", lw=2, c='green')
    ax2.legend(['Predicted Omega', 'True Omega'], loc="upper right")

    rms3 = sqrt(mean_squared_error(model3['true_v'], model3['pred_v']))
    print('rms_v = %s' %(rms3))
    rms6 = sqrt(mean_squared_error(model3['true_omega'], model3['pred_omega']))
    print('rms_omega = %s' %(rms6))

    # plt.show()

with sns.axes_style("darkgrid"):
    ####### plot model 3
    fig4 = plt.figure()

    # hide automatic x and y ticks
    plt.xticks([])
    plt.yticks([])

    ax1 = fig4.add_subplot(212)
    ax1.set_xlabel('Number of test images')
    ax1.set_ylabel('Velocity Omega')
    ax1.plot(model_only_omega[u'Unnamed: 0'], model_only_omega['pred_omega'], linestyle="-", lw=2, c='red')
    ax1.plot(model_only_omega[u'Unnamed: 0'], model_only_omega['true_omega'], linestyle="-", lw=2, c='green')
    ax1.legend(['Predicted Omega', 'True Omega'], loc="upper right")

    ax2 = fig4.add_subplot(211)
    plt.title('Predicted vs True Velocities')
    # ax2.set_xlabel('Number of test images')
    ax2.set_ylabel('Velocity V')
    ax2.plot(model_only_omega[u'Unnamed: 0'], model3['true_v'], linestyle="-", lw=2, c='red')
    ax2.plot(model_only_omega[u'Unnamed: 0'], model3['true_v'], linestyle="-", lw=2, c='green')
    ax2.legend(['Forced V', 'True V'], loc="upper right")

plt.show()

rmsnew = sqrt(mean_squared_error(model_only_omega['true_omega'], model_only_omega['pred_omega']))

print('rmsnew = %s' % (rmsnew))