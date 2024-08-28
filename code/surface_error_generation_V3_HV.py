# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 09:02:43 2023

@author: Administrator
"""
__author__ = "Fugui Yang"
__date__ = "28 February 2020"
import_lib = 1
if import_lib == 1:
    import os, sys
    import numpy as np
    import pickle, imageio, time
    #交互式画图选项
    import matplotlib as mpl
    if os.name == 'posix':
        print('the OS is Linux, interative plot is not used')
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from functions_def import *

cwd = os.getcwd()
strExDataFolderName = 'output'
Error_path = './surface error_DATA/'

nRecv = 10001
N = 400 #面形点数
rL = 230 #反射镜长度#beamLine.VKB.limPhysY[1]-beamLine.VKB.limPhysY[0]
slopes = 0.3*1e-6 # 0 - 0.5urad
clxs = 30 #60到90
dump = []
for i0 in range(nRecv):
    h=clxs*1e-3*slopes/np.sqrt(2)*1e3
    xi, yi, zi = rsgeng1D(N,rL,h,clxs)
    #保存数据，x，y，z（mm）
    name_1= Error_path +'HKB_'+str(i0)+'_'
    zi0=zi.T
    np.savetxt(name_1+'X.txt', xi,fmt='%.8e')
    np.savetxt(name_1+'Y.txt', yi,fmt='%.8e')
    np.savetxt(name_1+'Z.txt', zi0,fmt='%.8e')

    h_2 = clxs * 1e-3 * slopes / np.sqrt(2) * 1e3
    xi_2, yi_2, zi_2 = rsgeng1D(N, rL, h_2, clxs)
    # 保存数据，x，y，z（mm）
    name_2 = Error_path + 'VKB_' + str(i0) + '_'
    zi0_2 = zi_2.T

    np.savetxt(name_2 + 'X.txt', xi_2, fmt='%.8e')
    np.savetxt(name_2 + 'Y.txt', yi_2, fmt='%.8e')
    np.savetxt(name_2 + 'Z.txt', zi0_2, fmt='%.8e')


    # heightx = zi[0,:]
    # slopex = np.diff(heightx)/(yi[1]-yi[0])
    # slope = np.std(slopex)
    # plt.figure(130)
    # plt.subplot(211)
    # plt.plot(yi,heightx*1e6, label = '{0:.2f}nm and {1:.2f}nm'. format(np.std(heightx)*1e6, h*1e6))
    # plt.xlabel('x(mm)',fontsize=14)
    # plt.ylabel('height(nm)',fontsize=14)
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(yi[0:-1],slopex*1e6, label = '{0:.2f}urad and {1:.2f}urad'. format( np.std(slopex)*1e6, slope*1e6))
    # plt.xlabel('x(mm)',fontsize=14)
    # plt.ylabel('slope(urad)',fontsize=14)
    # plt.show()
    # plt.legend()
    # strTrajOutFileName1 = name_1+"Error.png"
    # plt.savefig(strTrajOutFileName1)
    # strTrajOutFileName2 = name_2+"Error.png"
    # plt.savefig(strTrajOutFileName2)
    # plt.close()
