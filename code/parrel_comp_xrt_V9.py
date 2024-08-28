# -*- coding: utf-8 -*-
"""
Note:
(1)建输出文件目录，output  
(2)确定误差文件目录：Error_path = './surface error_DATA/'  
"""
__author__ = "Fugui Yang"
__date__ = "28 February 2020"
import_lib = 1
if import_lib == 1:
    import os, sys
    import numpy as np
    import pickle, imageio, time
    import matplotlib as mpl
    mpl.use('Agg')
    from ray_tracing_unit_V9_change import *
    from functions_def import *
    #交互式画图选项
    import matplotlib as mpl
    import matplotlib.pyplot as plt

#%%mpi USING
nRecv = 4 #数据包

n_processor = 10 #确保使用的核数与cmd一致
MPI = None
comMPI = None
data = None
datas = None
try:
    from mpi4py import MPI #OC091014
    comMPI = MPI.COMM_WORLD
    rank = comMPI.Get_rank()
    nProc = comMPI.Get_size()
except:
    print('Calculation will be sequential (non-parallel), because "mpi4py" module can not be loaded')

#开始并行计算
if rank > 0:
    i_rank = rank
    while i_rank <= nRecv:  
        print(i_rank,rank)
        #%%%%%需要并行计算的代码
        data_re = []
        #%%差分测量
        n_dimension = 6  #需要随机评估的自由度
        #rand_num = (np.random.random(n_dimension)-0.5)*2 #
        rand_num = np.array([0,0,0,0,0,0])
        step_angle = np.array([0, 0, 0, 0, 0, 0])
        
        #%%选择合适的组面形误差做计算
        # i_error = 0
        i_error = i_rank
        
        prex = 'ray_first_'
        main(data_re, rand_num, step_angle*0, i_rank, i_error, rank, prex)

        prex = 'real_first_'
        data_re1 = data_re.copy()
        main(data_re1, rand_num, step_angle * 0, i_rank, i_error, rank, prex)
        #%%结束
        
        i_rank = i_rank + 9
        
#%%保存数据
if rank == 0:
    print('rank {0}'.format(rank))
    # if data == None:
    #     data = np.zeros(7).astype('float')
    # datas = np.zeros((nRecv,7)).astype('float')
    # for i in range(nRecv): #loop over messages from workers
    #     comMPI.Recv(data, source=MPI.ANY_SOURCE)
    #     datas[i,:] = data
    # datas1 = [Error_data, datas]
    # pickleName = os.path.join(os.getcwd(), 'output', 'scan_data.pickle')
    # with open(pickleName, 'wb') as f:
    #     pickle.dump(datas1, f, protocol=4)
        
