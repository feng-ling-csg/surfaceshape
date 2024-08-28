# -*- coding: utf-8 -*-

#2023.10.18
#波动计算，单电子，同时多个位置输出

#Note:
#(1)建输出文件目录，output  
#(2)确定误差文件目录：Error_path = './surface error_DATA/'  

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
    # xrt函数引用
    import xrt.backends.raycing as raycing
    import xrt.runner as xrtr
    import xrt.plotter as xrtp
    import xrt.backends.raycing.apertures as ra
    import xrt.backends.raycing.oes as roes
    import xrt.backends.raycing.screens as rsc

    import xrt.backends.raycing.run as rr
    import Distorted_Element as Distorted
    from functions_def import *
    import gc
    # raycing.targetOpenCL = (1, 0)

# what = 'rays'
what = 'hybrid'
if what != 'rays':
    import xrt.backends.raycing.waves as rw
    import xrt.backends.raycing.modes as rm
nElectrons = 50
nElectronsSave = 10
nsamples = (200, 200)
nModes = 4
# make_modes_R = True
make_modes_R = False
if isinstance(nsamples, (tuple, list)): 
    basename = 'B8_'+str(nsamples[0]*nsamples[1])+'_'
else:
    basename = 'B8_'+str(nsamples)+'_'
prefix = 'B8_'

oe_error_V = 1
oe_error_H = 1 

#**********************************************************#
#%%*******************文件路径********************

cwd = os.getcwd()
strExDataFolderName = 'output'
Error_path = './surface error_DATA/'

#从保存的图片中读取图片，并保存动态图片

#**********************************************************#
#%%*******************全局角度********************
accept_ang_x, accept_ang_z = 15e-6, 15e-6

p_slitWB = 38000
p_DCM = 39000
theta_KB = 3e-3
q_VKB= 1500
y_samp = 43999.9392003
y_VKB = p_DCM + 4000
p_VKB = y_VKB  
theta_HKB = theta_KB
y_HKB = y_VKB + 300
p_HKB = p_VKB + (y_HKB-y_VKB)/np.cos(2*theta_HKB)
q_HKB = q_VKB - (y_HKB-y_VKB)/np.cos(2*theta_HKB)


#**********************************************************#
#%%*******************光源参数********************

E0, n0 = 10000, 1  # 插入间工作能量点，及谐波级次
n1 = 1  # 待分析谐波的能量点
E1 = E0/n0*n1  # 待分析的能量点
nrays, repeats = nsamples, 1


dE = E1*0.0004/2
eMin0, eMax0 = E1-dE, E1+dE

kwargs_SR = dict(
    # nrays = nrays, 
    center = [0, 0, 0],
    eE = 6.0, eI = 0.2, 
    # eEspread = 0,
    # eEpsilonX = 0, eEpsilonZ = 0,
    eEspread = 0.00111,
    eEpsilonX = 0.02728, eEpsilonZ = 0.00276,
    betaX = 2.83871, betaZ = 1.91667,
    xPrimeMax = accept_ang_x, zPrimeMax = accept_ang_z,
    xPrimeMaxAutoReduce = False, zPrimeMaxAutoReduce = False,
    targetE = [E0, n0], eMin = eMin0, eMax = eMax0,
    # uniformRayDensity = uniformRayDensity,
    # filamentBeam = filamentBeam,
    K = r"auto", period = 16.8, n = 176)
    
#%%%输出屏幕的设置

x_lim, z_lim = 5, 5

zbins, zppb = 64, 2
xbins, xppb = 64, 2
#%%%输出屏幕的设置
edges = np.linspace(-x_lim, x_lim, xbins+1)*1e-3
fsmFX = (edges[:-1] + edges[1:]) * 0.5
edges1 = np.linspace(-z_lim, z_lim, zbins+1)*1e-3
fsmFZ = (edges1[:-1] + edges1[1:]) * 0.5

edges = np.linspace(-x_lim - 4, x_lim + 4, xbins+1)*1e-3
fsmFXA20= (edges[:-1] + edges[1:]) * 0.5
edges1 = np.linspace(-z_lim - 4, z_lim + 4, zbins+1)*1e-3
fsmFZA20= (edges1[:-1] + edges1[1:]) * 0.5

#%%材料特性
if True:
    crystalSi01 = raycing.materials.CrystalSi(name=None,hkl= (1, 1, 1))
    Si = raycing.materials.Material(elements=r"Si",kind=r"mirror",rho=2.33,name=None)
    crystalSi01 = raycing.materials.CrystalSi(name=None)

def define_beamline(i_error):
    beamLine = raycing.BeamLine(alignE=E0)
    beamLine.nrays = nrays
    #%%选择光源类型

    source_und = raycing.sources.Undulator(beamLine,  **kwargs_SR)#targetOpenCL = None,
    beamLine.source = source_und
   #%%第一光学元件：白光狭缝 


    slitDx, slitDz = accept_ang_x*p_slitWB, accept_ang_z*p_slitWB
    opening = [-slitDx/2, slitDx/2, -slitDz/2, slitDz/2]
    beamLine.slit_WB = ra.RectangularAperture(
        beamLine, 'Slit_WB',  [0, p_slitWB, 0], 
        ('left', 'right', 'bottom', 'top'), opening = opening )
   #%%第二光学元件：单色器 

    DCM_error = 0
    Surface_name = 'DCM_OE2_'
    fixedOffset = 20
    DCM01x = 0
    p_DCM01 = p_DCM
    theta_DCM = crystalSi01.get_Bragg_angle(E0)-crystalSi01.get_dtheta_symmetric_Bragg(E0)
    DCM02x = DCM01x+fixedOffset/np.sin(2*theta_DCM)*np.sin(2*(theta_DCM))
    DCM02z = DCM01x+fixedOffset/np.sin(2*theta_DCM)*np.cos(2*(theta_DCM))
    p_DCM02 = p_DCM01+DCM02z

    # 使用传统单色器模型
    kwargs = dict(
        name='DCM',
        center=[0, p_DCM, 0],
        bragg=r"auto",  positionRoll=np.pi/2,
        material=crystalSi01, material2=crystalSi01,
        fixedOffset=fixedOffset)
    beamLine.dcM1 = raycing.oes.DCM(bl=beamLine, **kwargs)

    # 使用双平晶代表，在tracing部分的代码也是不同的
    kwargs_DCM01 = dict(
        name='DCM01', center=[DCM01x, p_DCM01, 0],
        pitch=theta_DCM,#[E0],
        positionRoll=np.pi/2,
        material=crystalSi01,
        limPhysX=[-25, 25], limPhysY=[-25, 25])
    kwargs_DCM02 = dict(
        name='DCM02', center=[DCM02x, p_DCM02, 0],
        pitch=-theta_DCM,#[E0],
        positionRoll=-np.pi/2,
        material=crystalSi01,
        limPhysX=[-25, 25], limPhysY=[-25, 25])
    if DCM_error == 0:
        # 不带面形误差
        beamLine.DCM01 = raycing.oes.OE(bl=beamLine, **kwargs_DCM01)
    else:
        # 带面形误差
        kwargs_DCM01['get_distorted_surface'] = 'error'
        kwargs_DCM01['fname1'] = Surface_name+'X.txt'
        kwargs_DCM01['fname2'] = Surface_name+'Y.txt'
        kwargs_DCM01['fname3'] = Surface_name+'Z.txt'
        beamLine.DCM01 = Distorted.PlaneMirrorDistorted(
            bl=beamLine, **kwargs_DCM01)

    beamLine.DCM02 = raycing.oes.OE(bl=beamLine, **kwargs_DCM02)
   #%%第三光学元件：单色器狭缝

    p_slitDCM = p_DCM + 500
    beamLine.slit_DCM = raycing.apertures.RectangularAperture(
        beamLine, 'Slit_DCM',  [fixedOffset, p_slitDCM,
                                0], ('left', 'right', 'bottom', 'top'),
        [-slitDx/2, slitDx/2, -slitDz/2, slitDz/2])
   #%%聚焦光学元件：KB聚焦镜缝

    Surface_name= Error_path+'VKB_'+str(i_error)+'_'
    globalRoll = 0
    inclination = 0
    theta_VKB = theta_KB

    kwargs_OE = dict(
                name ='VKB', 
                center=[fixedOffset, y_VKB, 0],
#                material = Si,
                limPhysX=[-3, 3], limPhysY=[-150, 150],
                rotationSequence='RyRzRx',
                pitch= theta_VKB+ inclination*np.cos(globalRoll), 
                positionRoll = globalRoll,
                yaw = inclination*np.sin(globalRoll),
                q = q_VKB, p = p_VKB,
                isCylindrical=True)    
    if oe_error_V==0:        
        beamLine.VKB = raycing.oes.EllipticalMirrorParam(bl=beamLine,**kwargs_OE)
    else:            
        kwargs_OE['get_distorted_surface'] ='error'
        kwargs_OE['fname1'] =Surface_name+'X.txt'
        kwargs_OE['fname2'] =Surface_name+'Y.txt'
        kwargs_OE['fname3'] =Surface_name+'Z.txt'       
        beamLine.VKB = Distorted.EllipMirrorDistorted(beamLine, **kwargs_OE)
        
    inclination = 2 * theta_VKB
    globalRoll = - np.pi/2
    
    Surface_name=Error_path+'HKB_'+str(i_error)+'_'
    
    kwargs_OE = dict(
                name='HKB',
                center=[fixedOffset, y_HKB, np.tan(inclination)*(y_HKB-y_VKB)],
                pitch = theta_HKB+inclination*np.cos(globalRoll), 
                positionRoll=globalRoll,
                yaw = inclination*np.sin(globalRoll),                
                rotationSequence='RyRzRx',                                    
#                material = Si,
                limPhysX=[-3, 3], limPhysY=[-150, 150],
                q = q_HKB-0.007, p = p_HKB,
                isCylindrical=True)    
    if oe_error_H ==0:        
        beamLine.HKB = raycing.oes.EllipticalMirrorParam(bl=beamLine,**kwargs_OE)
    else:            
        kwargs_OE['get_distorted_surface'] ='error'
        kwargs_OE['fname1'] =Surface_name+'X.txt'
        kwargs_OE['fname2'] =Surface_name+'Y.txt'
        kwargs_OE['fname3'] =Surface_name+'Z.txt'       
        beamLine.HKB = Distorted.EllipMirrorDistorted(beamLine, **kwargs_OE)
    """ 观察屏组件 """

    t_out_HKB = np.array([-np.sin(2*theta_KB), (np.cos(2*theta_KB))**2, np.sin(4*theta_KB)/2])
    l_HKB = beamLine.HKB.center
    lsamp = l_HKB + t_out_HKB*q_HKB
    beamLine.dqs = np.array([0, 20])
    beamLine.fsm2f = rsc.Screen(beamLine, 'FSM2f')
    beamLine.fsm2A20 = rsc.Screen(beamLine, 'fsm2A20')


    beamLine.fsm2f.center = [0, lsamp[1], 0]
    beamLine.fsm2A20.center = [0, lsamp[1] + 20, 0]

    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM0')
    beamLine.fsm0.center = beamLine.slit_WB.center
    beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1')
    beamLine.fsm1.center = beamLine.slit_DCM.center
    beamLine.fsm2 = rsc.Screen(beamLine, 'FSM2')
    

    beamLine.fsmn = rsc.Screen(beamLine, name = 'FSMn')
    return beamLine

#%%光线追迹模块
def run_process_rays(beamLine):
    print('***************************')
    print('run_process_rays  user')
    isMono = False
    if isMono:
        fixedEnergy = E0
    else:
        fixedEnergy = False
    waveOnSlit = beamLine.slit_WB.prepare_wave(beamLine.source, nrays = int(beamLine.nrays))
    # waveOnSlit = beamLine.slit_WB.prepare_wave(beamLine.source, nrays = (256, 256))
    beamSource = beamLine.source.shine(wave=waveOnSlit, fixedEnergy=fixedEnergy)  # 直接得到白光狭缝slit后的光场分布       
   
    # strTrajOutFileName = 'source.pickle'
    # pickleName =  os.path.join(os.getcwd(), strExDataFolderName, strTrajOutFileName)
    # with open(pickleName, 'wb') as f:
    #     pickle.dump(beamSource, f, protocol=2) 
    
    beamFSM0 = beamLine.fsm0.expose(beam=beamSource)

    # DCM
    DCM_in = beamSource
    DCM01Global, DCM01Local = beamLine.DCM01.reflect(beam=DCM_in)
    DCM02Global, DCM02Local = beamLine.DCM02.reflect(beam=DCM01Global)
    # beamLine.slit_DCM.propagate(beam=DCM02Global)
    beamFSM1 = beamLine.fsm1.expose(beam=DCM02Global)
    DCM_OUT = DCM02Global

    #KB
    KB_in = DCM_OUT
    VKBGlobal, VKBLocal = beamLine.VKB.reflect(KB_in)
    HKBGlobal, HKBLocal = beamLine.HKB.reflect(VKBGlobal)
    
    outDict = {'beamSource': beamSource,
        'beamFSM0':beamFSM0,
        'DCM01Global':DCM01Global,
        'DCM01Local': DCM01Global,
        'DCM02Global': DCM02Global,
        'DCM02Local': DCM02Global,
        'beamFSM1': beamFSM1,
        'VKBGlobal': VKBGlobal,
        'VKBLocal': VKBLocal,
        'HKBLocal': HKBLocal,
        'HKBGlobal': HKBGlobal}
#1. 样品处 焦点

    beam_samp =  beamLine.fsm2f.expose(HKBGlobal)
    outDict['Samp_focus'] = beam_samp
    beam_samp = beamLine.fsm2A20.expose(HKBGlobal)
    outDict['Samp_A20'] = beam_samp

    return outDict

#%%波动传播模块
def run_process_hybr(beamLine):
    print('***************************')
    print('run_process_hybr  user')
    isMono = False
    if isMono:
        fixedEnergy = E0
    else:
        fixedEnergy = False
        
    beamLine.iBeam = beamLine.iBeam+1 if hasattr(beamLine, 'iBeam') else 0
    print(str(beamLine.iBeam)+ '/' + str(beamLine.repeats)) 

    if beamLine.mode_run == 'tradition':
        waveOnSlit = beamLine.slit_WB.prepare_wave(beamLine.source, nrays = nsamples)
        beamSource = beamLine.source.shine(wave=waveOnSlit, fixedEnergy=fixedEnergy)  # 直接得到白光狭缝slit后的光场分布    

    elif beamLine.mode_run == 'read':
        pickleName = os.path.join(os.getcwd(), 'output', 'source.pickle')
        with open(pickleName, 'rb') as f:
            beamSource = pickle.load(f)         
    else:
        beamSource = beamLine.savedBeams[beamLine.iBeam]
    
    beamFElocal = beamLine.slit_WB.propagate(beamSource)
    beamLine.fsm0.center = beamLine.slit_WB.center
    beamFSM0 = beamLine.fsm0.expose(beam=beamSource)

    # DCM
    DCM_in = beamSource
    DCM01Global, DCM01Local = beamLine.DCM01.reflect(beam=DCM_in)
    DCM02Global, DCM02Local = beamLine.DCM02.reflect(beam=DCM01Global)
    beamLine.slit_DCM.propagate(beam=DCM02Global)
    beamLine.fsm1.center = beamLine.slit_DCM.center
    beamFSM1 = beamLine.fsm1.expose(beam=DCM02Global)
    DCM_OUT = DCM02Global

    #单色器到KB镜的传播方式，波动还是几何
    if False:
        waveOnVKB = beamLine.VKB.prepare_wave(beamLine.slit_DCM, nrays)
        beamToVKB = rw.diffract(DCM_OUT, waveOnVKB)
        VKBGlobal, VKBLocal = beamLine.VKB.reflect(
            beamToVKB, noIntersectionSearch=True)

        waveOnHKB = beamLine.HKB.prepare_wave(beamLine.VKB, nrays)
        beamToHKB = rw.diffract(VKBLocal, waveOnHKB)
        HKBGlobal, HKBLocal = beamLine.HKB.reflect(
            beamToHKB, noIntersectionSearch=True)
    else:
        VKBGlobal, VKBLocal = beamLine.VKB.reflect(DCM_OUT)
        HKBGlobal, HKBLocal = beamLine.HKB.reflect(VKBGlobal)   


    outDict = {'beamSource': beamSource,
        'beamFSM0':beamFSM0,
        'DCM01Global':DCM01Global,
        'DCM01Local': DCM01Global,
        'DCM02Global': DCM02Global,
        'DCM02Local': DCM02Global,
        'beamFSM1': beamFSM1,
        'VKBLocal': VKBLocal,
        'VKBGlobal': VKBGlobal,
        'HKBLocal': HKBLocal,
        'HKBGlobal': HKBGlobal}
#从KB位置后开始波动传播
#1. 样品处 焦点

    waveOnSample = beamLine.fsm2f.prepare_wave(beamLine.HKB, beamLine.fsmFX, beamLine.fsmFZ)
    rw.diffract(HKBLocal, waveOnSample)

    waveOnSampleA20 = beamLine.fsm2A20.prepare_wave(beamLine.HKB, beamLine.fsmFXA20, beamLine.fsmFZA20)
    rw.diffract(HKBLocal, waveOnSampleA20)

    outDict['Samp_focus'] = waveOnSample
    outDict['Samp_A20'] = waveOnSampleA20

    return outDict


#%% 画图
def plot_image(what, beamLine):
    plots = []
    plots_FSMns = []
    #%%测试用的画图
    if False:
        plot = xrtp.XYCPlot(
            'beamSource', aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', unit=r"$\mu$m"),
            yaxis=xrtp.XYCAxis(r'$z$', unit=r"$\mu$m"),
            caxis=xrtp.XYCAxis('energy', 'eV', offset=E0,
                                limits=[eMin0-1, eMax0+1]),
            title='Source size'
            )
        plot.baseName = plot.title
        plots.append(plot)
        plot = xrtp.XYCPlot(
            'beamSource', aspect='auto',
            xaxis=xrtp.XYCAxis(r"x'", unit=r"$\mu$rad"),
            yaxis=xrtp.XYCAxis(r"z'", unit=r"$\mu$rad"),
            caxis=xrtp.XYCAxis('energy', 'eV', offset=E0,
                                limits=[eMin0-1, eMax0+1]),
            title='source angle', fluxFormatStr=r"%g")
        plot.xaxis.limits = [-20, 20]
        plot.yaxis.limits = [-20, 20]
        plot.baseName = plot.title
        plots.append(plot)    


        #Sample
        plot = xrtp.XYCPlot(
            'HKBLocal', aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', unit=r"mm", bins=xbins, ppb=xppb),
            yaxis=xrtp.XYCAxis(r'$y$', unit=r"mm", bins=zbins, ppb=zppb),
            caxis=xrtp.XYCAxis('energy', 'eV', offset=E0, limits=[eMin0-1, eMax0+1]),
            title='HKBLocal')    
        plot.baseName = plot.title
        plots.append(plot)

        plot = xrtp.XYCPlot(
            'VKBLocal', aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', unit=r"mm", bins=xbins, ppb=xppb),
            yaxis=xrtp.XYCAxis(r'$y$', unit=r"mm", bins=zbins, ppb=zppb),
            caxis=xrtp.XYCAxis('energy', 'eV', offset=E0, limits=[eMin0-1, eMax0+1]),
            title='VKBLocal')    
        plot.baseName = plot.title
        plots.append(plot)


        #Sample
        plot = xrtp.XYCPlot(
            'DCM02Global', aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', unit=r"mm", bins=xbins, ppb=xppb),
            yaxis=xrtp.XYCAxis(r'$z$', unit=r"mm", bins=zbins, ppb=zppb),
            caxis=xrtp.XYCAxis('energy', 'eV', offset=E0, limits=[eMin0-1, eMax0+1]),
            title='DCM02Global')    
        plot.baseName = plot.title
        plots.append(plot)
    if True:
#%%输出的画图
        plot = xrtp.XYCPlot(
            'Samp_focus', aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', unit=r"$\mu$m", bins=xbins, ppb=xppb),
            yaxis=xrtp.XYCAxis(r'$z$', unit=r"$\mu$m", bins=zbins, ppb=zppb),
            caxis=xrtp.XYCAxis('energy', 'eV', offset=E0, limits=[eMin0 - 1, eMax0 + 1]),
            title='Sample_focus')
        if what.startswith('hybr'):
            plot.xaxis.limits = [-x_lim, x_lim]
            plot.yaxis.limits = [-z_lim, z_lim]
            ax = plot.xaxis
            edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins + 1)
            beamLine.fsmFX = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
            ax = plot.yaxis
            edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins + 1)
            beamLine.fsmFZ = (edges[:-1] + edges[1:]) * 0.5 / ax.factor

        plot.baseName = plot.title
        plots.append(plot)
        plots_FSMns.append(plot)

# Sample

        plot = xrtp.XYCPlot(
            'Samp_A20', aspect='auto',
            xaxis=xrtp.XYCAxis(r'$x$', unit=r"$\mu$m", bins=xbins, ppb=xppb),
            yaxis=xrtp.XYCAxis(r'$z$', unit=r"$\mu$m", bins=zbins, ppb=zppb),
            caxis=xrtp.XYCAxis('energy', 'eV', offset=E0, limits=[eMin0 - 1, eMax0 + 1]),
            title='Sample_After_20')
        if what.startswith('hybr'):
            plot.xaxis.limits = [-x_lim - 3, x_lim + 3]
            plot.yaxis.limits = [-z_lim - 3, z_lim + 3]
            ax = plot.xaxis
            edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins + 1)
            beamLine.fsmFXA20 = (edges[:-1] + edges[1:]) * 0.5 / ax.factor
            ax = plot.yaxis
            edges = np.linspace(ax.limits[0], ax.limits[1], ax.bins + 1)
            beamLine.fsmFZA20 = (edges[:-1] + edges[1:]) * 0.5 / ax.factor

        plot.baseName = plot.title
        plots.append(plot)
        plots_FSMns.append(plot)

    for plot in plots:
        plot.saveName = os.path.join(os.getcwd(), strExDataFolderName, prefix + plot.title + '.png')#[prefix + plot.title + '.png', ]
        if plot.caxis.label.startswith('energy'):
            plot.caxis.limits = eMin0, eMax0
            plot.caxis.offset = E0
        if plot.fluxKind.startswith('power'):
            plot.fluxFormatStr = '%.0f'
        else:
            plot.fluxFormatStr = '%.1p'
    return plots, plots_FSMns

#%% 扫描程序Generator
def plot_generator(plots, plots_FSMns, beamLine, data_re, i_rank, prex):
    
    for plot in plots_FSMns:
        try:
            plot.textPanel.set_text(
                plot.textPanelTemplate)
        except AttributeError:
            pass            
        plot.xaxis.fwhmFormatStr = '%.2f'
        plot.yaxis.fwhmFormatStr = '%.2f'
        plot.caxis.fwhmFormatStr = '%.5f'
        
        plot.fluxFormatStr = '%.2p'
        strTrajOutFileName = prex+plot.title+"_{0}.png".format(i_rank)
        plot.saveName =  os.path.join(os.getcwd(), strExDataFolderName, strTrajOutFileName)

        if plot.caxis.label.startswith('energy'):
            plot.caxis.limits = eMin0, eMax0
            plot.caxis.offset = E0
        if plot.fluxKind.startswith('power'):
            plot.fluxFormatStr = '%.2f'
        else:
            plot.fluxFormatStr = '%.2p'
    yield
    dump = []
    dump.append(["three images(x,y,I2D(x,y)), error angle(HKB, VKB), title, i_rank"])
    
    for plot in plots_FSMns:
        x = plot.xaxis.binCenters
        y = plot.yaxis.binCenters
        Ixy = plot.total2D
        dump.append([x, y, Ixy,beamLine.HKB.extraPitch, beamLine.HKB.extraRoll,beamLine.HKB.extraYaw,
                     beamLine.VKB.extraPitch, beamLine.VKB.extraRoll,beamLine.VKB.extraYaw,plot.title, i_rank])
        data_re.append([plot.cx, plot.cy, plot.dx, plot.dy])
        
    dump.append(data_re)
    strTrajOutFileName = prex+'Sample{0}.pickle'.format(i_rank)
    pickleName =  os.path.join(os.getcwd(), strExDataFolderName, strTrajOutFileName)
    with open(pickleName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)

def afterScript(plots, data_re, i_rank, plots_FSMns):  
    # print(i_rank,' data =', data_re[0])
    # print('data...',i_rank)
    print('AfterScript done')

def main(data_re, err, step_angle, i_rank, i_error, rank_GPU, prex):

    beamLine = define_beamline(i_error)
   
    #%%添加光学器件误差
    beamLine.HKB.extraPitch = err[0] *30*1e-6 + step_angle[0]
    beamLine.HKB.extraRoll = err[1] *500*1e-6 + step_angle[1]
    beamLine.HKB.extraYaw = err[2] *500*1e-6 + step_angle[2] 
    
    beamLine.VKB.extraPitch = err[3] *30*1e-6 + step_angle[3] 
    beamLine.VKB.extraRoll = err[4] *500*1e-6 + step_angle[4] 
    beamLine.VKB.extraYaw = err[5] *500*1e-6 + step_angle[5]
    
    #%%光线追迹，
    #如果已经确定观察屏横向位置，data_re给出光斑位置
    #如果不确定观察屏横向位置，否则利用光线追迹寻找光斑位置，并输出给data_re

    if len(data_re)>0:
        print(i_rank)
        plots, plots_FSMns = plot_image(what, beamLine)
        if what.startswith('rays'):
            raycing.run.run_process = run_process_rays
        elif what.startswith('hybr'):
            raycing.run.run_process = run_process_hybr
 #调整三个屏幕的位置
        beamLine.fsm2f.center[0] = data_re[0][0]*1e-3
        beamLine.fsm2f.center[2] = data_re[0][1]*1e-3

        beamLine.fsm2A20.center[0] = data_re[1][0] * 1e-3
        beamLine.fsm2A20.center[2] = data_re[1][1] * 1e-3

        #%%生成模式场
        if what.startswith('hybr'):
            beamLine.repeats = nModes
            beamLine.source.ismono = True
            beamLine.source.filamentBeam = True
            beamLine.sourceuniformRayDensity = True
            beamLine.mode_run = 'modes'

            if make_modes_R == True:
                beamLine = define_beamline(0)
                beamLine.source.ismono = True
                beamLine.source.filamentBeam = True
                beamLine.sourceuniformRayDensity = True
                rm.make_and_save_modes(
                    beamLine, nsamples, nElectrons, nElectronsSave, nModes, E0, output='hybr',
                    basename = basename)
            else:
                beamLine.savedBeams, wAll, totalFlux = rm.use_saved(what+'-modes', basename)
                wAll1 = np.flip(wAll)
                print(wAll)
                # print(np.sum(wAll1[0:20]))
                # print(np.sum(wAll1[0:30]))
                # print(np.sum(wAll1[0:40]))
                # print(np.sum(wAll1[0:50]))
                # print(np.sum(wAll1[0:60]))
                # print(np.sum(wAll1[0:70]))
                # print(np.sum(wAll1[0:80]))

                xrtr.run_ray_tracing(plots, repeats=beamLine.repeats, beamLine=beamLine, 
                            generator=plot_generator, generatorArgs=[plots, plots_FSMns, beamLine, data_re,i_rank, prex],
                            afterScript=afterScript, afterScriptArgs=[plots, data_re, i_rank, plots_FSMns])
        else:
            beamLine.nrays = 2e5
            xrtr.run_ray_tracing(plots, repeats = 1, beamLine=beamLine, 
                    generator=plot_generator, generatorArgs=[plots, plots_FSMns, beamLine, data_re,i_rank, prex],
                    afterScript=afterScript, afterScriptArgs=[plots, data_re, i_rank, plots_FSMns])

    else:
        plots, plots_FSMns = plot_image('rays', beamLine)
        raycing.run.run_process = run_process_rays
        beamLine.nrays = 5e4
        
        beamLine.source.ismono = False
        beamLine.source.filamentBeam = False
        beamLine.sourceuniformRayDensity = False
        beamLine.fsm2f.center[0] = 0
        beamLine.fsm2f.center[2] = 0

        beamLine.fsm2A20.center[0] = 0
        beamLine.fsm2A20.center[2] = 0

        xrtr.run_ray_tracing(plots, repeats = 1, beamLine=beamLine,
                generator=plot_generator, generatorArgs=[plots, plots_FSMns, beamLine, data_re,i_rank, prex],
                afterScript=afterScript, afterScriptArgs=[plots, data_re, i_rank, plots_FSMns])
    del plots, plots_FSMns, beamLine
    gc.collect()
    plt.close('all')

#%%%
#DEGUG部分
