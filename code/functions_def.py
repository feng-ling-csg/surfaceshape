"""
函数库
V20221111：增加傅里叶变换功能
V2022/12/31:增加画图和Gaussian拟合
"""
import numpy as np
import numba
import math 
import os, sys, math
import pickle, imageio, time
from scipy.optimize import curve_fit
# import pyopencl as cl
import matplotlib as mpl
if os.name == 'posix':
    print('the OS is Linux, interative plot is not used')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Define the Gaussian function
def Gauss_fit(x,y):
    def Gauss(x, A, B, C, D):
        y = A*np.exp(-1*B*(x-C)**2)+D
        return y
    parameters, covariance = curve_fit(Gauss, x, y)
      
    fit_A = parameters[0]
    fit_B = parameters[1]
    fit_C = parameters[2]
    fit_D = parameters[3]
    fit_y = Gauss(x, fit_A, fit_B,fit_C,fit_D)
    sigma = 1/(2*fit_B)**0.5
    print(parameters)
    print('center = ',fit_C,', sigma = ', sigma)
    return fit_C,sigma, fit_y

# OPENCL函数引用
def dtft(x, omegas):
    """
    Exact evaluation the DTFT at the indicated points omega for the signal x
    Note this is incredibly slow
    
    Note x runs from 0 to N-1
    """
    N = len(x)
    ns = np.arange(N)
    W = np.zeros((len(omegas), N), dtype=np.complex128)
    for wi, w in enumerate(omegas):
        W[wi, :] = np.exp(-1.0j * w * ns)
        
    return np.dot(W, x)

#@numba.jit(nopython=True)
def nextpow2(n):
    # Return the smallest power of two greater than or equal to n.
    return int(math.ceil(math.log(n)/math.log(2)))


#@numba.jit(nopython=True)
def chirpz(x,A,W,M):
    # chirp z transform per Rabiner derivation pp1256
    # x is our (complex) signal of length N

    N = len(x)
    L = 2**(nextpow2(N + M -1))  # or nearest power of two
    yn = np.zeros(L, dtype=np.complex128)
    for n in range(N):
        yn_scale =  A**(-n) * W**((n**2.0)/2.0)
        yn[n] = x[n] * yn_scale
    Yr = np.fft.fft(yn)
    
    vn = np.zeros(L, dtype=np.complex128)
    for n in range(M):
        vn[n] = W**((-n**2.0)/2.0)
        
    for n in range(L-N+1, L):
        vn[n] = W**(-((L-n)**2.0)/2.0)
        
    Vr = np.fft.fft(vn)
    
    Gr = Yr * Vr
    
    gk = np.fft.ifft(Gr)
    #gk = np.convolve(yn, vn)
    
    Xk = np.zeros(M, dtype=np.complex128)
    for k in range(M):
        g_scale = W**((k**2.0)/2.0) 
        Xk[k] = g_scale * gk[k]
        
    return Xk

#@numba.jit(nopython=True)
def chirpz2(x,A_row,W_row,M_row,A_column,W_column,M_column):
    # Perform the Chirp z-Transform on a 2D signal.
    # x -- 2 dimensional input signal
    # A_row,W_row,M_row -- A, W and M applied to rows
    # A_column,W_column,M_column -- A, W and M applied to columns
    # Returns the Chirp z-Transform of dimension (M_row,M_column).
    # See also: chirpz
    y = np.array([chirpz(r,A_row,W_row,M_row) for r in x])
    y = np.ascontiguousarray(y.transpose())
    y1 = np.array([chirpz(r1,A_column,W_column,M_column) for r1 in y])
    return y1.transpose()

def zoom_fft(x, theta_start, step_size, M):
    """
    "zoomed" version of the fft, produces M step_sized samples
    around the unit circle starting at theta_start
    
    """
    A = np.exp(1j * theta_start)
    W = np.exp(-1j * step_size)
    
    return chirpz(x, A, W, M)

def Ampli_Phase_2D(Es0, dimx, dimz, title = None):
    #画二维分布图
    #一维中心轴上的光场分布
    #统计分布宽度（中心轴，及累积方式）
    Es = Es0.reshape([len(dimz),len(dimx)])
 
    fig = plt.figure(figsize=(12,5))
    fig.suptitle(title, fontsize=14)
    plt.subplot(1,2,1)
    plt.imshow(np.angle(Es), interpolation='bicubic', #cmap='jet',
                extent=[min(dimx)*1e3,max(dimx)*1e3,min(dimz)*1e3,max(dimz)*1e3], aspect='auto')#,vmin=vmin, vmax=vmax)
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('z ($\mu$m)')
    plt.title('Phase',size=12)
    
    plt.subplot(1,2,2)
    plt.imshow(np.abs(Es)**2, interpolation='bicubic', #cmap='jet',
                extent=[min(dimx)*1e3,max(dimx)*1e3,min(dimz)*1e3,max(dimz)*1e3], aspect='auto')#,vmin=vmin, vmax=vmax)
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('z ($\mu$m)')
    plt.title('Intensity',size=12)
    plt.savefig(title+'phase_intensity', dpi=300)
def FWHM_get(dimx, Ix):
    spline = UnivariateSpline(dimx, Ix-Ix.max()/2, s=0)
    try:
        r1, r2 = spline.roots()  # find the roots
    except:
        r2, r1 = 0, 0
    return abs(r2-r1)

def image_1D2D(I, dimx, dimz, title = None):
    #画二维分布图
    #一维中心轴上的光场分布
    #统计分布宽度（中心轴，及累积方式）
    I0 = I.reshape([len(dimz),len(dimx)])
    
    Ix = I0[len(dimz)//2,:]
    Iz = I0[:,len(dimx)//2]
    sigmax1 = ((Ix * dimx**2).sum() / Ix.sum())**0.5*1e3
    sigmaz1 = ((Iz * dimz**2).sum() / Iz.sum())**0.5*1e3
    FWHMx = FWHM_get(dimx, Ix)*1e3
    FWHMz = FWHM_get(dimz, Iz)*1e3 
    print('{0:.2f}    {1:.2f}    {2:.2f}    {3:.2f}'.format(sigmax1, sigmaz1,  FWHMx, FWHMz))
    Ix = I0.sum(axis=0)
    Iz = I0.sum(axis=1)
    sigmax2 = ((Ix * dimx**2).sum() / Ix.sum())**0.5*1e3
    sigmaz2 = ((Iz * dimz**2).sum() / Iz.sum())**0.5*1e3

    fig = plt.figure(figsize=(12,5))
    
    
    plt.subplot(1,2,1)
    plt.imshow(I0, interpolation='bicubic', #cmap='jet',
                extent=[min(dimx)*1e3,max(dimx)*1e3,min(dimz)*1e3,max(dimz)*1e3], aspect='auto')#,vmin=vmin, vmax=vmax)
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('z ($\mu$m)')
    plt.title('2D map',size=12)
    
    axis1 = fig.add_subplot(1,2,2)
    # axis2 = axis1.twinx()
    axis1.plot(dimx*1e3, I0[len(dimz)//2,:], label = "Horizontal slice. {0:.1f} $\mu$m rms ".format(sigmax1))
    axis1.plot(dimz*1e3, I0[:,len(dimx)//2], label = "Vertical slice. {0:.1f} $\mu$m rms ".format(sigmaz1))    
    # axis2.plot(dimx*1e3, I0.sum(axis=0), 'g--',label = "Horizontal integration {0:.1f} $\mu$m rms ".format(sigmax2))
    # axis2.plot(dimz*1e3, I0.sum(axis=1),'k--',label = "Vertical integration {0:.1f} $\mu$m rms ".format(sigmaz2))
    axis1.set_xlabel('x or z($\mu$m)')
    axis1.set_ylabel('Sliced Intensity')
    # axis2.set_ylabel('Integrated Intensity')  
    plt.title('1D plot',size=12)
    axis1.legend()
    # axis2.legend()
    fig.suptitle(title+': sigma:{0:.2f}$\mu$m {1:.2f}$\mu$m, FWHM:{2:.2f}$\mu$m {3:.2f}$\mu$m'.format(sigmax1, sigmaz1,  FWHMx, FWHMz), fontsize=14)
    plt.savefig(title, dpi=300)
    
    
# @numba.jit(nopython=True)    
def _diffraction_integral_conv_direct(U_in, x, y, z, x0, y0, z0, k, xbins, zbins):
    t0 = time.time()
    a1 = x[:, np.newaxis] - x0
    b1 = y[:, np.newaxis] - y0
    c1 = z[:, np.newaxis] - z0
    pathAfter = (a1**2 + b1**2 + c1**2)**0.5
    U1 = k*1j/(2*np.pi)  * np.exp(1j*k*(pathAfter)) / pathAfter
    U_out1 = (U_in * U1).sum(axis=1)
    print(time.time()-t0)
    Es1 = U_out1.reshape([xbins, zbins])
    return Es1

def _diffraction_integral_CL_Fresnel(U_in, x, y, z, x0, y0, z0, k0, waveCL, targetOpenCL = "GPU", precisionOpenCL = 'float64'):
#使用OPENCL计算
#U_in 输入光场
#x,y,z是一维输入光场点坐标
#x0, y0, z0是一维输出光场点坐标
#k是 波矢
    myfloat = waveCL.cl_precisionF
    mycomplex = waveCL.cl_precisionC
    frontRays = np.int32(len(x0))
    imageRays = np.int32(len(x))
    k = np.ones(len(x0)) * k0
    scalarArgs = [frontRays]
    
    slicedROArgs = [myfloat(x), myfloat(y), myfloat(z)]  # x,y,z
    
    
    nonSlicedROArgs = [mycomplex(U_in),  # Es_loc
                       myfloat(k),
                       np.array([x0, y0, z0, 0*z0], order='F', dtype=myfloat)] # 出射面坐标]  # surface_normal  
    
    slicedRWArgs = [np.zeros(imageRays, dtype=mycomplex)]  # Es_res

    Es_res = waveCL.run_parallel(
        'integrate_fresnel_kirchhoff', scalarArgs, slicedROArgs,
        nonSlicedROArgs, slicedRWArgs, None, imageRays)

    return Es_res


def Fresnel_Bluestain(U_in, dimx0, dimz0, dimx, dimz, p, k):
    #输入为方形矩阵
    #
    xbins0, zbins0 = len(dimx0), len(dimz0)
    xbins, zbins = len(dimx), len(dimz)
    PI2 = 2*np.pi
    Lambda=  PI2/k
    d1s0, d2s0 = np.meshgrid(dimx0, dimz0)
    dx0 = dimx0[1]-dimx0[0]
    dz0 = dimz0[1]-dimz0[0]
    x0 = d1s0.flatten()
    z0 = d2s0.flatten() 
    
    dx = dimx[1]-dimx[0]
    dz = dimz[1]-dimz[0]
    d1s, d2s = np.meshgrid(dimx, dimz)
    x = d1s.flatten()
    z = d2s.flatten()
    
     # 计算菲涅尔数
    if False: 
        # print("Positioni:",ii)
        y = np.zeros_like(x) + p
        r = (x**2 + z**2+ y**2)**0.5 
        a = x/r
        c = z/r
        NAx = (a.max() - a.min()) * 0.5
        NAz = (c.max() - c.min()) * 0.5
        fn = (NAx**2 + NAz**2) * r.mean() /Lambda  # Fresnel number
        # print('Effective Fresnel number = {0:.3g}'.format(fn))
    
    f1 = x.min()*dx0/Lambda/p
    f1z = z.min()*dz0/Lambda/p

    phase_startx = PI2*f1
    phase_stepx = PI2*dx*dx0/Lambda/p

    phase_startz = PI2*f1z
    phase_stepz = PI2*dz*dz0/Lambda/p

    U1 = U_in * np.exp(1j*k*(x0**2+z0**2)/p/2)   
    U2 = U1.reshape([zbins0, xbins0])

    A_row = np.exp(1j * phase_startx)
    W_row = np.exp(-1j * phase_stepx)
    M_row = xbins
    A_column = np.exp(1j * phase_startz)
    W_column = np.exp(-1j * phase_stepz)
    M_column = zbins    
    zoom_cz = chirpz2(U2, A_row, W_row, M_row, A_column, W_column, M_column)*np.exp(-1j*k*(d1s*x0.min()+d2s*z0.min())/p)
    U3 = zoom_cz*np.exp(1j*k*p)*np.exp(1j*k*(d1s**2+d2s**2)/p/2)/Lambda/p/(-1j)*dx0*dz0

    return U3


def sample_points(x_min,x_max,bins): 
    edges = np.linspace(x_min, x_max, bins+1)
    fsmExpX = (edges[:-1] + edges[1:]) * 0.5
    return fsmExpX

def FFT_zoom2(fmin, fmax, M, x, I):
    dx = x[1]-x[0]
    phase_start = np.pi*2*fmin*dx
    phase_step = np.pi*2*(fmax-fmin)*dx/M 
    omegas_cz = np.arange(M) * phase_step + phase_start
    zoom_cz = zoom_fft(I, phase_start, phase_step , int(M))
    return omegas_cz/np.pi/2/dx, zoom_cz
    
      
def rsgeng1D(N,rL,h,cl):
    x = np.linspace(-rL/2,rL/2,N)
    Z = h*np.random.randn(1,N)                      
    F = np.exp(-x**2/(cl**2/2))
    f = np.sqrt(2/np.sqrt(np.pi))*np.sqrt(rL/N/cl)*np.fft.ifft(np.fft.fft(Z)*np.fft.fft(F))
    z0=[]
    p = np.polyfit(x, f.real[0,:],0) 
    ry = np.polyval(p, x)
    zi = f.real[0,:] - ry  
    y= np.arange(-5,5,1)  # 原始
    for i in np.arange(0,len(y),1):
        z0.append(zi)
    z=np.asarray(z0)  
    # print(np.std(zi)*1e6,np.std(f.real[0,:])*1e6)
    return y,x,z

#生成1维误差，
#N采样点数，rL采样范围，cl相干长度，h高度rms值
def rsgeng1D2(N, rL, h, cl):
    x = np.linspace(-rL/2,rL/2,N)
    Z = h*np.random.randn(1,N)                      
    F = np.exp(-x**2/(cl**2/2))
    f = np.sqrt(2/np.sqrt(np.pi))*np.sqrt(rL/N/cl)*np.fft.ifft(np.fft.fft(Z)*np.fft.fft(F))
    p = np.polyfit(x, f.real[0,:],0) 
    ry = np.polyval(p, x)
    zi = f.real[0,:] - ry  
    return x, zi

def dynamic_images(Plots): 
    gif_images = []    
    for plot in Plots:
        filename = plot.saveName
        gif_images.append(imageio.imread(filename)) 
    strTrajOutFileName = "dynamic_images.gif"
    saveName_dynamic =  os.path.join(os.getcwd(), strExDataFolderName, strTrajOutFileName)
    imageio.mimsave(saveName_dynamic,gif_images,fps=1.5) 
    print("dynamic images : Done!")

#离焦曲线画图
def defocus_plot(Plots, dqs): 
    qCurve = []
    xCurve = []
    zCurve = []
    czCurve = []
    cxCurve = []
    for dq, plot in zip(dqs, Plots):
        qCurve.append(dq)            
        xCurve.append(plot.dx)
        zCurve.append(plot.dy)
        cxCurve.append(plot.cx)
        czCurve.append(plot.cy)

    plt.figure(100)
    ax1 = plt.subplot(111)
    ax1.set_title(r'FWHM size of beam cross-section near focal position')
    ax1.plot(qCurve, xCurve, 'o', label='Horizontal direction')  
    ax1.plot(qCurve, zCurve, '+', label='Vertical direction')
    ax1.set_xlabel(u'Shift (mm)', fontsize=14)
    ax1.set_ylabel(u'FWHM size ($\mu$m)', fontsize=14)
    plt.legend()          
    plt.show()
    strTrajOutFileName = "Depth_Of_Focus.png"
    path =  os.path.join(os.getcwd(), strExDataFolderName, strTrajOutFileName)
    plt.savefig(path, dpi = 200)

    plt.figure(300)
    ax1 = plt.subplot(111)
    ax1.plot(qCurve, cxCurve, 'o', label='Horizontal direction')  
    ax1.plot(qCurve, czCurve, '+', label='Vertical direction')
    ax1.set_xlabel(u'Shift (mm)', fontsize=14)
    ax1.set_ylabel(u'Position ($\mu$m)', fontsize=14)
    plt.legend()          
    plt.show()
    strTrajOutFileName = "position.png"
    path =  os.path.join(os.getcwd(), strExDataFolderName, strTrajOutFileName)
    plt.savefig(path, dpi = 200)
    print('plot_generator Done')

    #保存截面图数据
    dump = []
    I_x = []
    I_z = []
    for plot in Plots:
        x = plot.xaxis.binCenters
        y = plot.yaxis.binCenters
        Ixy = plot.total2D
        ny = len(y)
        nx = len(x)
        I_x.append(Ixy[ny//2,:])
        I_z.append(Ixy[:,nx//2])
    dump.append([x, y, dqs, I_x,I_z])
    strTrajOutFileName = "image_section.pickle"
    pickleName =  os.path.join(os.getcwd(), strExDataFolderName, strTrajOutFileName)
    with open(pickleName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)
    print(plot.title," Defocus Data save :Done")

#画图数据保存,强度分布及坐标轴信息
def save_data(plot):
    dump = []
    x = plot.xaxis.binCenters
    y = plot.yaxis.binCenters
    Ixy = plot.total2D
    dump.append([x, y, Ixy])
    fileName = '{0}'.format(plot.title)
    strTrajOutFileName = '{0}.pickle'.format(fileName)
    pickleName =  os.path.join(os.getcwd(), strExDataFolderName, strTrajOutFileName)
    with open(pickleName, 'wb') as f:
        pickle.dump(dump, f, protocol=2)
    print(plot.title," Data save :Done")