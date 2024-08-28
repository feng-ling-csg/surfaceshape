# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:55:05 2018
2019.12.30:
    更正CRL误差程序

"""

import os, sys;

sys.path.append(os.path.join('..', '..', '..'))  # analysis:ignore
import numpy as np
import matplotlib.pyplot as plt
# import re
from scipy import ndimage
# from scipy import optimize

import xrt.backends.raycing as raycing
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.materials as rm


# import xrt.backends.raycing.sources as rs
# import xrt.backends.raycing.screens as rsc
# import xrt.backends.raycing.run as rr
# import xrt.backends.raycing.waves as rw
# import xrt.plotter as xrtp
# import xrt.runner as xrtr

class rough_Material(rm.Material):
    def __init__(self, elements='Pd', quantities=None, rho=12.0, kind='mirror',
                 t=None, table='Chantler', efficiency=None, efficiencyFile=None,
                 name='', **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        rm.Material.__init__(self, elements=elements, quantities=quantities, rho=rho, kind=kind,
                             t=t, table=table, efficiency=efficiency, efficiencyFile=efficiencyFile,
                             name=name, **kwargs)

    def __pop_kwargs(self, **kwargs):
        """
        sigrough: sigma of roughness [m]
        """
        self.sigrough = kwargs.pop('sigrough')
        return kwargs

    def get_amplitude(self, E, beamInDotNormal, fromVacuum=True):
        sigrough = self.sigrough
        rs, rp, absob1, absorb2 = super(rough_Material, self).get_amplitude(E, beamInDotNormal)
        cosAlpha = abs(beamInDotNormal)
        sinAlpha = np.sqrt(1 - beamInDotNormal ** 2)
        waveLength = 12398. / E  # [Angstrom]
        waveLength *= 1.e-10  # [m]
        Qz = 4 * np.pi * cosAlpha / waveLength
        rs = rs * np.exp(-Qz ** 2 * sigrough ** 2)
        rp = rp * np.exp(-Qz ** 2 * sigrough ** 2)
        return (rs, rp, absob1, absorb2)  # Compared with xop, abs(rs)**2 and abs(rp)**2 is a little bit smaller!


def Read_distorted_surface(fname1, fname2, fname3):
    x = np.loadtxt(fname1, unpack=True)
    y = np.loadtxt(fname2, unpack=True)
    z = np.loadtxt(fname3, unpack=True)
    return x, y, z


# ========================================================================================================#
# ========================================================================================================#
class ToroidMirrorDistorted(roe.ToroidMirror):
    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        roe.ToroidMirror.__init__(self, *args, **kwargs)

        if self.get_distorted_surface is None:
            return
        self.warpX, self.warpY, self.warpZ = Read_distorted_surface(self.fname1, self.fname2, self.fname3)
        self.warpNX, self.warpNY = len(self.warpX), len(self.warpY)
        self.limPhysX0 = np.min(self.warpX) * 30, np.max(self.warpX) * 30
        self.limPhysY0 = np.min(self.warpY), np.max(self.warpY)
        self.get_surface_limits()
        self.warpA, self.warpB = np.gradient(self.warpZ)
        dx = self.warpX[1] - self.warpX[0]
        dy = self.warpY[1] - self.warpY[0]
        self.warpA = self.warpA / dx
        self.warpB = self.warpB / dy
        rmsA = ((self.warpA ** 2).sum() / (self.warpNX * self.warpNY)) ** 0.5
        rmsB = ((self.warpB ** 2).sum() / (self.warpNX * self.warpNY)) ** 0.5
        self.warpSplineZ = ndimage.spline_filter(self.warpZ)
        self.warpSplineA = ndimage.spline_filter(self.warpA)
        self.warpSplineB = ndimage.spline_filter(self.warpB)

        # print('########Informaton of the mirror surface##########')
        # print('##xy ranges[mm]:')
        # print('x: {0:.2f}, {1:.2f}'.format(self.warpX.min(), self.warpX.max()))
        # print('y: {0:.2f}, {1:.2f}'.format(self.warpY.min(), self.warpY.max()))
        # print('##z range[nm]')
        # print('z: {0:.2f}, {1:.2f}'.format(self.warpZ.min() * 1.e6, self.warpZ.max() * 1.e6))
        # print(r"##Slope error min & max[$\mu$rad]:")
        # print('dz/dx: {0:.2f}, {1:.2f}'.format(self.warpA.min() * 1.e6, self.warpA.max() * 1.e6))
        # print('dz/dy: {0:.2f}, {1:.2f}'.format(self.warpB.min() * 1.e6, self.warpB.max() * 1.e6))
        # print(r"RMS slope error[\murad]:")
        # print(rmsA * 1.e6, rmsB * 1.e6)

    def __pop_kwargs(self, **kwargs):
        self.fname1 = kwargs.pop('fname1')  # fname1 and fname2 is the ANSYS data file, the mirror surface profile.
        self.fname2 = kwargs.pop('fname2')
        self.fname3 = kwargs.pop('fname3')
        self.get_distorted_surface = kwargs.pop('get_distorted_surface')
        return kwargs

    def local_z_distorted(self, x, y):
        if self.get_distorted_surface is None:
            return
        coords = np.array(
            [(x - self.limPhysX0[0]) /
             (self.limPhysX0[1] - self.limPhysX0[0]) * (self.warpNX - 1),
             (y - self.limPhysY0[0]) /
             (self.limPhysY0[1] - self.limPhysY0[0]) * (self.warpNY - 1)])
        z = ndimage.map_coordinates(self.warpSplineZ, coords, prefilter=True)
        return z

    def local_n_distorted(self, x, y):
        if self.get_distorted_surface is None:
            return
        coords = np.array(
            [(x - self.limPhysX0[0]) /
             (self.limPhysX0[1] - self.limPhysX0[0]) * (self.warpNX - 1),
             (y - self.limPhysY0[0]) /
             (self.limPhysY0[1] - self.limPhysY0[0]) * (self.warpNY - 1)])

        a = ndimage.map_coordinates(self.warpSplineA, coords, prefilter=True)
        b = ndimage.map_coordinates(self.warpSplineB, coords, prefilter=True)
        return b, -a


# ===================================================================================================#
# ===================================================================================================#

class PlaneMirrorDistorted(roe.OE):
    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        roe.OE.__init__(self, *args, **kwargs)
        self.read_data()

    # end# here you specify the bump and its mesh ###

    def __pop_kwargs(self, **kwargs):
        self.fname1 = kwargs.pop('fname1')  # fname1 and fname2 is the ANSYS data file, the mirror surface profile.
        self.fname2 = kwargs.pop('fname2')
        self.fname3 = kwargs.pop('fname3')
        self.get_distorted_surface = kwargs.pop('get_distorted_surface')
        return kwargs

    def read_data(self):
        if self.get_distorted_surface is None:
            return
        self.warpX, self.warpY, self.warpZ = Read_distorted_surface(self.fname1, self.fname2, self.fname3)
        self.warpNX, self.warpNY = len(self.warpX), len(self.warpY)
        self.limPhysX0 = np.min(self.warpX), np.max(self.warpX)
        self.limPhysY0 = np.min(self.warpY), np.max(self.warpY)
        self.get_surface_limits()
        self.warpA, self.warpB = np.gradient(self.warpZ)
        dx = self.warpX[1] - self.warpX[0]
        dy = self.warpY[1] - self.warpY[0]
        self.warpA = self.warpA / dx
        self.warpB = self.warpB / dy
        rmsA = ((self.warpA ** 2).sum() / (self.warpNX * self.warpNY)) ** 0.5
        rmsB = ((self.warpB ** 2).sum() / (self.warpNX * self.warpNY)) ** 0.5
        self.warpSplineZ = ndimage.spline_filter(self.warpZ)
        self.warpSplineA = ndimage.spline_filter(self.warpA)
        self.warpSplineB = ndimage.spline_filter(self.warpB)

        # print('########Informaton of the mirror surface##########')
        # print('##xy ranges[mm]:')
        # print('x: {0:.2f}, {1:.2f}'.format(self.warpX.min(), self.warpX.max()))
        # print('y: {0:.2f}, {1:.2f}'.format(self.warpY.min(), self.warpY.max()))
        # print('##z range[nm]')
        # print('z: {0:.2f}, {1:.2f}'.format(self.warpZ.min() * 1.e6, self.warpZ.max() * 1.e6))
        # print(r"##Slope error min & max[$\mu$rad]:")
        # print('dz/dx: {0:.2f}, {1:.2f}'.format(self.warpA.min() * 1.e6, self.warpA.max() * 1.e6))
        # print('dz/dy: {0:.2f}, {1:.2f}'.format(self.warpB.min() * 1.e6, self.warpB.max() * 1.e6))
        # print(r"RMS slope error[urad]:")
        # print(rmsA * 1.e6, rmsB * 1.e6)

    def local_z(self, x, y):
        if self.get_distorted_surface is None:
            return
        coords = np.array(
            [(x - self.limPhysX0[0]) /
             (self.limPhysX0[1] - self.limPhysX0[0]) * (self.warpNX - 1),
             (y - self.limPhysY0[0]) /
             (self.limPhysY0[1] - self.limPhysY0[0]) * (self.warpNY - 1)])
        z = ndimage.map_coordinates(self.warpSplineZ, coords, prefilter=True)
        return z

    def local_n(self, x, y):
        if self.get_distorted_surface is None:
            return
        Ax = -self.warpSplineA
        By = -self.warpSplineB
        coords = np.array(
            [(x - self.limPhysX0[0]) /
             (self.limPhysX0[1] - self.limPhysX0[0]) * (self.warpNX - 1),
             (y - self.limPhysY0[0]) /
             (self.limPhysY0[1] - self.limPhysY0[0]) * (self.warpNY - 1)])
        a = ndimage.map_coordinates(Ax, coords, prefilter=True)
        b = ndimage.map_coordinates(By, coords, prefilter=True)
        c = np.ones_like(x)
        norm = (a ** 2 + b ** 2 + 1) ** 0.5
        return [a / norm, b / norm, c / norm]


# ===================================================================================================#
# ===================================================================================================#


class EllipMirrorDistorted(roe.EllipticalMirrorParam):
    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        roe.EllipticalMirrorParam.__init__(self, *args, **kwargs)
        if self.get_distorted_surface is None:
            return
        self.warpX, self.warpY, self.warpZ = Read_distorted_surface(self.fname1, self.fname2, self.fname3)
        self.warpNX, self.warpNY = len(self.warpX), len(self.warpY)
        self.limPhysX0 = np.min(self.warpX), np.max(self.warpX)
        self.limPhysY0 = np.min(self.warpY), np.max(self.warpY)
        self.get_surface_limits()
        self.warpA, self.warpB = np.gradient(self.warpZ)
        dx = self.warpX[1] - self.warpX[0]
        dy = self.warpY[1] - self.warpY[0]
        self.warpA = self.warpA / dx
        self.warpB = self.warpB / dy
        rmsA = ((self.warpA ** 2).sum() / (self.warpNX * self.warpNY)) ** 0.5
        rmsB = ((self.warpB ** 2).sum() / (self.warpNX * self.warpNY)) ** 0.5
        self.warpSplineZ = ndimage.spline_filter(self.warpZ)
        self.warpSplineA = ndimage.spline_filter(self.warpA)
        self.warpSplineB = ndimage.spline_filter(self.warpB)

        # print('########Informaton of the mirror surface##########')
        # print('##xy ranges[mm]:')
        # print('x: {0:.2f}, {1:.2f}'.format(self.warpX.min(), self.warpX.max()))
        # print('y: {0:.2f}, {1:.2f}'.format(self.warpY.min(), self.warpY.max()))
        # print('##z range[nm]')
        # print('z: {0:.2f}, {1:.2f}'.format(self.warpZ.min() * 1.e6, self.warpZ.max() * 1.e6))
        # print(r"##Slope error min & max[$\mu$rad]:")
        # print('dz/dx: {0:.2f}, {1:.2f}'.format(self.warpA.min() * 1.e6, self.warpA.max() * 1.e6))
        # print('dz/dy: {0:.2f}, {1:.2f}'.format(self.warpB.min() * 1.e6, self.warpB.max() * 1.e6))
        # print(r"RMS slope error[urad]:")
        # print(rmsA * 1.e6, rmsB * 1.e6)

    # end# here you specify the bump and its mesh ###

    def __pop_kwargs(self, **kwargs):
        self.fname1 = kwargs.pop('fname1')  # fname1 and fname2 is the ANSYS data file, the mirror surface profile.
        self.fname2 = kwargs.pop('fname2')
        self.fname3 = kwargs.pop('fname3')
        self.get_distorted_surface = kwargs.pop('get_distorted_surface')
        return kwargs

    def local_r_distorted(self, x, y):
        """
        x, y corresponds to s, phi in parametric surfaces
        """
        if self.get_distorted_surface is None:
            return
        r = self.local_r(x, y)
        x0, y0, z0 = self.param_to_xyz(x, y, r)
        coords = np.array(
            [(x0 - self.limPhysX0[0]) /
             (self.limPhysX0[1] - self.limPhysX0[0]) * (self.warpNX - 1),
             (y0 - self.limPhysY0[0]) /
             (self.limPhysY0[1] - self.limPhysY0[0]) * (self.warpNY - 1)])
        z0 += ndimage.map_coordinates(self.warpSplineZ, coords, prefilter=True)
        s, phi, r_new = self.xyz_to_param(x0, y0, z0)
        dr = r_new - r
        return dr

    def local_n_distorted(self, x, y):
        """
        x, y corresponds to s, phi in parametric surfaces
        """
        if self.get_distorted_surface is None:
            return
        #        a = np.zeros_like(x)
        #        b = np.ones_like(x)
        r = self.local_r(x, y)
        x0, y0, z0 = self.param_to_xyz(x, y, r)
        coords = np.array(
            [(x0 - self.limPhysX[0]) /
             (self.limPhysX0[1] - self.limPhysX0[0]) * (self.warpNX - 1),
             (y0 - self.limPhysY0[0]) /
             (self.limPhysY0[1] - self.limPhysY0[0]) * (self.warpNY - 1)])
        a = ndimage.map_coordinates(self.warpSplineA, coords, prefilter=True)
        b = ndimage.map_coordinates(self.warpSplineB, coords, prefilter=True)

        return b, -a

    # ===================================================================================================#


# ===================================================================================================#
class DoubleParaboloidLens_Distorted(roe.DoubleParaboloidLens):
 # 所有的self代表的是class的对象
    hiddenMethods = roe.DoubleParaboloidLens.hiddenMethods + ['double_refract']  # ?

    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        roe.DoubleParaboloidLens.__init__(self, *args, **kwargs)
        self.read_data()

    def __pop_kwargs(self, **kwargs):
        self.fname1 = kwargs.pop('fname1')  # fname1 and fname2 is the ANSYS data file, the mirror surface profile.
        self.fname2 = kwargs.pop('fname2')
        self.fname3 = kwargs.pop('fname3')
        self.get_distorted_surface = kwargs.pop('get_distorted_surface')
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'lens'

    def read_data(self):
        if self.get_distorted_surface is None:
            return
        self.warpX, self.warpY, self.warpZ = Read_distorted_surface(self.fname1, self.fname2, self.fname3)
        self.warpNX, self.warpNY = len(self.warpX), len(self.warpY)
        self.limPhysX0 = np.min(self.warpX), np.max(self.warpX)
        self.limPhysY0 = np.min(self.warpY), np.max(self.warpY)
        self.get_surface_limits()
        self.warpA, self.warpB = np.gradient(self.warpZ)
        dx = self.warpX[1] - self.warpX[0]
        dy = self.warpY[1] - self.warpY[0]
        self.warpA = self.warpA / dx
        self.warpB = self.warpB / dy

        self.warpSplineZ = ndimage.spline_filter(self.warpZ)
        self.warpSplineA = ndimage.spline_filter(self.warpA)
        self.warpSplineB = ndimage.spline_filter(self.warpB)

    def local_z1(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        z = (x ** 2 + y ** 2) / (4 * self.focus)
        coords = np.array(
            [(x - self.limPhysX0[0]) /
             (self.limPhysX0[1] - self.limPhysX0[0]) * (self.warpNX - 1),
             (y - self.limPhysY0[0]) /
             (self.limPhysY0[1] - self.limPhysY0[0]) * (self.warpNY - 1)])

        z = z + ndimage.map_coordinates(self.warpSplineZ, coords, prefilter=True)
        if self.zmax is not None:
            z[z > self.zmax] = self.zmax
        return z

    def local_z2(self, x, y):
        """Determines the surface of OE at (x, y) position."""
  # 认为第二个面和第一个面一样
        return self.local_z1(x, y)


    def local_n1(self, x, y):
        Ax = -self.warpSplineA
        By = -self.warpSplineB
        coords = np.array(
            [(x - self.limPhysX0[0]) /
             (self.limPhysX0[1] - self.limPhysX0[0]) * (self.warpNX - 1),
             (y - self.limPhysY0[0]) /
             (self.limPhysY0[1] - self.limPhysY0[0]) * (self.warpNY - 1)])
        a0 = ndimage.map_coordinates(Ax, coords, prefilter=True)
        b0 = ndimage.map_coordinates(By, coords, prefilter=True)
        if self.zmax is not None:
            z = (x ** 2 + y ** 2) / (4 * self.focus)
            z = z + ndimage.map_coordinates(self.warpSplineZ, coords, prefilter=True)
            
            a = -x / (2 * self.focus) + a0  # -dz/dx  含误差部分
            b = -y / (2 * self.focus) + b0  # -dz/dy  含误差部分

            if isinstance(a, np.ndarray):
                a[z > self.zmax] = 0
            if isinstance(b, np.ndarray):
                b[z > self.zmax] = 0

            c = np.ones_like(x)
            norm = (a ** 2 + b ** 2 + 1) ** 0.5
        return [a / norm, b / norm, c / norm]

    def local_n2(self, x, y):
        return self.local_n1(x, y)


class DoubleParaboloidLensCy(roe.ParaboloidFlatLens):
    """Implements a refractive lens or a stack of lenses (CRL) with two equal
    paraboloids from both sides."""

    cl_local_z = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float res;
        res = 0.25 * (x * x + y * y) / cl_plist.s1;
        if (res > cl_plist.s0) res = cl_plist.s0;
        return res;
    }"""
    cl_local_n = """
    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res;
        res.s0 = -x / (2*cl_plist.s1);
        res.s1 = -y / (2*cl_plist.s1);
        float z = (x*x + y*y) / (4*cl_plist.s1);
        if (z > cl_plist.s0)
        {
            res.s0 = 0;
            res.s1 = 0;
        }
        res.s2 = 1.;
        return normalize(res);
    }"""

    def local_z1(self, x, y):
        return roe.ParaboloidFlatLens.local_z1(self, 0, y)

    def local_n1(self, x, y):
        return roe.ParaboloidFlatLens.local_n1(self, 0, y)

    def local_z2(self, x, y):
        return roe.ParaboloidFlatLens.local_z1(self, 0, y)

    def local_n2(self, x, y):
        return roe.ParaboloidFlatLens.local_n1(self, 0, y)


# ===================================================================================================#
# ===================================================================================================#
class DoubleParaboloidLensCy0(roe.DoubleParaboloidLens):
    # 面形误差在第二个透镜上
    hiddenMethods = roe.DoubleParaboloidLens.hiddenMethods + ['double_refract0']

    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        roe.DoubleParaboloidLens.__init__(self, *args, **kwargs)

    def __pop_kwargs(self, **kwargs):
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'lens'

    def local_z1(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        z = (y ** 2) / (4 * self.focus)
        z[z > self.zmax] = self.zmax
        return z

    def local_z2(self, x, y):
        """Determines the surface of OE at (x, y) position."""

        z = (y ** 2) / (4 * self.focus)
        z[z > self.zmax] = self.zmax
        return z

    def local_n1(self, x, y):
        z = (y ** 2) / (4 * self.focus)

        a = 0  # -dz/dx
        b = -y / (2 * self.focus)  # -dz/dy

        a[z > self.zmax] = 0
        b[z > self.zmax] = 0
        c = np.ones_like(x)
        norm = (a ** 2 + b ** 2 + 1) ** 0.5
        return [a / norm, b / norm, c / norm]

    def local_n2(self, x, y):
        z = (y ** 2) / (4 * self.focus)

        a = 0  # -dz/dx
        b = -y / (2 * self.focus)  # -dz/dy
        a[z > self.zmax] = 0
        b[z > self.zmax] = 0
        c = np.ones_like(x)
        norm = (a ** 2 + b ** 2 + 1) ** 0.5
        return [a / norm, b / norm, c / norm]

    # ===================================================================================================#


# ===================================================================================================#
class AnyPlate_Distorted(roe.Plate):
    # 面形误差在第一个透镜上hiddenMethods = roe.Plate.hiddenMethods + ['double_refract']
    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        roe.Plate.__init__(self, *args, **kwargs)
        self.read_data()

    def __pop_kwargs(self, **kwargs):
        self.fname1 = kwargs.pop('fname1')  # fname1 and fname2 is the ANSYS data file, the mirror surface profile.
        self.fname2 = kwargs.pop('fname2')
        self.fname3 = kwargs.pop('fname3')
        self.get_distorted_surface = kwargs.pop('get_distorted_surface')
        self.zmax = kwargs.pop('zmax', None)  # 面形曲线的最大值，从零开始，器件实际起始厚度为t
        kwargs['pitch'] = kwargs.get('pitch', np.pi / 2)
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'lens'

    def read_data(self):
        if self.get_distorted_surface is None:
            return
        warpX, warpY, warpZ = Read_distorted_surface(self.fname1, self.fname2, self.fname3)
        self.warpNX, self.warpNY = len(warpX), len(warpY)
        self.limPhysX = np.min(warpX), np.max(warpX)
        self.limPhysY = np.min(warpY), np.max(warpY)
        self.get_surface_limits()
        warpA, warpB = np.gradient(warpZ)
        dx = warpX[1] - warpX[0]
        dy = warpY[1] - warpY[0]
        warpA = warpA / dx
        warpB = warpB / dy
        self.warpSplineZ = ndimage.spline_filter(warpZ)
        self.warpSplineA = ndimage.spline_filter(warpA)
        self.warpSplineB = ndimage.spline_filter(warpB)

    def local_z1(self, x, y):
        """第一个表面为为外界输入的自由曲面数据z = (x**2 + y**2) / (4 * self.focus)"""
        coords = np.array(
            [(x - self.limPhysX[0]) /
             (self.limPhysX[1] - self.limPhysX[0]) * (self.warpNX - 1),
             (y - self.limPhysY[0]) /
             (self.limPhysY[1] - self.limPhysY[0]) * (self.warpNY - 1)])
        z = ndimage.map_coordinates(self.warpSplineZ, coords, prefilter=True)

        z[z > self.zmax] = self.zmax
        return z

    def local_z2(self, x, y):
        """第二个表面为平面"""
        return self.local_z(x, y)

    def local_n1(self, x, y):
        Ax = -self.warpSplineA
        By = -self.warpSplineB
        coords = np.array(
            [(x - self.limPhysX[0]) /
             (self.limPhysX[1] - self.limPhysX[0]) * (self.warpNX - 1),
             (y - self.limPhysY[0]) /
             (self.limPhysY[1] - self.limPhysY[0]) * (self.warpNY - 1)])
        a = ndimage.map_coordinates(Ax, coords, prefilter=True)
        b = ndimage.map_coordinates(By, coords, prefilter=True)
        z = ndimage.map_coordinates(self.warpSplineZ, coords, prefilter=True)

        a[z > self.zmax] = 0
        b[z > self.zmax] = 0
        c = np.ones_like(x)
        norm = (a ** 2 + b ** 2 + 1) ** 0.5
        return [a / norm, b / norm, c / norm]

    def local_n2(self, x, y):
        return self.local_n(x, y)


# ===================================================================================================#
# ===================================================================================================#


def see_the_bump():
    Surface_name = 'Surface_'
    beamLine = raycing.BeamLine()
    Surface_name = 'OE_'
    kwargs_OE = dict(
        name='Mirror', center=[0, 100000, 0],
        limPhysX=[-0.1, 0.1], limPhysY=[-100, 100],
        p=40000, q=30000, isCylindrical=True,
        targetOpenCL=r"auto")
    kwargs_OE['get_distorted_surface'] = 'error'
    kwargs_OE['fname1'] = Surface_name + 'X.txt'
    kwargs_OE['fname2'] = Surface_name + 'Y.txt'
    kwargs_OE['fname3'] = Surface_name + 'Z.txt'
    oe = EllipMirrorDistorted(beamLine, **kwargs_OE)

    xi = oe.warpX
    yi = oe.warpY
    zi = oe.warpZ * 1e6
    print(xi.shape, yi.shape, zi.shape)
    rmsA = ((oe.warpA ** 2).sum() / (oe.warpNX * oe.warpNY)) ** 0.5
    rmsB = ((oe.warpB ** 2).sum() / (oe.warpNX * oe.warpNY)) ** 0.5

    fig = plt.figure(figsize=(6, 8))

    fig.suptitle('{0}\n'.format('Surface Error Profile') +
                 r'rms slope errors:' + '\n' + r'dz/dx = {0:.2f} $\mu$rad, '
                                               r'dz/dy = {1:.2f} $\mu$rad'.format(rmsA * 1e6, rmsB * 1e6),
                 fontsize=14)
    rect_2D = [0.15, 0.08, 0.75, 0.80]
    ax = plt.axes(rect_2D)
    ax.contour(xi, yi, zi.T, 15, linewidths=0.5, colors='k')
    c = ax.contourf(xi, yi, zi.T, 15, cmap=plt.cm.jet)
    cbar = fig.colorbar(c)  # draw colorbar
    cbar.ax.set_ylabel(u'z (nm)', fontsize=14)
    ax.set_xlabel(u'x (mm)', fontsize=14)
    ax.set_ylabel(u'y (mm)', fontsize=14)


if __name__ == "__main__":
    print('good')
    # see_the_bump()
