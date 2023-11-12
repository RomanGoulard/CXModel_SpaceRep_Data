#!/usr/bin/env python3


import os
import math
# import cv2
from OpenGL.GL import *
import pyglet
from pyglet.gl import *
# from pyglet.gl.wglext_nv import *
import numpy as np
import pyautogui
from PIL import Image
import time
import random
import shutil
from csv import reader
import tkinter as tk
from tkinter import filedialog, ttk
import tkinter.font as font
import scipy.io as spio
# from scipy import stats
import matplotlib.pyplot as mp
import datetime
import warnings

warnings.filterwarnings("ignore")

RNG = np.random.RandomState(2018)

sideCube = 1000

verticies = ((sideCube, -sideCube, -sideCube),
             (sideCube, sideCube, -sideCube),
             (-sideCube, sideCube, -sideCube),
             (-sideCube, -sideCube, -sideCube),
             (sideCube, -sideCube, sideCube),
             (sideCube, sideCube, sideCube),
             (-sideCube, -sideCube, sideCube),
             (-sideCube, sideCube, sideCube))

surfaces = ((0, 1, 2, 3),
            (3, 2, 7, 6),
            (6, 7, 5, 4),
            (4, 5, 1, 0),
            (1, 5, 7, 2),
            (4, 0, 3, 6))

ground_surface = ((sideCube, 0, sideCube),
                  (sideCube, 0, -sideCube),
                  (-sideCube, 0, -sideCube),
                  (-sideCube, 0, sideCube),
                  (sideCube, -20, sideCube),
                  (sideCube, -20, -sideCube),
                  (-sideCube, -20, -sideCube),
                  (-sideCube, -20, sideCube))

sky_color = (0.0, 0.0, 1.0)
ground_color = (0.5, 0.3, 0.0)

# sky_color = (.0, .5, 1.0)
# ground_color = (.6, .3, .0)
#########################
#### Few custom functions
def angle_to_coords(theta, phi, radius):
    """return coordinates of point on sphere given angles and radius"""

    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    return x * radius, y * radius, z * radius


def wrapTo180(angle):
    if not isinstance(angle, float):
        angle_comp = angle.copy()

        if isinstance(angle_comp, np.float64):
            while angle_comp > 180:
                angle_comp -= 360
            while angle_comp < -180:
                angle_comp += 360
        elif len(angle_comp) == 1:
            while angle_comp > 180:
                angle_comp -= 360
            while angle_comp < -180:
                angle_comp += 360
        else:
            while np.any(angle_comp > 180):
                angle_comp[angle_comp > 180] -= 360
            while np.any(angle_comp < -180):
                angle_comp[angle_comp < -180] += 360

    else:
        angle_comp = angle
        while angle_comp > 180:
            angle_comp -= 360
        while angle_comp < -180:
            angle_comp += 360

    return angle_comp


# def read_texture(filename):
#     size = (512, 512)
#
#     img = Image.open(filename)
#
#     lim = 90# * np.sqrt(2)
#
#     img_RotCoord = [np.linspace(-lim, lim, img.size[0]),
#                     np.linspace(-lim, lim, img.size[1])]
#
#     Xdisp = np.linspace(0, 180, img.size[0])
#     Ydisp = np.linspace(0, 360, img.size[1]) - 180
#     disp_RotCoord = [Xdisp,
#                      Ydisp]
#     # disp_RotCoord[0][disp_RotCoord[0] > 180] -= 360
#     # print(disp_RotCoord)
#
#     img_mat = np.array(img.getdata(), np.uint8).reshape(img.size[0], img.size[1], 3)
#     DispMap = np.zeros(img_mat.shape)
#
#     for ix in range(img_mat.shape[0]):
#         for iy in range(img_mat.shape[1]):
#             CoordInput_theta = np.degrees(np.arctan2(img_RotCoord[0][ix], img_RotCoord[1][iy]))
#             CoordInput_rho = np.sqrt(img_RotCoord[0][ix]**2 + img_RotCoord[1][iy]**2)# * np.tan(np.radians(CoordInput_theta))
#
#             Disp_x = int(np.argmin(abs(disp_RotCoord[0][:] - CoordInput_rho)))
#             Disp_y = int(np.argmin(abs(disp_RotCoord[1][:] - CoordInput_theta)))
#
#             DispMap[Disp_x, Disp_y, :] = img_mat[ix, iy, :]
#
#     img2 = cv2.resize(DispMap, size, interpolation=cv2.INTER_AREA)
#     # img2filt = np.array(img.getdata(), np.float).reshape(img.size[0], img.size[1], 3)
#
#     # img2 = filtImg(img2, 'lowpass', (2, 1.0))
#     # img2 = filtImg(img2, 'highpass', (20, 1.0))
#
#     img = Image.fromarray(np.uint8(img2))
#
#     # print(img)
#
#     img_data = np.array(list(img.getdata()), np.uint8)
#
#     # print(np.unique(img_data))
#
#     textID = glGenTextures(1)
#     return textID, img_data, img


###################################
#### 3D objects rendering functions
def Polygons_generate(x, y, z, color, dimension, batch=None):
    x = x - np.mean(x)
    y = y - np.mean(y)
    ratio = dimension/np.max([np.max(abs(x)), np.max(abs(y))])
    x = x * ratio
    y = y * ratio
    z = z * ratio
    z[z < 0] = 0
    if not batch:
        glBegin(GL_QUADS)
        for surface in range(x.shape[0]):
            col_surf = (color[surface, 0], color[surface, 1], color[surface, 2])
            glColor3fv(col_surf)
            for vertex in range(4):
                if vertex < 3:
                    xyz = x[surface, vertex], z[surface, vertex], y[surface, vertex]
                    glVertex3fv(xyz)
                else:
                    xyz = x[surface, 0], z[surface, 0], y[surface, 0]
                    glVertex3fv(xyz)
        glEnd()
    else:
        for surface in range(x.shape[0]):
            vertices = [int(x[surface, 0]*10), int(z[surface, 0]*10), int(y[surface, 0]*10),
                        int(x[surface, 0]*10), int(z[surface, 0]*10), int(y[surface, 0]*10),
                        int(x[surface, 1]*10), int(z[surface, 1]*10), int(y[surface, 1]*10),
                        int(x[surface, 2]*10), int(z[surface, 2]*10), int(y[surface, 2]*10),
                        int(x[surface, 0]*10), int(z[surface, 0]*10), int(y[surface, 0]*10),
                        int(x[surface, 0]*10), int(z[surface, 0]*10), int(y[surface, 0]*10)]

            batch.add(len(vertices) // 3, GL_TRIANGLE_STRIP, None, ("v3i", vertices))


def Cube(color):
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x += 1
            glColor3f(color[0], color[1], color[2])
            glVertex3f(verticies[vertex][0], verticies[vertex][1], verticies[vertex][2])
    glEnd()


def Cube2(position, width, height, Color=(0, 0, 0)):
    X1 = -(position[1] - width/2)
    X2 = -(position[1] + width/2)
    Y1 = position[0] - width/2
    Y2 = position[0] + width/2
    Z1 = 0
    Z2 = height

    verticies_cube = ((X1, Z1, Y1),
                      (X1, Z1, Y2),
                      (X2, Z1, Y2),
                      (X2, Z1, Y1),
                      (X1, Z2, Y1),
                      (X1, Z2, Y2),
                      (X2, Z2, Y2),
                      (X2, Z2, Y1))

    surfaces_cube = ((0, 1, 2, 3),
                     (0, 1, 5, 4),
                     (1, 2, 6, 5),
                     (2, 3, 7, 6),
                     (3, 0, 4, 7),
                     (4, 5, 6, 7))

    glBegin(GL_QUADS)
    for surface in surfaces_cube:
        x = 0
        for vertex in surface:
            x += 1
            glColor3fv(Color)
            glVertex3fv(verticies_cube[vertex])
    glEnd()


def Surface(color):
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x += 1
            glColor3f(color[0], color[1], color[2])
            glVertex3f(ground_surface[vertex][0], ground_surface[vertex][1], ground_surface[vertex][2])
    glEnd()


def Cylinder(center, radius, height, num_slices, Color=(0, 0, 0)):
    r = radius
    h = height
    n = float(num_slices)

    center = [center[0], -center[1]]

    circle_pts = []
    for i in range(int(n) + 1):
        angle = 2 * math.pi * (i / n)
        x = r * math.cos(angle) + center[1]
        y = r * math.sin(angle) + center[0]
        pt = (x, y)
        circle_pts.append(pt)

    glBegin(GL_TRIANGLE_FAN)  # drawing the back circle
    glColor3f(Color[0], Color[1], Color[2])
    # glVertex(0, h / 2.0, 0)
    for (x, y) in circle_pts:
        z = 0.0
        xyz = (x, z, y)
        glVertex3f(x, z, y)
    glEnd()

    glBegin(GL_TRIANGLE_FAN)  # drawing the front circle
    glColor3f(Color[0], Color[1], Color[2])
    # glVertex(0, h / 2.0, 0)
    for (x, y) in circle_pts:
        z = h
        xyz = (x, z, y)
        glVertex3f(x, z, y)
    glEnd()

    glBegin(GL_TRIANGLE_STRIP)  # draw the tube
    glColor3f(Color[0], Color[1], Color[2])
    for (x, y) in circle_pts:
        z = h
        x0y = (x, 0, y)
        glVertex3f(x, 0, y)
        xyz = (x, z, y)
        glVertex3f(x, z, y)
    glEnd()


def Cone(center, radius, height, num_slices, Color=(0, 0, 0)):
    r = radius
    h = height
    n = float(num_slices)

    center = [center[0], -center[1]]

    circle_pts = []
    for i in range(int(n) + 1):
        angle = 2 * math.pi * (i / n)
        x = r * math.cos(angle) + center[1]
        y = r * math.sin(angle) + center[0]
        pt = (x, y)
        circle_pts.append(pt)

    glBegin(GL_TRIANGLE_FAN)  # drawing the back circle
    glColor3f(Color[0], Color[1], Color[2])
    # glVertex(0, h / 2.0, 0)
    for (x, y) in circle_pts:
        z = 0.0
        xyz = (x, z, y)
        glVertex3f(x, z, y)
    glEnd()

    glBegin(GL_TRIANGLE_FAN)  # drawing the front circle
    glColor3f(Color[0], Color[1], Color[2])
    # glVertex(0, h / 2.0, 0)
    for (x, y) in circle_pts:
        z = h
        czc = (center[1], z, center[0])
        glVertex3f(center[1], z, center[0])
    glEnd()

    glBegin(GL_TRIANGLE_STRIP)  # draw the tube
    glColor3f(Color[0], Color[1], Color[2])
    for (x, y) in circle_pts:
        z = h
        x0y = (x, 0, y)
        glVertex3f(x, 0, y)
        czc = (center[1], z, center[0])
        glVertex3f(center[1], z, center[0])
    glEnd()


# def Sphere(center, radius, facets, Color=(0, 0, 0), elevation=0.0):
#     """approximate a sphere using a certain number of facets"""
#
#     center = [center[0], -center[1]]
#
#     dtheta = 180.0 / facets
#     dphi = 360.0 / facets
#
#     sphere_list = glGenLists(2)
#     glNewList(sphere_list, GL_COMPILE)
#
#     glBegin(GL_QUADS)
#
#     for y in range(facets):
#         theta = y * dtheta - 90
#         for x in range(facets):
#             phi = x * dphi
#             a1 = theta, phi
#             a2 = theta + dtheta, phi
#             a3 = theta + dtheta, phi + dphi
#             a4 = theta, phi + dphi
#
#             angles = [a1, a2, a3, a4]
#
#
#             for angle in angles:
#                 x, y, z = angle_to_coords(np.radians(angle[0]), np.radians(angle[1]), radius)
#                 x += center[0]
#                 y += center[1]
#                 z += elevation
#                 glColor3fv(Color)
#                 glVertex(x, z, y)
#
#     glEnd()
#
#     glEndList()


# def Tree(center, radius, height, num_slices, Color=(0, 0, 0)):
#     Cylinder(center, radius/3, height*0.8, num_slices, Color=Color)
#     Sphere(center, radius, num_slices, Color=Color, elevation=height*0.9)


################################################
#### Eye model -> Geometry and post-process (LI)
class Eye_obj:
    def __init__(self, ori_gen, radius, nb_ommat, dist_met='uniform', fovea_loc=None):
        self.type = 'eye'
        self.nb_ommat = nb_ommat
        self.nb_ommat_tot = self.nb_ommat[0] * self.nb_ommat[1]

        # self.PixMap_3d, self.PixMap_PhiTheta = self.cube_pix(0.1, cubeface_size)

        self.ommatidies_dir, self.ommatidies_acc, self.ommatidies_pos, self.ommatidies_ID = self.generate_eye(ori_gen, radius, nb_ommat, method=dist_met, fovea_loc=fovea_loc)
        self.LI_net = self.generate_LInet(InhibRange=8, MaxDist=-1)

        if dist_met == 'dispersion':
            self.nb_ommat_tot = len(self.ommatidies_ID)

        # self.ReceptiveField = self.receptfield_calculation(self.ommatidies_dir, self.ommatidies_acc, self.PixMap_PhiTheta)

        self.output_size = self.nb_ommat_tot

    def process(self, image_vec, RF_list):
        ommatidies_val = np.zeros(self.nb_ommat_tot)

        image_vec = image_vec.astype(np.float32)
        image_vec -= np.min(image_vec)
        if np.max(image_vec) != 0:
            image_vec /= np.max(image_vec)

        for iommat in range(len(self.ommatidies_ID)):
            if len(RF_list[iommat]) > 0:
                list_pix = np.asarray(RF_list[iommat])
                cropped_im = image_vec[list_pix[:, 0]]
                # cropped_im = cropped_im
                if np.max(image_vec) != 0:
                    ommatidies_val[iommat] = np.mean(cropped_im)# / len(list_pix[:, 0])#
                else:
                    ommatidies_val[iommat] = np.mean(cropped_im)
            else:
                ommatidies_val[iommat] = 0

        return ommatidies_val

    def postprocess_LI(self, ommatidies_val, method='absolute', thresh=0.1):
        if len(ommatidies_val.shape) == 1:
            LI_val = np.reshape(np.zeros_like(ommatidies_val, dtype=np.float32), (1, ommatidies_val.shape[0]))
            ommatidies_val = np.reshape(ommatidies_val.astype(np.float32), (1, ommatidies_val.shape[0]))
        else:
            LI_val = np.zeros_like(ommatidies_val, dtype=np.float32)
            ommatidies_val = ommatidies_val.copy()

        # PN activity calculation
        pn_float = (ommatidies_val @ self.LI_net)  # / (np.ones_like(ommat) @ abs(self.LatInhib))
        pn_float[pn_float < 0.0] = 0.0

        if method == 'absolute':
            pn_out_float = pn_float.copy()
            pn_out_float[pn_out_float < thresh] = 0.0
        elif method == 'binary':
            pn_out_float = pn_float.copy()
            pn_out_float[pn_out_float < thresh] = 0.0
            pn_out_float[pn_out_float >= thresh] = 1.0
            pn_out_float = pn_out_float.astype(np.uint8)
        elif method == 'changerate':
            pn_out_float = abs(pn_float - self.pn_memo) / (pn_float + self.pn_memo + 0.0000000001)
            # pn_out_float[(pn_float + self.pn_memo) == 0.0] = 0.0
            pn_out_float[pn_out_float < thresh] = 0.0
            pn_out_float[pn_out_float >= thresh] = 1.0
            self.pn_memo = pn_float.copy()
        else:
            pn_out_float = pn_float.copy()
            print('Edge detection method not recognize: absolute used')

        LI_val = pn_out_float

        return LI_val

    #### Function to build the eye geometry
    def generate_eye(self, ori_gen, radius, nb_ommat, method='uniform', fovea_loc=None):
        if method == 'dispersion':
            pass
        else:
            nb_ommat_tot = nb_ommat[0]*nb_ommat[1]
            iommat_vec = np.zeros([nb_ommat_tot, 2])
            iommat_vec.fill(np.nan)
            accept_vec = np.zeros([nb_ommat_tot, 2])
            accept_vec.fill(np.nan)
            iommat_pos = np.zeros([nb_ommat_tot, 2])
            iommat_ID = np.zeros(nb_ommat_tot)

        if (abs(fovea_loc[0])-abs(ori_gen[0])) > (abs(fovea_loc[0])-radius[0]):
            std_hor = radius[0]/2
        else:
            std_hor = radius[0]

        if (abs(fovea_loc[1])-abs(ori_gen[1])) > (abs(fovea_loc[1])-radius[1]):
            std_ver = radius[1] / 2
        else:
            std_ver = radius[1]

        ##############################
        if method == 'uniform':
            angle_y = np.linspace(-radius[0], radius[0], num=nb_ommat[0])
            angle_z = np.linspace(-radius[1], radius[1], num=nb_ommat[1])

        ##############################
        # elif method == 'gaussian':
        #     if fovea_loc is None:
        #         fovea_loc = [0, 0]
        #
        #     fovea_loc = [fovea_loc[0] - ori_gen[0],
        #                  fovea_loc[1] - ori_gen[1]]
        #
        #     distribution_y = stats.norm(loc=fovea_loc[0], scale=std_hor)
        #     # percentile point, the range for the inverse cumulative distribution function:
        #     bounds_for_range_y = distribution_y.cdf([-radius[0], radius[0]])
        #
        #     distribution_z = stats.norm(loc=fovea_loc[1], scale=std_ver)
        #     # percentile point, the range for the inverse cumulative distribution function:
        #     bounds_for_range_z = distribution_z.cdf([-radius[1], radius[1]])
        #
        #     # Linspace for the inverse cdf:
        #     angle_y = distribution_y.ppf(np.linspace(*bounds_for_range_y, num=nb_ommat[0]))
        #     angle_z = distribution_z.ppf(np.linspace(*bounds_for_range_z, num=nb_ommat[1]))

        elif method == 'dispersion':
            angle_correction = wrapTo180(np.random.uniform(0, 360, 1)[0])
            nb_ommat = 8
            r = 3
            inter_ommat = 1.5
            dispersive_rate = 0.5

            OpticalAxes = [[0.0, 0.0]]
            InterOmmatidialAngle = [[inter_ommat, inter_ommat]]

            nb_ommat_tot = 1

            while r < 180:
                angles = wrapTo180(np.linspace(0, 360 - 360 / (np.floor(nb_ommat / 2) * 2), int(np.floor(nb_ommat / 2) * 2)) + angle_correction)
                noise_az = np.random.normal(0.0, 0.1, int(np.floor(nb_ommat / 2) * 2))
                azimuth = noise_az + np.cos(np.radians(angles)) * r
                noise_el = np.random.normal(0.0, 0.1, int(np.floor(nb_ommat / 2) * 2))
                elevation = noise_el + np.sin(np.radians(angles)) * r

                for i_newommat in range(len(azimuth)):
                    if (abs(elevation[i_newommat]) < radius[1]-inter_ommat/2) & (abs(azimuth[i_newommat]) < radius[0]-inter_ommat/2):
                        OpticalAxes.append([elevation[i_newommat], azimuth[i_newommat]])
                        InterOmmatidialAngle.append([inter_ommat, inter_ommat])
                        nb_ommat_tot += 1

                inter_ommat *= (1 + dispersive_rate/10)
                r += (1 + dispersive_rate + 0.1) * inter_ommat
                nb_ommat = int((np.pi*r)/(inter_ommat)*0.95)
                angle_correction += 180 / (np.floor(nb_ommat / 2) * 2)

            # print(nb_ommat_tot)

            iommat_vec = np.zeros([nb_ommat_tot, 2])
            iommat_vec.fill(np.nan)
            accept_vec = np.zeros([nb_ommat_tot, 2])
            accept_vec.fill(np.nan)
            iommat_pos = np.zeros([nb_ommat_tot, 2])
            iommat_ID = np.zeros(nb_ommat_tot)

        else:
            angle_y = np.random.uniform(-radius[0], radius[0], nb_ommat[0])
            angle_z = np.random.uniform(-radius[1], radius[1], nb_ommat[1])
            print('Ommatidies construction: Distribution not recognize -> random assignment (uniform)')

        iy = 0
        iz = 0

        if method != 'dispersion':
            for iommat in range(nb_ommat_tot):
                ##############################
                iy_rev = iy
                theta = angle_y[iy_rev]
                while theta > 180:
                    theta -= 360
                while theta < -180:
                    theta += 360

                # if (iy_rev != 0) & (iy != nb_ommat[0]-1):
                #     accept_theta = (abs(angle_y[iy_rev+1] - angle_y[iy_rev-1])/2)
                # elif iy_rev == 0:
                #     accept_theta = abs(angle_y[iy_rev + 1] - angle_y[iy_rev])
                # elif iy_rev == nb_ommat[0]-1:
                #     accept_theta = abs(angle_y[iy_rev] - angle_y[iy_rev-1])

                ##############################
                iz_rev = angle_z.shape[0] - 1 - iz
                phi = angle_z[iz_rev]
                while phi > 180:
                    phi -= 360
                while phi < -180:
                    phi += 360

                accept_theta = abs(theta)**0.5/1.2
                if accept_theta < 2.0:
                    accept_theta = 2.0
                elif accept_theta > 10.0:
                    accept_theta = 10.0

                accept_phi = abs(phi)**0.5/1.2
                if accept_phi < 2.0:
                    accept_phi = 2.0
                if accept_phi > 10.0:
                    accept_phi = 10.0

                # if (iz_rev != 0) & (iz_rev != nb_ommat[1]-1):
                #     accept_phi = (abs(angle_z[iz_rev + 1] - angle_z[iz_rev - 1])/2)
                # elif iz_rev == 0:
                #     accept_phi = abs(angle_z[iz_rev + 1] - angle_z[iz_rev])
                # elif iz_rev == nb_ommat[1]-1:
                #     accept_phi = abs(angle_z[iz_rev] - angle_z[iz_rev-1])

                ##############################
                iommat_vec[iommat, :] = [phi, theta]
                accept_vec[iommat, :] = [accept_phi, accept_theta]
                iommat_pos[iommat, :] = [iz, iy]
                iommat_ID[iommat] = iommat

                ##############################
                iy += 1
                if iy >= nb_ommat[0]:
                    iy = 0
                    iz += 1

        else:
            iommat_vec = np.asarray(OpticalAxes)
            accept_vec = np.asarray(InterOmmatidialAngle)

            OpticalAxes_np = np.asarray(OpticalAxes)
            idx_sort_az = np.argsort(OpticalAxes_np[:, 0])
            idx_sort_el = np.argsort(OpticalAxes_np[:, 1])
            for iom in range(len(idx_sort_az)):
                iommat_pos[iom, :] = [idx_sort_el[iom], idx_sort_az[iom]]
                iommat_ID[iom] = iom

        return iommat_vec, accept_vec, iommat_pos, iommat_ID

    def generate_LInet(self, InhibRange=10, MaxDist=15):
        self.pn_memo = np.zeros_like(self.ommatidies_ID)

        ## Lateral Inhibition (Eye v2.1 | dispersion)
        LI_net = np.zeros((len(self.ommatidies_ID), len(self.ommatidies_ID)))
        for iom1 in range(len(self.ommatidies_ID)):
            # print('\n\n%%%%%')
            # print(iom1, self.Monocular_eye.ommatidies_dir[iom1, 1], self.Monocular_eye.ommatidies_dir[iom1, 0])

            # LI_visu = np.zeros((1800, 3600, 3), np.uint8)
            dist = np.sqrt((self.ommatidies_dir[:, 0] - self.ommatidies_dir[iom1, 0])**2
                           + (self.ommatidies_dir[:, 1] - self.ommatidies_dir[iom1, 1])**2)
            minimal_dists = np.argsort(dist)
            if (MaxDist != -1) & (dist[minimal_dists[InhibRange]] > MaxDist):
                InhibRange = np.sum(dist < MaxDist)
            elif MaxDist == -1:
                InhibRange = np.sum(dist < 1.5 * np.sum(self.ommatidies_acc[iom1, :]))
            RatioInhib = np.mean(dist[minimal_dists[1:InhibRange]])
            NormFactor = np.sum(np.exp(-0.5 * (dist[minimal_dists[1:InhibRange]] / RatioInhib)**2))
            # print('\n££££££')
            for imins in range(InhibRange+1):
                iom2 = minimal_dists[imins]
                # print(iom2, self.Monocular_eye.ommatidies_dir[iom2, 1], self.Monocular_eye.ommatidies_dir[iom2, 0])
                if imins == 0:
                    LI_net[iom1, iom2] = NormFactor * 0.8
                    # LI_net[iom1, iom2] = InhibRange - 1.5

                    # LI_visu = cv2.circle(LI_visu,
                    #                      (int(10 * (self.ommatidies_dir[iom2, 1] + 180)),
                    #                       int(10 * (-self.ommatidies_dir[iom2, 0] + 90))),
                    #                      int(8 * self.ommatidies_acc[iom2, 0]),
                    #                      color=(0, 255, 0),
                    #                      thickness=-1)
                else:
                    LI_net[iom1, iom2] = -np.exp(-0.5 * (dist[minimal_dists[imins]]/RatioInhib)**2)
                    # LI_net[iom1, iom2] = -1.0
                    # print(dist[minimal_dists[imins]], RatioInhib, self.LatInhib[iom1, iom2])

                    # LI_visu = cv2.circle(LI_visu,
                    #                      (int(10 * (self.ommatidies_dir[iom2, 1] + 180)),
                    #                       int(10 * (-self.ommatidies_dir[iom2, 0] + 90))),
                    #                      int(10 * self.ommatidies_acc[iom2, 0]),
                    #                      color=(-LI_net[iom1, iom2]*255, 0, 0),
                    #                      thickness=-1)

            # cv2.imshow('LI show', cv2.resize(LI_visu, (1440, 720)))
            # cv2.waitKey()

        return LI_net


######################
#### World-Agent model
class Agent_sim(pyglet.window.Window):
    def __init__(self, name_saved,
                 Obj_names, Obj_position, Obj_size,
                 MB_thresh,
                 width, height,
                 Display, Display_neurons,
                 Scenario, BrainType,
                 Gain_global, Gain_local,
                 MotorNoise,
                 *args, **kwargs):
        super(Agent_sim, self).__init__(width, height, *args, **kwargs)

        print('** Model circuits initialisation **')

        self.BrainType = BrainType

        self.Gain_global = Gain_global
        self.Gain_local = Gain_local

        self.MotorNoise = MotorNoise

        if len(Scenario) == 2:
            self.Scenario = Scenario
            self.Blink = False
        elif len(Scenario) == 3:
            self.Scenario = Scenario[0:2]
            if Scenario[2] == 'B':
                self.Blink = True
            elif Scenario[2] == '2':
                self.Scenario = 'VM2'
                self.Blink = False
            else:
                self.Blink = False

        self.Display = Display
        self.Display_neurons = Display_neurons

        if self.Scenario == 'VM':
            self.Target_on = False
            angle_random = 45#np.random.uniform(0.0, 360.0, 1)[0]
            self.Target_location = [200 * np.cos(np.radians(angle_random)), 200 * np.sin(np.radians(angle_random))]
        elif self.Scenario == 'MB':
            self.Target_on = False
            angle_random = np.random.uniform(0.0, 360.0, 1)[0]
            self.Target_location = [400 * np.cos(np.radians(angle_random)), 400 * np.sin(np.radians(angle_random))]
        else:
            self.Target_on = True
            angle_random = 45#np.random.uniform(0.0, 360.0, 1)[0]
            self.Target_location = [200 * np.cos(np.radians(angle_random)), 200 * np.sin(np.radians(angle_random))]
        self.Target_size = [15, 160]

        self.winOmmat = 'Ommatidies activity'
        self.winLI = 'vPNs activity'
        self.winOF = 'OF calculation'

        # self.lightfv = ctypes.c_float * 4

        ## Dark = empty world
        self.dark = False
        self.main_batch = pyglet.graphics.Batch() #-> To-do: move object simulation to batch

        ## 3D world unpacking
        if Obj_names == []:
            self.dark = True
            self.angle2Obj = 0.0
        elif Obj_names[0] == 'Polygons':
            self.Obj_names = Obj_names
            self.Poly_C = Obj_position[0]
            self.Poly_X = Obj_position[1]
            self.Poly_Y = Obj_position[2]
            self.Poly_Z = Obj_position[3]
            self.angle2Obj = 0.0

            self.Poly_X = self.Poly_X - np.mean(self.Poly_X)
            self.Poly_Y = self.Poly_Y - np.mean(self.Poly_Y)
            self.Poly_Z = self.Poly_Z - np.mean(self.Poly_Z)

            ratio = 500 / np.max([np.max(abs(self.Poly_X)), np.max(abs(self.Poly_Y))])
            self.Poly_X = self.Poly_X * ratio
            self.Poly_Y = self.Poly_Y * ratio
            self.Poly_Z = self.Poly_Z * ratio

            for ivert in range(self.Poly_X.shape[0]):
                x1, y1, z1 = int(self.Poly_X[ivert, 0]), int(self.Poly_Y[ivert, 0]), int(
                    self.Poly_Z[ivert, 0])
                x2, y2, z2 = int(self.Poly_X[ivert, 1]), int(self.Poly_Y[ivert, 1]), int(
                    self.Poly_Z[ivert, 1])
                x3, y3, z3 = int(self.Poly_X[ivert, 2]), int(self.Poly_Y[ivert, 2]), int(
                    self.Poly_Z[ivert, 2])
                p1 = [x1, z1, y1, x1, z1, y1, x2, z2, y2, x3, z3, y3, x1, z1, y1, x1, z1, y1]

                c1 = (int(self.Poly_C[ivert, 0] * 255), int(self.Poly_C[ivert, 1] * 255),
                      int(self.Poly_C[ivert, 2] * 255)) * 6

                self.main_batch.add(len(p1) // 3, pyglet.gl.GL_TRIANGLE_STRIP, None, ('v3i', p1), ('c3B', c1))
        else:
            self.Obj_names = Obj_names
            self.Obj_position = Obj_position
            self.Obj_size = Obj_size
            self.angle2Obj = np.arctan2(Obj_position[0][1], Obj_position[0][0])

        ## Neuron record allocation
        self.name_saved = name_saved
        self.Pose_Mat = []
        self.EPG_activity = []
        self.PEN_activity = []
        self.PEG_activity = []
        self.D7_activity = []
        self.NO_activity = []
        self.Polar_activity = []
        self.PFN_activity = []
        self.hDc_activity = []
        self.PFN_in_activity = []
        self.hDc_in_activity = []
        self.PFL_activity = []
        self.LAL_activity = []
        self.PImemory_PFL = []
        self.PImemory_PFLloc = []
        self.PImemory_PFLint = []
        self.PImemory_PFLglo = []
        self.PImemory_hDc = []
        self.FBt_memory = []
        self.FBt_hDc_memory = []

        self.Width = width
        self.Height = height

        ## Position allocation
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.translation_z = -0.5  # Height
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = np.random.uniform(0.0, 360.0, 1)
        # self.rotation_z = np.arctan2((self.position_feeder[1] - self.translation_y),
        #                              (self.position_feeder[0] - self.translation_x)) #orientation_relat # Yaw
        self.speed = 0.25 #0.25
        self.it = 0

        ## Camera parameters
        self.nb_segments = 4
        angle_adjust_base = -180 + 360 / self.nb_segments / 2  # -157.5
        self.NEAR_CLIPPING_PLANE = 0.1
        self.H_CLIPPING_PLANE = [-0.5, 0.5]
        self.W_CLIPPING_PLANE = [-0.1, 0.1]
        self.FAR_CLIPPING_PLANE = 5000
        PixMap_angular = np.zeros((self.Height*self.Width, 2))
        Top_cam = self.H_CLIPPING_PLANE[1]
        Bottom_cam = self.H_CLIPPING_PLANE[0]
        Right_cam = self.W_CLIPPING_PLANE[1]
        Left_cam = self.W_CLIPPING_PLANE[0]
        width = round(self.Width / self.nb_segments)
        height = round(self.Height)
        pixratio_H = abs(Top_cam - Bottom_cam) / height
        pixratio_W = abs(Right_cam - Left_cam) / width

        print('\t-> Ommatidia angular mapping')
        ipix = 0
        for ih in range(height):
            for icam in range(self.nb_segments):
                angle_adjust = angle_adjust_base + icam*360/self.nb_segments
                for iw in range(width):
                    PixMap_angular[ipix] = [np.degrees(np.arctan2(Top_cam - (ih+0.5) * pixratio_H,
                                                                  self.NEAR_CLIPPING_PLANE)),
                                            np.degrees(np.arctan2(Left_cam + (iw+0.5) * pixratio_W,
                                                                  self.NEAR_CLIPPING_PLANE)) + angle_adjust]

                    ipix += 1

        # PixMap_X_img = PixMap_angular[:, 1].reshape((Height, Width))
        # PixMap_Y_img = PixMap_angular[:, 0].reshape((Height, Width))

        self.Monocular_eye = Eye_obj([0, 0], [170.0, 80.0], [40, 20], dist_met='dispersion', fovea_loc=[0, 0])

        # imEyeMercator = np.zeros((1800, 3600, 3), np.uint8)
        # for iom in range(len(self.Monocular_eye.ommatidies_ID)):
        #     imEyeMercator = cv2.circle(imEyeMercator,
        #                                 (int(10 * (self.Monocular_eye.ommatidies_dir[iom, 1] + 180)),
        #                                  int(10 * (-self.Monocular_eye.ommatidies_dir[iom, 0] + 90))),
        #                                 int(10 * self.Monocular_eye.ommatidies_acc[iom, 0]),
        #                                 color=(255, 255, 255),
        #                                 thickness=2)
        #
        # imEyeMercator = cv2.resize(imEyeMercator,
        #                            (Width, Height),
        #                            interpolation=cv2.INTER_AREA)
        # cv2.imshow('Eye model', imEyeMercator)
        # cv2.waitKey(0)

        ## Save eye model parameters (Ommatidia direction & acceptance angles) -> Each simulation generates different parameters
        name_savedfile = self.name_saved + 'Ommat_receptfields.csv'
        EyeModel_save = np.concatenate((self.Monocular_eye.ommatidies_dir, self.Monocular_eye.ommatidies_acc), axis=1)
        np.savetxt(name_savedfile, EyeModel_save, delimiter=",")

        self.pn_memo = np.zeros(len(self.Monocular_eye.ommatidies_ID))
        self.Eye_rf_list = []
        self.VisRew_att_rf = np.zeros(len(self.Monocular_eye.ommatidies_ID))

        ## Generate receptive field of each ommatidia (list of pixels/ommatidia from viewport)
        # -> To-do: register receptive field to eye model
        PixDist_ellipse_img = np.zeros((Height, Width))
        # Cam2Ommat = np.zeros((len(self.Monocular_eye.ommatidies_ID), len(PixMap_angular[:, 0])))
        for iommat in range(len(self.Monocular_eye.ommatidies_ID)):
            PixDist_ellipse = ((wrapTo180(PixMap_angular[:, 0] - self.Monocular_eye.ommatidies_dir[iommat, 0]) / self.Monocular_eye.ommatidies_acc[iommat, 0])**2
                               + (wrapTo180(PixMap_angular[:, 1] - self.Monocular_eye.ommatidies_dir[iommat, 1]) / self.Monocular_eye.ommatidies_acc[iommat, 1])**2)

            PixDist_ellipse_matrix = PixDist_ellipse.reshape((Height, Width))

            PixDist_ellipse_img[PixDist_ellipse_matrix <= 1] = np.random.randint(1, 255, 1)#((iommat + 10) / (self.Monocular_eye.nb_ommat_tot + 9)) * 255

            pixel_list = np.argwhere(PixDist_ellipse <= 1)
            # Cam2Ommat[iommat, pixel_list] = 1.0

            self.Eye_rf_list.append(pixel_list)
            azimuth = self.Monocular_eye.ommatidies_dir[iommat, 1]
            elevation = self.Monocular_eye.ommatidies_dir[iommat, 0]
            if (abs(azimuth) <= 10.0) & (elevation > 0.0):
                self.VisRew_att_rf[iommat] = 1.0
            else:
                self.VisRew_att_rf[iommat] = 0.0

        ## Old eye model
        # self.Ommat_pixelRes = (40, 40)
        # self.Eye_rf, self.ID_ommatidies = eye_generate2(self.Ommat_pixelRes, (self.Height, self.Width))
        #
        # self.Eye_rf_ang = np.zeros((len(self.Eye_rf[:, 0]), 2))
        # self.Eye_rf_ang[:, 0] = (self.Eye_rf[:, 2] + self.Eye_rf[:, 3])/2
        # self.Eye_rf_ang[:, 0] = -(self.Eye_rf_ang[:, 0]/max(self.Eye_rf[:, 3]) * 360 - 180)
        #
        # self.Eye_rf_ang[:, 1] = (self.Eye_rf[:, 0] + self.Eye_rf[:, 1])/2
        # # ratio_HW = height/width
        # ratio_HW = float(width / 4) / float(height)
        # # print(ratio_HW)
        # self.Eye_rf_ang[:, 1] = -(self.Eye_rf_ang[:, 1]/max(self.Eye_rf[:, 1]) * 360 - 180) * ratio_HW
        #
        # self.Eye_rf_center = self.ID_ommatidies

        ## Lateral Inhibition (Eye v2.1 | dispersion)
        # ===> Now moved to eye model processes
        print('\t-> Lateral inhibition circuit')
        self.LatInhib = np.zeros((len(self.Monocular_eye.ommatidies_ID), len(self.Monocular_eye.ommatidies_ID)))
        for iom1 in range(len(self.Monocular_eye.ommatidies_ID)):
            # LI_visu = np.zeros((1800, 3600, 3), np.uint8)
            dist = np.sqrt((self.Monocular_eye.ommatidies_dir[:, 0] - self.Monocular_eye.ommatidies_dir[iom1, 0])**2
                           + (self.Monocular_eye.ommatidies_dir[:, 1] - self.Monocular_eye.ommatidies_dir[iom1, 1])**2)
            minimal_dists = np.argsort(dist)
            InhibRange = 12
            if dist[minimal_dists[InhibRange]] > 15:
                InhibRange = sum(dist < 15)
            RatioInhib = np.mean(dist[minimal_dists[1:InhibRange]])
            NormFactor = np.sum(np.exp(-0.5 * (dist[minimal_dists[1:InhibRange]] / RatioInhib)**2))
            for imins in range(InhibRange+1):
                iom2 = minimal_dists[imins]
                # print(iom2, self.Monocular_eye.ommatidies_dir[iom2, 1], self.Monocular_eye.ommatidies_dir[iom2, 0])
                if imins == 0:
                    self.LatInhib[iom1, iom2] = NormFactor * 0.9
                else:
                    self.LatInhib[iom1, iom2] = -np.exp(-0.5 * (dist[minimal_dists[imins]]/RatioInhib)**2)

        print('\t==> Visiual pipeline generated')
        ## CX nets
        ## (eye v2.0)
        self.CX_pn = np.zeros((1, len(self.Monocular_eye.ommatidies_ID)))
        self.CX_cards = (np.arange(8)+0.5)/8 * 360 - 180
        self.CX_net = np.zeros((len(self.Monocular_eye.ommatidies_ID), len(self.CX_cards)))
        shiftVin = 0.0#np.random.uniform(-180, 180, 1)
        for iom in range(len(self.Monocular_eye.ommatidies_ID)):
            dist = abs(wrapTo180(wrapTo180(self.Monocular_eye.ommatidies_dir[iom, 1] + shiftVin) - self.CX_cards))
            minimum = np.argmin(dist, axis=0)
            self.CX_net[iom, minimum] = 1 #retinotopic

        self.CX_Vin2Rew = self.Monocular_eye.ommatidies_dir[:, 1].copy()
        self.CX_Vin2Rew[self.CX_Vin2Rew > 180] -= 360
        self.CX_Vin2Rew[self.CX_Vin2Rew < -180] += 360
        ## Discrete
        limitVisRew = 15.0
        self.CX_Vin2Rew[abs(self.CX_Vin2Rew) <= limitVisRew] = 1.0
        self.CX_Vin2Rew[abs(self.CX_Vin2Rew) > limitVisRew] = 0.0 #Pos
        
        ##########################
        ## CX connectomic matrixes
        # fig, axs = mp.subplots(3, 3)
        # igraph = 0

        ## Ellipsoid Body compass
        self.CX_Visin = np.zeros(16)
        self.CX_Polin = np.zeros(16)
        self.CX_EB_EPG = np.zeros(16)
        self.CX_PB_PEG = np.zeros(16)
        self.CX_PB_PEN = np.zeros(16)
        self.CX_PB_Delta7 = np.zeros(16)
        self.CX_NO = np.zeros(2)
        self.CX_NOt = np.zeros(1)

        print('\t-> EB - PB connectivity')

        self.CX_EPG2PEG = np.zeros((len(self.CX_EB_EPG), len(self.CX_PB_PEG)))
        for ineuron1 in range(self.CX_EPG2PEG.shape[0]):
            for ineuron2 in range(self.CX_EPG2PEG.shape[1]):
                neuron_distance = np.remainder(ineuron1-ineuron2, 16)
                if neuron_distance == 0:
                    self.CX_EPG2PEG[ineuron1, ineuron2] = 1.0

        self.CX_PEG2EPG = np.zeros((len(self.CX_PB_PEG), len(self.CX_EB_EPG)))
        for ineuron1 in range(self.CX_PEG2EPG.shape[0]):
            for ineuron2 in range(self.CX_PEG2EPG.shape[1]):
                neuron_distance = np.remainder(ineuron1-ineuron2, 16)
                if neuron_distance == 0:
                    self.CX_PEG2EPG[ineuron1, ineuron2] = 1.0

        self.CX_EPG2PEN = np.zeros((len(self.CX_EB_EPG), len(self.CX_PB_PEN)))
        for ineuron1 in range(self.CX_EPG2PEN.shape[0]):
            for ineuron2 in range(self.CX_EPG2PEN.shape[1]):
                neuron_distance = np.remainder(ineuron1-ineuron2, 16)
                if neuron_distance == 0:
                    self.CX_EPG2PEN[ineuron1, ineuron2] = 1.0

        self.CX_PEN2EPG = np.zeros((len(self.CX_PB_PEN), len(self.CX_EB_EPG)))
        shift = 1
        idx_input_half = np.arange(int(len(self.CX_PB_PEN)/2))
        idx_input_half1 = np.remainder(idx_input_half+shift, 8)
        idx_input_half2 = np.remainder(idx_input_half-shift, 8)
        idx_input = np.concatenate((idx_input_half1, idx_input_half2+1000))
        idx_output_half = np.arange(int(len(self.CX_EB_EPG)/2))
        idx_output_half = np.remainder(idx_output_half, 8)
        idx_output = np.concatenate((idx_output_half, idx_output_half+1000))
        for ineuron1 in range(self.CX_PEN2EPG.shape[0]):
            for ineuron2 in range(self.CX_PEN2EPG.shape[1]):
                if idx_input[ineuron1] == idx_output[ineuron2]:
                    self.CX_PEN2EPG[ineuron1, ineuron2] = 1.0

        self.CX_EPG2D7 = np.zeros((len(self.CX_EB_EPG), len(self.CX_PB_Delta7)))
        idx_input = np.arange(len(self.CX_EB_EPG))
        span_in = 1
        idx_output = np.arange(len(self.CX_PB_Delta7))
        span_out = 7
        for ineuron1 in range(self.CX_EPG2D7.shape[0]):
            modulator1 = np.zeros(idx_input.shape)
            imax = int(span_in/2)
            for ispan in range(imax+1):
                idx1 = np.remainder(idx_input[ineuron1]+ispan, 8)
                idx2 = np.remainder(idx_input[ineuron1]-ispan, 8)
                modulator1[idx_input == idx1] = 1.0 - ispan/int((span_in+1)/2)
                modulator1[idx_input == idx2] = 1.0 - ispan/int((span_in+1)/2)

            for ineuron2 in range(self.CX_EPG2D7.shape[1]):
                modulator2 = np.zeros(idx_output.shape)
                imax = int(span_out/2)
                for ispan in range(imax+1):
                    idx1 = np.remainder(idx_output[ineuron2]+ispan, 8)
                    idx2 = np.remainder(idx_output[ineuron2]-ispan, 8)
                    modulator2[idx_output == idx1] = 1.0 - ispan/int((span_out+1)/2)
                    modulator2[idx_output == idx2] = 1.0 - ispan/int((span_out+1)/2)

                modulator1[modulator1 < 0] = 0
                modulator2[modulator2 < 0] = 0
                if (ineuron1 < 8) & (ineuron2 < 8):
                    self.CX_EPG2D7[ineuron1, ineuron2] = sum(modulator1*modulator2)
                elif (ineuron1 >= 8) & (ineuron2 >= 8):
                    self.CX_EPG2D7[ineuron1, ineuron2] = sum(modulator1*modulator2)

        self.CX_D72D7 = np.zeros((len(self.CX_PB_Delta7), len(self.CX_PB_Delta7)))
        idx_input_half = np.arange(int(len(self.CX_PB_Delta7)/2))
        shift = 4
        idx_input_half1 = np.remainder(idx_input_half+shift, 8)
        idx_input_half2 = np.remainder(idx_input_half-shift, 8)
        idx_input = np.concatenate((idx_input_half1, idx_input_half2))
        span_in = 3
        idx_output_half = np.arange(int(len(self.CX_PB_Delta7)/2))
        idx_output_half = np.remainder(idx_output_half, 8)
        idx_output = np.concatenate((idx_output_half, idx_output_half))
        span_out = 7
        for ineuron1 in range(self.CX_D72D7.shape[0]):
            modulator1 = np.zeros(idx_output.shape)
            imax = int(span_in / 2)
            for ispan in range(imax + 1):
                idx1 = np.remainder(idx_input[ineuron1] + ispan, 8)
                idx2 = np.remainder(idx_input[ineuron1] - ispan, 8)
                modulator1[idx_output == idx1] = 1.0 - ispan / int((span_in + 1) / 2)
                modulator1[idx_output == idx2] = 1.0 - ispan / int((span_in + 1) / 2)

            for ineuron2 in range(self.CX_D72D7.shape[1]):
                modulator2 = np.zeros(idx_output.shape)
                imax = int(span_out / 2)
                for ispan in range(imax + 1):
                    idx1 = np.remainder(idx_output[ineuron2] + ispan, 8)
                    idx2 = np.remainder(idx_output[ineuron2] - ispan, 8)
                    modulator2[idx_output == idx1] = 1.0 - ispan / int((span_out + 1) / 2)
                    modulator2[idx_output == idx2] = 1.0 - ispan / int((span_out + 1) / 2)

                modulator1[modulator1 < 0] = 0
                modulator2[modulator2 < 0] = 0
                if (ineuron1 < 8) & (ineuron2 < 8):
                    self.CX_D72D7[ineuron1, ineuron2] = -sum(modulator1 * modulator2)
                elif (ineuron1 >= 8) & (ineuron2 >= 8):
                    self.CX_D72D7[ineuron1, ineuron2] = -sum(modulator1 * modulator2)

        self.CX_D72PEG = np.zeros((len(self.CX_PB_Delta7), len(self.CX_PB_PEG)))
        idx_input_half = np.arange(int(len(self.CX_PB_Delta7)/2))
        shift = 4
        idx_input_half1 = np.remainder(idx_input_half+shift, 8)
        idx_input_half2 = np.remainder(idx_input_half-shift, 8)
        idx_input = np.concatenate((idx_input_half1, idx_input_half2))
        span_in = 3
        idx_output_half = np.arange(int(len(self.CX_PB_PEG)/2))
        idx_output_half = np.remainder(idx_output_half, 8)
        idx_output = np.concatenate((idx_output_half, idx_output_half))
        span_out = 1
        for ineuron1 in range(self.CX_D72PEG.shape[0]):
            modulator1 = np.zeros(idx_output.shape)
            imax = int(span_in / 2)
            for ispan in range(imax + 1):
                idx1 = np.remainder(idx_input[ineuron1] + ispan, 8)
                idx2 = np.remainder(idx_input[ineuron1] - ispan, 8)
                modulator1[idx_output == idx1] = 1.0 - ispan / int((span_in + 1) / 2)
                modulator1[idx_output == idx2] = 1.0 - ispan / int((span_in + 1) / 2)

            for ineuron2 in range(self.CX_D72PEG.shape[1]):
                modulator2 = np.zeros(idx_output.shape)
                imax = int(span_out / 2)
                for ispan in range(imax + 1):
                    idx1 = np.remainder(idx_output[ineuron2] + ispan, 8)
                    idx2 = np.remainder(idx_output[ineuron2] - ispan, 8)
                    modulator2[idx_output == idx1] = 1.0 - ispan / int((span_out + 1) / 2)
                    modulator2[idx_output == idx2] = 1.0 - ispan / int((span_out + 1) / 2)

                modulator1[modulator1 < 0] = 0
                modulator2[modulator2 < 0] = 0
                if (ineuron1 < 8) & (ineuron2 < 8):
                    self.CX_D72PEG[ineuron1, ineuron2] = -sum(modulator1 * modulator2)
                elif (ineuron1 >= 8) & (ineuron2 >= 8):
                    self.CX_D72PEG[ineuron1, ineuron2] = -sum(modulator1 * modulator2)

        self.CX_D72PEN = np.zeros((len(self.CX_PB_Delta7), len(self.CX_PB_PEN)))
        idx_input_half = np.arange(int(len(self.CX_PB_Delta7)/2))
        shift = 4
        idx_input_half1 = np.remainder(idx_input_half+shift, 8)
        idx_input_half2 = np.remainder(idx_input_half-shift, 8)
        idx_input = np.concatenate((idx_input_half1, idx_input_half2))
        span_in = 3
        idx_output_half = np.arange(int(len(self.CX_PB_PEN)/2))
        idx_output_half = np.remainder(idx_output_half, 8)
        idx_output = np.concatenate((idx_output_half, idx_output_half))
        span_out = 1
        for ineuron1 in range(self.CX_D72PEN.shape[0]):
            modulator1 = np.zeros(idx_output.shape)
            imax = int(span_in / 2)
            for ispan in range(imax + 1):
                idx1 = np.remainder(idx_input[ineuron1] + ispan, 8)
                idx2 = np.remainder(idx_input[ineuron1] - ispan, 8)
                modulator1[idx_output == idx1] = 1.0 - ispan / int((span_in + 1) / 2)
                modulator1[idx_output == idx2] = 1.0 - ispan / int((span_in + 1) / 2)

            for ineuron2 in range(self.CX_D72PEN.shape[1]):
                modulator2 = np.zeros(idx_output.shape)
                imax = int(span_out / 2)
                for ispan in range(imax + 1):
                    idx1 = np.remainder(idx_output[ineuron2] + ispan, 8)
                    idx2 = np.remainder(idx_output[ineuron2] - ispan, 8)
                    modulator2[idx_output == idx1] = 1.0 - ispan / int((span_out + 1) / 2)
                    modulator2[idx_output == idx2] = 1.0 - ispan / int((span_out + 1) / 2)

                modulator1[modulator1 < 0] = 0
                modulator2[modulator2 < 0] = 0
                if (ineuron1 < 8) & (ineuron2 < 8):
                    self.CX_D72PEN[ineuron1, ineuron2] = -sum(modulator1 * modulator2)
                elif (ineuron1 >= 8) & (ineuron2 >= 8):
                    self.CX_D72PEN[ineuron1, ineuron2] = -sum(modulator1 * modulator2)

        self.CX_NO2PEN = np.zeros((len(self.CX_NO), len(self.CX_PB_PEN)))
        for ineuron1 in range(len(self.CX_NO)):
            for ineuron2 in range(len(self.CX_PB_PEN)):
                if (ineuron1 == 0) & (ineuron2 < 8):
                    self.CX_NO2PEN[ineuron1, ineuron2] = 1.0
                elif (ineuron1 == 1) & (ineuron2 >= 8):
                    self.CX_NO2PEN[ineuron1, ineuron2] = 1.0

        print('\t-> PB - FB connectivity')
        self.CX_PB_PFNc = np.zeros(16)
        self.CX_PB_PFL3 = np.zeros(16)
        self.CX_LAL = np.zeros(2)

        self.CX_D72PFN = np.zeros((len(self.CX_PB_Delta7), len(self.CX_PB_PFNc)))
        idx_input_half = np.arange(int(len(self.CX_PB_Delta7)/2))
        shift = 4
        idx_input_half1 = np.remainder(idx_input_half+shift, 8)
        idx_input_half2 = np.remainder(idx_input_half-shift, 8)
        idx_input = np.concatenate((idx_input_half1, idx_input_half2))
        span_in = 3
        idx_output_half = np.arange(int(len(self.CX_PB_PFNc)/2))
        idx_output_half = np.remainder(idx_output_half, 8)
        idx_output = np.concatenate((idx_output_half, idx_output_half))
        span_out = 1
        for ineuron1 in range(self.CX_D72PFN.shape[0]):
            modulator1 = np.zeros(idx_output.shape)
            imax = int(span_in / 2)
            for ispan in range(imax + 1):
                idx1 = np.remainder(idx_input[ineuron1] + ispan, 8)
                idx2 = np.remainder(idx_input[ineuron1] - ispan, 8)
                modulator1[idx_output == idx1] = 1.0 - ispan / int((span_in + 1) / 2)
                modulator1[idx_output == idx2] = 1.0 - ispan / int((span_in + 1) / 2)

            for ineuron2 in range(self.CX_D72PFN.shape[1]):
                modulator2 = np.zeros(idx_output.shape)
                imax = int(span_out / 2)
                for ispan in range(imax + 1):
                    idx1 = np.remainder(idx_output[ineuron2] + ispan, 8)
                    idx2 = np.remainder(idx_output[ineuron2] - ispan, 8)
                    modulator2[idx_output == idx1] = 1.0 - ispan / int((span_out + 1) / 2)
                    modulator2[idx_output == idx2] = 1.0 - ispan / int((span_out + 1) / 2)

                modulator1[modulator1 < 0] = 0
                modulator2[modulator2 < 0] = 0
                if (ineuron1 < 8) & (ineuron2 < 8):
                    self.CX_D72PFN[ineuron1, ineuron2] = -sum(modulator1 * modulator2)
                elif (ineuron1 >= 8) & (ineuron2 >= 8):
                    self.CX_D72PFN[ineuron1, ineuron2] = -sum(modulator1 * modulator2)

        self.CX_NOt2PFN = np.zeros((len(self.CX_NOt), len(self.CX_PB_PFNc)))
        for ineuron1 in range(self.CX_NOt2PFN.shape[0]):
            for ineuron2 in range(self.CX_NOt2PFN.shape[1]):
                    self.CX_NOt2PFN[ineuron1, ineuron2] = 1.0

        self.CX_D72PFL = np.zeros((len(self.CX_PB_Delta7), len(self.CX_PB_PFL3)))
        idx_input_half = np.arange(int(len(self.CX_PB_Delta7)/2))
        shift = 4
        idx_input_half1 = np.remainder(idx_input_half+shift, 8)
        idx_input_half2 = np.remainder(idx_input_half-shift, 8)
        idx_input = np.concatenate((idx_input_half1, idx_input_half2))
        span_in = 3
        idx_output_half = np.arange(int(len(self.CX_PB_PFL3)/2))
        idx_output_half = np.remainder(idx_output_half, 8)
        idx_output = np.concatenate((idx_output_half, idx_output_half))
        span_out = 1
        for ineuron1 in range(self.CX_D72PFL.shape[0]):
            modulator1 = np.zeros(idx_output.shape)
            imax = int(span_in / 2)
            for ispan in range(imax + 1):
                idx1 = np.remainder(idx_input[ineuron1] + ispan, 8)
                idx2 = np.remainder(idx_input[ineuron1] - ispan, 8)
                modulator1[idx_output == idx1] = 1.0 - ispan / int((span_in + 1) / 2)
                modulator1[idx_output == idx2] = 1.0 - ispan / int((span_in + 1) / 2)

            for ineuron2 in range(self.CX_D72PFL.shape[1]):
                modulator2 = np.zeros(idx_output.shape)
                imax = int(span_out / 2)
                for ispan in range(imax + 1):
                    idx1 = np.remainder(idx_output[ineuron2] + ispan, 8)
                    idx2 = np.remainder(idx_output[ineuron2] - ispan, 8)
                    modulator2[idx_output == idx1] = 1.0 - ispan / int((span_out + 1) / 2)
                    modulator2[idx_output == idx2] = 1.0 - ispan / int((span_out + 1) / 2)

                modulator1[modulator1 < 0] = 0
                modulator2[modulator2 < 0] = 0
                if (ineuron1 < 8) & (ineuron2 < 8):
                    self.CX_D72PFL[ineuron1, ineuron2] = -sum(modulator1 * modulator2)
                elif (ineuron1 >= 8) & (ineuron2 >= 8):
                    self.CX_D72PFL[ineuron1, ineuron2] = -sum(modulator1 * modulator2)

        self.CX_PFN2PFL = np.zeros((len(self.CX_PB_PFNc), len(self.CX_PB_PFL3)))
        idx_input_half = np.arange(int(len(self.CX_PB_PFNc) / 2))
        shift = -1
        idx_input_half1 = np.remainder(idx_input_half - shift, 8)
        idx_input_half2 = np.remainder(idx_input_half + shift, 8)
        idx_input = np.concatenate((idx_input_half1, idx_input_half2))
        span_in = 1
        idx_output_half = np.arange(int(len(self.CX_PB_PFL3) / 2))
        idx_output_half = np.remainder(idx_output_half, 8)
        idx_output = np.concatenate((idx_output_half, idx_output_half))
        span_out = 1
        for ineuron1 in range(self.CX_PFN2PFL.shape[0]):
            modulator1 = np.zeros(idx_output.shape)
            imax = int(span_in / 2)
            for ispan in range(imax + 1):
                idx1 = np.remainder(idx_input[ineuron1] + ispan, 8)
                idx2 = np.remainder(idx_input[ineuron1] - ispan, 8)
                modulator1[idx_output == idx1] = 0.5 - ispan / int((span_in + 1) / 2) / 2
                modulator1[idx_output == idx2] = 0.5 - ispan / int((span_in + 1) / 2) / 2

            for ineuron2 in range(self.CX_PFN2PFL.shape[1]):
                modulator2 = np.zeros(idx_output.shape)
                imax = int(span_out / 2)
                for ispan in range(imax + 1):
                    idx1 = np.remainder(idx_output[ineuron2] + ispan, 8)
                    idx2 = np.remainder(idx_output[ineuron2] - ispan, 8)
                    modulator2[idx_output == idx1] = 0.5 - ispan / int((span_out + 1) / 2) / 2
                    modulator2[idx_output == idx2] = 0.5 - ispan / int((span_out + 1) / 2) / 2

                modulator1[modulator1 < 0] = 0
                modulator2[modulator2 < 0] = 0
                if (ineuron1 < 8) & (ineuron2 < 8):
                    self.CX_PFN2PFL[ineuron1, ineuron2] = sum(modulator1 * modulator2)
                elif (ineuron1 >= 8) & (ineuron2 >= 8):
                    self.CX_PFN2PFL[ineuron1, ineuron2] = sum(modulator1 * modulator2)

        self.CX_PFL2LAL = np.zeros((len(self.CX_PB_PFL3), len(self.CX_LAL)))
        for ineuron1 in range(self.CX_PFL2LAL.shape[0]):
            for ineuron2 in range(self.CX_PFL2LAL.shape[1]):
                if (ineuron2 == 0) & (ineuron1 < 8):
                    self.CX_PFL2LAL[ineuron1, ineuron2] = 1.0
                if (ineuron2 == 1) & (ineuron1 >= 8):
                    self.CX_PFL2LAL[ineuron1, ineuron2] = 1.0

        print('\t-> intra-FB connectivity')
        self.CX_FB_hDc = np.ones(16) * 0.5
        self.CX_PFN2hDc = np.zeros((len(self.CX_PB_PFNc), len(self.CX_FB_hDc)))
        idx_input_half = np.arange(int(len(self.CX_PB_PFNc) / 2))
        shift = -1
        idx_input_half1 = np.remainder(idx_input_half - shift, 8)
        idx_input_half2 = np.remainder(idx_input_half + shift, 8)
        idx_input = np.concatenate((idx_input_half1, idx_input_half2))
        span_in = 1
        idx_output_half = np.arange(int(len(self.CX_FB_hDc) / 2))
        idx_output_half = np.remainder(idx_output_half, 8)
        idx_output = np.concatenate((idx_output_half, idx_output_half))
        span_out = 1
        for ineuron1 in range(self.CX_PFN2hDc.shape[0]):
            modulator1 = np.zeros(idx_output.shape)
            imax = int(span_in / 2)
            for ispan in range(imax + 1):
                idx1 = np.remainder(idx_input[ineuron1] + ispan, 8)
                idx2 = np.remainder(idx_input[ineuron1] - ispan, 8)
                modulator1[idx_output == idx1] = 0.5 - ispan / int((span_in + 1) / 2) / 2
                modulator1[idx_output == idx2] = 0.5 - ispan / int((span_in + 1) / 2) / 2

            for ineuron2 in range(self.CX_PFN2hDc.shape[1]):
                modulator2 = np.zeros(idx_output.shape)
                imax = int(span_out / 2)
                for ispan in range(imax + 1):
                    idx1 = np.remainder(idx_output[ineuron2] + ispan, 8)
                    idx2 = np.remainder(idx_output[ineuron2] - ispan, 8)
                    modulator2[idx_output == idx1] = 0.5 - ispan / int((span_out + 1) / 2) / 2
                    modulator2[idx_output == idx2] = 0.5 - ispan / int((span_out + 1) / 2) / 2

                modulator1[modulator1 < 0] = 0
                modulator2[modulator2 < 0] = 0
                if (ineuron1 < 8) & (ineuron2 < 8):
                    self.CX_PFN2hDc[ineuron1, ineuron2] = sum(modulator1 * modulator2)
                elif (ineuron1 >= 8) & (ineuron2 >= 8):
                    self.CX_PFN2hDc[ineuron1, ineuron2] = sum(modulator1 * modulator2)

        self.CX_hDc2PFL = np.zeros((len(self.CX_FB_hDc), len(self.CX_PB_PFL3)))
        idx_input = np.arange(len(self.CX_FB_hDc))
        shift = 8
        idx_input = np.remainder(idx_input + shift, 16)
        span_in = 1
        idx_output = np.arange(int(len(self.CX_PB_PFL3)))
        span_out = 1
        for ineuron1 in range(self.CX_hDc2PFL.shape[0]):
            modulator1 = np.zeros(idx_output.shape)
            imax = int(span_in / 2)
            for ispan in range(imax + 1):
                idx1 = np.remainder(idx_input[ineuron1] + ispan, 16)
                idx2 = np.remainder(idx_input[ineuron1] - ispan, 16)
                modulator1[idx_output == idx1] = 1.0 - ispan / int((span_in + 1) / 2)
                modulator1[idx_output == idx2] = 1.0 - ispan / int((span_in + 1) / 2)

            for ineuron2 in range(self.CX_hDc2PFL.shape[1]):
                modulator2 = np.zeros(idx_output.shape)
                imax = int(span_out / 2)
                for ispan in range(imax + 1):
                    idx1 = np.remainder(idx_output[ineuron2] + ispan, 16)
                    idx2 = np.remainder(idx_output[ineuron2] - ispan, 16)
                    modulator2[idx_output == idx1] = 1.0 - ispan / int((span_out + 1) / 2)
                    modulator2[idx_output == idx2] = 1.0 - ispan / int((span_out + 1) / 2)

                modulator1[modulator1 < 0] = 0
                modulator2[modulator2 < 0] = 0
                self.CX_hDc2PFL[ineuron1, ineuron2] = sum(modulator1 * modulator2)

        self.Max_DeltaCX = 20

        print('\t-> FBt memory circuit')
        ## Memory circuit
        self.CX_FB_FBt = np.zeros(3)
        self.CX_FB_FBt_pfn = np.zeros(3)
        self.DAN_FBt_pfn = np.zeros(3)
        self.CX_FB_FBt_hdc = np.zeros(3)
        self.DAN_FBt_hdc = np.zeros(3)
        self.CX_FBt2PFN = -np.ones((len(self.CX_FB_FBt_pfn), len(self.CX_PB_PFNc))) * 0.5
        self.CX_FBt2hDc = -np.ones((len(self.CX_FB_FBt_hdc), len(self.CX_FB_hDc))) * 0.5

        ## Motivational circuit(s)
        self.Exploration = 0
        self.Explo2FBt = np.zeros((2, 3))
        self.Explo2FBt[0, 0] = 0.5
        self.Explo2FBt[1, 0] = 0.5
        if self.BrainType == 'Settled':
            self.Return = 0
            self.Ret2FBt = np.zeros((2, 3))
            self.Ret2FBt[0, 1] = 2.0
            self.Ret2FBt[1, 1] = 0.0
        elif self.BrainType == 'Nomadic':
            self.Return = 0
            self.Ret2FBt = np.zeros((2, 3))
            self.Ret2FBt[0, 1] = 2.0
            self.Ret2FBt[1, 1] = 2.0
        self.VecMemo = 0
        self.VM2FBt = np.zeros((2, 3))
        self.VM2FBt[0, 0] = 0.5
        self.VM2FBt[1, 0] = 0.5

        if self.Scenario == 'CT':
            self.Explo2FBt = np.zeros((2, 3))
            self.Ret2FBt = np.zeros((2, 3))
            self.VM2FBt = np.zeros((2, 3))

        self.Brain = np.zeros((0, 0))

        self.learning = False
        self.antilearning = False

        print('\t==> CX connectivity generated')
        ## MB nets
        # MB model is divided in two lateral MBs but used combined output
        # To-do: test with lateral inputs
        self.ref_thresh = MB_thresh

        if self.Scenario == 'MB':
            if len(Scenario) == 3:
                if Scenario[2] == 'Z':
                    self.MB_LearnParadigm = 'ZigzagRoute'
                else:
                    self.MB_LearnParadigm = 'StraightRoute'
            else:
                self.MB_LearnParadigm = 'StraightRoute'

        self.MB_pn = np.zeros((1, self.Monocular_eye.nb_ommat_tot))
        self.MBleft_kc = np.zeros((1, 10000))
        self.MBright_kc = np.zeros((1, 10000))
        self.MBleft_kc_memo = np.zeros((1, 10000))
        self.MBright_kc_memo = np.zeros((1, 10000))
        self.MB_pn_memo = np.zeros_like(self.MB_pn)

        self.MBleft_kc2en = np.ones((10000, 1))
        self.MBright_kc2en = np.ones((10000, 1))
        self.MBright_kc2en_fam = np.ones((10000, 1))
        self.MBleft_kc2en_fam = np.ones((10000, 1))
        self.MBleft_kc2en_neg = np.ones((10000, 1))
        self.MBright_kc2en_neg = np.ones((10000, 1))
        self.MBleft_en = np.zeros(self.MBleft_kc2en.shape[1])
        self.MBright_en = np.zeros(self.MBright_kc2en.shape[1])
        self.MBleft_en_neg = np.zeros(self.MBleft_kc2en_neg.shape[1])
        self.MBright_en_neg = np.zeros(self.MBright_kc2en_neg.shape[1])

        self.MB_habituation = False
        self.MB_sensitization = False
        self.KCright_shut = 0
        self.KCleft_shut = 0

        self.StartRoute = np.random.uniform(-100.0, 100.0, 2)*0

        ## Panoramic PN2KC
        self.MBright_net, self.MBright_PNcnt = generate_random_kc3(len(self.Monocular_eye.ommatidies_ID),
                                                                   self.MBright_kc.shape[1],
                                                                   limits='stricts',
                                                                   potential='homo',
                                                                   min_pn=2,
                                                                   max_pn=5,
                                                                   dtype=np.float32)
        self.MBleft_net, self.MBleft_PNcnt = generate_random_kc3(len(self.Monocular_eye.ommatidies_ID),
                                                                 self.MBleft_kc.shape[1],
                                                                 limits='stricts',
                                                                 potential='homo',
                                                                 min_pn=2,
                                                                 max_pn=5,
                                                                 dtype=np.float32)

        self.MBmemo = np.zeros(self.Monocular_eye.ommatidies_ID.shape[0])

        print('\t==> MB connectivity generated')

        self.familiarity = 0.0
        self.proprioception = 0.0
        self.init = True

        self.CX_updating = True

        if self.Scenario == 'VD':
            self.Exploration = 1
        elif self.Scenario == 'VM':
            self.VecMemo = 1
            self.FindFood = False
        elif self.Scenario == 'VM2':
            self.VecMemo = 1
        elif self.Scenario == 'MB':
            self.Exploration = 1

        self.PI_noise = 0.0

        # self.PI_rule = 'ActDependent'
        self.PI_rule = 'hDc'

        if self.Display_neurons:
            self.tableau2bord, self.ax_tableau2bord = mp.subplots(3, 3)
            self.tableau2bord.set_size_inches(16, 10)
            mp.ion()
            mp.show()

        self.Back = True

        self.VisRew = 0.0
        self.Visatt_rewardvalue = 0.0

        self.Food = 0

        self.LearningWalk = True
        self.HomingTest = False
        self.NewLoc_LW = True
        self.it_LW = 0
        self.LW_n = 0

        self.it_Screenshot = 0
        self.rotation_reminder = 0
        if self.Scenario == 'FS':
            self.FindFood = False
            self.LearningWalk = False
            self.it_return = 0
            self.Exploration = 1
            self.Food_sources_nb = 15
            self.Food_sources_locationxy = []
            for isource in range(self.Food_sources_nb):
                rho = 100 + np.random.uniform(0.0, 300.0, 1)[0]
                phi = np.radians(np.random.uniform(0, 90, 1))[0]
                Xsource = np.cos(phi) * rho
                Ysource = np.sin(phi) * rho
                ReplaceSource = False
                if isource > 0:
                    FoodsourcesXY_np = np.asarray(self.Food_sources_locationxy)
                    dist_intersource = np.zeros(isource)
                    for iso in range(len(dist_intersource)):
                        dist_intersource[iso] = np.sqrt((Xsource - FoodsourcesXY_np[iso, 0]) ** 2
                                                        + (Ysource - FoodsourcesXY_np[iso, 1]) ** 2)
                        if dist_intersource[iso] < 40:
                            ReplaceSource = True
                    while ReplaceSource:
                        rho = 100 + np.random.uniform(0.0, 300.0, 1)[0]
                        phi = np.radians(np.random.uniform(0, 90, 1))[0]
                        Xsource = np.cos(phi) * rho
                        Ysource = np.sin(phi) * rho
                        FoodsourcesXY_np = np.asarray(self.Food_sources_locationxy)
                        ReplaceSource = False
                        dist_intersource = np.zeros(isource)
                        for iso in range(len(dist_intersource)):
                            dist_intersource[iso] = np.sqrt((Xsource - FoodsourcesXY_np[iso, 0]) ** 2
                                                            + (Ysource - FoodsourcesXY_np[iso, 1]) ** 2)
                            # print(dist_intersource[iso], dist_intersource[iso] < 30)
                            if dist_intersource[iso] < 40:
                                ReplaceSource = True

                self.Food_sources_locationxy.append([Xsource, Ysource])
        else:
            self.Food_sources_locationxy = []
            self.Food_sources_locationxy.append('NaN')

        name_savedfile = self.name_saved + 'FoodSources.csv'
        a = np.asarray(self.Food_sources_locationxy, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")

        self.SourceExplored = 0

        self.Sight = False

        print('==> Model initiated')

    def on_draw(self):
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glLightfv(GL_LIGHT0, GL_POSITION, self.lightfv(0.0, 0.0, 0.0, 0.0))
        # glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        # glShadeModel(GL_SMOOTH)

        nb_segments = self.nb_segments

        angle_adjust = -180 + 360/nb_segments/2#-157.5

        NEAR_CLIPPING_PLANE = self.NEAR_CLIPPING_PLANE
        FAR_CLIPPING_PLANE = self.FAR_CLIPPING_PLANE

        for icam in range(nb_segments):

            glViewport(icam * round(self.Width / nb_segments), 0, round(self.Width / nb_segments), round(self.Height))
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()

            glFrustum(self.W_CLIPPING_PLANE[0], self.W_CLIPPING_PLANE[1],
                      self.H_CLIPPING_PLANE[0], self.H_CLIPPING_PLANE[1],
                      NEAR_CLIPPING_PLANE, FAR_CLIPPING_PLANE)

            glMatrixMode(GL_MODELVIEW)  # // Select The Modelview Matrix
            glLoadIdentity()  # // Reset The Modelview Matrix
            glClear(GL_DEPTH_BUFFER_BIT)

            glTranslated(0, 0, 0)
            glRotated(0, 0, 1, 0)
            glRotated(0, 1, 0, 0)
            glRotated(0, 0, 0, 1)
            glRotated(self.rotation_z + 180 + angle_adjust, 0, 1, 0)
            # glRotated(90, 1, 0, 0)
            glTranslated(self.translation_y, self.translation_z, -self.translation_x)

            angle_adjust += 360/nb_segments

            if not self.dark:

                ## Simple world
                Cube(sky_color)
                Surface(ground_color)
                if self.Obj_names[0] == 'Polygons':
                    self.main_batch.draw()
                else:
                    for iobj in range(len(self.Obj_names)):
                        if self.Obj_names[iobj] == 'Cylinder':
                            Cylinder(self.Obj_position[iobj], self.Obj_size[iobj][1], self.Obj_size[iobj][0], 50, Color=(1, 0, 0))
                        elif self.Obj_names[iobj] == 'Cone':
                            Cone(self.Obj_position[iobj], self.Obj_size[iobj][1], self.Obj_size[iobj][0], 50, Color=(1, 0, 0))
                        elif self.Obj_names[iobj] == 'Cube':
                            Cube2(self.Obj_position[iobj], self.Obj_size[iobj][1], self.Obj_size[iobj][0], Color=(1, 0, 0))

                if self.Target_on & (self.Scenario != 'FS'):
                    Cylinder(self.Target_location, self.Target_size[0], self.Target_size[1], 50, Color=(0, 1, 0))
                elif self.Scenario == 'FS':
                    for isource in range(self.Food_sources_nb):
                        Cylinder(self.Food_sources_locationxy[isource], 15, 160, 50, Color=(0, 1, 0))

            else:
                Cube(sky_color)
                Surface(ground_color)

                if self.Target_on & (self.Scenario != 'FS'):
                    Cylinder(self.Target_location, self.Target_size[0], self.Target_size[1], 50, Color=(0, 1, 0))
                elif self.Scenario == 'FS':
                    for isource in range(self.Food_sources_nb):
                        Cylinder(self.Food_sources_locationxy[isource], 15, 160, 50, Color=(0, 1, 0))

        return True

    def MB_update(self, learning=False, antilearning=False, rate=1.0):
        MB_learningrate = rate
        self.MB_pn = np.concatenate(self.vision)

        ##APL version
        self.MBright_kc = self.MB_pn @ self.MBright_net
        self.MBright_kc /= np.ones_like(self.MB_pn) @ self.MBright_net
        self.MBright_kc = self.MBright_kc# + np.random.uniform(-0.0005, 0.0005, self.MBright_kc.shape)
        self.MBright_kc[self.MBright_kc < 0.0] = 0.0
        MBright_kc_sorted = sorted(self.MBright_kc)
        self.threshright_R = MBright_kc_sorted[int(0.99 * len(MBright_kc_sorted))]
        self.MBright_kc[self.MBright_kc < self.threshright_R] = 0.0
        self.MBright_kc[self.MBright_kc >= self.threshright_R] = 1.0
        Nb_KCact_right = np.sum(self.MBright_kc)

        self.MBleft_kc = self.MB_pn @ self.MBleft_net
        self.MBleft_kc /= np.ones_like(self.MB_pn) @ self.MBleft_net
        self.MBleft_kc = self.MBleft_kc# + np.random.uniform(-0.0005, 0.0005, self.MBleft_kc.shape)
        self.MBleft_kc[self.MBleft_kc < 0.0] = 0.0
        MBleft_kc_sorted = sorted(self.MBleft_kc)
        self.threshleft_L = MBleft_kc_sorted[int(0.99*len(MBleft_kc_sorted))]
        self.MBleft_kc[self.MBleft_kc < self.threshleft_L] = 0.0
        self.MBleft_kc[self.MBleft_kc >= self.threshleft_L] = 1.0
        Nb_KCact_left = np.sum(self.MBleft_kc)

        if learning:
            self.MBright_kc2en[self.MBright_kc == 1, 0] -= MB_learningrate
            self.MBright_kc2en[self.MBright_kc2en <= 0] = 0
            self.MBleft_kc2en[self.MBleft_kc == 1, 0] -= MB_learningrate
            self.MBleft_kc2en[self.MBleft_kc2en <= 0] = 0

        elif antilearning:
            self.MBright_kc2en_neg[self.MBright_kc == 1, 0] -= MB_learningrate
            self.MBright_kc2en_neg[self.MBright_kc2en_neg <= 0] = 0
            self.MBleft_kc2en_neg[self.MBleft_kc == 1, 0] -= MB_learningrate
            self.MBleft_kc2en_neg[self.MBleft_kc2en_neg <= 0] = 0
        else:
            self.MBright_kc2en_fam[self.MBright_kc == 1, 0] -= MB_learningrate
            self.MBright_kc2en_fam[self.MBright_kc2en_fam <= 0] = 0
            self.MBleft_kc2en_fam[self.MBleft_kc == 1, 0] -= MB_learningrate
            self.MBleft_kc2en_fam[self.MBleft_kc2en_fam <= 0] = 0

        ## APL version
        if Nb_KCact_right > 20:
            self.MBright_en = np.sum(self.MBright_kc @ self.MBright_kc2en)/(len(self.MBright_kc)*0.005) #APL
            self.MBright_en_neg = np.sum(self.MBright_kc @ self.MBright_kc2en_neg)/(len(self.MBright_kc)*0.005) #APL
            self.MBright_en = np.min([self.MBright_en, 1])
            self.MBright_en_neg = np.min([self.MBright_en_neg, 1])
        else:
            self.MBright_en = 1.0
            self.MBright_en_neg = 1.0

        if Nb_KCact_left > 20:
            self.MBleft_en = np.sum(self.MBleft_kc @ self.MBleft_kc2en)/(len(self.MBleft_kc)*0.005) #APL
            self.MBleft_en_neg = np.sum(self.MBleft_kc @ self.MBleft_kc2en_neg)/(len(self.MBleft_kc)*0.005) #APL
            self.MBleft_en = np.min([self.MBleft_en, 1])
            self.MBleft_en_neg = np.min([self.MBleft_en_neg, 1])
        else:
            self.MBleft_en = 1.0
            self.MBleft_en_neg = 1.0

        ## Perfect image memory case
        self.MBin = np.concatenate(self.vision)
        if learning:
            self.MBmemo += np.sign(self.MBin)

        return self.MBright_en, self.MBleft_en

    def CX_update(self, Updating=True):
        if np.max(self.vision) == 0:
            CX_Visin = np.concatenate(self.vision)
        else:
            CX_Visin = np.concatenate(self.vision/np.max(self.vision))

        self.MB_output = (self.MBright_en + self.MBleft_en)/2

        self.CX_Vis_att = self.vision_att * self.VisRew_att_rf
        VisRew = np.sum(self.CX_Vis_att)

        Visin_8Cards = (CX_Visin @ self.CX_net)
        Visin_16Cards = np.concatenate((Visin_8Cards, Visin_8Cards))
        self.CX_Visin = Visin_16Cards
        Compass = wrapTo180(self.rotation_z)
        Compass_8Cards = np.zeros_like(self.CX_cards)
        CompPref_diff = abs(wrapTo180(Compass - self.CX_cards))
        Compass_8Cards[np.argmin(CompPref_diff)] = 1.0

        Compass_16Cards = np.concatenate((Compass_8Cards, Compass_8Cards))
        self.CX_Polin = Compass_16Cards

        proprioception = 1.0 * float(self.proprioception)
        rotRight = np.max((-proprioception, 0.0)) / 10
        if rotRight > 1.0:
            rotRight = 1.0
        rotLeft = np.max((proprioception, 0.0)) / 10
        if rotLeft > 1.0:
            rotLeft = 1.0
        self.CX_NO = np.asarray((rotLeft, rotRight))

        self.CX_NOt = np.asarray([np.sign(self.speed)])

        EB_EPG_temp = (0.8 * self.CX_Polin
                       + 0.5 * self.CX_PB_PEN @ self.CX_PEN2EPG
                       + 0.6 * self.CX_PB_PEG @ self.CX_PEG2EPG)
        EB_EPG_temp[EB_EPG_temp < 0.0] = 0.0
        EB_EPG_temp[EB_EPG_temp > 1.0] = 1.0

        PB_PEN_temp = (0.5 * (self.CX_EB_EPG @ self.CX_EPG2PEN)
                       + 0.8 * self.CX_NO @ self.CX_NO2PEN
                       + 0.3 * self.CX_PB_Delta7 @ self.CX_D72PEN)
        PB_PEN_temp[PB_PEN_temp < 0.0] = 0.0
        PB_PEN_temp[PB_PEN_temp > 1.0] = 1.0

        PB_PEG_temp = (1.0 * self.CX_EB_EPG @ self.CX_EPG2PEG
                       + 0.1 * self.CX_PB_Delta7 @ self.CX_D72PEG)
        PB_PEG_temp[PB_PEG_temp < 0.0] = 0.0
        PB_PEG_temp[PB_PEG_temp > 1.0] = 1.0

        PB_D7_temp = (1.2 * self.CX_EB_EPG @ self.CX_EPG2D7
                      + 0.1 * self.CX_PB_Delta7 @ self.CX_D72D7)
        PB_D7_temp[PB_D7_temp < 0.0] = 0.0
        PB_D7_temp[PB_D7_temp > 1.0] = 1.0

        PB_PFNc_temp = (0.25 * self.CX_PB_Delta7 @ self.CX_D72PFN
                        + 1.0 * self.CX_NOt @ self.CX_NOt2PFN)
        PB_PFNc_temp[PB_PFNc_temp < 0.0] = 0.0
        PB_PFNc_temp[PB_PFNc_temp > 1.0] = 1.0

        IndivPathway = True
        Use_local = 0.0
        Use_global = 0.0
        if IndivPathway:
            if (self.Gain_global > 0) & (self.Gain_local > 0):
                Use_local = 1.0
                Use_global = 1.0
            elif (self.Gain_global == 0) & (self.Gain_local == 0):
                Use_local = 0.0
                Use_global = 0.0
            elif self.Gain_global == 0:
                Use_local = 1.0
                Use_global = 0.0
            elif self.Gain_local == 0:
                Use_local = 0.0
                Use_global = 1.0
        else:
            Use_local = 1.0
            Use_global = 1.0

        FB_hDc_temp = (Use_global * self.CX_PB_PFNc @ self.CX_PFN2hDc)
        FB_hDc_temp[FB_hDc_temp < 0.0] = 0.0
        FB_hDc_temp[FB_hDc_temp > 1.0] = 1.0

        PB_PFNc_input = (Use_local * self.CX_PB_PFNc @ self.CX_PFN2PFL
                         + 1.0 * self.CX_FB_FBt_pfn @ self.CX_FBt2PFN)
        PB_PFNc_input[PB_PFNc_input < 0.0] = 0.0

        FB_hDc_input = (self.CX_FB_hDc
                        + 1.0 * self.CX_FB_FBt_hdc @ self.CX_FBt2hDc) @ self.CX_hDc2PFL
        FB_hDc_input[FB_hDc_input < 0.0] = 0.0

        PB_PFL3_temp = (0.6 * self.CX_PB_Delta7 @ self.CX_D72PFL
                        + 0.5 * PB_PFNc_input
                        + 0.5 * FB_hDc_input)
        PB_PFL3_temp[PB_PFL3_temp < 0.0] = 0.0
        PB_PFL3_temp[PB_PFL3_temp > 1.0] = 1.0

        FBt_pfn_temp = (np.array([float(self.Exploration)], ndmin=2) @ np.array(self.Explo2FBt[0, :], ndmin=2)
                        + np.array([float(self.Return)], ndmin=2) @ np.array(self.Ret2FBt[0, :], ndmin=2)
                        + np.array([float(self.VecMemo)], ndmin=2) @ np.array(self.VM2FBt[0, :], ndmin=2))
        FBt_pfn_temp = np.squeeze(FBt_pfn_temp)

        FBt_hdc_temp = (np.array([float(self.Exploration)], ndmin=2) @ np.array(self.Explo2FBt[1, :], ndmin=2)
                        + np.array([float(self.Return)], ndmin=2) @ np.array(self.Ret2FBt[1, :], ndmin=2)
                        + np.array([float(self.VecMemo)], ndmin=2) @ np.array(self.VM2FBt[1, :], ndmin=2))
        FBt_hdc_temp = np.squeeze(FBt_hdc_temp)

        ## Memory circuit
        ## Path Integration (PFN-to-PFL weights)
        if Updating:
            if self.PI_rule == 'hDc':
                RefAct = np.mean(PB_PFNc_temp)  # 0.5 #
                RefAct_hDc = np.mean(self.CX_FB_hDc)
                for iFBt in range(self.CX_FBt2PFN.shape[0]):
                    if (self.DAN_FBt_pfn[iFBt] == 1) & (np.sum(PB_PFNc_temp) != 0):
                        self.CX_FBt2PFN[iFBt, :] = (-np.ones_like(self.CX_FBt2PFN[iFBt, :]) * 0.5
                                                    - self.Gain_local * (PB_PFNc_temp - RefAct))

                        self.CX_FBt2PFN[iFBt, self.CX_FBt2PFN[iFBt, :] > 0.0] = 0.0
                        self.CX_FBt2PFN[iFBt, self.CX_FBt2PFN[iFBt, :] < -1.0] = -1.0

                for iFBt in range(self.CX_FBt2hDc.shape[0]):
                    if (self.DAN_FBt_hdc[iFBt] == 1) & (np.sum(self.CX_FB_hDc) != 0):
                        self.CX_FBt2hDc[iFBt, :] = (-np.ones_like(self.CX_FBt2PFN[iFBt, :]) * 0.5
                                                    - self.Gain_global * (self.CX_FB_hDc - RefAct_hDc))

                        self.CX_FBt2hDc[iFBt, self.CX_FBt2hDc[iFBt, :] > 0.0] = 0.0
                        self.CX_FBt2hDc[iFBt, self.CX_FBt2hDc[iFBt, :] < -1.0] = -1.0

                self.CX_FB_hDc += 0.001 * (FB_hDc_temp - np.mean(FB_hDc_temp))
                self.CX_FB_hDc[self.CX_FB_hDc < 0.0] = 0.0
                self.CX_FB_hDc[self.CX_FB_hDc > 1.0] = 1.0
                self.CX_FB_hDc_in = FB_hDc_input.copy()

        self.CX_LAL = (self.CX_PB_PFL3 @ self.CX_PFL2LAL)
        self.CX_LAL[self.CX_LAL < 0.0] = 0.0
        Delta_CX = 25 * np.diff(self.CX_LAL)# * (1 + np.random.normal(0.0, 0.05, 1))  # 5% noise
        if abs(Delta_CX) > self.Max_DeltaCX:
            Delta_CX = np.sign(Delta_CX) * self.Max_DeltaCX

        self.CX_PB_PEN = PB_PEN_temp.copy()
        self.CX_EB_EPG = EB_EPG_temp.copy()
        self.CX_PB_PEG = PB_PEG_temp.copy()
        self.CX_PB_Delta7 = PB_D7_temp.copy()
        self.CX_PB_PFNc = PB_PFNc_temp.copy()
        self.CX_PB_PFNc_in = PB_PFNc_input.copy()
        self.CX_PB_PFL3 = PB_PFL3_temp.copy()
        self.CX_FB_FBt_hdc = FBt_hdc_temp.copy()
        self.CX_FB_FBt_pfn = FBt_pfn_temp.copy()
        self.VisRew = VisRew.copy()

        if (self.Scenario == 'VD') & (self.VisRew >= 5.0) & (self.Exploration == 1):
            self.DAN_FBt_pfn[0] = 1.0
            self.DAN_FBt_hdc[0] = 1.0
            self.Visatt_rewardvalue = self.VisRew
            if not self.Sight:
                self.Sight = True

        elif (self.Scenario == 'MB') & (self.MB_output < 0.01):
            self.DAN_FBt_pfn[0] = 1.0
            self.DAN_FBt_hdc[0] = 1.0
            self.Visatt_rewardvalue = 1.0#self.MB_output

        elif self.Scenario == 'VM':
            if self.FindFood:
                self.DAN_FBt_hdc[2] = 1.0
                self.FindFood = False

        elif (self.Scenario == 'FS'):
            if self.BrainType == 'Settled':
                if (self.VisRew >= 5.0) & (self.Exploration == 1):
                    self.DAN_FBt_pfn[0] = 1.0
                    self.DAN_FBt_hdc[0] = 1.0
                    self.Visatt_rewardvalue = 1.0
                elif self.FindFood:
                    self.DAN_FBt_pfn[0] = 0.0
                    self.DAN_FBt_hdc[0] = 1.0
                    self.Visatt_rewardvalue = 0.0
                else:
                    self.DAN_FBt_pfn[0] = 0.0
                    self.DAN_FBt_hdc[0] = 0.0
                    self.Visatt_rewardvalue = 0.0
            elif self.BrainType == 'Nomadic':
                if self.FindFood & (self.Exploration == 1):
                    self.DAN_FBt_pfn[0] = 0.0
                    self.DAN_FBt_hdc[0] = 1.0
                    self.Visatt_rewardvalue = -1.0
                    self.it_return = 0
                elif (self.VisRew >= 5.0) & (self.Exploration == 1):
                    self.DAN_FBt_pfn[0] = 1.0
                    self.DAN_FBt_hdc[0] = 1.0
                    self.Visatt_rewardvalue = 1.0
                else:
                    self.DAN_FBt_pfn[0] = 0.0
                    self.Visatt_rewardvalue = 0.0

        else:
            self.DAN_FBt_pfn[0] = 0.0
            self.DAN_FBt_hdc[0] = 0.0
            self.Visatt_rewardvalue = 0.0

        return Delta_CX

    def update(self, dt):
    #################
    #### Image buffer
        viewport = (GLint * 4)()
        glGetIntegerv(GL_VIEWPORT, viewport)
        buff = (GLubyte * (self.Width * self.Height * 3))(0)
        glReadPixels(0, 0, self.Width, self.Height, GL_RGB, GL_UNSIGNED_BYTE, buff)
        image_array = np.frombuffer(buff, np.uint8)
        image = image_array.reshape(self.Height, self.Width, 3)

    #############################
    #### Color channel separation
        image_red_array = image[:, :, 0]
        image_green_array = image[:, :, 1]
        image_blue_array = image[:, :, 2]

        image_green = np.flipud(image_green_array).reshape(self.Height*self.Width)

        self.vision_att_brut = self.Monocular_eye.process(image_green, self.Eye_rf_list)
        self.vision_att = self.vision_att_brut.copy()
        self.vision_att[self.vision_att >= 0.1] = 1.0
        self.vision_att[self.vision_att < 0.1] = 0.0

        image_array2 = np.flipud(image_blue_array).reshape(self.Height * self.Width)
        self.vision_brut = self.Monocular_eye.process(image_array2, self.Eye_rf_list)
        self.vision = self.Monocular_eye.postprocess_LI(self.vision_brut, method='binary', thresh=0.01)

        dist2nest = np.sqrt(self.translation_x ** 2 + self.translation_y ** 2)
        if self.it < 50:
            self.CX_update(Updating=False)
            self.it += 1
            self.it_Route = 0
            if self.it == 1:
                print('** Behaviour **')

        elif self.it == 50:
            print('\t--> Start walking')
            self.it += 1
            self.snapcount = 0
        else:
            t1 = time.time()

        ############################
        #### Vision pipeline display
            # if self.Display:
            #     imSimu = np.flipud(cv2.resize(cv2.cvtColor(image_bluechannel, cv2.COLOR_RGB2BGR),
            #                                   (int(Width/2), int(Height/2)),
            #                                   interpolation=cv2.INTER_AREA))
            #
            #     imGreenChannel = np.zeros((1800, 3600, 3), np.uint8)
            #     for iom in range(len(self.Monocular_eye.ommatidies_ID)):
            #         color_val = int(self.vision_att[iom] * 255)
            #
            #         if (abs(self.Monocular_eye.ommatidies_dir[iom, 1]) < 10) & (self.Monocular_eye.ommatidies_dir[iom, 0] > 0):
            #             imGreenChannel = cv2.circle(imGreenChannel,
            #                                         (int(10 * (self.Monocular_eye.ommatidies_dir[iom, 1]+180)),
            #                                          int(10 * (-self.Monocular_eye.ommatidies_dir[iom, 0]+90))),
            #                                         int(9 * self.Monocular_eye.ommatidies_acc[iom, 0]),
            #                                         color=(0, color_val, 0),
            #                                         thickness=-1)
            #             imGreenChannel = cv2.circle(imGreenChannel,
            #                                         (int(10 * (self.Monocular_eye.ommatidies_dir[iom, 1] + 180)),
            #                                          int(10 * (-self.Monocular_eye.ommatidies_dir[iom, 0] + 90))),
            #                                         int(10 * self.Monocular_eye.ommatidies_acc[iom, 0]),
            #                                         color=(255, 255, 255),
            #                                         thickness=2)
            #
            #     imGreenChannel = cv2.resize(imGreenChannel,
            #                                 (int(Width/2), int(Height/2)),
            #                                 interpolation=cv2.INTER_AREA)
            #
            #     imPN_visionbrut = np.zeros((1800, 3600, 3), np.uint8)
            #
            #     for iom in range(len(self.Monocular_eye.ommatidies_ID)):
            #         color_val = (self.vision_brut[iom] * 255)
            #         if color_val > 255:
            #             color_val = 255
            #
            #         imPN_visionbrut = cv2.circle(imPN_visionbrut,
            #                                      (int(10 * (self.Monocular_eye.ommatidies_dir[iom, 1]+180)),
            #                                       int(10 * (-self.Monocular_eye.ommatidies_dir[iom, 0]+90))),
            #                                      int(10 * self.Monocular_eye.ommatidies_acc[iom, 0]),
            #                                      color=(color_val, color_val, color_val),
            #                                      thickness=-1)
            #         imPN_visionbrut = cv2.circle(imPN_visionbrut,
            #                                      (int(10 * (self.Monocular_eye.ommatidies_dir[iom, 1]+180)),
            #                                       int(10 * (-self.Monocular_eye.ommatidies_dir[iom, 0]+90))),
            #                                      int(10 * self.Monocular_eye.ommatidies_acc[iom, 0]),
            #                                      color=(255, 255, 255),
            #                                      thickness=2)
            #
            #     imPN_visionbrut = cv2.resize(imPN_visionbrut,
            #                                  (int(Width/2), int(Height/2)),
            #                                  interpolation=cv2.INTER_AREA)
            #
            #     imPN_vision = np.zeros((1800, 3600, 3), np.uint8)
            #
            #     for iom in range(len(self.Monocular_eye.ommatidies_ID)):
            #         color_val = int(self.vision[0, iom] * 255)
            #         if color_val > 255:
            #             color_val = 255
            #
            #         imPN_vision = cv2.circle(imPN_vision,
            #                                  (int(10 * (self.Monocular_eye.ommatidies_dir[iom, 1]+180)),
            #                                   int(10 * (-self.Monocular_eye.ommatidies_dir[iom, 0]+90))),
            #                                  int(10 * self.Monocular_eye.ommatidies_acc[iom, 0]),
            #                                  color=(color_val, color_val, color_val),
            #                                  thickness=-1)
            #         imPN_vision = cv2.circle(imPN_vision,
            #                                  (int(10 * (self.Monocular_eye.ommatidies_dir[iom, 1]+180)),
            #                                   int(10 * (-self.Monocular_eye.ommatidies_dir[iom, 0]+90))),
            #                                  int(10 * self.Monocular_eye.ommatidies_acc[iom, 0]),
            #                                  color=(255, 255, 255),
            #                                  thickness=2)
            #
            #     imPN_vision = cv2.resize(imPN_vision,
            #                              (int(Width/2), int(Height/2)),
            #                              interpolation=cv2.INTER_AREA)
            #
            #     UpperLine = cv2.hconcat([imSimu, imPN_visionbrut])
            #     BottomLine = cv2.hconcat([imPN_vision, imGreenChannel])
            #     Quadrant4Img = cv2.vconcat([UpperLine, BottomLine])
            #
            #     cv2.imshow('Visual Process pipeline', Quadrant4Img)
            #     cv2.waitKey(1)

            ##############
            #### Behaviour

            dist2target = float('inf')
            if self.Scenario == 'CT':
                Delta_CX = self.CX_update()
                orientation_initial = self.rotation_z.copy()
                Str_cmd = Delta_CX + np.random.normal(0.0, self.MotorNoise, 1)
                self.rotation_z += Str_cmd
                self.translation_y += np.sin(np.radians(self.rotation_z)) * self.speed
                self.translation_x += np.cos(np.radians(self.rotation_z)) * self.speed
                orientation_final = self.rotation_z.copy()
                self.proprioception = orientation_final - orientation_initial

                self.it += 1

            elif self.Scenario == 'FS':

                if self.LearningWalk:

                    if (self.it_LW % 30 == 0) & (self.it_LW != 0) & (self.it_Screenshot != 5):
                        self.CX_update(Updating=False)
                        self.rotation_z = np.degrees(np.arctan2(-self.translation_y, -self.translation_x))
                        self.it_Screenshot += 1
                    else:
                        self.CX_update()
                        self.translation_x += np.cos(self.rotation_z) * self.speed
                        self.translation_y += np.sin(self.rotation_z) * self.speed
                        Ori2Nest = wrapTo180(np.degrees(np.arctan2(-self.translation_y, -self.translation_x)) - self.rotation_z)
                        Cmd_loop = np.min([abs(Ori2Nest), 10.0]) * np.sign(Ori2Nest)
                        self.rotation_z += 0.1 * Cmd_loop + np.random.normal(0.0, 2.5, 1)[0]
                        if self.it_Screenshot == 5:
                            self.rotation_z = self.rotation_reminder
                            self.it_Screenshot = 0
                        self.it_LW += 1
                        self.rotation_reminder = self.rotation_z

                    if self.it_LW >= 1000:
                        self.LW_n += 1
                        self.it_LW = 0

                        if self.LW_n > 1:
                            self.CX_FBt2PFN[0, :] = -np.ones_like(self.CX_FBt2PFN[0, :]) * 0.5
                            self.LearningWalk = False
                            self.Exploration = 1
                            self.Food = 0

                        self.translation_x = 0.0
                        self.translation_y = 0.0
                        self.rotation_z = np.random.uniform(0.0, 360.0, 1)[0]

                else:
                    self.it += 1
                    Delta_CX = self.CX_update()
                    if (self.Food == 1) & (self.Exploration == 1):
                        self.Exploration = 0
                        self.Return = 1

                    orientation_initial = self.rotation_z.copy()
                    self.rotation_z += Delta_CX + np.random.normal(0.0, self.MotorNoise, 1)
                    self.translation_y += np.sin(np.radians(self.rotation_z)) * self.speed
                    self.translation_x += np.cos(np.radians(self.rotation_z)) * self.speed
                    orientation_final = self.rotation_z.copy()
                    self.proprioception = orientation_final - orientation_initial

                    FindFood = np.zeros(self.Food_sources_nb)
                    for isource in range(self.Food_sources_nb):
                        dist2source = np.sqrt((self.Food_sources_locationxy[isource][0] - self.translation_x)**2
                                              + (self.Food_sources_locationxy[isource][1] - self.translation_y)**2)
                        if (self.Food == 0) & (dist2source < 15):
                            self.it = 51
                            self.Food = 1
                            # self.DAN_FBt_hdc[0] = 1
                            FindFood[isource] = 1
                            self.Food_sources_locationxy.pop(isource)
                            self.Food_sources_nb -= 1

                            break
                        else:
                            FindFood[isource] = 0
                    self.FindFood = FindFood.any() > 0

                    if (self.BrainType == 'Nomadic') & (self.Return == 1):
                        self.it_return += 1
                        if (self.Gain_global != 0.0) & (self.it_return > 1000):
                            self.Exploration = 1
                            self.Return = 0
                            self.Food = 0
                            self.it = 51
                        elif (self.Gain_global == 0.0) & (self.it_return > 200):
                            self.Exploration = 1
                            self.Return = 0
                            self.Food = 0
                            self.it = 51

                    elif (self.BrainType == 'Settled') & (dist2nest < 10) & (self.Return == 1):
                        # self.LearningWalk = False
                        self.Exploration = 1
                        self.Return = 0

                        self.Food = 0

                        self.translation_x = 0.0
                        self.translation_y = 0.0
                        self.rotation_z = np.random.uniform(0.0, 360.0, 1)[0]

                        self.it = 51

            elif self.Scenario == 'MB':
                if self.LearningWalk:
                    self.CX_update()
                    if self.MB_LearnParadigm == 'StraightRoute':
                        if self.NewLoc_LW:
                            self.CX_FB_hDc = np.ones(16) * 0.5
                            Dmax = np.sqrt(self.Target_location[1]**2 + self.Target_location[0]**2)
                            Distance = self.LW_n * (200.0/30.0)#np.random.uniform(10, Dmax-10, 1)[0]
                            print('\rLearning location...' + str(self.LW_n+1) + '/30 (' + str(Distance) + ')', end=' ')
                            Orientation = np.arctan2(self.Target_location[1], self.Target_location[0])
                            self.translation_x = np.cos(Orientation) * Distance + self.StartRoute[0]
                            self.translation_y = np.sin(Orientation) * Distance + self.StartRoute[1]
                            self.rot_mean = np.degrees(Orientation)
                            self.rotation_z = self.rot_mean + np.random.uniform(-5.0, 5.0, 1)[0]

                            self.NewLoc_LW = False
                        else:
                            self.rotation_z = self.rot_mean + np.random.uniform(-5.0, 5.0, 1)[0]
                            self.it_LW += 1
                            if (self.it_LW > 3) & (self.it_LW < 13):
                                self.MB_update(learning=True, rate=1.0)

                            if self.it_LW > 15:
                                self.NewLoc_LW = True
                                self.it_LW = 0
                                self.LW_n += 1

                        if self.LW_n > 30:
                            self.HomingTest = True
                            self.LearningWalk = False
                            self.Homing_n = 1
                            self.reset_Homing = True
                            self.it = 0

                    elif self.MB_LearnParadigm == 'ZigzagRoute':
                        if self.it_Route == 0:
                            self.turn_sign = np.sign(np.random.uniform(-1, 1, 1)[0])
                            self.translation_x = 0.0
                            self.translation_y = 0.0
                            self.direction_route = np.random.uniform(0.0, 360.0, 1)[0]
                            self.rotation_z = self.direction_route
                        elif self.it_Route == 400:
                            self.direction_route += np.random.uniform(80.0, 100.0, 1)[0] * self.turn_sign
                            self.rotation_z = self.direction_route
                        elif self.it_Route == 1200:
                            self.direction_route += np.random.uniform(80.0, 100.0, 1)[0] * -self.turn_sign
                            self.rotation_z = self.direction_route
                        elif self.it_Route >= 2000:
                            self.HomingTest = True
                            self.LearningWalk = False

                            self.Homing_n = 0
                            self.reset_Homing = True
                            self.it = 0
                        else:
                            self.translation_y += np.sin(np.radians(self.direction_route)) * self.speed
                            self.translation_x += np.cos(np.radians(self.direction_route)) * self.speed
                            self.rotation_z = self.direction_route + np.random.uniform(-5.0, 5.0, 1)[0]

                        self.MB_update(learning=True, rate=0.1)
                        self.it_Route += 1

                elif self.HomingTest:
                    if self.it > 8000:
                        if self.Homing_n >= 15:
                            print('\r ', end='')
                            self.stop()
                        self.reset_Homing = True

                    if self.reset_Homing:
                        self.CX_FB_hDc = np.ones(16) * 0.5
                        self.CX_FBt2PFN[0, :] = -np.ones_like(self.CX_FBt2PFN[0, :]) * 0.5
                        self.CX_FBt2hDc[0, :] = -np.ones_like(self.CX_FBt2hDc[0, :]) * 0.5
                        self.Homing_n += 1
                        self.reset_Homing = False
                        Distance = np.random.uniform(0.0, 20.0, 1)[0]
                        Orientation = np.random.uniform(0.0, 360.0, 1)[0]
                        self.translation_x = np.cos(np.radians(Orientation)) * Distance
                        self.translation_y = np.sin(np.radians(Orientation)) * Distance
                        self.rotation_z = np.random.uniform(0.0, 360.0, 1)[0]
                        self.it = 0
                        print('\rTest... ' + str(self.Homing_n) + '/15', end='')
                    else:

                        self.MB_update()
                        Delta_CX = self.CX_update()
                        orientation_initial = self.rotation_z.copy()
                        self.rotation_z += Delta_CX + np.random.normal(0.0, self.MotorNoise, 1)
                        self.translation_y += np.sin(np.radians(self.rotation_z)) * self.speed
                        self.translation_x += np.cos(np.radians(self.rotation_z)) * self.speed
                        orientation_final = self.rotation_z.copy()
                        self.proprioception = orientation_final - orientation_initial
                        self.it += 1

            elif self.Scenario == 'VD':
                if self.Blink & (dist2target < 50):
                    self.Target_on = False

                dist2target = np.sqrt((self.translation_x - self.Target_location[0]) ** 2 + (self.translation_y - self.Target_location[1]) ** 2)

                if (self.Exploration == 1) & (dist2target < self.Target_size[0]) & (not self.Blink) & self.Back:
                    self.Food = 1
                    self.DAN_FBt_hdc[2] = 1
                    print('\t==> Target reached after', str(self.it), 'steps')
                    self.it = 0
                    self.CX_FBt2PFN[0, :] = -np.ones_like(self.CX_FBt2PFN[0, :]) * 0.5 #reinitialize
                else:
                    self.DAN_FBt_hdc[2] = 0

                if ((self.Exploration == 0) & (self.VecMemo == 0)) & (dist2nest < 20):
                    print('\t==> Home reached after', str(self.it), 'steps')
                    self.Food = 0
                    # self.Exploration = 1 #2nd go using sensory-driven behaviour
                    self.VecMemo = 1 #2nd go using vector memory
                    self.it = 0
                    self.Return = 0
                    self.translation_x = 0.0
                    self.translation_y = 0.0
                    self.rotation_z = np.random.uniform(0.0, 360.0, 1)[0] #random orientation: coming out of the nest?
                    self.Back = False # No stop & homing at the food anymore
                    self.CX_FBt2PFN[0, :] = -np.ones_like(self.CX_FBt2PFN[0, :]) * 0.5 #reinitialize


                Delta_CX = self.CX_update()
                orientation_initial = self.rotation_z.copy()
                k_CX = abs(Delta_CX) / self.Max_DeltaCX
                inst_noise = (1 - k_CX) * self.MotorNoise
                self.rotation_z += Delta_CX + np.random.normal(0.0, self.MotorNoise, 1)
                self.translation_y += np.sin(np.radians(self.rotation_z)) * self.speed
                self.translation_x += np.cos(np.radians(self.rotation_z)) * self.speed
                orientation_final = self.rotation_z.copy()
                self.proprioception = orientation_final - orientation_initial

                if (self.Exploration == 1) & (self.Food == 1):
                    self.Exploration = 0
                    self.Return = 1
                    print('\t==> Homing engaged')

            elif self.Scenario == 'VM':
                if self.Target_on:
                    dist2target = np.sqrt((self.translation_x - self.Target_location[0]) ** 2 + (self.translation_y - self.Target_location[1]) ** 2)
                else:
                    dist2target = float('inf')

                if (self.VecMemo == 1) & (dist2nest > 200) & self.Back:
                    self.Food = 1
                    # self.FindFood = True
                    self.DAN_FBt_hdc[0] = 1
                    print('\t==> Food source reached at coordinates:', str(self.translation_x), '| ', str(self.translation_y))
                    self.Target_location = [self.translation_x , self.translation_y]
                    self.Target_on = True
                    self.it = 0
                else:
                    self.DAN_FBt_hdc[0] = 0

                if (self.VecMemo == 0) & (dist2nest < 20):
                    print('\t==> Home reached after', str(self.it), 'steps')
                    self.Food = 0
                    self.VecMemo = 1
                    self.Turn = 0
                    self.it = 0
                    self.Return = 0
                    self.translation_x = 0.0
                    self.translation_y = 0.0
                    self.rotation_z = np.random.uniform(0.0, 360.0, 1)[0]
                    self.Back = False
                    self.CX_FBt2PFN[0, :] = -np.ones_like(self.CX_FBt2PFN[0, :]) * 0.5


                if self.Back & (self.VecMemo == 1):
                    ## Zig-Zag pattern
                    Delta_CX = self.CX_update()
                    orientation_initial = self.rotation_z.copy()
                    if self.it == 51:
                        self.DirTravel = np.random.uniform(0.0, 360.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 0
                        self.TurnWay = np.sign(np.random.uniform(-1, 1, 1)[0])
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 80) & (self.Turn == 0):
                        self.DirTravel += self.TurnWay * np.random.uniform(100.0, 120.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 1
                        self.TurnWay *= -1
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 160) & (self.Turn == 1):
                        self.DirTravel += self.TurnWay * np.random.uniform(60.0, 90.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 2
                        self.TurnWay *= -1
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 240) & (self.Turn == 2):
                        self.DirTravel += self.TurnWay * np.random.uniform(20.0, 45.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 3
                        self.TurnWay *= -1
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 290) & (self.Turn == 3):
                        self.DirTravel += self.TurnWay * np.random.uniform(20.0, 40.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 4
                        self.TurnWay *= -1
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 350) & (self.Turn == 4):
                        self.DirTravel += self.TurnWay * np.random.uniform(10.0, 30.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 5
                        self.TurnWay *= -1

                    self.rotation_z += np.random.normal(0.0, 1.0, 1)
                    self.translation_y += np.sin(np.radians(self.rotation_z)) * self.speed
                    self.translation_x += np.cos(np.radians(self.rotation_z)) * self.speed
                    orientation_final = self.rotation_z.copy()
                    self.proprioception = orientation_final - orientation_initial

                else:
                    Delta_CX = self.CX_update()
                    orientation_initial = self.rotation_z.copy()
                    k_CX = abs(Delta_CX) / self.Max_DeltaCX
                    inst_noise = (1 - k_CX) * self.MotorNoise
                    self.rotation_z += Delta_CX + np.random.normal(0.0, self.MotorNoise, 1)
                    self.translation_y += np.sin(np.radians(self.rotation_z)) * self.speed
                    self.translation_x += np.cos(np.radians(self.rotation_z)) * self.speed
                    orientation_final = self.rotation_z.copy()
                    self.proprioception = orientation_final - orientation_initial

                if (self.VecMemo == 1) & (self.Food == 1):
                    self.VecMemo = 0
                    self.Return = 1
                    print('\t==> Homing engaged')

            elif self.Scenario == 'VM2':
                if self.Target_on:
                    dist2target = np.sqrt((self.translation_x - self.Target_location[0]) ** 2 + (self.translation_y - self.Target_location[1]) ** 2)
                else:
                    dist2target = float('inf')

                if (self.VecMemo == 1) & (dist2nest > 200) & self.Back:
                    self.Food = 1
                    self.DAN_FBt_hdc[self.SourceExplored] = 1
                    self.SourceExplored += 1
                    print('\t==> Food source #', self.SourceExplored, 'reach at coordinates:', str(self.translation_x),
                          '| ', str(self.translation_y))
                    self.Target_location = [self.translation_x, self.translation_y]
                    self.Target_on = True
                    self.it = 0
                else:
                    self.DAN_FBt_hdc[0] = 0
                    self.DAN_FBt_hdc[1] = 0

                if ((self.Exploration == 0) & (self.VecMemo == 0)) & (dist2nest < 20):
                    print('\t==> Home reached after', str(self.it), 'steps')
                    self.Food = 0
                    self.VecMemo = 1
                    self.Turn = 0
                    self.it = 0
                    self.Return = 0
                    self.translation_x = 0.0
                    self.translation_y = 0.0
                    self.rotation_z = np.random.uniform(0.0, 360.0, 1)[0]
                    self.CX_FBt2PFN[0, :] = -np.ones_like(self.CX_FBt2PFN[0, :]) * 0.5
                    if self.SourceExplored < 2:
                        self.Back = True
                    else:
                        self.Back = False

                if self.Back & (self.VecMemo == 1):
                    ## Zig-Zag pattern
                    Delta_CX = self.CX_update()
                    orientation_initial = self.rotation_z.copy()
                    if self.it == 51:
                        self.DirTravel = np.random.uniform(0.0, 360.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 0
                        self.TurnWay = np.sign(np.random.uniform(-1, 1, 1)[0])
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 80) & (self.Turn == 0):
                        self.DirTravel += self.TurnWay * np.random.uniform(100.0, 120.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 1
                        self.TurnWay *= -1
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 160) & (self.Turn == 1):
                        self.DirTravel += self.TurnWay * np.random.uniform(60.0, 90.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 2
                        self.TurnWay *= -1
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 240) & (self.Turn == 2):
                        self.DirTravel += self.TurnWay * np.random.uniform(20.0, 45.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 3
                        self.TurnWay *= -1
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 290) & (self.Turn == 3):
                        self.DirTravel += self.TurnWay * np.random.uniform(20.0, 40.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 4
                        self.TurnWay *= -1
                    elif (np.sqrt(self.translation_x ** 2 + self.translation_y ** 2) > 350) & (self.Turn == 4):
                        self.DirTravel += self.TurnWay * np.random.uniform(10.0, 30.0, 1)[0]
                        self.rotation_z = self.DirTravel
                        self.Turn = 5
                        self.TurnWay *= -1

                    self.rotation_z += np.random.normal(0.0, 1.0, 1)
                    self.translation_y += np.sin(np.radians(self.rotation_z)) * self.speed
                    self.translation_x += np.cos(np.radians(self.rotation_z)) * self.speed
                    orientation_final = self.rotation_z.copy()
                    self.proprioception = orientation_final - orientation_initial

                else:
                    Delta_CX = self.CX_update()
                    orientation_initial = self.rotation_z.copy()
                    k_CX = abs(Delta_CX) / self.Max_DeltaCX
                    inst_noise = (1 - k_CX) * self.MotorNoise
                    self.rotation_z += Delta_CX + np.random.normal(0.0, self.MotorNoise, 1)
                    self.translation_y += np.sin(np.radians(self.rotation_z)) * self.speed
                    self.translation_x += np.cos(np.radians(self.rotation_z)) * self.speed
                    orientation_final = self.rotation_z.copy()
                    self.proprioception = orientation_final - orientation_initial

                if (self.VecMemo == 1) & (self.Food == 1):
                    self.Exploration = 0
                    self.VecMemo = 0
                    self.Return = 1
                    print('\t==> Homing engaged')
        ###############################
        #### Neurons activity recording
            # if self.Display_neurons:
            #     thetas = self.CX_cards/180 * np.pi
            #     for irow in range(self.ax_tableau2bord.shape[0]):
            #         for icol in range(self.ax_tableau2bord.shape[1]):
            #             self.ax_tableau2bord[irow, icol].cla()
            #     self.ax_tableau2bord[0, 0].bar(np.arange(1, 9)+0.15, self.CX_EB_EPG[0:8], width=0.3, color='r')
            #     self.ax_tableau2bord[0, 0].bar(np.arange(1, 9)-0.15, self.CX_EB_EPG[8:], width=0.3, color='g')
            #     self.ax_tableau2bord[0, 0].set_ylim([0.0, 1.0])
            #     self.ax_tableau2bord[0, 0].title.set_text('EPG')
            #     self.ax_tableau2bord[0, 1].bar(np.arange(1, 9)+0.15, self.CX_PB_Delta7[0:8], width=0.3, color='r')
            #     self.ax_tableau2bord[0, 1].bar(np.arange(1, 9)-0.15, self.CX_PB_Delta7[8:], width=0.3, color='g')
            #     self.ax_tableau2bord[0, 1].set_ylim([0.0, 1.0])
            #     self.ax_tableau2bord[0, 1].title.set_text('\u03947')
            #     self.ax_tableau2bord[1, 0].bar(np.arange(1, 9)+0.15, self.CX_PB_PFN_glo[0:8], width=0.3, color='r')
            #     self.ax_tableau2bord[1, 0].bar(np.arange(1, 9)-0.15, self.CX_PB_PFN_glo[8:], width=0.3, color='g')
            #     self.ax_tableau2bord[1, 0].set_ylim([0.0, 1.0])
            #     self.ax_tableau2bord[1, 0].title.set_text('PFNc')
            #     self.ax_tableau2bord[1, 1].bar(np.arange(1, 9)+0.15, self.CX_FB_hDc[0:8], width=0.3, color='r')
            #     self.ax_tableau2bord[1, 1].bar(np.arange(1, 9)-0.15, self.CX_FB_hDc[8:], width=0.3, color='g')
            #     self.ax_tableau2bord[1, 1].set_ylim([0.0, 1.0])
            #     self.ax_tableau2bord[1, 1].title.set_text('h\u0394c')
            #     self.ax_tableau2bord[1, 2].bar(np.arange(1, 9)+0.15, self.CX_PB_PFL3[0:8], width=0.3, color='r')
            #     self.ax_tableau2bord[1, 2].bar(np.arange(1, 9)-0.15, self.CX_PB_PFL3[8:], width=0.3, color='g')
            #     self.ax_tableau2bord[1, 2].set_ylim([0.0, 1.0])
            #     self.ax_tableau2bord[1, 2].title.set_text('PFL3')
            #     self.ax_tableau2bord[2, 0].imshow(self.CX_FBt2PFN)
            #     self.ax_tableau2bord[2, 0].set_xlabel('FBt')
            #     self.ax_tableau2bord[2, 0].set_ylabel('PFN')
            #     self.ax_tableau2bord[2, 0].title.set_text('FBt-to-PFNc')
            #     self.ax_tableau2bord[2, 1].imshow(self.CX_FBt2hDc)
            #     self.ax_tableau2bord[2, 1].set_xlabel('FBt')
            #     self.ax_tableau2bord[2, 1].set_ylabel('h\u0394c')
            #     self.ax_tableau2bord[2, 1].title.set_text('FBt-to-h\u0394c')
            #
            #     self.ax_tableau2bord[2, 2].bar([2, 1], self.CX_LAL, tick_label=['Right', 'Left'])
            #     self.ax_tableau2bord[2, 2].set_ylim([0, 8])
            #     self.ax_tableau2bord[2, 2].title.set_text('LAL')
            #     if self.Exploration == 1:
            #         color_fleche = 'r'
            #     else:
            #         color_fleche = 'g'
            #     if self.it >= 2:
            #         self.ax_tableau2bord[0, 2].plot(np.asarray(self.Pose_Mat)[:, 0], np.asarray(self.Pose_Mat)[:, 1], 'b')
            #         self.ax_tableau2bord[0, 2].plot(np.asarray([self.translation_x, self.translation_x + np.cos(np.radians(self.rotation_z)) * 15]),
            #                                             np.asarray([self.translation_y, self.translation_y + np.sin(np.radians(self.rotation_z)) * 15]),
            #                                             color_fleche)
            #
            #         for isource in range(self.Food_sources_nb):
            #             self.ax_tableau2bord[0, 2].plot(self.Food_sources_locationxy[isource][0], self.Food_sources_locationxy[isource][1], 'xc')
            #             # self.ax_tableau2bord[0, 2].text(self.Food_sources_locationxy[isource][0] + 2, self.Food_sources_locationxy[isource][1] + 2,
            #             #                                 str(np.round(P_odor[isource], 3)), fontsize=4)
            #
            #         if bool(self.Exploration):
            #             self.ax_tableau2bord[0, 2].text(298, 298, 'Exploration', color='r', fontsize=6, ha='right', va='top')
            #         elif bool(self.Return):
            #             self.ax_tableau2bord[0, 2].text(298, 298, 'Return', color='g', fontsize=6, ha='right', va='top')
            #
            #         self.ax_tableau2bord[0, 2].title.set_text('XY')
            #         self.ax_tableau2bord[0, 2].set_xlim([-300, 300])
            #         self.ax_tableau2bord[0, 2].set_ylim([-300, 300])
            #         self.ax_tableau2bord[0, 2].set_aspect('equal', adjustable='box')
            #
            #     mp.draw()
            #     mp.pause(0.001)

            self.NO_activity.append(self.CX_NO.tolist())
            self.Polar_activity.append(self.CX_Polin.tolist())
            self.EPG_activity.append(self.CX_EB_EPG.tolist())
            self.PEG_activity.append(self.CX_PB_PEG.tolist())
            self.PEN_activity.append(self.CX_PB_PEN.tolist())
            self.D7_activity.append(self.CX_PB_Delta7.tolist())
            self.PFN_activity.append(self.CX_PB_PFNc.tolist())
            self.PFN_in_activity.append(self.CX_PB_PFNc_in.tolist())
            self.hDc_activity.append(self.CX_FB_hDc.tolist())
            self.hDc_in_activity.append(self.CX_FB_hDc_in.tolist())
            self.PFL_activity.append(self.CX_PB_PFL3.tolist())
            self.LAL_activity.append(self.CX_LAL.tolist())
            self.FBt_memory.append(self.CX_FBt2PFN.reshape(self.CX_FBt2PFN.shape[0]*self.CX_FBt2PFN.shape[1]).tolist())
            self.FBt_hDc_memory.append(self.CX_FBt2hDc.reshape(self.CX_FBt2hDc.shape[0]*self.CX_FBt2hDc.shape[1]).tolist())

        #############################################
        #### 6D pose (Position+Orientation) recording
            self.Pose_Mat.append([float(np.round(self.translation_x, 4)),
                                  float(np.round(self.translation_y, 4)),
                                  float(np.round(self.translation_z, 4)),
                                  float(np.round(self.rotation_x, 4)),
                                  float(np.round(self.rotation_y, 4)),
                                  float(np.round(self.rotation_z, 4)),
                                  float(self.Visatt_rewardvalue)])

        #################
        #### Time counter
            if self.Scenario == 'MB':
                if self.it > 100000:
                    self.stop()
            elif self.Scenario == 'FS':
                if self.it > 8000:
                    self.stop()
            elif self.Scenario == 'CT':
                if self.it > 4000:
                    self.stop()
            else:
                self.it += 1
                if self.Back & (self.it >= 8000):
                    self.stop()
                elif (not self.Back) & (self.it >= 8000):
                    self.stop()
                elif dist2nest > 1500:
                    self.stop()

    def stop(self):
        name_savedfile = self.name_saved + 'Results_XY.csv'
        a = np.asarray(self.Pose_Mat, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")

        ## Inputs activity
        name_savedfile = self.name_saved + 'NOD.csv'
        a = np.asarray(self.NO_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'CIN.csv'
        a = np.asarray(self.Polar_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")

        ## Compass activity
        name_savedfile = self.name_saved + 'EPG.csv'
        a = np.asarray(self.EPG_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'PEN.csv'
        a = np.asarray(self.PEN_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'PEG.csv'
        a = np.asarray(self.PEG_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'D7.csv'
        a = np.asarray(self.D7_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")

        ## Steering activity
        name_savedfile = self.name_saved + 'PFN.csv'
        a = np.asarray(self.PFN_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'hDc.csv'
        a = np.asarray(self.hDc_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'PinFN.csv'
        a = np.asarray(self.PFN_in_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'hinDc.csv'
        a = np.asarray(self.hDc_in_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'PFL.csv'
        a = np.asarray(self.PFL_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'LAL.csv'
        a = np.asarray(self.LAL_activity, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")

        ## FBt Memory
        name_savedfile = self.name_saved + 'FBt_HD_memo.csv'
        a = np.asarray(self.FBt_memory, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")
        name_savedfile = self.name_saved + 'FBt_PI_memo.csv'
        a = np.asarray(self.FBt_hDc_memory, dtype=np.float32)
        np.savetxt(name_savedfile, a, delimiter=",")

        # cv2.destroyAllWindows()
        self.close()


def eye_generate2(nb_pn_size, image_size):
    # print(image_size[0] / nb_pn_size[0])
    nb_pn_ver = int(image_size[0]/nb_pn_size[0])
    nb_pn_hor = int(image_size[1]/nb_pn_size[1])
    nb_pn = nb_pn_ver * nb_pn_hor
    receptive_fields = np.zeros([nb_pn, 4])
    id_ommat = np.zeros((nb_pn, 2))
    cpt = 0
    for i in range(nb_pn_ver):
        for j in range(nb_pn_hor):
            limit_bottom = i*image_size[0]/nb_pn_ver
            limit_top = (i+1)*image_size[0]/nb_pn_ver+1
            limit_left = j*image_size[1]/nb_pn_hor
            limit_right = (j+1)*image_size[1]/nb_pn_hor+1
            receptive_fields[cpt, :] = [limit_bottom, limit_top, limit_left, limit_right]
            id_ommat[cpt, :] = [i, j]

            cpt += 1

    return receptive_fields, id_ommat


def generate_random_kc3(nb_pn, nb_kc, min_pn=10, max_pn=21, limits='strict', potential='hetero', vicinity='off', pn_map=None,
                rnd=RNG, dtype=np.single):
    """
    Create the synaptic weights among the Projection Neurons (PNs) and the Kenyon Cells (KCs).
    Choose the first sample that has dispersion below the baseline (early stopping), or the
    one with the lower dispersion (in case non of the samples' dispersion is less than the
    baseline).

    :param nb_pn:       the number of the Projection Neurons (PNs)
    :param nb_kc:       the number of the Kenyon Cells (KCs)
    :param min_pn:
    :param max_pn:
    :param aff_pn2kc:   the number of the PNs connected to every KC (usually 28-34)
                        if the number is less than or equal to zero it creates random values
                        for each KC in range [28, 34]
    :param nb_trials:   the number of trials in order to find a acceptable sample
    :param baseline:    distance between max-min number of projections per PN
    :param rnd:
    :type rnd: np.random.RandomState
    :param dtype:
    """

    pn2kc = np.zeros((nb_pn, nb_kc), dtype=dtype)
    pn_cnt = np.zeros(nb_kc, dtype=np.uint)
    potential_max = max_pn

    if vicinity == 'off':
        for i in range(nb_kc):
            # nb_con = rnd.randint(min_pn, max_pn+1)
            center = min_pn + (max_pn-min_pn)/2
            std = (max_pn-min_pn)/4
            nb_con = int(np.random.normal(center, std))
            if limits == 'stricts':
                while (nb_con < min_pn) | (nb_con > max_pn):
                    nb_con = int(np.random.normal(center, std))

            vaff_pn2kc = rnd.permutation(nb_pn)
            pn_con = vaff_pn2kc[0:nb_con]

            if potential == 'hetero':
                pn2kc[pn_con, i] = 1
                pn_cnt[i] = nb_con
            elif potential == 'homo':
                con_value = float(nb_con)/float(potential_max)
                pn2kc[pn_con, i] = con_value
                pn_cnt[i] = potential_max

    elif vicinity == 'on':
        if pn_map is None:
            print('Error: No PN map specified')
            print('Cannot process vicinity model')
            quit()
        for i in range(nb_kc):
            nb_con = int(random.normalvariate(min_pn + (max_pn - min_pn) / 2, (max_pn - min_pn) / 10))
            central_con = rnd.permutation(nb_pn)[0]

            possib_con = central_con

    # if non of the samples have dispersion lower than the baseline,
    # return the less dispersed one
    return pn2kc, pn_cnt


if __name__ == "__main__":
    screensize = pyautogui.size()
    screensize = (round(screensize[0] / 2), round(screensize[1] / 2))

    file_root = str(__file__)
    search = True
    target = -1
    start = 0

    while search:
        look = file_root.find('\\', start)
        if look == -1:
            search = False
        else:
            target = look
            start = look + 1

    savepath_default = file_root[0:target + 1]

    Width = screensize[0]
    Height = screensize[1]

    display_vision = False
    display_neurons = False
    Gain_height = 3.0

    MotorNoise = 10.0

    Gain_global = 2.118#0.0#
    Gain_local = 1.5#'R'

    # Scenario = 'CT'
    # Scenario = 'VD'
    # Scenario = 'VDB'
    # Scenario = 'VM'
    # Scenario = 'VM2'
    # Scenario = 'MB'
    # Scenario = 'FS'

    # BrainType = 'Nomadic'
    BrainType = 'Settled'

    now = datetime.datetime.now()
    timetag = now.strftime('_%H%M%S_%Y%m%d')
    root_paramModel = tk.Tk()

    tk.Label(root_paramModel, text='CX model - Spatial representation of goal',
             font=font.Font(size=12, weight='bold')).pack(fill='x')

    save_frame = tk.Frame(root_paramModel, highlightbackground="black", highlightthickness=1)
    save_frame.pack(fill='x', padx=5, pady=10)

    tk.Label(save_frame, text='Save folder', font=font.Font(size=9), padx=10).pack(fill='x')

    path_frame = tk.Frame(save_frame)
    path_frame.pack(fill='x')
    path_init = tk.StringVar()
    path_init.set(savepath_default)
    tk.Label(path_frame, text='Main folder', padx=5).pack(side='left')
    tk.Entry(path_frame, textvariable=path_init, width=60).pack(side='left', fill='x', expand=1)

    def browse_rootpath():
        rootpath = filedialog.askdirectory(initialdir=path_init.get())
        if rootpath[-1] != '\\':
            rootpath += '\\'
        path_init.set(rootpath)

    tk.Button(path_frame, text='Browse', command=browse_rootpath).pack(side='left')

    folder_frame = tk.Frame(save_frame)
    folder_frame.pack(fill='x')
    folder_init = tk.StringVar(value='CXModel_SpaceRep' + timetag)
    tk.Label(folder_frame, text='Save folder', padx=5).pack(side='left')
    tk.Entry(folder_frame, textvariable=folder_init).pack(side='left', fill='x', expand=1)

    param_frame = tk.Frame(root_paramModel, highlightbackground="black", highlightthickness=1)
    param_frame.pack(fill='x', padx=5, pady=5)
    param_frame1 = tk.Frame(param_frame)
    param_frame1.pack()
    param_frame2 = tk.Frame(param_frame, height=10)
    param_frame2.pack()
    param_frame3 = tk.Frame(param_frame, padx=10, pady=10)
    param_frame3.pack()

    tk.Label(param_frame1, text='Scenario', padx=20).grid(row=0, column=0)
    scenario_select = tk.StringVar()
    scenario_list = ['Control', 'PI memory', 'Senso Nav', 'Senso Nav (Blink)', 'Route Nav (Straight)',
                     'Route Nav (Zigzag)', 'Multi feeder']
    scenario_select.set(scenario_list[0])
    scenario_combo = ttk.Combobox(param_frame1, textvariable=scenario_select, width=15)
    scenario_combo['values'] = scenario_list
    scenario_combo['state'] = 'readonly'
    scenario_combo.grid(row=1, column=0, padx=20)

    root_paramModel.worldpath = ''

    world_name = tk.StringVar(value='Empty')

    def worldchoice():
        file_path = filedialog.askopenfilename(initialdir=savepath_default + '/Worlds_pyOpenGL',
                                               filetypes=(('CSV files', '*.csv'),
                                                          ('MAT files', '*.mat'),
                                                          ('All files', '*.*')))

        if file_path != '':
            search = True
            target = -1
            start = 0

            while search:
                look = file_path.find('/', start)
                if look == -1:
                    search = False
                else:
                    target = look
                    start = look + 1

            filename = file_path[target + 1:-4]

            world_name.set(filename)
            root_paramModel.worldpath = file_path
        else:
            world_name.set('Empty')
            root_paramModel.worldpath = ''

    tk.Label(param_frame1, text='\u03B2PFN', padx=20).grid(row=0, column=2)
    betaPFN = tk.StringVar(value=str(1.0))
    tk.Entry(param_frame1, textvariable=betaPFN, width=5, justify='center').grid(row=1, column=2)

    tk.Label(param_frame1, text='\u03B2h\u0394', padx=20).grid(row=0, column=3)
    betahDc = tk.StringVar(value=str(2.1))
    tk.Entry(param_frame1, textvariable=betahDc, width=5, justify='center').grid(row=1, column=3)

    tk.Label(param_frame1, text='Nb Exp', padx=20).grid(row=0, column=4)
    NB_exp = tk.StringVar(value=str(10))
    tk.Entry(param_frame1, textvariable=NB_exp, width=5, justify='center').grid(row=1, column=4)

    dispvision = tk.BooleanVar(value=display_vision)
    dispbrain = tk.BooleanVar(value=display_neurons)
    # tk.Checkbutton(param_frame3, text='Display Vision pipeline', variable=dispvision).pack(side='left')
    # tk.Checkbutton(param_frame3, text='Display Brain activity', variable=dispbrain).pack(side='right')

    tk.Button(param_frame3, text='3-D World', command=worldchoice, padx=10).pack(side='left')
    tk.Label(param_frame3, textvariable=world_name, font=font.Font(size=8, slant='italic'),
             justify='left', anchor='w', width=40, padx=10).pack(side='left', fill='x', expand=1)

    scenar = tk.StringVar(value='CT')
    root_paramModel.launch = False

    def startsimu():
        if scenario_select.get() == scenario_list[0]:
            scenar.set('CT')  # Control simulation (innate motor control)
        elif scenario_select.get() == scenario_list[1]:
            scenar.set('VM')  # PI vector memory (1 location)
            # Scenario = 'VM2'  # PI vector memory (2 locations + fusion)
        elif scenario_select.get() == scenario_list[2]:
            scenar.set('VD')  # Sensory navigation
        elif scenario_select.get() == scenario_list[3]:
            scenar.set('VDB')  # Sensory navigation with Blink
        elif scenario_select.get() == scenario_list[4]:
            scenar.set('MB')  # MB route straight
        elif scenario_select.get() == scenario_list[5]:
            scenar.set('MBZ')  # MB route zigzag
        elif scenario_select.get() == scenario_list[6]:
            scenar.set('FS')  # Multiple food source

        root_paramModel.launch = True

        root_paramModel.quit()

    tk.Button(root_paramModel, text='Launch', command=startsimu, padx=5).pack(side='bottom')

    root_paramModel.mainloop()

    if not root_paramModel.launch:
        quit()

    Scenario = scenar.get()
    NB_exp = int(NB_exp.get())
    if NB_exp < 1:
        NB_exp = 1

    if betahDc.get() != 'R':
        Gain_global = float(betahDc.get())
    else:
        Gain_global = 'R'

    if betaPFN.get() != 'R':
        Gain_local = float(betaPFN.get())
    else:
        Gain_local = 'R'

    display_neurons = dispbrain.get()
    display_vision = dispvision.get()

    name_folder = path_init.get() + folder_init.get()
    if name_folder[-1] != '/':
        name_folder += '/'

    ######################################

    os.mkdir(name_folder)

    root_paramModel.destroy()

    filetxt_info = open(name_folder + "Simulation_info.txt", "a")
    filetxt_info.write('Scenario' + '\t' + Scenario + '\n'
                       + 'Brain type' + '\t' + BrainType + '\n'
                       + 'Beta_PFN' + '\t' + str(Gain_local) + '\n'
                       + 'Beta_hDelta' + '\t' + str(Gain_global) + '\n'
                       + 'Motor noise' + '\t' + str(MotorNoise) + '\n'
                       + '3D world' + '\t' + world_name.get() + '\n')
    filetxt_info.close()

    print('')
    print('Folder name: ' + name_folder)
    print('--------------')

    filetxt = open(name_folder + "steer_params.txt", "a")
    filetxt.write('Exp#' + '\t'
                  + 'Vin_Kp' + '\t'
                  + 'MB_Kp' + '\n')
    filetxt.close()

    ScriptRoot = str(__file__)
    search = 'ON'
    start = 0
    target = -1
    while search == 'ON':
        target_temp = ScriptRoot.find('/', start, len(ScriptRoot))
        if target_temp != -1:
            target = target_temp
            start = target + 1
        else:
            search = 'OFF'

    # ScriptName = __file__[target+1:]
    # src = ScriptName
    # dst = name_folder + 'SimuCode.py'
    # shutil.copyfile(src, dst)
    filetxt_params = open(name_folder + "Obj_parameters_exp.txt", "a")
    filetxt_params.write('Exp' + '\t'
                         + 'Object' + '\t'
                         + 'X' + '\t'
                         + 'Y' + '\n')
    filetxt_params.close()

    MB_thresh = 1.75

    print('')
    print('3D World simulation:', root_paramModel.worldpath)
    print('')

    if root_paramModel.worldpath[-3:] == 'csv':
        src = root_paramModel.worldpath
        dst = name_folder + 'World_items.csv'
        shutil.copyfile(src, dst)
    elif root_paramModel.worldpath[-3:] == 'mat':
        src = root_paramModel.worldpath
        dst = name_folder + 'World_polygon.mat'
        shutil.copyfile(src, dst)

    filetxt_gains = open(name_folder + "Gains_list.txt", "a")
    filetxt_gains.write('Glo' + '\t'
                        + 'Loc' + '\n')
    filetxt_gains.close()

    for iexp in range(NB_exp):

        if Gain_global == 'R':
            Gain_global_param = float(np.random.uniform(0.0, 2.0, 1)[0])
            if Gain_global_param < 0.1:
                Gain_global_param = 0.0
        else:
            Gain_global_param = Gain_global
        if Gain_local == 'R':
            Gain_local_param = float(np.random.uniform(1.5, 2.5, 1)[0])
        else:
            Gain_local_param = Gain_local

        filetxt_gains = open(name_folder + "Gains_list.txt", "a")
        filetxt_gains.write(str(Gain_global_param) + '\t'
                            + str(Gain_local_param) + '\n')
        filetxt_gains.close()

        Obj_names = []
        Obj_position = []
        Obj_size = []
        if root_paramModel.worldpath[-3:] == 'csv':
            with open(root_paramModel.worldpath, 'r') as read_obj:
                csv_reader = reader(read_obj)
                irow = 0
                for row in csv_reader:
                    if irow >= 1:
                        # print(row[0])
                        Obj_names.append(row[0])
                        position = [np.round(np.cos(np.radians(float(row[2]))) * float(row[1])),
                                    np.round(np.sin(np.radians(float(row[2]))) * float(row[1]))]
                        Obj_position.append(position)
                        size = [float(row[3]), float(row[4])]
                        Obj_size.append(size)
                    irow += 1

        elif root_paramModel.worldpath[-3:] == 'mat':
            matlab_struct = spio.loadmat(root_paramModel.worldpath)
            Poly_C = np.asarray(matlab_struct['C'])
            Poly_C[:, 1] = 0.0
            Poly_C[:, 2] = 0.0
            Poly_X = np.asarray(matlab_struct['X'])
            Poly_Y = np.asarray(matlab_struct['Y'])
            Poly_Z = np.asarray(matlab_struct['Z'])
            Poly_Z *= Gain_height
            Obj_names.append('Polygons')
            Obj_position = np.asarray([Poly_C, Poly_X, Poly_Y, Poly_Z])

        print('\n##########')
        print('Experiment #' + str(iexp+1).zfill(3) + '/' + str(NB_exp))
        Name_record = name_folder + 'Result_files/' + 'Exp' + str(iexp + 1).zfill(3)
        if not os.path.isdir(name_folder + 'Result_files/'):
            os.mkdir(name_folder + 'Result_files/')

        Agent = Agent_sim(Name_record,
                          Obj_names, Obj_position, Obj_size,
                          MB_thresh,
                          Width, Height,
                          display_vision, display_neurons,
                          Scenario, BrainType,
                          Gain_global_param, Gain_local_param,
                          MotorNoise,
                          fullscreen=False, resizable=True)

        pyglet.clock.schedule_interval(Agent.update, 1.0/1000.0)
        pyglet.app.run()
        pyglet.clock.unschedule(Agent.update)

        Agent = None

        # MB_thresh -= 0.25

    # input('Press any key to quit...')