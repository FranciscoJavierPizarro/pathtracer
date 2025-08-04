"""
gpu_renderer.py - Renderizado acelerado por GPU usando CUDA (Versión Mejorada)
"""

import numpy as np
import time
import os
from numba import cuda
import math
from basic_ops import *
try:
    from numba import cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Numba no está disponible. Solo se usará CPU.")

if CUDA_AVAILABLE:
    # Configuración de entorno CUDA
    os.environ['NUMBAPRO_NVVM'] = '/usr/lib/x86_64-linux-gnu/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/lib/nvidia-cuda-toolkit/libdevice'
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')


    @cuda.jit(device=True)
    def hit_sphere_gpu(origin, direction, center, radius):
        oc = subtract_gpu(origin, center)
        a = dot_gpu(direction, direction)
        half_b = dot_gpu(oc, direction)
        c = dot_gpu(oc, oc) - radius * radius
        
        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return False, 0.0
        
        sqrtd = math.sqrt(discriminant)
        
        # Probar ambas raíces
        root = (-half_b - sqrtd) / a
        if root < 0.001:
            root = (-half_b + sqrtd) / a
            if root < 0.001:
                return False, 0.0
        
        return True, root

    @cuda.jit(device=True)
    def hit_plane_gpu(origin, direction, point, normal):
        denom = dot_gpu(normal, direction)
        if abs(denom) < 1e-6:
            return False, 0.0
        
        diff = subtract_gpu(point, origin)
        t = dot_gpu(diff, normal) / denom
        
        if t < 0.001:
            return False, 0.0
        
        return True, t

    @cuda.jit(device=True)
    def hit_triangle_gpu(origin, direction, pt1, pt2, pt3):
        EPSILON = 1e-5

        # Edge vectors
        edge1 = subtract_gpu(pt2, pt1)
        edge2 = subtract_gpu(pt3, pt1)

        # Begin calculating determinant
        h = cross_gpu(direction, edge2)
        a = dot_gpu(edge1, h)

        if abs(a) < EPSILON:
            return False, 0.0  # Ray is parallel to triangle

        f = 1.0 / a
        s = subtract_gpu(origin, pt1)
        u = f * dot_gpu(s, h)

        if u < 0.0 or u > 1.0:
            return False, 0.0

        q = cross_gpu(s, edge1)
        v = f * dot_gpu(direction, q)

        if v < 0.0 or u + v > 1.0:
            return False, 0.0

        t = f * dot_gpu(edge2, q)

        if t > EPSILON:
            return True, t
        else:
            return False, 0.0
        


    @cuda.jit
    def nearest_intersection(current_origin,current_direction,
                      sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_count,
                      plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_count,
                      triangle_pt1s, triangle_pt2s, triangle_pt3s, 
                      triangle_colors, triangle_materials, triangle_iors, triangle_count, 
                      ):
        hit_anything = False
        closest_t = 1e30
        hit_color = (0.05, 0.05, 0.1)  # Color de fondo
        hit_normal = (0.0, 0.0, 0.0)
        hit_material = (1.0, 0.0, 0.0)  # Por defecto difuso
        hit_ior = 1.0

        for i in range(sphere_count):
            center = (sphere_centers[i, 0], sphere_centers[i, 1], sphere_centers[i, 2])
            hit, t = hit_sphere_gpu(current_origin, current_direction, center, sphere_radii[i])
            
            if hit and t < closest_t:
                closest_t = t
                hit_anything = True
                
                hit_point = add_gpu(current_origin, scale_gpu(current_direction, closest_t))

                center = (sphere_centers[i, 0], 
                        sphere_centers[i, 1], 
                        sphere_centers[i, 2])
                normal_vec = subtract_gpu(hit_point, center)
                hit_normal = normalize_gpu(normal_vec)

                hit_color = (sphere_colors[i, 0], sphere_colors[i, 1], sphere_colors[i, 2])
                hit_material = (sphere_materials[i, 0], sphere_materials[i, 1], sphere_materials[i, 2])
                hit_ior = sphere_iors[i]
                    
        for i in range(plane_count):
            point = (plane_points[i, 0], plane_points[i, 1], plane_points[i, 2])
            normal = (plane_normals[i, 0], plane_normals[i, 1], plane_normals[i, 2])
            hit, t = hit_plane_gpu(current_origin, current_direction, point, normal)
            
            if hit and t < closest_t:
                closest_t = t
                hit_anything = True
                hit_color = (plane_colors[i, 0], plane_colors[i, 1], plane_colors[i, 2])
                hit_material = (plane_materials[i, 0], plane_materials[i, 1], plane_materials[i, 2])
                hit_ior = plane_iors[i]
                hit_normal = normal


        for i in range(triangle_count):
            pt1 = (triangle_pt1s[i, 0], triangle_pt1s[i, 1], triangle_pt1s[i, 2])
            pt2 = (triangle_pt2s[i, 0], triangle_pt2s[i, 1], triangle_pt2s[i, 2])
            pt3 = (triangle_pt3s[i, 0], triangle_pt3s[i, 1], triangle_pt3s[i, 2])
            
            hit, t = hit_triangle_gpu(current_origin,current_direction,pt1,pt2,pt3)
            
            if hit and t < closest_t:
                closest_t = t
                hit_anything = True
                hit_color = (triangle_colors[i, 0], triangle_colors[i, 1], triangle_colors[i, 2])
                hit_material = (triangle_materials[i, 0], triangle_materials[i, 1], triangle_materials[i, 2])
                hit_ior = triangle_iors[i]
                hit_normal = normal

        return hit_anything, closest_t, hit_color, hit_normal, hit_material, hit_ior