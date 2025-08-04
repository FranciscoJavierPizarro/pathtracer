"""
gpu_renderer.py - Renderizado acelerado por GPU usando CUDA (Versión Mejorada)
"""

import numpy as np
import time
import os
from numba import cuda
import math

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
    def dot_gpu(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    @cuda.jit(device=True)
    def length_gpu(v):
        return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

    @cuda.jit(device=True)
    def normalize_gpu(v):
        norm = length_gpu(v)
        if norm < 1e-6:
            return v
        return (v[0]/norm, v[1]/norm, v[2]/norm)

    @cuda.jit(device=True)
    def subtract_gpu(a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

    @cuda.jit(device=True)
    def add_gpu(a, b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

    @cuda.jit(device=True)
    def scale_gpu(v, s):
        return (v[0]*s, v[1]*s, v[2]*s)

    @cuda.jit(device=True)
    def cross_gpu(a, b):
        return (
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        )

    @cuda.jit(device=True)
    def reflect_gpu(incident, normal):
        dot_in = dot_gpu(incident, normal)
        return subtract_gpu(incident, scale_gpu(normal, 2.0 * dot_in))

    @cuda.jit(device=True)
    def refract_gpu(incident, normal, ior):
        cos_i = -dot_gpu(incident, normal)
        sin_t2 = ior * ior * (1.0 - cos_i * cos_i)
        
        if sin_t2 > 1.0:  # Reflexión total interna
            return False, (0.0, 0.0, 0.0)
        
        cos_t = math.sqrt(1.0 - sin_t2)
        refracted = add_gpu(
            scale_gpu(incident, ior),
            scale_gpu(normal, ior * cos_i - cos_t)
        )
        return True, refracted

    @cuda.jit(device=True)
    def wang_hash(seed):
        """Generador de hash Wang mejorado para mejor distribución"""
        seed = (seed ^ 61) ^ (seed >> 16)
        seed *= 9
        seed = seed ^ (seed >> 4)
        seed *= 0x27d4eb2d
        seed = seed ^ (seed >> 15)
        return seed

    @cuda.jit(device=True)
    def random_uniform(state):
        """Generador de números aleatorios mejorado usando xorshift"""
        # Xorshift32
        x = state[0]
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        x = x & 0xffffffff  # Mantener 32 bits
        state[0] = x
        return x / 4294967296.0  # 2^32

    @cuda.jit(device=True)
    def multiply_vectors_gpu(v1, v2):
        """Multiplica dos vectores elemento por elemento"""
        return (v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2])

    @cuda.jit(device=True)
    def scale_vector_gpu(v, scale):
        """Escala un vector por un escalar"""
        return (v[0] * scale, v[1] * scale, v[2] * scale)

    @cuda.jit(device=True)
    def process_refraction_setup(direction, normal, ior):
        """Procesa la configuración inicial para refracción"""
        n = normal
        cosi = dot_gpu(direction, n)
        
        if cosi > 0:
            n = scale_gpu(normal, -1.0)
            ior = 1.0 / ior
            cosi = -cosi
        
        return n, ior, cosi

    @cuda.jit(device=True)
    def should_reflect_fresnel(ior, cosi, state):
        """Determina si debe ocurrir reflexión basado en Fresnel"""
        r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
        reflect_prob = r0 + (1.0 - r0) * (1.0 - abs(cosi)) ** 5
        return random_uniform(state) < reflect_prob
    

