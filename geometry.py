"""
geometry.py - Definición de figuras geométricas y objetos 3D
"""

import numpy as np


class Light:
    """Fuente de luz puntual."""
    def __init__(self, position, intensity=1.0, color=(1.0, 1.0, 1.0)):
        self.position = np.array(position, dtype=np.float32)
        self.intensity = intensity
        self.color = np.array(color, dtype=np.float32)


def reflect(v, n):
    return v - 2 * np.dot(v, n) * n


def refract(v, n, eta):
    cosi = -np.dot(n, v)
    cost2 = 1 - eta*eta*(1 - cosi*cosi)
    if cost2 < 0:
        return None  # total internal reflection
    t = eta * v + (eta * cosi - np.sqrt(cost2)) * n
    return t


class Sphere:
    """Esfera en la escena."""
    def __init__(self, center, radius, color, material=[1,0,0], ior=1.5):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.color = np.array(color, dtype=np.float32)
        self.material = np.array(material, dtype=np.float32)
        self.ior = ior


class Plane:
    """Plano infinito en la escena."""
    def __init__(self, point, normal, color, material=[1,0,0], ior=1.5):
        self.point = np.array(point, dtype=np.float32)
        self.normal = np.array(normal, dtype=np.float32) / np.linalg.norm(normal)
        self.color = np.array(color, dtype=np.float32)
        self.material = np.array(material, dtype=np.float32)
        self.ior = ior

class Triangle:
    """Triángulo en la escena."""
    def __init__(self, pt1, pt2, pt3, 
                #  uv1, uv2, uv3,
                color, material=[1,0,0], ior=1.5):
        self.pt1 = np.array(pt1, dtype=np.float32)
        self.pt2 = np.array(pt2, dtype=np.float32)
        self.pt3 = np.array(pt3, dtype=np.float32)
        # self.uv1 = np.array(uv1, dtype=np.float32)
        # self.uv2 = np.array(uv2, dtype=np.float32)
        # self.uv3 = np.array(uv3, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.material = np.array(material, dtype=np.float32)
        self.ior = ior

class Camera:
    """Cámara con perspectiva configurable."""
    def __init__(self, fov_deg=90, aspect_ratio=16/9, lookfrom=(0, 0, 0), lookat=(0, 0, -1), vup=(0, 1, 0)):
        self.aspect_ratio = aspect_ratio
        self.origin = np.array(lookfrom, dtype=np.float32)
        
        # Configurar el sistema de coordenadas de la cámara
        theta = np.radians(fov_deg)
        h = np.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height
        
        w = self._normalize(np.array(lookfrom, dtype=np.float32) - np.array(lookat, dtype=np.float32))
        u = self._normalize(np.cross(vup, w))
        v = np.cross(w, u)
        
        self.horizontal = viewport_width * u
        self.vertical = viewport_height * v
        self.lower_left_corner = self.origin - self.horizontal / 2 - self.vertical / 2 - w
    
    @staticmethod
    def _normalize(v):
        """Normalizar vector."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else v


class Scene:
    """Contenedor para todos los objetos de la escena."""
    def __init__(self):
        self.spheres = []
        self.planes = []
        self.triangles = []
        self.lights = []
    
    def add_sphere(self, sphere):
        """Añadir esfera a la escena."""
        self.spheres.append(sphere)
    
    def add_plane(self, plane):
        """Añadir plano a la escena."""
        self.planes.append(plane)
    
    def add_triangle(self, triangle):
        """Añadir triángulo a la escena."""
        self.triangles.append(triangle)
        
    def add_light(self, light):
        """Añadir luz a la escena."""
        self.lights.append(light)
    
    def clear(self):
        """Limpiar todos los objetos de la escena."""
        self.spheres.clear()
        self.planes.clear()
        self.triangles.clear()
        self.lights.clear()