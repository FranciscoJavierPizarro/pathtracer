"""
raytracer.py - Clase principal del Ray Tracer
"""

from geometry import Camera
from gpu_renderer import GPURenderer
from cpu_renderer import CPURenderer
import traceback

class RayTracer:
    """Ray tracer principal con soporte GPU/CPU."""
    
    def __init__(self, width, height, prefer_gpu=True, samples=1, bounces=2):
        self.width = width
        self.height = height
        self.samples = samples
        self.bounces = bounces

        self.gpu_renderer = GPURenderer(samples_per_pixel=samples, max_bounces=bounces)
        self.cpu_renderer = CPURenderer(samples_per_pixel=samples, max_bounces=bounces)
        
        self.use_gpu = prefer_gpu and self.gpu_renderer.is_available()
        
        if prefer_gpu and not self.use_gpu:
            print("GPU solicitada pero no disponible, usando CPU")
        
        print(f"Ray Tracer: {width}x{height}, usando {'GPU' if self.use_gpu else 'CPU'}")
        
        self.camera = Camera(
            fov_deg=90, 
            aspect_ratio=width/height,
            lookfrom=(0, 0, 0.5),
            lookat=(0, 0, -1)
        )
    
    def set_camera(self, camera):
        self.camera = camera
    
    def force_cpu(self):
        self.use_gpu = False
        print("Forzando CPU")
    
    def force_gpu(self):
        if self.gpu_renderer.is_available():
            self.use_gpu = True
            print("Forzando GPU")
        else:
            print("GPU no disponible")
    
    def render(self, scene):
        try:
            if self.use_gpu:
                return self.gpu_renderer.render(scene, self.camera, self.width, self.height)
            else:
                return self.cpu_renderer.render(scene, self.camera, self.width, self.height)
        except Exception as e:
            print(f"Error en renderizado: {e}")
            traceback.print_exc()
            if self.use_gpu:
                print("Intentando fallback a CPU...")
                self.use_gpu = False
                return self.cpu_renderer.render(scene, self.camera, self.width, self.height)
            else:
                raise