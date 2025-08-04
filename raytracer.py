"""
raytracer.py - Clase principal del Ray Tracer
"""

from geometry import Camera
from gpu_renderer import GPURenderer
from cpu_renderer import CPURenderer
import numpy as np
import traceback

def load_texture(image_path):
    """Carga una textura desde un archivo PNG/JPG"""
    from PIL import Image
    
    try:
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        
        # Convertir a array numpy normalizado [0,1]
        texture_data = np.array(img, dtype=np.float32) / 255.0
        texture_data = texture_data.flatten()  # Aplanar a 1D
        
        texture_info = {
            'data': texture_data,
            'width': width,
            'height': height
        }
        
        return texture_info  # Retornar índice de la textura
    
    except Exception as e:
        print(f"Error cargando textura {image_path}: {e}")
        return -1

class RayTracer:
    """Ray tracer principal con soporte GPU/CPU."""
    
    def __init__(self, width, height, prefer_gpu=True, samples=1, bounces=2):
        self.width = width
        self.height = height
        self.samples = samples
        self.bounces = bounces

        self.gpu_renderer = GPURenderer(samples_per_pixel=samples, max_bounces=bounces, textures=[load_texture("./imgs/tiopaco.jpg")])
        self.cpu_renderer = CPURenderer(samples_per_pixel=samples, max_bounces=bounces)
        # self.gpu_renderer.textures = load_texture("./imgs/tiopaco.jpg")  # Cargar textura de ejemplo
        
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