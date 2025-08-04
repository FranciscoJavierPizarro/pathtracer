"""
main.py - Script principal para ejecutar el Ray Tracer
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from math import pi
from geometry import Camera, Scene, Sphere, Plane, Triangle, Light
from raytracer import RayTracer
from auxiliar import load_obj_file, convert_to_triangles, compose, scale, rotate_x, rotate_y, rotate_z, translate

def create_demo_scene():
    """Crear una escena de demostración."""
    scene = Scene()
    
    # Luz principal
    scene.add_light(Light([1.0, 1.0, 1.0], intensity=2.0))
    
    # Esferas
    scene.add_triangle(Triangle([-2,0,-2],[0,2,-2],[2,0,-2],[0,1.0,0]))

    # vertex, triangles_index = load_obj_file("./cañon.obj")
    # # scaled_vertex = scale_vertex(vertex, scale_factor=(1/(50 * 16)))
    # transform = compose(translate(dy=-0.2, dz=-0.2),rotate_y(3*pi/4),scale(1/7))
    # modified_vertex = list(map(transform, vertex))
    # triangles = convert_to_triangles(modified_vertex, triangles_index, [0,1.0,0])
    # for triangle in triangles:
    #     scene.add_triangle(triangle)
    # print(len(scene.triangles))
    scene.add_sphere(Sphere([0, 0, -30], 0.5, [1.0, 0.0, 0.0], texture_id=0))      # Roja
    # scene.add_sphere(Sphere([-1, 0, -1], 0.5, [0.0, 1.0, 0.0]))     # Verde
    # scene.add_sphere(Sphere([1, 0, -1], 0.5, [0.0, 0.0, 1.0]))      # Azul
    # scene.add_sphere(Sphere([0.3, 0.7, -0.8], 0.2, [1.0, 1.0, 0.0])) # Amarilla
    
    # Plano del suelo
    scene.add_plane(Plane([0, -0.5, -1], [0, 1, 0], [0.8, 0.8, 0.8]))
    
    return scene


def create_cornell_box():
    """Crear una escena estilo Cornell Box."""
    scene = Scene()
    
    # Luz en el techo
    # scene.add_light(Light([0.0, 1.8, -1.0], intensity=1.2))
    scene.add_light(Light([0.0, 2, -1.0], intensity=1.2))
    
    # Esferas centrales
    scene.add_sphere(Sphere([-0.3, -0.2, -0.8], 0.3, [0.9, 0.9, 0.9]))  # Blanca
    scene.add_sphere(Sphere([0, 0, -3], 2, [0.9, 0.9, 0.9], texture_id=0))  # Blanca
    scene.add_sphere(Sphere([0.4, -0.3, -1.2], 0.2, [0.8, 0.2, 0.2]))   # Roja
    scene.add_triangle(Triangle([-2,-1,-2],[0,1,-2],[2,-1,-2],[1.0,1.0,0]))

    # Paredes Cornell Box
    scene.add_plane(Plane([0, -0.5, 0], [0, 1, 0], [0.9, 0.9, 0.9]))    # Suelo
    scene.add_plane(Plane([0, 2.0, 0], [0, -1, 0], [0.9, 0.9, 0.9]))    # Techo
    scene.add_plane(Plane([-1.5, 0, 0], [1, 0, 0], [0.8, 0.2, 0.2]))    # Pared izq roja
    scene.add_plane(Plane([1.5, 0, 0], [-1, 0, 0], [0.2, 0.8, 0.2]))    # Pared der verde
    scene.add_plane(Plane([0, 0, -2.5], [0, 0, 1], [0.9, 0.9, 0.9]))    # Pared trasera
    
    return scene


def display_image(image, title="Ray Tracing", save_path=None):
    """Mostrar imagen renderizada y opcionalmente guardarla."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Imagen guardada en: {save_path}")
    
    plt.show()


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Ray Tracer con CUDA')
    parser.add_argument('--mode', choices=['demo', 'cornell'], 
                        default='demo', help='Escena a renderizar')
    parser.add_argument('--width', type=int, default=400, help='Ancho de la imagen')
    parser.add_argument('--height', type=int, default=225, help='Alto de la imagen')
    parser.add_argument('--cpu', action='store_true', help='Forzar uso de CPU')
    parser.add_argument('--save', type=str, help='Ruta para guardar la imagen (ej: imagen.png)')
    
    args = parser.parse_args()
    
    print("Ray Tracer con CUDA")
    print("="*40)
    
    # Crear ray tracer
    use_gpu = not args.cpu
    tracer = RayTracer(width=args.width, height=args.height, prefer_gpu=use_gpu)
    
    # Seleccionar escena
    if args.mode == 'demo':
        scene = create_demo_scene()
        title = f"Escena Demo - {args.width}x{args.height}"
    elif args.mode == 'cornell':
        scene = create_cornell_box()
        # Configurar cámara para Cornell Box
        camera = Camera(
            fov_deg=60,
            aspect_ratio=args.width/args.height,
            lookfrom=(0, 0.5, 1.5),
            lookat=(0, 0, -1),
            vup=(0, 1, 0)
        )
        tracer.set_camera(camera)
        title = f"Cornell Box - {args.width}x{args.height}"
    
    # Renderizar
    print(f"\nRenderizando {args.mode}...")
    start_time = time.time()
    image = tracer.render(scene)
    end_time = time.time()
    
    print(f"Completado en {end_time - start_time:.2f} segundos")
    
    # Mostrar resultado y guardar si se especifica
    title += f" - {'GPU' if tracer.use_gpu else 'CPU'}"
    display_image(image, title, save_path=args.save)


if __name__ == "__main__":
    main()