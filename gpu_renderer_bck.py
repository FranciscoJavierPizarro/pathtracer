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
    def uniform_hemisphere_sample_gpu(normal, state):
        """Muestreo uniforme del hemisferio - distribución más natural"""
        # Generar punto uniforme en hemisferio usando coordenadas esféricas
        u1 = random_uniform(state)
        u2 = random_uniform(state)
        
        # Coordenadas esféricas para hemisferio uniforme
        cos_theta = u1  # Distribución uniforme en [0,1]
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        phi = 2.0 * math.pi * u2
        
        # Coordenadas locales (z hacia arriba)
        local_x = sin_theta * math.cos(phi)
        local_y = sin_theta * math.sin(phi)
        local_z = cos_theta
        
        # Crear base ortonormal con la normal
        # Encontrar un vector que no sea paralelo a la normal
        abs_x = abs(normal[0])
        abs_y = abs(normal[1])
        abs_z = abs(normal[2])
        
        if abs_x <= abs_y and abs_x <= abs_z:
            temp = (1.0, 0.0, 0.0)
        elif abs_y <= abs_z:
            temp = (0.0, 1.0, 0.0)
        else:
            temp = (0.0, 0.0, 1.0)
        
        # Crear tangente y bitangente
        tangent = normalize_gpu(cross_gpu(normal, temp))
        bitangent = cross_gpu(normal, tangent)
        
        # Transformar de coordenadas locales a mundiales
        world_dir = add_gpu(
            add_gpu(
                scale_gpu(tangent, local_x),
                scale_gpu(bitangent, local_y)
            ),
            scale_gpu(normal, local_z)
        )
        
        return normalize_gpu(world_dir)

    @cuda.jit(device=True)
    def stratified_hemisphere_sample_gpu(normal, state, sample_idx, total_samples):
        """Muestreo estratificado para reducir patrones"""
        # Dividir el espacio en estratos
        strata_x = int(math.sqrt(total_samples))
        strata_y = strata_x
        
        stratum_x = sample_idx % strata_x
        stratum_y = sample_idx // strata_x
        
        # Jitter dentro del estrato
        jitter_x = random_uniform(state)
        jitter_y = random_uniform(state)
        
        u1 = (stratum_x + jitter_x) / strata_x
        u2 = (stratum_y + jitter_y) / strata_y
        
        # Coordenadas esféricas
        cos_theta = u1
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        phi = 2.0 * math.pi * u2
        
        # Resto igual que uniform_hemisphere_sample_gpu
        local_x = sin_theta * math.cos(phi)
        local_y = sin_theta * math.sin(phi)
        local_z = cos_theta
        
        abs_x = abs(normal[0])
        abs_y = abs(normal[1])
        abs_z = abs(normal[2])
        
        if abs_x <= abs_y and abs_x <= abs_z:
            temp = (1.0, 0.0, 0.0)
        elif abs_y <= abs_z:
            temp = (0.0, 1.0, 0.0)
        else:
            temp = (0.0, 0.0, 1.0)
        
        tangent = normalize_gpu(cross_gpu(normal, temp))
        bitangent = cross_gpu(normal, tangent)
        
        world_dir = add_gpu(
            add_gpu(
                scale_gpu(tangent, local_x),
                scale_gpu(bitangent, local_y)
            ),
            scale_gpu(normal, local_z)
        )
        
        return normalize_gpu(world_dir)

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
    def is_in_shadow_gpu(hit_point, light_pos,
                         sphere_centers, sphere_radii, sphere_count,
                         plane_points, plane_normals, plane_count):
        light_dir = subtract_gpu(light_pos, hit_point)
        light_distance = length_gpu(light_dir)
        light_dir = normalize_gpu(light_dir)
        
        for i in range(sphere_count):
            center = (sphere_centers[i, 0], sphere_centers[i, 1], sphere_centers[i, 2])
            hit, t = hit_sphere_gpu(hit_point, light_dir, center, sphere_radii[i])
            if hit and t < light_distance - 0.001:
                return True
        
        for i in range(plane_count):
            point = (plane_points[i, 0], plane_points[i, 1], plane_points[i, 2])
            normal = (plane_normals[i, 0], plane_normals[i, 1], plane_normals[i, 2])
            hit, t = hit_plane_gpu(hit_point, light_dir, point, normal)
            if hit and t < light_distance - 0.001:
                return True
            
        
        
        return False

    @cuda.jit(device=True)
    def compute_direct_light_gpu(hit_point, hit_normal, light_pos,
                                sphere_centers, sphere_radii, sphere_count,
                                plane_points, plane_normals, plane_count):
        """Calcula la intensidad de luz directa en el punto de intersección."""
        light_vec = subtract_gpu(light_pos, hit_point)
        light_dir = normalize_gpu(light_vec)
        bias_point = add_gpu(hit_point, scale_gpu(hit_normal, 0.001))
        
        if is_in_shadow_gpu(bias_point, light_pos,
                           sphere_centers, sphere_radii, sphere_count,
                           plane_points, plane_normals, plane_count):
            return 0.1
        else:
            return max(0.0, dot_gpu(hit_normal, light_dir))

    @cuda.jit(device=True)
    def trace_ray_gpu(origin, direction, light_pos,
                      sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_count,
                      plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_count,
                     
                      depth, max_depth, state, sample_idx, total_samples):
        
        color = (0.0, 0.0, 0.0)
        throughput = (1.0, 1.0, 1.0)
        current_origin = (origin[0], origin[1], origin[2])
        current_direction = (direction[0], direction[1], direction[2])
        
        for bounce in range(max_depth + 1):
            hit_anything = False
            closest_t = 1e30
            hit_color = (0.05, 0.05, 0.1)  # Color de fondo
            hit_normal = (0.0, 0.0, 0.0)
            hit_material = (1.0, 0.0, 0.0)  # Por defecto difuso
            hit_ior = 1.0
            hit_sphere_idx = -1
            hit_plane_idx = -1

            
            # Encontrar intersección más cercana
            for i in range(sphere_count):
                center = (sphere_centers[i, 0], sphere_centers[i, 1], sphere_centers[i, 2])
                hit, t = hit_sphere_gpu(current_origin, current_direction, center, sphere_radii[i])
                
                if hit and t < closest_t:
                    closest_t = t
                    hit_anything = True
                    hit_color = (sphere_colors[i, 0], sphere_colors[i, 1], sphere_colors[i, 2])
                    hit_material = (sphere_materials[i, 0], sphere_materials[i, 1], sphere_materials[i, 2])
                    hit_ior = sphere_iors[i]
                    hit_sphere_idx = i
                    hit_plane_idx = -1

            
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
                    hit_sphere_idx = -1
                    hit_plane_idx = i
 
                    hit_normal = normal
            
            

            if not hit_anything:
                # Agregar contribución del fondo
                color = add_gpu(color, (
                    throughput[0] * hit_color[0],
                    throughput[1] * hit_color[1],
                    throughput[2] * hit_color[2]
                ))
                break
            
            # Calcular punto de intersección y normal
            hit_point = add_gpu(current_origin, scale_gpu(current_direction, closest_t))
            
            if hit_sphere_idx >= 0:
                center = (sphere_centers[hit_sphere_idx, 0], 
                         sphere_centers[hit_sphere_idx, 1], 
                         sphere_centers[hit_sphere_idx, 2])
                normal_vec = subtract_gpu(hit_point, center)
                hit_normal = normalize_gpu(normal_vec)
            
            # Calcular luz directa
            direct_light_intensity = compute_direct_light_gpu(
                hit_point, hit_normal, light_pos,
                sphere_centers, sphere_radii, sphere_count,
                plane_points, plane_normals, plane_count,
                
            )
            
            # Agregar contribución de luz directa
            kd = hit_material[0]
            direct_contribution = (
                throughput[0] * hit_color[0] * kd * direct_light_intensity,
                throughput[1] * hit_color[1] * kd * direct_light_intensity,
                throughput[2] * hit_color[2] * kd * direct_light_intensity
            )
            color = add_gpu(color, direct_contribution)
            
            # Decidir el tipo de rebote usando ruleta rusa
            if bounce >= max_depth:
                break
            
            # Normalizar probabilidades de material
            total_prob = hit_material[0] + hit_material[1] + hit_material[2]
            if total_prob < 1e-6:
                break
            
            rand_val = random_uniform(state)
            prob_kd = hit_material[0] / total_prob
            prob_kr = hit_material[1] / total_prob
            
            if rand_val < prob_kd:
                # Rebote difuso - MUESTREO UNIFORME DEL HEMISFERIO
                bounce_dir = uniform_hemisphere_sample_gpu(hit_normal, state)
                bias_point = add_gpu(hit_point, scale_gpu(hit_normal, 0.001))
                
                # BRDF Lambertiana con muestreo uniforme
                # PDF = 1/(2*pi), BRDF = albedo/pi
                # Factor = BRDF * cos(theta) / PDF = (albedo/pi) * cos(theta) / (1/(2*pi)) = 2 * albedo * cos(theta)
                cos_theta = dot_gpu(bounce_dir, hit_normal)
                brdf_factor = 2.0 * cos_theta  # Factor correcto para muestreo uniforme
                
                throughput = (
                    throughput[0] * hit_color[0] * brdf_factor,
                    throughput[1] * hit_color[1] * brdf_factor,
                    throughput[2] * hit_color[2] * brdf_factor
                )
                
                current_origin = (bias_point[0], bias_point[1], bias_point[2])
                current_direction = (bounce_dir[0], bounce_dir[1], bounce_dir[2])
                
            elif rand_val < prob_kd + prob_kr:
                # Rebote especular (espejo)
                reflect_dir = reflect_gpu(current_direction, hit_normal)
                bias_point = add_gpu(hit_point, scale_gpu(hit_normal, 0.001))
                
                current_origin = (bias_point[0], bias_point[1], bias_point[2])
                current_direction = (reflect_dir[0], reflect_dir[1], reflect_dir[2])
                
            else:
                # Rebote de refracción (cristal)
                n = hit_normal
                ior = hit_ior
                cosi = dot_gpu(current_direction, n)
                
                if cosi > 0:
                    n = scale_gpu(hit_normal, -1.0)
                    ior = 1.0 / ior
                    cosi = -cosi
                
                can_refract, refr_dir = refract_gpu(current_direction, n, ior)
                
                if not can_refract:
                    # Reflexión total interna
                    reflect_dir = reflect_gpu(current_direction, hit_normal)
                    bias_point = add_gpu(hit_point, scale_gpu(hit_normal, 0.001))
                    current_origin = (bias_point[0], bias_point[1], bias_point[2])
                    current_direction = (reflect_dir[0], reflect_dir[1], reflect_dir[2])
                else:
                    # Calcular Fresnel (aproximación Schlick)
                    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
                    reflect_prob = r0 + (1.0 - r0) * (1.0 - abs(cosi)) ** 5
                    
                    if random_uniform(state) < reflect_prob:
                        # Reflexión
                        reflect_dir = reflect_gpu(current_direction, hit_normal)
                        bias_point = add_gpu(hit_point, scale_gpu(hit_normal, 0.001))
                        current_origin = (bias_point[0], bias_point[1], bias_point[2])
                        current_direction = (reflect_dir[0], reflect_dir[1], reflect_dir[2])
                    else:
                        # Refracción
                        bias_point = add_gpu(hit_point, scale_gpu(refr_dir, 0.001))
                        current_origin = (bias_point[0], bias_point[1], bias_point[2])
                        current_direction = (refr_dir[0], refr_dir[1], refr_dir[2])
        
        return color

    @cuda.jit
    def render_kernel(image, width, height, origin, lower_left_corner,
                      horizontal, vertical, light_pos,
                      sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_count,
                      plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_count,
                      
                      samples_per_pixel, max_depth):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return
        
        # Inicializar estado del generador de números aleatorios mejorado
        state = cuda.local.array(1, dtype=np.uint32)
        # Usar wang hash para mejor distribución inicial
        pixel_id = y * width + x
        state[0] = wang_hash(pixel_id * 123456789 + 987654321)
        
        pixel_color = (0.0, 0.0, 0.0)
        
        for sample in range(samples_per_pixel):
            # Antialiasing con jitter
            jitter_x = random_uniform(state) - 0.5
            jitter_y = random_uniform(state) - 0.5
            
            u = (x + 0.5 + jitter_x) / width
            v = (height - y - 0.5 + jitter_y) / height
            
            ray_dir = subtract_gpu(
                add_gpu(
                    add_gpu(lower_left_corner, scale_gpu(horizontal, u)),
                    scale_gpu(vertical, v)
                ),
                origin
            )
            ray_dir = normalize_gpu(ray_dir)
            
            # Convertir a tuplas para mantener consistencia de tipos
            origin_tuple = (origin[0], origin[1], origin[2])
            ray_dir_tuple = (ray_dir[0], ray_dir[1], ray_dir[2])
            light_pos_tuple = (light_pos[0], light_pos[1], light_pos[2])
            
            sample_color = trace_ray_gpu(
                origin_tuple, ray_dir_tuple, light_pos_tuple,
                sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_count,
                plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_count,
                
                0, max_depth, state, sample, samples_per_pixel
            )
            
            pixel_color = add_gpu(pixel_color, sample_color)
        
        # Promediar samples
        pixel_color = scale_gpu(pixel_color, 1.0 / samples_per_pixel)
        
        # Tone mapping simple (gamma correction)
        gamma = 1.0 / 2.2
        pixel_color = (
            math.pow(max(0.0, min(1.0, pixel_color[0])), gamma),
            math.pow(max(0.0, min(1.0, pixel_color[1])), gamma),
            math.pow(max(0.0, min(1.0, pixel_color[2])), gamma)
        )
        
        # Convertir a enteros
        image[y, x, 0] = int(255 * pixel_color[0])
        image[y, x, 1] = int(255 * pixel_color[1])
        image[y, x, 2] = int(255 * pixel_color[2])


class GPURenderer:
    """Renderizador GPU avanzado con muestreo uniforme mejorado."""
    
    def __init__(self, samples_per_pixel=1, max_bounces=3):
        self.available = CUDA_AVAILABLE and cuda.is_available() if CUDA_AVAILABLE else False
        self.samples_per_pixel = samples_per_pixel
        self.max_bounces = max_bounces
    
    def is_available(self):
        return self.available
    
    def render(self, scene, camera, width, height):
        if not self.available:
            raise RuntimeError("CUDA no está disponible")
        
        # Preparar datos de esferas
        sphere_centers = np.array([s.center for s in scene.spheres], dtype=np.float32)
        sphere_radii = np.array([s.radius for s in scene.spheres], dtype=np.float32)
        sphere_colors = np.array([s.color for s in scene.spheres], dtype=np.float32)
        sphere_materials = np.array([s.material for s in scene.spheres], dtype=np.float32)
        sphere_iors = np.array([s.ior for s in scene.spheres], dtype=np.float32)
        
        # Preparar datos de planos
        plane_points = np.array([p.point for p in scene.planes], dtype=np.float32)
        plane_normals = np.array([p.normal for p in scene.planes], dtype=np.float32)
        plane_colors = np.array([p.color for p in scene.planes], dtype=np.float32)
        plane_materials = np.array([p.material for p in scene.planes], dtype=np.float32)
        plane_iors = np.array([p.ior for p in scene.planes], dtype=np.float32)


        

        
        # Posición de la luz
        light_position = np.array([-2.0, 2.0, 0.0], dtype=np.float32)
        if scene.lights:
            light_position = scene.lights[0].position
        
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Configuración de bloques y hilos
        threads_per_block = (16, 16)
        blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        print(f"Renderizando en GPU: {width}x{height}, {self.samples_per_pixel} SPP, {self.max_bounces} rebotes")
        print("Usando muestreo hemisférico uniforme para rebotes difusos")
        start_time = time.time()
        render_kernel[blocks_per_grid, threads_per_block](
            image, width, height,
            camera.origin,
            camera.lower_left_corner,
            camera.horizontal,
            camera.vertical,
            light_position,
            sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, len(scene.spheres),
            plane_points, plane_normals, plane_colors, plane_materials, plane_iors, len(scene.planes),
            
            self.samples_per_pixel, self.max_bounces
        )
        
        cuda.synchronize()
        end_time = time.time()
        print(f"Renderizado GPU completado en {end_time - start_time:.2f} segundos")
        
        return image