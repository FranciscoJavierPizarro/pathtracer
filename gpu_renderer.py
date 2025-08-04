"""
gpu_renderer.py - Renderizado acelerado por GPU usando CUDA con soporte de texturas
"""

import numpy as np
import time
import os
from numba import cuda
import math
from basic_ops import *
from collisions import *
try:
    from numba import cuda
except ImportError:
    CUDA_AVAILABLE = False
    print("Numba no está disponible. Solo se usará CPU.")

if CUDA_AVAILABLE:
    # Configuración de entorno CUDA
    os.environ['NUMBAPRO_NVVM'] = '/usr/lib/x86_64-linux-gnu/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/lib/nvidia-cuda-toolkit/libdevice'
    os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

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
    def get_sphere_uv_gpu(hit_point, center, radius):
        """Calcula coordenadas UV para una esfera usando mapeo esférico"""
        # Vector desde el centro hacia el punto de intersección
        p = subtract_gpu(hit_point, center)
        p = normalize_gpu(p)
        
        # Coordenadas esféricas
        # u = atan2(z, x) / (2*pi) + 0.5
        # v = asin(y) / pi + 0.5
        u = math.atan2(p[2], p[0]) / (2.0 * math.pi) + 0.5
        v = math.asin(p[1]) / math.pi + 0.5
        
        return (u, v)

    @cuda.jit(device=True)
    def get_plane_uv_gpu(hit_point, plane_point, plane_normal):
        """Calcula coordenadas UV para un plano"""
        # Vector desde el punto del plano al punto de intersección
        local_point = subtract_gpu(hit_point, plane_point)
        
        # Crear base ortonormal para el plano
        abs_x = abs(plane_normal[0])
        abs_y = abs(plane_normal[1])
        abs_z = abs(plane_normal[2])
        
        if abs_x <= abs_y and abs_x <= abs_z:
            temp = (1.0, 0.0, 0.0)
        elif abs_y <= abs_z:
            temp = (0.0, 1.0, 0.0)
        else:
            temp = (0.0, 0.0, 1.0)
        
        u_axis = normalize_gpu(cross_gpu(plane_normal, temp))
        v_axis = cross_gpu(plane_normal, u_axis)
        
        # Proyectar en los ejes UV
        u = dot_gpu(local_point, u_axis) * 0.1  # Factor de escala ajustable
        v = dot_gpu(local_point, v_axis) * 0.1
        
        # Convertir a rango [0,1] con repetición
        u = u - math.floor(u)
        v = v - math.floor(v)
        
        return (u, v)

    @cuda.jit(device=True)
    def get_triangle_uv_gpu(hit_point, pt1, pt2, pt3, uv1, uv2, uv3):
        """Calcula coordenadas UV para un triángulo usando coordenadas baricéntricas"""
        # Calcular coordenadas baricéntricas
        v0 = subtract_gpu(pt3, pt1)
        v1 = subtract_gpu(pt2, pt1)
        v2 = subtract_gpu(hit_point, pt1)
        
        dot00 = dot_gpu(v0, v0)
        dot01 = dot_gpu(v0, v1)
        dot02 = dot_gpu(v0, v2)
        dot11 = dot_gpu(v1, v1)
        dot12 = dot_gpu(v1, v2)
        
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u_bary = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v_bary = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w_bary = 1.0 - u_bary - v_bary
        
        # Interpolar coordenadas UV
        u = w_bary * uv1[0] + v_bary * uv2[0] + u_bary * uv3[0]
        v = w_bary * uv1[1] + v_bary * uv2[1] + u_bary * uv3[1]
        
        return (u, v)

    @cuda.jit(device=True)
    def sample_texture_gpu(texture_data, texture_width, texture_height, u, v):
        """Muestrea una textura en las coordenadas UV dadas"""
        # Asegurar que u y v estén en [0,1]
        u = u - math.floor(u)
        v = v - math.floor(v)
        
        # Convertir a coordenadas de píxel
        x = int(u * (texture_width - 1))
        y = int(v * (texture_height - 1))
        
        # Clamp para evitar desbordamiento
        x = max(0, min(x, texture_width - 1))
        y = max(0, min(y, texture_height - 1))
        
        # Obtener color del píxel
        idx = y * texture_width + x
        r = texture_data[idx * 3 + 0]
        g = texture_data[idx * 3 + 1]
        b = texture_data[idx * 3 + 2]
        
        return (r, g, b)

    @cuda.jit(device=True)
    def nearest_intersection_with_texture(origin, direction,
                      sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_texture_ids, sphere_count,
                      plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_texture_ids, plane_count,
                      triangle_pt1s, triangle_pt2s, triangle_pt3s, 
                      triangle_colors, triangle_materials, triangle_iors, triangle_texture_ids, triangle_count,
                      texture_data, texture_widths, texture_heights, texture_count):
        """
        Encuentra la intersección más cercana y devuelve el color apropiado (textura o color sólido)
        """
        
        closest_t = float('inf')
        hit_anything = False
        final_color = (0.8, 0.8, 1.0)  # Color de fondo por defecto
        hit_normal = (0.0, 0.0, 0.0)
        hit_material = (0.0, 0.0, 0.0)
        hit_ior = 1.0

        # Verificar intersecciones con esferas
        for i in range(sphere_count):
            center = (sphere_centers[i, 0], sphere_centers[i, 1], sphere_centers[i, 2])
            hit, t = hit_sphere_gpu(origin, direction, center, sphere_radii[i])
            
            if hit and t < closest_t and t > 0.001:
                closest_t = t
                hit_anything = True
                
                # Calcular punto de intersección y normal
                hit_point = add_gpu(origin, scale_gpu(direction, t))
                normal = normalize_gpu(subtract_gpu(hit_point, center))
                hit_normal = normal
                hit_material = (sphere_materials[i, 0], sphere_materials[i, 1], sphere_materials[i, 2])
                hit_ior = sphere_iors[i]
                
                # Verificar si tiene textura
                texture_id = sphere_texture_ids[i]
                if texture_id >= 0 and texture_id < texture_count:
                    # Calcular UV y muestrear textura
                    u, v = get_sphere_uv_gpu(hit_point, center, sphere_radii[i])
                    
                    # Calcular offset en el array de texturas
                    texture_offset = 0
                    for tex_idx in range(texture_id):
                        texture_offset += texture_widths[tex_idx] * texture_heights[tex_idx] * 3
                    
                    final_color = sample_texture_gpu(
                        texture_data[texture_offset:], 
                        texture_widths[texture_id], 
                        texture_heights[texture_id], 
                        u, v
                    )
                else:
                    # Usar color sólido
                    final_color = (sphere_colors[i, 0], sphere_colors[i, 1], sphere_colors[i, 2])

        # Verificar intersecciones con planos
        for i in range(plane_count):
            point = (plane_points[i, 0], plane_points[i, 1], plane_points[i, 2])
            normal = (plane_normals[i, 0], plane_normals[i, 1], plane_normals[i, 2])
            hit, t = hit_plane_gpu(origin, direction, point, normal)
            
            if hit and t < closest_t and t > 0.001:
                closest_t = t
                hit_anything = True
                
                # Calcular punto de intersección
                hit_point = add_gpu(origin, scale_gpu(direction, t))
                hit_normal = normal
                hit_material = (plane_materials[i, 0], plane_materials[i, 1], plane_materials[i, 2])
                hit_ior = plane_iors[i]
                
                # Verificar si tiene textura
                texture_id = plane_texture_ids[i]
                if texture_id >= 0 and texture_id < texture_count:
                    # Calcular UV y muestrear textura
                    u, v = get_plane_uv_gpu(hit_point, point, normal)
                    
                    # Calcular offset en el array de texturas
                    texture_offset = 0
                    for tex_idx in range(texture_id):
                        texture_offset += texture_widths[tex_idx] * texture_heights[tex_idx] * 3
                    
                    final_color = sample_texture_gpu(
                        texture_data[texture_offset:], 
                        texture_widths[texture_id], 
                        texture_heights[texture_id], 
                        u, v
                    )
                else:
                    # Usar color sólido
                    final_color = (plane_colors[i, 0], plane_colors[i, 1], plane_colors[i, 2])

        # Verificar intersecciones con triángulos
        for i in range(triangle_count):
            pt1 = (triangle_pt1s[i, 0], triangle_pt1s[i, 1], triangle_pt1s[i, 2])
            pt2 = (triangle_pt2s[i, 0], triangle_pt2s[i, 1], triangle_pt2s[i, 2])
            pt3 = (triangle_pt3s[i, 0], triangle_pt3s[i, 1], triangle_pt3s[i, 2])
            
            hit, t = hit_triangle_gpu(origin, direction, pt1, pt2, pt3)
            
            if hit and t < closest_t and t > 0.001:
                closest_t = t
                hit_anything = True
                
                # Calcular punto de intersección y normal
                hit_point = add_gpu(origin, scale_gpu(direction, t))
                edge1 = subtract_gpu(pt2, pt1)
                edge2 = subtract_gpu(pt3, pt1)
                normal = normalize_gpu(cross_gpu(edge1, edge2))
                hit_normal = normal
                hit_material = (triangle_materials[i, 0], triangle_materials[i, 1], triangle_materials[i, 2])
                hit_ior = triangle_iors[i]
                
                # Verificar si tiene textura
                texture_id = triangle_texture_ids[i]
                if texture_id >= 0 and texture_id < texture_count:
                    # Para triángulos, necesitarías coordenadas UV por vértice
                    # Por simplicidad, usamos coordenadas UV calculadas automáticamente
                    uv1 = (0.0, 0.0)
                    uv2 = (1.0, 0.0)
                    uv3 = (0.5, 1.0)
                    u, v = get_triangle_uv_gpu(hit_point, pt1, pt2, pt3, uv1, uv2, uv3)
                    
                    # Calcular offset en el array de texturas
                    texture_offset = 0
                    for tex_idx in range(texture_id):
                        texture_offset += texture_widths[tex_idx] * texture_heights[tex_idx] * 3
                    
                    final_color = sample_texture_gpu(
                        texture_data[texture_offset:], 
                        texture_widths[texture_id], 
                        texture_heights[texture_id], 
                        u, v
                    )
                else:
                    # Usar color sólido
                    final_color = (triangle_colors[i, 0], triangle_colors[i, 1], triangle_colors[i, 2])

        return hit_anything, closest_t, final_color, hit_normal, hit_material, hit_ior

    @cuda.jit(device=True)
    def is_in_shadow_gpu(hit_point, light_pos,
                         sphere_centers, sphere_radii, sphere_count,
                         plane_points, plane_normals, plane_count,
                         triangle_pt1s, triangle_pt2s, triangle_pt3s, triangle_count):
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
        
        for i in range(triangle_count):
            pt1 = (triangle_pt1s[i, 0], triangle_pt1s[i, 1], triangle_pt1s[i, 2])
            pt2 = (triangle_pt2s[i, 0], triangle_pt2s[i, 1], triangle_pt2s[i, 2])
            pt3 = (triangle_pt3s[i, 0], triangle_pt3s[i, 1], triangle_pt3s[i, 2])

            hit, t = hit_triangle_gpu(hit_point, light_dir, pt1, pt2, pt3)
            if hit and t < light_distance - 0.001:
                return True

        return False

    @cuda.jit(device=True)
    def compute_direct_light_gpu(hit_point, hit_normal, light_pos,
                                sphere_centers, sphere_radii, sphere_count,
                                plane_points, plane_normals, plane_count,
                                triangle_pt1s, triangle_pt2s, triangle_pt3s, triangle_count):
        """Calcula la intensidad de luz directa en el punto de intersección."""
        light_vec = subtract_gpu(light_pos, hit_point)
        light_dir = normalize_gpu(light_vec)
        bias_point = add_gpu(hit_point, scale_gpu(hit_normal, 0.001))
        
        if is_in_shadow_gpu(bias_point, light_pos,
                           sphere_centers, sphere_radii, sphere_count,
                           plane_points, plane_normals, plane_count,
                           triangle_pt1s, triangle_pt2s, triangle_pt3s, triangle_count):
            return 0.1
        else:
            return max(0.0, dot_gpu(hit_normal, light_dir))

    @cuda.jit(device=True)
    def trace_ray_gpu(origin, direction, light_pos,
                      sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_texture_ids, sphere_count,
                      plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_texture_ids, plane_count,
                      triangle_pt1s, triangle_pt2s, triangle_pt3s, 
                      triangle_colors, triangle_materials, triangle_iors, triangle_texture_ids, triangle_count,
                      texture_data, texture_widths, texture_heights, texture_count,
                      depth, max_depth, state):
        
        color = (0.0, 0.0, 0.0)
        throughput = (1.0, 1.0, 1.0)
        current_origin = (origin[0], origin[1], origin[2])
        current_direction = (direction[0], direction[1], direction[2])
        
        for bounce in range(max_depth + 1):
            
            # Encontrar intersección más cercana con soporte de texturas
            hit_anything, closest_t, hit_color, hit_normal, hit_material, hit_ior = nearest_intersection_with_texture(
                current_origin, current_direction,
                sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_texture_ids, sphere_count,
                plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_texture_ids, plane_count,
                triangle_pt1s, triangle_pt2s, triangle_pt3s, 
                triangle_colors, triangle_materials, triangle_iors, triangle_texture_ids, triangle_count,
                texture_data, texture_widths, texture_heights, texture_count
            )

            if not hit_anything:
                color = add_gpu(color, multiply_vectors_gpu(throughput, hit_color))
                break
            
            hit_point = add_gpu(current_origin, scale_gpu(current_direction, closest_t))
            
            # Luz directa
            direct_intensity = compute_direct_light_gpu(
                hit_point, hit_normal, light_pos,
                sphere_centers, sphere_radii, sphere_count,
                plane_points, plane_normals, plane_count,
                triangle_pt1s, triangle_pt2s, triangle_pt3s, triangle_count
            )
            
            # Contribución de luz directa
            kd = hit_material[0]
            direct_contrib = scale_vector_gpu(
                multiply_vectors_gpu(throughput, hit_color), 
                kd * direct_intensity * 0.5
            )
            color = add_gpu(color, direct_contrib)
            
            # Terminar si alcanzamos máxima profundidad
            if bounce >= max_depth:
                break
            
            # Selección de tipo de rebote usando material probabilities
            total_prob = hit_material[0] + hit_material[1] + hit_material[2]
            if total_prob <= 0.0:
                break
                
            rand_val = random_uniform(state)
            prob_diffuse = hit_material[0] / total_prob
            prob_specular = hit_material[1] / total_prob
            
            # Aplicar Russian Roulette progresiva
            russian_prob = 1.0 if bounce <= 2 else max(0.1, 1.0 - (bounce - 2) * 0.15)
            if bounce > 2 and random_uniform(state) > russian_prob:
                break
            
            bias_point = add_gpu(hit_point, scale_gpu(hit_normal, 0.001))
            
            if rand_val < prob_diffuse:
                # Rebote difuso
                bounce_dir = uniform_hemisphere_sample_gpu(hit_normal, state)
                cos_theta = dot_gpu(bounce_dir, hit_normal)
                if cos_theta <= 0.0:
                    break
                
                # BRDF factor con Russian Roulette compensation
                brdf_factor = (2.0 * cos_theta / 3.14159265359) / russian_prob
                throughput = scale_vector_gpu(
                    multiply_vectors_gpu(throughput, hit_color), 
                    brdf_factor
                )
                
                current_origin = bias_point
                current_direction = bounce_dir
                
            elif rand_val < prob_diffuse + prob_specular:
                # Rebote especular
                current_direction = reflect_gpu(current_direction, hit_normal)
                current_origin = bias_point
                
            else:
                # Rebote de refracción
                n, ior, cosi = process_refraction_setup(current_direction, hit_normal, hit_ior)
                can_refract, refr_dir = refract_gpu(current_direction, n, ior)
                
                if not can_refract:
                    # Reflexión total interna
                    current_direction = reflect_gpu(current_direction, hit_normal)
                    current_origin = bias_point
                else:
                    # Fresnel decision
                    if should_reflect_fresnel(ior, cosi, state):
                        current_direction = reflect_gpu(current_direction, hit_normal)
                        current_origin = bias_point
                    else:
                        current_direction = refr_dir
                        current_origin = add_gpu(hit_point, scale_gpu(refr_dir, 0.001))
        
        return color
    
    @cuda.jit
    def render_kernel(image, width, height, origin, lower_left_corner,
                      horizontal, vertical, light_pos,
                      sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_texture_ids, sphere_count,
                      plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_texture_ids, plane_count,
                      triangle_pt1s, triangle_pt2s, triangle_pt3s, 
                      triangle_colors, triangle_materials, triangle_iors, triangle_texture_ids, triangle_count,
                      texture_data, texture_widths, texture_heights, texture_count,
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
                sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_texture_ids, sphere_count,
                plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_texture_ids, plane_count,
                triangle_pt1s, triangle_pt2s, triangle_pt3s, 
                triangle_colors, triangle_materials, triangle_iors, triangle_texture_ids, triangle_count,
                texture_data, texture_widths, texture_heights, texture_count,
                0, max_depth, state
            )
            
            pixel_color = add_gpu(pixel_color, sample_color)
        
        # Promediar samples
        pixel_color = scale_gpu(pixel_color, 1.0 / samples_per_pixel)
        
        # Convertir a enteros
        image[y, x, 0] = pixel_color[0]
        image[y, x, 1] = pixel_color[1]
        image[y, x, 2] = pixel_color[2]

class GPURenderer:
    """Renderizador GPU avanzado con soporte completo de texturas."""
    
    def __init__(self, samples_per_pixel=4, max_bounces=5, textures=None):
        self.available = CUDA_AVAILABLE and cuda.is_available() if CUDA_AVAILABLE else False
        self.samples_per_pixel = samples_per_pixel
        self.max_bounces = max_bounces
        self.textures = textures  # Lista de texturas cargadas
    
    def is_available(self):
        return self.available
    
    def prepare_texture_arrays(self):
        """Prepara los arrays de texturas para CUDA"""
        if not self.textures:
            # Si no hay texturas, crear arrays vacíos
            return (np.array([], dtype=np.float32), 
                    np.array([], dtype=np.int32), 
                    np.array([], dtype=np.int32))
        
        # Concatenar todas las texturas en un solo array
        all_texture_data = []
        texture_widths = []
        texture_heights = []
        
        for texture in self.textures:
            all_texture_data.extend(texture['data'])
            texture_widths.append(texture['width'])
            texture_heights.append(texture['height'])
        
        return (np.array(all_texture_data, dtype=np.float32),
                np.array(texture_widths, dtype=np.int32),
                np.array(texture_heights, dtype=np.int32))
    
    def render(self, scene, camera, width, height):
        if not self.available:
            raise RuntimeError("CUDA no está disponible")
        
        # Preparar datos de esferas
        sphere_centers = np.array([s.center for s in scene.spheres], dtype=np.float32)
        sphere_radii = np.array([s.radius for s in scene.spheres], dtype=np.float32)
        sphere_colors = np.array([s.color for s in scene.spheres], dtype=np.float32)
        sphere_materials = np.array([s.material for s in scene.spheres], dtype=np.float32)
        sphere_iors = np.array([s.ior for s in scene.spheres], dtype=np.float32)
        tmp = [s.texture_id for s in scene.spheres]
        print(tmp)
        sphere_texture_ids = np.array(tmp, dtype=np.int32)
        
        # Preparar datos de planos
        plane_points = np.array([p.point for p in scene.planes], dtype=np.float32)
        plane_normals = np.array([p.normal for p in scene.planes], dtype=np.float32)
        plane_colors = np.array([p.color for p in scene.planes], dtype=np.float32)
        plane_materials = np.array([p.material for p in scene.planes], dtype=np.float32)
        plane_iors = np.array([p.ior for p in scene.planes], dtype=np.float32)
        # Nuevo: IDs de texturas para planos
        plane_texture_ids = np.array([getattr(p, 'texture_id', -1) for p in scene.planes], dtype=np.int32)
        
        # Preparar los datos de triángulos
        triangle_pt1s = np.array([t.pt1 for t in scene.triangles], dtype=np.float32)
        triangle_pt2s = np.array([t.pt2 for t in scene.triangles], dtype=np.float32)
        triangle_pt3s = np.array([t.pt3 for t in scene.triangles], dtype=np.float32)
        triangle_colors = np.array([t.color for t in scene.triangles], dtype=np.float32)
        triangle_materials = np.array([t.material for t in scene.triangles], dtype=np.float32)
        triangle_iors = np.array([t.ior for t in scene.triangles], dtype=np.float32)
        # Nuevo: IDs de texturas para triángulos
        triangle_texture_ids = np.array([getattr(t, 'texture_id', -1) for t in scene.triangles], dtype=np.int32)

        # Preparar datos de texturas
        texture_data, texture_widths, texture_heights = self.prepare_texture_arrays()
        texture_count = len(self.textures)

        # Posición de la luz
        light_position = np.array([-2.0, 2.0, 0.0], dtype=np.float32)
        if scene.lights:
            light_position = scene.lights[0].position

        image = np.zeros((height, width, 3), dtype=np.float32)

        # Configuración de bloques y hilos
        threads_per_block = (16, 16)
        blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        print(f"Renderizando en GPU: {width}x{height}, {self.samples_per_pixel} SPP, {self.max_bounces} rebotes")
        print(f"Texturas cargadas: {texture_count}")
        start_time = time.time()
        
        render_kernel[blocks_per_grid, threads_per_block](
            image, width, height,
            camera.origin,
            camera.lower_left_corner,
            camera.horizontal,
            camera.vertical,
            light_position,
            sphere_centers, sphere_radii, sphere_colors, sphere_materials, sphere_iors, sphere_texture_ids, len(scene.spheres),
            plane_points, plane_normals, plane_colors, plane_materials, plane_iors, plane_texture_ids, len(scene.planes),
            triangle_pt1s, triangle_pt2s, triangle_pt3s, 
            triangle_colors, triangle_materials, triangle_iors, triangle_texture_ids, len(scene.triangles),
            texture_data, texture_widths, texture_heights, texture_count,
            self.samples_per_pixel, self.max_bounces
        )
        
        cuda.synchronize()
        end_time = time.time()
        print(f"Renderizado GPU completado en {end_time - start_time:.2f} segundos")
        
        return image