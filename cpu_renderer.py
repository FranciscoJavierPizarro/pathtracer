"""
cpu_renderer.py - Renderizado en CPU
"""

import numpy as np

from geometry import reflect, refract


def fast_norm(v):
    return np.sqrt(np.dot(v, v))


def normalize(v):
    return v / fast_norm(v)


def hit_sphere(origin, direction, center, radius):
    oc = origin - center
    a = np.dot(direction, direction)
    half_b = np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius
    discriminant = half_b**2 - a * c

    if discriminant < 0:
        return False, 0.0

    sqrt_disc = np.sqrt(discriminant)
    for sign in (-1, 1):
        root = (-half_b + sign * sqrt_disc) / a
        if root >= 0.001:
            return True, root

    return False, 0.0


def hit_plane(origin, direction, point, normal):
    denom = np.dot(normal, direction)
    if abs(denom) < 1e-6:
        return False, 0.0
    t = np.dot(point - origin, normal) / denom
    return (t >= 0.001), t


def is_in_shadow(point, light_pos, spheres, planes, light_distance):
    shadow_ray = light_pos - point
    light_dir = normalize(shadow_ray)

    for s in spheres:
        hit, t = hit_sphere(point, light_dir, s.center, s.radius)
        if hit and t < light_distance - 0.001:
            return True

    for p in planes:
        hit, t = hit_plane(point, light_dir, p.point, p.normal)
        if hit and t < light_distance - 0.001:
            return True

    return False


def cosine_weighted_hemisphere_sample(normal) -> np.ndarray:
    while True:
        a, b = np.random.uniform(-1, 1, 2)
        if a * a + b * b < 1:
            break
    c = np.sqrt(1 - a * a - b * b)
    local_dir = np.array([2 * a * c, 2 * b * c, 1 - 2 * (a * a + b * b)], dtype=np.float32)

    # Create tangent space
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(normal[0]) > 0.9:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    tangent = normalize(np.cross(normal, up))
    bitangent = np.cross(normal, tangent)

    world_dir = local_dir[0] * tangent + local_dir[1] * bitangent + local_dir[2] * normal
    return normalize(world_dir)

def direct_light(direction, hit_color, hit_point, hit_normal, hit_ior, light_pos, spheres, planes, hit_material, depth, max_depth) -> np.ndarray:
    kd, kr, kt = hit_material
    color_kd = kd * compute_direct_light(hit_point, hit_normal, light_pos, spheres, planes) * hit_color if kd > 0 else np.array([0, 0, 0], dtype=np.float32)
    color_kr = kr * shade_mirror(direction, hit_point, hit_normal, spheres, planes, light_pos, depth, max_depth) if kr > 0 else np.array([0, 0, 0], dtype=np.float32)
    color_kt = kt * shade_glass(direction, hit_point, hit_normal, hit_ior, spheres, planes, light_pos, depth, max_depth) if kt > 0 else np.array([0, 0, 0], dtype=np.float32)

    return color_kd + color_kr + color_kt

def compute_direct_light(hit_point, hit_normal, light_pos, spheres, planes) -> float:
    light_vec = light_pos - hit_point
    light_dist = fast_norm(light_vec)
    light_dir = light_vec / light_dist
    bias_point = hit_point + hit_normal * 0.001

    if is_in_shadow(bias_point, light_pos, spheres, planes, light_dist):
        return 0.1
    else:
        return max(0.0, np.dot(hit_normal, light_dir))

def shade_diffuse(hit_point, hit_normal, hit_color, spheres, planes, light_pos, depth, max_depth) -> np.ndarray:
    bounce_dir = cosine_weighted_hemisphere_sample(hit_normal)
    bias_point = hit_point + hit_normal * 0.001
    bounced_color = trace_ray(bias_point, bounce_dir, spheres, planes, light_pos, depth + 1, max_depth)
    return hit_color * bounced_color * 0.5


def shade_mirror(direction, hit_point, hit_normal, spheres, planes, light_pos, depth, max_depth) -> np.ndarray:
    reflect_dir = reflect(direction, hit_normal)
    bias_point = hit_point + hit_normal * 0.001
    return trace_ray(bias_point, reflect_dir, spheres, planes, light_pos, depth + 1, max_depth)


def shade_glass(direction, hit_point, hit_normal, hit_ior, spheres, planes, light_pos, depth, max_depth) -> np.ndarray:
    ior = hit_ior
    n = hit_normal.copy()
    cosi = np.dot(direction, n)
    if cosi > 0:
        n = -hit_normal
        ior = 1.0 / ior
        cosi = -cosi

    refr_dir = refract(direction, n, ior)
    bias_point = hit_point + refr_dir * 0.001

    if refr_dir is None:
        # Reflexión total
        reflect_dir = reflect(direction, hit_normal)
        return trace_ray(bias_point, reflect_dir, spheres, planes, light_pos, depth + 1, max_depth)
    else:
        # Fresnel (aproximación Schlick)
        r0 = ((1 - ior) / (1 + ior)) ** 2
        reflect_prob = r0 + (1 - r0) * (1 - abs(cosi)) ** 5

        if np.random.rand() < reflect_prob:
            reflect_dir = reflect(direction, hit_normal)
            bias_point = hit_point + reflect_dir * 0.001
            return trace_ray(bias_point, reflect_dir, spheres, planes, light_pos, depth + 1, max_depth)
        else:
            return trace_ray(bias_point, refr_dir, spheres, planes, light_pos, depth + 1, max_depth)


def trace_ray(origin, direction, spheres, planes, light_pos, depth, max_depth) -> np.ndarray:
    closest_t = float('inf')
    hit_point, hit_normal = None, None
    hit_material = None

    # Encontrar intersección más cercana
    for s in spheres:
        hit, t = hit_sphere(origin, direction, s.center, s.radius)
        if hit and t < closest_t:
            closest_t = t
            hit_point = origin + t * direction
            hit_normal = normalize(hit_point - s.center)
            hit_material = s.material
            hit_color = s.color
            hit_ior = s.ior

    for p in planes:
        hit, t = hit_plane(origin, direction, p.point, p.normal)
        if hit and t < closest_t:
            closest_t = t
            hit_point = origin + t * direction
            hit_normal = p.normal
            hit_material = p.material
            hit_color = p.color
            hit_ior = p.ior

    if hit_point is None:
        return np.array([0.05, 0.05, 0.1])  # color fondo

    # Luz directa
    final_color = direct_light(direction, hit_color, hit_point, hit_normal, hit_ior, light_pos, spheres, planes, hit_material, depth, max_depth)

    # Rebotes con ruleta rusa según material
    if depth < max_depth:
        probs = hit_material
        probs = probs / probs.sum()  # Normalizar
        chosen_type_idx = np.random.choice(3, p=probs)

        if chosen_type_idx == 0:  # Difuso
            final_color += shade_diffuse(hit_point, hit_normal, hit_color, spheres, planes, light_pos, depth, max_depth)
        elif chosen_type_idx == 1:  # Espejo
            final_color += shade_mirror(direction, hit_point, hit_normal, spheres, planes, light_pos, depth, max_depth)
        else:  # Cristal
            final_color += shade_glass(direction, hit_point, hit_normal, hit_ior, spheres, planes, light_pos, depth, max_depth)
    assert len(final_color) == 3, f"final_color shape = {final_color} hit_point = {hit_point} hit_normal = {hit_normal} hit_material = {hit_material}"
    return np.clip(final_color, 0, 1)




class CPURenderer:
    def __init__(self, samples_per_pixel=1, max_bounces=0):
        self.samples_per_pixel = samples_per_pixel
        self.max_bounces = max_bounces

    def render(self, scene, camera, width, height):
        light_pos = scene.lights[0].position if scene.lights else np.array([-2.0, 2.0, 0.0], dtype=np.float32)
        image = np.zeros((height, width, 3), dtype=np.float32)  # float32 para acumulación

        for y in range(height):
            for x in range(width):
                u = (x + 0.5) / width
                v = (height - y - 0.5) / height

                pixel_color = np.zeros(3, dtype=np.float32)
                for _ in range(self.samples_per_pixel):
                    ray_dir = (
                        camera.lower_left_corner +
                        u * camera.horizontal +
                        v * camera.vertical -
                        camera.origin
                    )
                    ray_dir = normalize(ray_dir)

                    pixel_color += trace_ray(
                        camera.origin, ray_dir,
                        scene.spheres, scene.planes, light_pos,
                        depth=0, max_depth=self.max_bounces
                    )

                pixel_color /= self.samples_per_pixel
                image[y, x] = pixel_color
        return image
