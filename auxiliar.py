from geometry import Triangle

def load_obj_file(file_path):
    vertex = []
    triangles_index = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) == 4:
                    x, y, z = map(float, parts[1:])
                    vertex.append((x, y, z))
            elif line.startswith('f '):
                parts = line.strip().split()
                if len(parts) == 4:
                    # OBJ indices start at 1, subtract 1 for 0-based indexing
                    i1, i2, i3 = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    triangles_index.append((i1, i2, i3))
    
    return vertex, triangles_index

def convert_to_triangles(vertex, triangle_index, color):
    return [Triangle(vertex[i1], vertex[i2], vertex[i3], color) for i1, i2, i3 in triangle_index]

def scale(factor):
    def inner(v):
        return (v[0] * factor, v[1] * factor, v[2] * factor)
    return inner

def translate(dx=0, dy=0, dz=0):
    def inner(v):
        x, y, z = v
        return (x + dx, y + dy, z + dz)
    return inner

def rotate_x(theta):
    def inner(v):
        from math import cos, sin
        x, y, z = v
        return (x, y * cos(theta) - z * sin(theta), y * sin(theta) + z * cos(theta))
    return inner

def rotate_y(theta):
    def inner(v):
        from math import cos, sin
        x, y, z = v
        return (x * cos(theta) + z * sin(theta), y, -x * sin(theta) + z * cos(theta))
    return inner

def rotate_z(theta):
    def inner(v):
        from math import cos, sin
        x, y, z = v
        return (x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta), z)
    return inner


def compose(*functions):
    def composed(x):
        for f in reversed(functions):  # Apply right to left
            x = f(x)
        return x
    return composed