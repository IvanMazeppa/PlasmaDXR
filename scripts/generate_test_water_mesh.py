#!/usr/bin/env python3
"""
Generate test water meshes for PlasmaDX-Clean development.

Creates simple geometric shapes (bowl, sphere, plane) in the raw binary format
expected by the water mesh loader. Useful for testing without Blender.

Binary format:
- Header: uint32 vertex_count, uint32 index_count
- Vertices: interleaved float3 position + float3 normal (6 floats per vertex)
- Indices: uint32 triangle indices
"""

import struct
import math
import os
import argparse


def normalize(v):
    """Normalize a 3D vector."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length < 1e-10:
        return (0, 1, 0)
    return (v[0]/length, v[1]/length, v[2]/length)


def generate_bowl(radius=100.0, depth=50.0, segments=32, rings=16):
    """
    Generate a bowl-shaped mesh (hemisphere open at top).

    Args:
        radius: Bowl radius
        depth: Bowl depth (how deep the bowl is)
        segments: Number of horizontal segments
        rings: Number of vertical rings

    Returns:
        (vertices, indices) - vertices is list of (px,py,pz,nx,ny,nz) tuples
    """
    vertices = []
    indices = []

    # Generate vertices
    for ring in range(rings + 1):
        # Ring goes from bottom (full depth) to top (0 depth)
        t = ring / rings  # 0 at bottom, 1 at top
        y = -depth * (1 - t)  # -depth at bottom, 0 at top
        ring_radius = radius * math.sin(t * math.pi * 0.5)  # Expands from 0 to radius

        for seg in range(segments):
            theta = (seg / segments) * math.pi * 2

            x = ring_radius * math.cos(theta)
            z = ring_radius * math.sin(theta)

            # Normal points outward and slightly up for bowl shape
            nx = math.cos(theta) * math.cos(t * math.pi * 0.5)
            ny = math.sin(t * math.pi * 0.5)
            nz = math.sin(theta) * math.cos(t * math.pi * 0.5)
            nx, ny, nz = normalize((nx, ny, nz))

            vertices.append((x, y, z, nx, ny, nz))

    # Generate indices (triangulate rings)
    for ring in range(rings):
        for seg in range(segments):
            current = ring * segments + seg
            next_seg = ring * segments + (seg + 1) % segments
            above = (ring + 1) * segments + seg
            above_next = (ring + 1) * segments + (seg + 1) % segments

            # Two triangles per quad
            indices.extend([current, above, next_seg])
            indices.extend([next_seg, above, above_next])

    return vertices, indices


def generate_sphere(radius=50.0, segments=32, rings=16):
    """
    Generate a UV sphere mesh.

    Args:
        radius: Sphere radius
        segments: Number of horizontal segments
        rings: Number of vertical rings

    Returns:
        (vertices, indices)
    """
    vertices = []
    indices = []

    # Generate vertices
    for ring in range(rings + 1):
        phi = (ring / rings) * math.pi  # 0 to PI (top to bottom)
        y = radius * math.cos(phi)
        ring_radius = radius * math.sin(phi)

        for seg in range(segments):
            theta = (seg / segments) * math.pi * 2

            x = ring_radius * math.cos(theta)
            z = ring_radius * math.sin(theta)

            # Normal = normalized position for sphere
            nx, ny, nz = normalize((x, y, z))

            vertices.append((x, y, z, nx, ny, nz))

    # Generate indices
    for ring in range(rings):
        for seg in range(segments):
            current = ring * segments + seg
            next_seg = ring * segments + (seg + 1) % segments
            below = (ring + 1) * segments + seg
            below_next = (ring + 1) * segments + (seg + 1) % segments

            # Two triangles per quad (except at poles)
            if ring > 0:
                indices.extend([current, below, next_seg])
            if ring < rings - 1:
                indices.extend([next_seg, below, below_next])

    return vertices, indices


def generate_plane(width=200.0, depth=200.0, subdivisions=10, y_offset=0.0):
    """
    Generate a flat plane mesh (for testing reflections).

    Args:
        width: Plane width (X)
        depth: Plane depth (Z)
        subdivisions: Grid subdivisions
        y_offset: Y position offset

    Returns:
        (vertices, indices)
    """
    vertices = []
    indices = []

    # Generate grid vertices
    for zi in range(subdivisions + 1):
        for xi in range(subdivisions + 1):
            x = (xi / subdivisions - 0.5) * width
            z = (zi / subdivisions - 0.5) * depth
            y = y_offset

            # Normal points up
            vertices.append((x, y, z, 0.0, 1.0, 0.0))

    # Generate triangles
    for zi in range(subdivisions):
        for xi in range(subdivisions):
            current = zi * (subdivisions + 1) + xi
            right = current + 1
            below = current + (subdivisions + 1)
            below_right = below + 1

            indices.extend([current, below, right])
            indices.extend([right, below, below_right])

    return vertices, indices


def generate_wavy_plane(width=200.0, depth=200.0, subdivisions=32, amplitude=10.0, frequency=2.0):
    """
    Generate a wavy plane mesh (simulates calm water surface).

    Args:
        width: Plane width (X)
        depth: Plane depth (Z)
        subdivisions: Grid subdivisions
        amplitude: Wave height
        frequency: Number of wave cycles

    Returns:
        (vertices, indices)
    """
    vertices = []
    indices = []

    # Generate grid vertices with wave displacement
    for zi in range(subdivisions + 1):
        for xi in range(subdivisions + 1):
            u = xi / subdivisions
            v = zi / subdivisions

            x = (u - 0.5) * width
            z = (v - 0.5) * depth

            # Simple sine wave pattern
            y = amplitude * math.sin(u * frequency * math.pi * 2) * math.cos(v * frequency * math.pi * 2)

            # Calculate normal from wave gradient
            dydx = amplitude * frequency * math.pi * 2 * math.cos(u * frequency * math.pi * 2) * math.cos(v * frequency * math.pi * 2)
            dydz = -amplitude * frequency * math.pi * 2 * math.sin(u * frequency * math.pi * 2) * math.sin(v * frequency * math.pi * 2)

            # Normal = cross product of tangents
            nx, ny, nz = normalize((-dydx, 1.0, -dydz))

            vertices.append((x, y, z, nx, ny, nz))

    # Generate triangles
    for zi in range(subdivisions):
        for xi in range(subdivisions):
            current = zi * (subdivisions + 1) + xi
            right = current + 1
            below = current + (subdivisions + 1)
            below_right = below + 1

            indices.extend([current, below, right])
            indices.extend([right, below, below_right])

    return vertices, indices


def write_mesh(filepath, vertices, indices):
    """
    Write mesh data to binary file.

    Args:
        filepath: Output path
        vertices: List of (px,py,pz,nx,ny,nz) tuples
        indices: List of triangle indices
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    vertex_count = len(vertices)
    index_count = len(indices)

    # Flatten vertices to float array
    flat_vertices = []
    for v in vertices:
        flat_vertices.extend(v)

    with open(filepath, 'wb') as f:
        # Header
        f.write(struct.pack('II', vertex_count, index_count))
        # Vertices (6 floats per vertex)
        f.write(struct.pack(f'{len(flat_vertices)}f', *flat_vertices))
        # Indices
        f.write(struct.pack(f'{len(indices)}I', *indices))

    file_size = os.path.getsize(filepath)
    print(f"Generated: {filepath}")
    print(f"  Vertices: {vertex_count}")
    print(f"  Triangles: {index_count // 3}")
    print(f"  File size: {file_size} bytes ({file_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description='Generate test water meshes for PlasmaDX-Clean')
    parser.add_argument('--type', choices=['bowl', 'sphere', 'plane', 'wavy', 'all'], default='bowl',
                        help='Mesh type to generate')
    parser.add_argument('--output', default='assets/water/',
                        help='Output directory')
    parser.add_argument('--radius', type=float, default=100.0,
                        help='Mesh radius/size')
    parser.add_argument('--segments', type=int, default=32,
                        help='Number of segments')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.type in ('bowl', 'all'):
        vertices, indices = generate_bowl(radius=args.radius, depth=args.radius*0.5,
                                          segments=args.segments, rings=args.segments//2)
        write_mesh(os.path.join(args.output, 'test_bowl.bin'), vertices, indices)

    if args.type in ('sphere', 'all'):
        vertices, indices = generate_sphere(radius=args.radius*0.5,
                                            segments=args.segments, rings=args.segments//2)
        write_mesh(os.path.join(args.output, 'test_sphere.bin'), vertices, indices)

    if args.type in ('plane', 'all'):
        vertices, indices = generate_plane(width=args.radius*2, depth=args.radius*2,
                                           subdivisions=args.segments)
        write_mesh(os.path.join(args.output, 'test_plane.bin'), vertices, indices)

    if args.type in ('wavy', 'all'):
        vertices, indices = generate_wavy_plane(width=args.radius*2, depth=args.radius*2,
                                                subdivisions=args.segments*2,
                                                amplitude=args.radius*0.05, frequency=3.0)
        write_mesh(os.path.join(args.output, 'test_wavy.bin'), vertices, indices)


if __name__ == '__main__':
    main()
