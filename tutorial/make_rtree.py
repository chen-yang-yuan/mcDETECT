from rtree import index
from shapely.geometry import Point

def make_rtree(spheres):
    p = index.Property()
    idx = index.Index(properties = p)
    for i, sphere in enumerate(spheres.itertuples()):
        center = Point(sphere.sphere_x, sphere.sphere_y)
        bounds = (
            center.x - sphere.sphere_r,
            center.y - sphere.sphere_r,
            center.x + sphere.sphere_r,
            center.y + sphere.sphere_r
        )
        idx.insert(i, bounds)
    return idx