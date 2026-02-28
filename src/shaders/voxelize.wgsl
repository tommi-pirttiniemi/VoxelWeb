// Voxelization compute shader
// For each voxel, casts a ray along +Y and counts triangle crossings (parity → solid).
// Outputs a flat u32 grid: 0 = empty, (rgb565 << 9) | 1 = filled.

struct VoxParams {
    gridMin:   vec4f,  // xyz = world min of grid, w = voxelSize (world units per voxel)
    gridDims:  vec4u,  // xyz = grid dims in voxels, w = triCount
    color:     u32,    // default packed value: (rgb565 << 9) | 1
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
};

@group(0) @binding(0) var<uniform>            params:     VoxParams;
@group(0) @binding(1) var<storage, read>      vertices:   array<f32>;  // packed xyz
@group(0) @binding(2) var<storage, read>      indices:    array<u32>;
@group(0) @binding(3) var<storage, read_write> flatGrid:  array<u32>;

// 2D point-in-triangle test in XZ plane.
// Returns true + barycentric u,v if point (px,pz) is inside triangle (a,b,c) in XZ.
fn pointInTriXZ(px: f32, pz: f32,
                ax: f32, az: f32,
                bx: f32, bz: f32,
                cx: f32, cz: f32) -> vec3f {
    // v0 = C-A, v1 = B-A, v2 = P-A  (all in XZ)
    let v0x = cx - ax; let v0z = cz - az;
    let v1x = bx - ax; let v1z = bz - az;
    let v2x = px - ax; let v2z = pz - az;

    let d00 = v0x*v0x + v0z*v0z;
    let d01 = v0x*v1x + v0z*v1z;
    let d11 = v1x*v1x + v1z*v1z;
    let d02 = v0x*v2x + v0z*v2z;
    let d12 = v1x*v2x + v1z*v2z;

    let denom = d00*d11 - d01*d01;
    if abs(denom) < 1e-10 { return vec3f(-1.0, -1.0, -1.0); } // degenerate

    let inv = 1.0 / denom;
    let u = (d11*d02 - d01*d12) * inv;
    let v = (d00*d12 - d01*d02) * inv;
    return vec3f(u, v, 1.0); // z=1 signals valid
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let voxelIdx = gid.x;
    let dx = params.gridDims.x;
    let dy = params.gridDims.y;
    let dz = params.gridDims.z;
    let total = dx * dy * dz;
    if voxelIdx >= total { return; }

    let vx = voxelIdx % dx;
    let vy = (voxelIdx / dx) % dy;
    let vz = voxelIdx / (dx * dy);

    let voxSize = params.gridMin.w;
    let gmin    = params.gridMin.xyz;

    // Voxel center in world space
    let cx = gmin.x + (f32(vx) + 0.5) * voxSize;
    let cy = gmin.y + (f32(vy) + 0.5) * voxSize;
    let cz = gmin.z + (f32(vz) + 0.5) * voxSize;

    let triCount = params.gridDims.w;
    var crossings: u32 = 0u;

    for (var i = 0u; i < triCount; i += 1u) {
        let i0 = indices[i * 3u];
        let i1 = indices[i * 3u + 1u];
        let i2 = indices[i * 3u + 2u];

        let ax = vertices[i0 * 3u];     let ay = vertices[i0 * 3u + 1u]; let az = vertices[i0 * 3u + 2u];
        let bx = vertices[i1 * 3u];     let by = vertices[i1 * 3u + 1u]; let bz = vertices[i1 * 3u + 2u];
        let ccx = vertices[i2 * 3u];    let ccy = vertices[i2 * 3u + 1u]; let ccz = vertices[i2 * 3u + 2u];

        // Skip triangles entirely below this voxel
        if max(ay, max(by, ccy)) <= cy { continue; }

        let uv = pointInTriXZ(cx, cz, ax, az, bx, bz, ccx, ccz);
        if uv.z < 0.5 { continue; } // degenerate
        let u = uv.x; let v = uv.y;
        if u < 0.0 || v < 0.0 || u + v > 1.0 { continue; }

        // Y at crossing
        let hitY = (1.0 - u - v) * ay + u * ccy + v * by;
        if hitY > cy { crossings += 1u; }
    }

    if crossings % 2u == 1u {
        flatGrid[voxelIdx] = params.color;
    }
}
