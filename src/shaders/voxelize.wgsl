// Voxelization compute shaders — exact WGSL port of Voxelizer3.compute
// Three entry points:
//   initializeGrid   — clears buffer, marks edge voxels as Air
//   voxelizeSurface  — per-triangle surface sampling with texture colors
//   resolveAverage   — thresholds hit count, computes average/mode color

const SAMPLES_PER_UNIT: f32 = 5.0;

// ── Params uniform ────────────────────────────────────────────────────────────

struct Params {
    boundsMin:     vec3f,   // offset  0 (bytes 0-11)
    voxelSize:     f32,     // offset 12 (byte 12-15)
    gridSize:      vec3u,   // offset 16 (bytes 16-27)
    triCount:      u32,     // offset 28 (bytes 28-31)
    alphaCutoff:   f32,     // offset 32
    edgeThreshold: f32,     // offset 36
    averageColor:  u32,     // offset 40  (1=average, 0=mode)
    _pad:          u32,     // offset 44
}

// ── Voxel storage struct ──────────────────────────────────────────────────────
// Layout matches Unity's stride=28 (7 × 4 bytes)
// State: 0=Unknown, 1=Shell, 2=Air

struct Voxel {
    accR:          atomic<u32>,   // offset  0
    accG:          atomic<u32>,   // offset  4
    accB:          atomic<u32>,   // offset  8
    count:         atomic<u32>,   // offset 12
    modeCandidate: atomic<u32>,   // offset 16  high16=vote count, low16=RGB565
    state:         i32,           // offset 20
    colorPacked:   u32,           // offset 24  (r<<24|g<<16|b<<8|255) after resolve
}

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>            params:   Params;
@group(0) @binding(1) var<storage, read_write> voxels:  array<Voxel>;

// Bindings 2-5 only used by voxelizeSurface
@group(0) @binding(2) var<storage, read>       vertices: array<f32>;  // flat xyz
@group(0) @binding(3) var<storage, read>       indices:  array<u32>;
@group(0) @binding(4) var<storage, read>       uvs:      array<f32>;  // flat uv
@group(0) @binding(5) var                      tex:      texture_2d<f32>;

// ── Helper: flat grid index ───────────────────────────────────────────────────

fn getIndex(p: vec3i) -> i32 {
    let gs = vec3i(params.gridSize);
    if any(p < vec3i(0)) || any(p >= gs) { return -1; }
    return p.x + gs.x * (p.y + gs.y * p.z);
}

// ── Helper: RGB565 pack/unpack ────────────────────────────────────────────────

fn toRGB565(r: u32, g: u32, b: u32) -> u32 {
    return ((r >> 3u) << 11u) | ((g >> 2u) << 5u) | (b >> 3u);
}

fn fromRGB565(c: u32) -> vec3u {
    return vec3u(
        ((c >> 11u) & 0x1Fu) << 3u,
        ((c >> 5u)  & 0x3Fu) << 2u,
        (c & 0x1Fu) << 3u,
    );
}

// ── Helper: linear → sRGB (matches Unity LinearToSRGB) ───────────────────────

fn linearToSRGB(lin: vec3f) -> vec3f {
    let lo = lin * 12.92;
    let hi = 1.055 * pow(abs(lin), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, lin <= vec3f(0.0031308));
}

// ── Helper: bilinear texture sample with repeat wrap ─────────────────────────
// textureSampleLevel is forbidden in compute shaders; use textureLoad instead.

fn sampleTex(uv: vec2f) -> vec4f {
    let dims = vec2i(textureDimensions(tex, 0));
    let uvw  = fract(uv);                          // repeat wrap
    let px   = uvw.x * f32(dims.x) - 0.5;
    let py   = uvw.y * f32(dims.y) - 0.5;
    let ix   = i32(floor(px));
    let iy   = i32(floor(py));
    let fx   = px - floor(px);
    let fy   = py - floor(py);
    let x0   = clamp(ix,     0, dims.x - 1);
    let x1   = clamp(ix + 1, 0, dims.x - 1);
    let y0   = clamp(iy,     0, dims.y - 1);
    let y1   = clamp(iy + 1, 0, dims.y - 1);
    let c00  = textureLoad(tex, vec2i(x0, y0), 0);
    let c10  = textureLoad(tex, vec2i(x1, y0), 0);
    let c01  = textureLoad(tex, vec2i(x0, y1), 0);
    let c11  = textureLoad(tex, vec2i(x1, y1), 0);
    return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

// ── KERNEL 1: InitializeGrid ──────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 8)
fn initializeGrid(@builtin(global_invocation_id) id: vec3u) {
    let gs = params.gridSize;
    if any(id >= gs) { return; }
    let idx = i32(id.x) + i32(gs.x) * (i32(id.y) + i32(gs.y) * i32(id.z));

    let isEdge = (id.x == 0u || id.y == 0u || id.z == 0u ||
                  id.x == gs.x - 1u || id.y == gs.y - 1u || id.z == gs.z - 1u);

    atomicStore(&voxels[idx].accR,          0u);
    atomicStore(&voxels[idx].accG,          0u);
    atomicStore(&voxels[idx].accB,          0u);
    atomicStore(&voxels[idx].count,         0u);
    atomicStore(&voxels[idx].modeCandidate, 0u);
    voxels[idx].state       = select(0i, 2i, isEdge);
    voxels[idx].colorPacked = 0u;
}

// ── KERNEL 2: VoxelizeSurface ─────────────────────────────────────────────────
// One thread per triangle. Walks the triangle surface in barycentric space
// at stepSize = voxelSize/SAMPLES_PER_UNIT, samples the texture at interpolated
// UV, and accumulates color into the voxel with atomics.

@compute @workgroup_size(64, 1, 1)
fn voxelizeSurface(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= params.triCount { return; }

    let base = id.x * 3u;
    let i0 = indices[base];
    let i1 = indices[base + 1u];
    let i2 = indices[base + 2u];

    let v0 = vec3f(vertices[i0*3u], vertices[i0*3u+1u], vertices[i0*3u+2u]);
    let v1 = vec3f(vertices[i1*3u], vertices[i1*3u+1u], vertices[i1*3u+2u]);
    let v2 = vec3f(vertices[i2*3u], vertices[i2*3u+1u], vertices[i2*3u+2u]);

    let uv0 = vec2f(uvs[i0*2u], uvs[i0*2u+1u]);
    let uv1 = vec2f(uvs[i1*2u], uvs[i1*2u+1u]);
    let uv2 = vec2f(uvs[i2*2u], uvs[i2*2u+1u]);

    let edge0 = v1 - v0;
    let edge1 = v2 - v0;
    let len0  = length(edge0);
    let len1  = length(edge1);

    let stepSize = params.voxelSize / SAMPLES_PER_UNIT;
    let steps0   = max(1i, i32(ceil(len0 / stepSize)));
    let steps1   = max(1i, i32(ceil(len1 / stepSize)));
    let gs       = vec3i(params.gridSize);

    for (var i = 0i; i <= steps0; i += 1i) {
        let u = f32(i) / f32(steps0);
        let curSteps1 = i32(ceil((1.0 - u) * f32(steps1)));

        for (var j = 0i; j <= curSteps1; j += 1i) {
            let v = select(0.0, f32(j) / f32(steps1), steps1 > 0i);
            if u + v > 0.99999 { continue; }

            let pos     = v0 + edge0 * u + edge1 * v;
            let gridPos = min(vec3i(floor((pos - params.boundsMin) / params.voxelSize)), gs - vec3i(1));
            let idx     = getIndex(gridPos);
            if idx == -1i { continue; }

            let texUV = uv0 + (uv1 - uv0) * u + (uv2 - uv0) * v;
            var col   = sampleTex(texUV);

            // Premultiplied alpha unpack
            if col.a > 0.0 {
                col = vec4f(saturate(col.rgb / col.a), col.a);
            }

            col = vec4f(linearToSRGB(col.rgb), col.a);

            if col.a > params.alphaCutoff {
                let r = u32(col.r * 255.0);
                let g = u32(col.g * 255.0);
                let b = u32(col.b * 255.0);

                atomicAdd(&voxels[idx].accR,  r);
                atomicAdd(&voxels[idx].accG,  g);
                atomicAdd(&voxels[idx].accB,  b);
                atomicAdd(&voxels[idx].count, 1u);

                // Boyer-Moore majority-vote candidate (mode color tracking)
                if params.averageColor == 0u {
                    let rgb565    = toRGB565(r, g, b);
                    let packed    = atomicLoad(&voxels[idx].modeCandidate);
                    let candColor = packed & 0xFFFFu;
                    let candCount = packed >> 16u;

                    if candCount == 0u {
                        // Slot empty — try to claim it (CAS from 0)
                        _ = atomicCompareExchangeWeak(&voxels[idx].modeCandidate,
                                                      0u, (1u << 16u) | rgb565);
                    } else if candColor == rgb565 {
                        atomicAdd(&voxels[idx].modeCandidate, 1u << 16u);
                    } else {
                        atomicSub(&voxels[idx].modeCandidate, 1u << 16u);
                    }
                }
            }
        }
    }
}

// ── KERNEL 3: ResolveAverage ──────────────────────────────────────────────────
// Thresholds voxels by hit count. Below threshold → Air. Above → Shell with
// average or mode color packed into colorPacked (RGBA8888).

@compute @workgroup_size(8, 8, 8)
fn resolveAverage(@builtin(global_invocation_id) id: vec3u) {
    let gs = params.gridSize;
    if any(id >= gs) { return; }
    let idx = i32(id.x) + i32(gs.x) * (i32(id.y) + i32(gs.y) * i32(id.z));

    let count        = atomicLoad(&voxels[idx].count);
    let baselineHits = SAMPLES_PER_UNIT * SAMPLES_PER_UNIT;           // 25
    let requiredHits = u32(ceil(baselineHits * params.edgeThreshold)); // ceil(25 * 0.6) = 15

    if count > 0u {
        if count < requiredHits {
            voxels[idx].state       = 2i;  // Air — noise / thin edge
            voxels[idx].colorPacked = 0u;
        } else {
            voxels[idx].state = 1i;  // Shell

            var r: u32;
            var g: u32;
            var b: u32;

            if params.averageColor == 1u {
                r = atomicLoad(&voxels[idx].accR) / count;
                g = atomicLoad(&voxels[idx].accG) / count;
                b = atomicLoad(&voxels[idx].accB) / count;
            } else {
                let cand = fromRGB565(atomicLoad(&voxels[idx].modeCandidate) & 0xFFFFu);
                r = cand.x;
                g = cand.y;
                b = cand.z;
            }

            // colorPacked = (r<<24) | (g<<16) | (b<<8) | 255
            voxels[idx].colorPacked = (r << 24u) | (g << 16u) | (b << 8u) | 255u;
        }
    }
}
