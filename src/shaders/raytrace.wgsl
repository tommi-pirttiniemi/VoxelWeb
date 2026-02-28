// Two-level DDA voxel raymarcher — WGSL port of VoxelAssetRotateLightGI.shader
// Grid is in world space; no separate model matrix.
//
// Coordinate spaces:
//   WS  = world space
//   CU  = chunk-unit space: 1 unit = 1 chunk (8×8×8 voxels)
//   VS  = voxel space within a chunk (0..7)

const MAX_CHUNK_STEPS: i32 = 128;
const MAX_VOXEL_STEPS: i32 = 27;
const EPS: f32 = 1e-6;

// ── Uniforms ─────────────────────────────────────────────────────────────────

struct Uniforms {
    mvp:          mat4x4f,
    // xyz = grid world-space min, w = chunkWorldSize
    gridMin:      vec4f,
    // xyz = chunk grid dimensions (float), w = chunkWorldSizeInv
    gridDims:     vec4f,
    cameraPos:    vec4f,  // xyz = world pos, w unused
};

// ── Storage buffers ───────────────────────────────────────────────────────────

// uint2 per chunk: x = voxel buffer chunk offset (0xFFFFFFFF = empty),
//                  y = packed AABB (18 bits: minX|minY|minZ|maxX|maxY|maxZ, 3 bits each)
@group(0) @binding(0) var<uniform>       uni:         Uniforms;
@group(0) @binding(1) var<storage, read> chunkGrid:   array<vec2u>;
@group(0) @binding(2) var<storage, read> voxelBuffer: array<u32>;

// ── Vertex stage ──────────────────────────────────────────────────────────────

// Unit cube vertices for a bounding-box proxy mesh.
// positionOS in [-0.5, 0.5]; the vertex shader maps it to the grid AABB.
const CUBE_POS = array<vec3f, 8>(
    vec3f(-0.5, -0.5, -0.5),
    vec3f( 0.5, -0.5, -0.5),
    vec3f(-0.5,  0.5, -0.5),
    vec3f( 0.5,  0.5, -0.5),
    vec3f(-0.5, -0.5,  0.5),
    vec3f( 0.5, -0.5,  0.5),
    vec3f(-0.5,  0.5,  0.5),
    vec3f( 0.5,  0.5,  0.5),
);

// 12 triangles × 3 indices = 36
const CUBE_IDX = array<u32, 36>(
    0u,2u,1u, 1u,2u,3u,   // -Z
    4u,5u,6u, 5u,7u,6u,   // +Z
    0u,1u,4u, 1u,5u,4u,   // -Y
    2u,6u,3u, 3u,6u,7u,   // +Y
    0u,4u,2u, 2u,4u,6u,   // -X
    1u,3u,5u, 3u,7u,5u,   // +X
);

struct VSOut {
    @builtin(position)  posH:     vec4f,
    @location(0)        worldPos: vec3f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    let idx     = CUBE_IDX[vi];
    let posUnit = CUBE_POS[idx];

    let bMin = uni.gridMin.xyz - 0.001;
    let bMax = uni.gridMin.xyz + uni.gridDims.xyz * uni.gridMin.w + 0.001;
    let worldPos = posUnit * (bMax - bMin) + (bMin + bMax) * 0.5;

    var out: VSOut;
    out.posH     = uni.mvp * vec4f(worldPos, 1.0);
    out.worldPos = worldPos;
    return out;
}

// ── Helper functions ──────────────────────────────────────────────────────────

fn rayBoxIntersect(ro: vec3f, rdi: vec3f, bMin: vec3f, bMax: vec3f) -> vec2f {
    let t0   = (bMin - ro) * rdi;
    let t1   = (bMax - ro) * rdi;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2f(
        max(0.0, max(tmin.x, max(tmin.y, tmin.z))),
        min(tmax.x, min(tmax.y, tmax.z))
    );
}

fn unpackAABB(aabb: u32) -> array<vec3i, 2> {
    let mn = vec3i(i32(aabb & 7u), i32((aabb >> 3u) & 7u), i32((aabb >> 6u) & 7u));
    let mx = vec3i(i32((aabb >> 9u) & 7u), i32((aabb >> 12u) & 7u), i32((aabb >> 15u) & 7u));
    return array<vec3i, 2>(mn, mx);
}

fn decodeRGB565(c: u32) -> vec3f {
    let r = f32((c >> 11u) & 31u)  / 31.0;
    let g = f32((c >> 5u)  & 63u)  / 63.0;
    let b = f32( c         & 31u)  / 31.0;
    return vec3f(r, g, b);
}

fn srgbToLinear(c: vec3f) -> vec3f {
    return c * c * (c * 0.2 + 0.8);
}

// ── Inner voxel DDA ──────────────────────────────────────────────────────────

struct VoxelHit {
    hit:    bool,
    color:  vec3f,
    normal: vec3f,
    t:      f32,
};

fn rayMarchChunk(
    ro: vec3f, rd: vec3f,
    gPos: vec3i, cOffset: u32,
    aabbMin: vec3i, aabbMax: vec3i,
    tEntry: f32, hitAxis: i32
) -> VoxelHit {
    var result: VoxelHit;
    result.hit = false;

    let lPos = (ro + rd * tEntry) - vec3f(gPos);
    var vPos  = clamp(vec3i(floor(lPos * 8.0)), aabbMin, aabbMax);
    let rayStep = vec3i(sign(rd));
    let tDelta  = abs(vec3f(0.125) / (rd + EPS * sign(rd)));
    let nextB   = vec3f(vPos + max(vec3i(0), rayStep)) * 0.125;

    var tMax = vec3f(
        select(1e30, (nextB.x - lPos.x) / rd.x, abs(rd.x) > EPS),
        select(1e30, (nextB.y - lPos.y) / rd.y, abs(rd.y) > EPS),
        select(1e30, (nextB.z - lPos.z) / rd.z, abs(rd.z) > EPS),
    );

    var cT = 0.0;
    var ha = hitAxis;

    for (var i = 0; i < MAX_VOXEL_STEPS; i++) {
        if (any(vPos < aabbMin) || any(vPos > aabbMax)) { break; }

        let pVal = voxelBuffer[cOffset + u32(vPos.x) + (u32(vPos.y) << 3u) + (u32(vPos.z) << 6u)];
        if pVal > 0u {
            let colorData = (pVal >> 9u) & 0xFFFFu;
            result.hit    = true;
            result.color  = decodeRGB565(colorData);
            result.normal = vec3f(
                select(0.0, -sign(rd.x), ha == 0),
                select(0.0, -sign(rd.y), ha == 1),
                select(0.0, -sign(rd.z), ha == 2),
            );
            result.t = tEntry + cT;
            return result;
        }

        if tMax.x < tMax.y && tMax.x < tMax.z {
            cT += tMax.x; vPos.x += rayStep.x; tMax.x += tDelta.x; ha = 0;
        } else if tMax.y < tMax.z {
            cT += tMax.y; vPos.y += rayStep.y; tMax.y += tDelta.y; ha = 1;
        } else {
            cT += tMax.z; vPos.z += rayStep.z; tMax.z += tDelta.z; ha = 2;
        }
    }
    return result;
}

// ── Fragment stage ────────────────────────────────────────────────────────────

struct FSOut {
    @location(0)            color: vec4f,
    @builtin(frag_depth)    depth: f32,
};

@fragment
fn fs_main(in: VSOut, @builtin(front_facing) isFront: bool) -> FSOut {
    let chunkWorldSize    = uni.gridMin.w;
    let chunkWorldSizeInv = uni.gridDims.w;
    let gridMin           = uni.gridMin.xyz;
    let gridDims          = uni.gridDims.xyz;
    let camPos            = uni.cameraPos.xyz;

    // Ray in chunk-unit space
    let camPosCU       = (camPos - gridMin) * chunkWorldSizeInv;
    let isCameraInside = all(camPosCU >= vec3f(0.0)) && all(camPosCU <= gridDims);

    // Discard back faces when outside, front faces when inside
    if  isCameraInside && isFront  { discard; }
    if !isCameraInside && !isFront { discard; }

    let rayDirWS  = normalize(in.worldPos - camPos);
    let rayDirCU  = rayDirWS * chunkWorldSizeInv;
    let rayDirInv = 1.0 / (rayDirCU + EPS * sign(rayDirCU));

    let assetHit = rayBoxIntersect(camPosCU, rayDirInv, vec3f(0.0), gridDims);
    if assetHit.x > assetHit.y { discard; }

    let gDims = vec3i(gridDims);
    var sPos  = camPosCU + rayDirCU * assetHit.x;
    var gPos  = clamp(vec3i(floor(sPos)), vec3i(0), gDims - vec3i(1));
    let gStep = vec3i(sign(rayDirCU));
    let tDelta = abs(1.0 / (rayDirCU + EPS * sign(rayDirCU)));
    let nBounds = vec3f(gPos + max(vec3i(0), gStep));

    var tMax = vec3f(
        select(1e30, (nBounds.x - sPos.x) / rayDirCU.x, abs(rayDirCU.x) > EPS),
        select(1e30, (nBounds.y - sPos.y) / rayDirCU.y, abs(rayDirCU.y) > EPS),
        select(1e30, (nBounds.z - sPos.z) / rayDirCU.z, abs(rayDirCU.z) > EPS),
    );

    var out: FSOut;

    for (var i = 0; i < MAX_CHUNK_STEPS; i++) {
        if (u32(gPos.x) >= u32(gDims.x) || u32(gPos.y) >= u32(gDims.y) || u32(gPos.z) >= u32(gDims.z)) { break; }

        let ci    = u32(gPos.x) + u32(gPos.y) * u32(gDims.x) + u32(gPos.z) * u32(gDims.x) * u32(gDims.y);
        let cInfo = chunkGrid[ci];

        if cInfo.x != 0xFFFFFFFFu {
            let bounds  = unpackAABB(cInfo.y & 0x3FFFFu);
            let aabbMin = bounds[0];
            let aabbMax = bounds[1];

            let t0    = (vec3f(gPos) + vec3f(aabbMin) * 0.125 - camPosCU) * rayDirInv;
            let t1    = (vec3f(gPos) + vec3f(aabbMax + vec3i(1)) * 0.125 - camPosCU) * rayDirInv;
            let tmin  = min(t0, t1);
            let tmax2 = max(t0, t1);
            let tNear = max(tmin.x, max(tmin.y, tmin.z));
            let tFar  = min(tmax2.x, min(tmax2.y, tmax2.z));

            if max(0.0, tNear) <= tFar {
                let hAxis = select(select(2, 1, tmin.y >= tmin.z), 0, tmin.x >= tmin.y && tmin.x >= tmin.z);
                let vHit  = rayMarchChunk(camPosCU, rayDirCU, gPos, cInfo.x * 512u, aabbMin, aabbMax, max(0.0, tNear), hAxis);

                if vHit.hit {
                    // Reconstruct world position from chunk-unit t
                    let hPosCU = camPosCU + rayDirCU * vHit.t;
                    let hPosWS = gridMin + hPosCU * chunkWorldSize;

                    // Clip-space depth
                    let clipPos = uni.mvp * vec4f(hPosWS, 1.0);
                    out.depth   = clipPos.z / clipPos.w;

                    // Simple Lambertian + ambient lighting
                    let lightDir = normalize(vec3f(0.5, 1.0, 0.3));
                    let albedo   = srgbToLinear(vHit.color);
                    let ndotl    = max(dot(vHit.normal, lightDir), 0.0);
                    let lit      = albedo * (0.25 + ndotl * 0.75);

                    out.color = vec4f(lit, 1.0);
                    return out;
                }
            }
        }

        if tMax.x < tMax.y && tMax.x < tMax.z {
            gPos.x += gStep.x; tMax.x += tDelta.x;
        } else if tMax.y < tMax.z {
            gPos.y += gStep.y; tMax.y += tDelta.y;
        } else {
            gPos.z += gStep.z; tMax.z += tDelta.z;
        }
    }

    discard;
}
