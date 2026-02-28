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
// Total: 144 bytes

struct Uniforms {
    mvp:          mat4x4f,        // offset   0 (64 bytes)
    // xyz = grid world-space min, w = chunkWorldSize
    gridMin:      vec4f,          // offset  64
    // xyz = chunk grid dimensions (float), w = chunkWorldSizeInv
    gridDims:     vec4f,          // offset  80
    cameraPos:    vec4f,          // offset  96  xyz = world pos, w unused
    lightDir:     vec4f,          // offset 112  xyz = sun dir (normalized), w = ambient strength
    metallic:     f32,            // offset 128
    smoothness:   f32,            // offset 132
    emissive:     u32,            // offset 136  0=off 1=on
    paletteMode:  u32,            // offset 140  0=Full565 1=PICO8 2=C64 3=EGA 4=Gray8 5=Gray32 6=Web216
};

// ── Storage buffers ───────────────────────────────────────────────────────────

// uint2 per chunk: x = voxel buffer chunk offset (0xFFFFFFFF = empty),
//                  y = packed AABB (18 bits: minX|minY|minZ|maxX|maxY|maxZ, 3 bits each)
@group(0) @binding(0) var<uniform>       uni:         Uniforms;
@group(0) @binding(1) var<storage, read> chunkGrid:   array<vec2u>;
@group(0) @binding(2) var<storage, read> voxelBuffer: array<u32>;

// ── Vertex stage ──────────────────────────────────────────────────────────────

struct VSOut {
    @builtin(position) posH:     vec4f,
    @location(0)       worldPos: vec3f,
}

// Unit cube — arrays inside function so runtime indexing is safe
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var cubeIdx = array<u32, 36>(
        0u,2u,1u, 1u,2u,3u,   // -Z
        4u,5u,6u, 5u,7u,6u,   // +Z
        0u,1u,4u, 1u,5u,4u,   // -Y
        2u,6u,3u, 3u,6u,7u,   // +Y
        0u,4u,2u, 2u,4u,6u,   // -X
        1u,3u,5u, 3u,7u,5u,   // +X
    );
    var cubePos = array<vec3f, 8>(
        vec3f(-0.5, -0.5, -0.5),
        vec3f( 0.5, -0.5, -0.5),
        vec3f(-0.5,  0.5, -0.5),
        vec3f( 0.5,  0.5, -0.5),
        vec3f(-0.5, -0.5,  0.5),
        vec3f( 0.5, -0.5,  0.5),
        vec3f(-0.5,  0.5,  0.5),
        vec3f( 0.5,  0.5,  0.5),
    );
    let posUnit = cubePos[cubeIdx[vi]];

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

struct ChunkAABB { mn: vec3i, mx: vec3i }

fn unpackAABB(aabb: u32) -> ChunkAABB {
    return ChunkAABB(
        vec3i(i32(aabb        & 7u), i32((aabb >> 3u)  & 7u), i32((aabb >> 6u)  & 7u)),
        vec3i(i32((aabb >> 9u) & 7u), i32((aabb >> 12u) & 7u), i32((aabb >> 15u) & 7u)),
    );
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

// ── Palette quantization ──────────────────────────────────────────────────────
// Palettes are in sRGB [0..1] space, matching voxel colors from decodeRGB565.

fn quantizePico8(c: vec3f) -> vec3f {
    var pal = array<vec3f, 16>(
        vec3f(0.0000, 0.0000, 0.0000), vec3f(0.1137, 0.1686, 0.3255),
        vec3f(0.4941, 0.1451, 0.3255), vec3f(0.0000, 0.5294, 0.3176),
        vec3f(0.6706, 0.3216, 0.2118), vec3f(0.3725, 0.3412, 0.3098),
        vec3f(0.7608, 0.7647, 0.7804), vec3f(1.0000, 0.9451, 0.9098),
        vec3f(1.0000, 0.0000, 0.3020), vec3f(1.0000, 0.6392, 0.0000),
        vec3f(1.0000, 0.9255, 0.1529), vec3f(0.0000, 0.8941, 0.2118),
        vec3f(0.1608, 0.6784, 1.0000), vec3f(0.5137, 0.4627, 0.6118),
        vec3f(1.0000, 0.4667, 0.6588), vec3f(1.0000, 0.8000, 0.6667),
    );
    var best = pal[0];
    var bestD = dot(c - pal[0], c - pal[0]);
    for (var i = 1u; i < 16u; i += 1u) {
        let d = dot(c - pal[i], c - pal[i]);
        if d < bestD { bestD = d; best = pal[i]; }
    }
    return best;
}

fn quantizeC64(c: vec3f) -> vec3f {
    var pal = array<vec3f, 16>(
        vec3f(0.0000, 0.0000, 0.0000), vec3f(1.0000, 1.0000, 1.0000),
        vec3f(0.5333, 0.2235, 0.1961), vec3f(0.4039, 0.7137, 0.7412),
        vec3f(0.5451, 0.2471, 0.5882), vec3f(0.3333, 0.6275, 0.2863),
        vec3f(0.2510, 0.1922, 0.5529), vec3f(0.7490, 0.8078, 0.4471),
        vec3f(0.5451, 0.3294, 0.1608), vec3f(0.3412, 0.2588, 0.0000),
        vec3f(0.7216, 0.4118, 0.3843), vec3f(0.3137, 0.3137, 0.3137),
        vec3f(0.4706, 0.4706, 0.4706), vec3f(0.5804, 0.8784, 0.5373),
        vec3f(0.4706, 0.4118, 0.7686), vec3f(0.6235, 0.6235, 0.6235),
    );
    var best = pal[0];
    var bestD = dot(c - pal[0], c - pal[0]);
    for (var i = 1u; i < 16u; i += 1u) {
        let d = dot(c - pal[i], c - pal[i]);
        if d < bestD { bestD = d; best = pal[i]; }
    }
    return best;
}

fn quantizeEGA(c: vec3f) -> vec3f {
    var pal = array<vec3f, 16>(
        vec3f(0.0000, 0.0000, 0.0000), vec3f(0.0000, 0.0000, 0.6667),
        vec3f(0.0000, 0.6667, 0.0000), vec3f(0.0000, 0.6667, 0.6667),
        vec3f(0.6667, 0.0000, 0.0000), vec3f(0.6667, 0.0000, 0.6667),
        vec3f(0.6667, 0.3333, 0.0000), vec3f(0.6667, 0.6667, 0.6667),
        vec3f(0.3333, 0.3333, 0.3333), vec3f(0.3333, 0.3333, 1.0000),
        vec3f(0.3333, 1.0000, 0.3333), vec3f(0.3333, 1.0000, 1.0000),
        vec3f(1.0000, 0.3333, 0.3333), vec3f(1.0000, 0.3333, 1.0000),
        vec3f(1.0000, 1.0000, 0.3333), vec3f(1.0000, 1.0000, 1.0000),
    );
    var best = pal[0];
    var bestD = dot(c - pal[0], c - pal[0]);
    for (var i = 1u; i < 16u; i += 1u) {
        let d = dot(c - pal[i], c - pal[i]);
        if d < bestD { bestD = d; best = pal[i]; }
    }
    return best;
}

fn applyPalette(c: vec3f, mode: u32) -> vec3f {
    switch mode {
        case 1u: { return quantizePico8(c); }
        case 2u: { return quantizeC64(c); }
        case 3u: { return quantizeEGA(c); }
        case 4u: {
            // Gray 8 (8 steps)
            let g = round(dot(c, vec3f(0.299, 0.587, 0.114)) * 7.0) / 7.0;
            return vec3f(g);
        }
        case 5u: {
            // Gray 32 (32 steps)
            let g = round(dot(c, vec3f(0.299, 0.587, 0.114)) * 31.0) / 31.0;
            return vec3f(g);
        }
        case 6u: {
            // Web 216: 6 steps per channel (0, 51, 102, 153, 204, 255)
            return round(c * 5.0) / 5.0;
        }
        default: { return c; }
    }
}

// ── Lighting ──────────────────────────────────────────────────────────────────
// Blinn-Phong with metallic/dielectric workflow.
// Reads metallic, smoothness, and ambient from uniforms.

fn litColor(albedo: vec3f, N: vec3f, L: vec3f, V: vec3f) -> vec3f {
    let H        = normalize(L + V);
    let ndotl    = max(dot(N, L), 0.0);
    let ndoth    = max(dot(N, H), 0.0);
    let specPow  = exp2(uni.smoothness * 10.0 + 1.0);
    let F0       = mix(vec3f(0.04), albedo, uni.metallic);
    let diff     = albedo * (1.0 - uni.metallic) * ndotl;
    let spec     = select(vec3f(0.0), F0 * pow(ndoth, specPow), ndotl > 0.0);
    let amb      = uni.lightDir.w;
    // Scale direct light by (1-ambient) so the total at full ndotl ≈ albedo
    return albedo * amb + (diff + spec) * (1.0 - amb);
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

    for (var i = 0; i < MAX_VOXEL_STEPS; i += 1) {
        if any(vPos < aabbMin) || any(vPos > aabbMax) { break; }

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
            cT = tMax.x; vPos.x += rayStep.x; tMax.x += tDelta.x; ha = 0;
        } else if tMax.y < tMax.z {
            cT = tMax.y; vPos.y += rayStep.y; tMax.y += tDelta.y; ha = 1;
        } else {
            cT = tMax.z; vPos.z += rayStep.z; tMax.z += tDelta.z; ha = 2;
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
    var out: FSOut;  // declare first so all early returns can use it

    let chunkWorldSize    = uni.gridMin.w;
    let chunkWorldSizeInv = uni.gridDims.w;
    let gridMin           = uni.gridMin.xyz;
    let gridDims          = uni.gridDims.xyz;
    let camPos            = uni.cameraPos.xyz;

    // Ray in chunk-unit space
    let camPosCU       = (camPos - gridMin) * chunkWorldSizeInv;
    let isCameraInside = all(camPosCU >= vec3f(0.0)) && all(camPosCU <= gridDims);

    // Discard back faces when outside, front faces when inside
    if  isCameraInside && isFront  { discard; return out; }
    if !isCameraInside && !isFront { discard; return out; }

    let rayDirWS  = normalize(in.worldPos - camPos);
    let rayDirCU  = rayDirWS * chunkWorldSizeInv;
    let rayDirInv = 1.0 / (rayDirCU + EPS * sign(rayDirCU));

    let assetHit = rayBoxIntersect(camPosCU, rayDirInv, vec3f(0.0), gridDims);
    if assetHit.x > assetHit.y { discard; return out; }

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

    for (var i = 0; i < MAX_CHUNK_STEPS; i += 1) {
        if u32(gPos.x) >= u32(gDims.x) || u32(gPos.y) >= u32(gDims.y) || u32(gPos.z) >= u32(gDims.z) { break; }

        let ci    = u32(gPos.x) + u32(gPos.y) * u32(gDims.x) + u32(gPos.z) * u32(gDims.x) * u32(gDims.y);
        let cInfo = chunkGrid[ci];

        if cInfo.x != 0xFFFFFFFFu {
            let ab    = unpackAABB(cInfo.y & 0x3FFFFu);
            let t0    = (vec3f(gPos) + vec3f(ab.mn) * 0.125 - camPosCU) * rayDirInv;
            let t1    = (vec3f(gPos) + vec3f(ab.mx + vec3i(1)) * 0.125 - camPosCU) * rayDirInv;
            let tmin  = min(t0, t1);
            let tmax2 = max(t0, t1);
            let tNear = max(tmin.x, max(tmin.y, tmin.z));
            let tFar  = min(tmax2.x, min(tmax2.y, tmax2.z));

            if max(0.0, tNear) <= tFar {
                let hAxis = select(select(2, 1, tmin.y >= tmin.z), 0, tmin.x >= tmin.y && tmin.x >= tmin.z);
                let vHit  = rayMarchChunk(camPosCU, rayDirCU, gPos, cInfo.x * 512u, ab.mn, ab.mx, max(0.0, tNear), hAxis);

                if vHit.hit {
                    // Reconstruct world position from chunk-unit t
                    let hPosCU = camPosCU + rayDirCU * vHit.t;
                    let hPosWS = gridMin + hPosCU * chunkWorldSize;

                    // Clip-space depth
                    let clipPos = uni.mvp * vec4f(hPosWS, 1.0);
                    out.depth   = clipPos.z / clipPos.w;

                    // Palette (applied in sRGB space, matching Unity DecodeVoxelColor)
                    let paletteColor = applyPalette(vHit.color, uni.paletteMode);

                    // Convert to linear for lighting
                    let albedo = srgbToLinear(paletteColor);

                    // PBR Blinn-Phong lighting
                    let L   = normalize(uni.lightDir.xyz);
                    let V   = normalize(camPos - hPosWS);
                    var lit = litColor(albedo, vHit.normal, L, V);

                    // Emissive: voxel colour acts as self-emission (×4 matching Unity)
                    if uni.emissive != 0u {
                        lit = lit + albedo * 4.0;
                    }

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

    // No voxel hit — discard fragment.
    // Explicit return required by WGSL control flow rules even though discard terminates.
    discard;
    return out;
}
