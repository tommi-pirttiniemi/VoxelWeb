// GPU voxelization pipeline — exact port of BoxVoxelizerRuntime.VoxelizeHDSparse
//
// Pipeline per call:
//   1. initializeGrid  — clears voxel buffer, marks edge voxels as Air
//   2. voxelizeSurface — per-triangle, one dispatch per mesh primitive
//   3. resolveAverage  — thresholds, computes average color
//   4. CPU readback    — extract Shell voxels (state==1, colorPacked!=0)
//
// No flood-fill / fill-inside (matches GlbToBoxConverter: fillInside=false).

import voxelizeWgsl from './shaders/voxelize.wgsl?raw';
import type { MeshPrimitive } from './glb-parse';

// ── Constants matching Unity ──────────────────────────────────────────────────

const ALPHA_CUTOFF    = 0.5;
const EDGE_THRESHOLD  = 0.6;
const AVERAGE_COLOR   = 1;          // 1 = average, 0 = Boyer-Moore mode

// ── Output type ───────────────────────────────────────────────────────────────

export interface VoxelGrid {
  data:      Uint32Array;            // flat, indexed x + dx*(y + dy*z)
  dx: number; dy: number; dz: number;
  gridMin:   [number, number, number];
  voxelSize: number;
}

// Voxel struct stride in bytes (7 × u32/i32 = 28)
const VOXEL_STRIDE = 28;

// ── Public entry point ────────────────────────────────────────────────────────

export async function voxelizeMesh(
  device: GPUDevice,
  primitives: MeshPrimitive[],
  resolution = 64,
): Promise<VoxelGrid> {

  // ── Global AABB across all primitives ──────────────────────────────────────
  let mnx = Infinity, mny = Infinity, mnz = Infinity;
  let mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (const prim of primitives) {
    const p = prim.positions;
    for (let i = 0; i < p.length; i += 3) {
      if (p[i]   < mnx) mnx = p[i];   if (p[i]   > mxx) mxx = p[i];
      if (p[i+1] < mny) mny = p[i+1]; if (p[i+1] > mxy) mxy = p[i+1];
      if (p[i+2] < mnz) mnz = p[i+2]; if (p[i+2] > mxz) mxz = p[i+2];
    }
  }

  // Uniform voxel size: largest axis = resolution voxels
  const span      = Math.max(mxx - mnx, mxy - mny, mxz - mnz) * 1.005; // tiny margin
  const voxelSize = span / resolution;
  const cx = (mnx + mxx) / 2, cy = (mny + mxy) / 2, cz = (mnz + mxz) / 2;
  const gridMin: [number, number, number] = [cx - span / 2, cy - span / 2, cz - span / 2];

  // Per-axis grid size (may be non-cubic for non-square meshes)
  const dx = Math.max(1, Math.ceil((mxx - mnx) / voxelSize));
  const dy = Math.max(1, Math.ceil((mxy - mny) / voxelSize));
  const dz = Math.max(1, Math.ceil((mxz - mnz) / voxelSize));
  const totalVoxels = dx * dy * dz;

  // ── Params uniform (48 bytes, std140) ─────────────────────────────────────
  const uniformData = new ArrayBuffer(48);
  const u = new DataView(uniformData);
  u.setFloat32( 0, gridMin[0],    true);
  u.setFloat32( 4, gridMin[1],    true);
  u.setFloat32( 8, gridMin[2],    true);
  u.setFloat32(12, voxelSize,     true);
  u.setUint32( 16, dx,            true);
  u.setUint32( 20, dy,            true);
  u.setUint32( 24, dz,            true);
  u.setUint32( 28, 0,             true);   // triCount — written per dispatch
  u.setFloat32(32, ALPHA_CUTOFF,  true);
  u.setFloat32(36, EDGE_THRESHOLD,true);
  u.setUint32( 40, AVERAGE_COLOR, true);
  u.setUint32( 44, 0,             true);   // pad

  const uniformBuf = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuf, 0, uniformData);

  // ── Voxel grid buffer ─────────────────────────────────────────────────────
  const gridBuf = device.createBuffer({
    size: totalVoxels * VOXEL_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // ── Compile shader (once) ─────────────────────────────────────────────────
  const module = device.createShaderModule({ code: voxelizeWgsl });

  const compInfo = await module.getCompilationInfo();
  for (const msg of compInfo.messages) {
    const tag = msg.type === 'error' ? '🔴 WGSL' : '⚠️ WGSL';
    console.error(`${tag} line ${msg.lineNum}:${msg.linePos} — ${msg.message}`);
  }
  if (compInfo.messages.some(m => m.type === 'error')) {
    throw new Error('Voxelize WGSL compilation failed');
  }

  // ── Pipelines (3 entry points, each with its own auto-layout) ─────────────
  const [initPipeline, surfacePipeline, resolvePipeline] = await Promise.all([
    device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'initializeGrid' },
    }),
    device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'voxelizeSurface' },
    }),
    device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'resolveAverage' },
    }),
  ]);

  // ── Bind groups for init and resolve (bindings 0+1 only) ──────────────────
  const initBindGroup = device.createBindGroup({
    layout: initPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: gridBuf } },
    ],
  });

  const resolveBindGroup = device.createBindGroup({
    layout: resolvePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: gridBuf } },
    ],
  });

  // ── Workgroup counts for 4³ kernels (4×4×4=64 ≤ 256 default limit) ───────
  const wgX = Math.ceil(dx / 4);
  const wgY = Math.ceil(dy / 4);
  const wgZ = Math.ceil(dz / 4);

  // ── Dispatch: initializeGrid ───────────────────────────────────────────────
  {
    const enc  = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(initPipeline);
    pass.setBindGroup(0, initBindGroup);
    pass.dispatchWorkgroups(wgX, wgY, wgZ);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // ── Dispatch: voxelizeSurface — one pass per primitive ────────────────────
  for (const prim of primitives) {
    const triCount = prim.indices.length / 3;
    if (triCount === 0) continue;

    // Upload geometry
    const vertBuf = device.createBuffer({
      size: prim.positions.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertBuf, 0, prim.positions);

    // Indices as u32
    const idxBuf = device.createBuffer({
      size: prim.indices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(idxBuf, 0, prim.indices);

    // UVs
    const uvBuf = device.createBuffer({
      size: prim.uvs.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uvBuf, 0, prim.uvs);

    // Texture
    const gpuTex = await makeTexture(device, prim);

    // Write triCount into uniform at offset 28
    const triCountBuf = new Uint32Array([triCount]);
    device.queue.writeBuffer(uniformBuf, 28, triCountBuf);

    const surfBindGroup = device.createBindGroup({
      layout: surfacePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: gridBuf } },
        { binding: 2, resource: { buffer: vertBuf } },
        { binding: 3, resource: { buffer: idxBuf } },
        { binding: 4, resource: { buffer: uvBuf } },
        { binding: 5, resource: gpuTex.createView() },
      ],
    });

    const enc  = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(surfacePipeline);
    pass.setBindGroup(0, surfBindGroup);
    pass.dispatchWorkgroups(Math.ceil(triCount / 64));
    pass.end();
    device.queue.submit([enc.finish()]);

    // Cleanup per-primitive buffers (GPU still running; destroy is deferred)
    vertBuf.destroy();
    idxBuf.destroy();
    uvBuf.destroy();
    gpuTex.destroy();
  }

  // ── Dispatch: resolveAverage ───────────────────────────────────────────────
  {
    const enc  = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(resolvePipeline);
    pass.setBindGroup(0, resolveBindGroup);
    pass.dispatchWorkgroups(wgX, wgY, wgZ);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // ── Readback ───────────────────────────────────────────────────────────────
  const readBuf = device.createBuffer({
    size: totalVoxels * VOXEL_STRIDE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(gridBuf, 0, readBuf, 0, totalVoxels * VOXEL_STRIDE);
    device.queue.submit([enc.finish()]);
  }

  await readBuf.mapAsync(GPUMapMode.READ);
  const raw = new DataView(readBuf.getMappedRange());

  // ── Extract Shell voxels → flat output grid ───────────────────────────────
  // Voxel layout: accR(0) accG(4) accB(8) count(12) modeCandidate(16) state(20) colorPacked(24)
  // Output: (rgb565 << 9) | 1   for filled voxels, 0 for empty
  // Index:  x + dx*(y + dy*z)  — same as shader getIndex

  const outputGrid = new Uint32Array(dx * dy * dz);

  for (let z = 0; z < dz; z++) {
    for (let y = 0; y < dy; y++) {
      for (let x = 0; x < dx; x++) {
        const voxIdx  = x + dx * (y + dy * z);
        const base    = voxIdx * VOXEL_STRIDE;
        const state   = raw.getInt32(base + 20, true);
        const packed  = raw.getUint32(base + 24, true);

        if (state === 1 && packed !== 0) {
          const r8 = (packed >>> 24) & 0xFF;
          const g8 = (packed >>> 16) & 0xFF;
          const b8 = (packed >>>  8) & 0xFF;
          const rgb565 = ((r8 >> 3) << 11) | ((g8 >> 2) << 5) | (b8 >> 3);
          outputGrid[voxIdx] = (rgb565 << 9) | 1;
        }
      }
    }
  }

  readBuf.unmap();

  // Cleanup
  uniformBuf.destroy();
  gridBuf.destroy();
  readBuf.destroy();

  return { data: outputGrid, dx, dy, dz, gridMin, voxelSize };
}

// ── Upload primitive texture to GPU ──────────────────────────────────────────

async function makeTexture(device: GPUDevice, prim: MeshPrimitive): Promise<GPUTexture> {
  if (prim.textureData) {
    try {
      const blob   = new Blob([prim.textureData], { type: prim.mimeType || 'image/jpeg' });
      const bitmap = await createImageBitmap(blob);
      const tex = device.createTexture({
        size:   [bitmap.width, bitmap.height, 1],
        format: 'rgba8unorm',
        usage:  GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST        |
                GPUTextureUsage.RENDER_ATTACHMENT,
      });
      device.queue.copyExternalImageToTexture(
        { source: bitmap, flipY: false },
        { texture: tex },
        [bitmap.width, bitmap.height, 1],
      );
      bitmap.close();
      return tex;
    } catch (e) {
      console.warn('Texture decode failed, using fallback color', e);
    }
  }

  // Fallback: 1×1 texture with the primitive's base color
  const [r, g, b] = prim.color;
  const tex = device.createTexture({
    size:   [1, 1, 1],
    format: 'rgba8unorm',
    usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: tex },
    new Uint8Array([Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 255]),
    { bytesPerRow: 4 },
    [1, 1, 1],
  );
  return tex;
}
