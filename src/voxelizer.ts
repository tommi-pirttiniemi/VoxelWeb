// GPU voxelization pipeline.
// Runs a compute shader to fill a flat u32 grid, then reads it back
// to CPU so pack-chunks.ts can build the chunk format for the renderer.

import voxelizeWgsl from './shaders/voxelize.wgsl?raw';
import type { MeshGeometry } from './glb-parse';

export interface VoxelGrid {
  data: Uint32Array;
  dx: number; dy: number; dz: number;
  gridMin: [number, number, number];
  voxelSize: number;
}

/** Voxelize `mesh` into a `resolution`³ grid using the GPU. */
export async function voxelizeMesh(
  device: GPUDevice,
  mesh: MeshGeometry,
  resolution = 64
): Promise<VoxelGrid> {
  // ── Compute AABB of mesh ─────────────────────────────────────────────────
  const pos = mesh.positions;
  let mnx = Infinity, mny = Infinity, mnz = Infinity;
  let mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let i = 0; i < pos.length; i += 3) {
    if (pos[i]   < mnx) mnx = pos[i];   if (pos[i]   > mxx) mxx = pos[i];
    if (pos[i+1] < mny) mny = pos[i+1]; if (pos[i+1] > mxy) mxy = pos[i+1];
    if (pos[i+2] < mnz) mnz = pos[i+2]; if (pos[i+2] > mxz) mxz = pos[i+2];
  }
  // Uniform voxel size so cubes are cubes
  const span     = Math.max(mxx-mnx, mxy-mny, mxz-mnz) * 1.02; // 1% margin
  const voxelSize = span / resolution;
  const cx = (mnx+mxx)/2, cy = (mny+mxy)/2, cz = (mnz+mxz)/2;
  const gridMin: [number,number,number] = [cx - span/2, cy - span/2, cz - span/2];
  const dx = resolution, dy = resolution, dz = resolution;
  const triCount = mesh.indices.length / 3;

  // ── Encode default voxel color from mesh base color ──────────────────────
  const [r, g, b] = mesh.color;
  const r5  = Math.round(r * 31)  & 0x1F;
  const g6  = Math.round(g * 63)  & 0x3F;
  const b5  = Math.round(b * 31)  & 0x1F;
  const rgb565 = (r5 << 11) | (g6 << 5) | b5;
  const colorPacked = (rgb565 << 9) | 1;

  // ── GPU buffers ──────────────────────────────────────────────────────────
  const uniformData = new ArrayBuffer(48);
  const uv = new DataView(uniformData);
  // gridMin (vec4f): xyz + w=voxelSize
  uv.setFloat32(0,  gridMin[0], true); uv.setFloat32(4,  gridMin[1], true);
  uv.setFloat32(8,  gridMin[2], true); uv.setFloat32(12, voxelSize,  true);
  // gridDims (vec4u): xyz + w=triCount
  uv.setUint32(16, dx, true); uv.setUint32(20, dy, true);
  uv.setUint32(24, dz, true); uv.setUint32(28, triCount, true);
  // color + padding
  uv.setUint32(32, colorPacked, true);

  const uniformBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniformBuf, 0, uniformData);

  const vertBuf = device.createBuffer({ size: pos.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertBuf, 0, pos);

  const idxData = mesh.indices;
  const idxBuf  = device.createBuffer({ size: idxData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(idxBuf, 0, idxData);

  const gridBytes = dx * dy * dz * 4;
  const gridBuf   = device.createBuffer({ size: gridBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const readBuf = device.createBuffer({ size: gridBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  // ── Pipeline ─────────────────────────────────────────────────────────────
  const module = device.createShaderModule({ code: voxelizeWgsl });
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: vertBuf } },
      { binding: 2, resource: { buffer: idxBuf } },
      { binding: 3, resource: { buffer: gridBuf } },
    ],
  });

  // ── Dispatch ─────────────────────────────────────────────────────────────
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(dx * dy * dz / 64));
  pass.end();
  enc.copyBufferToBuffer(gridBuf, 0, readBuf, 0, gridBytes);
  device.queue.submit([enc.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  // Destroy temporary buffers
  uniformBuf.destroy(); vertBuf.destroy(); idxBuf.destroy();
  gridBuf.destroy(); readBuf.destroy();

  return { data: result, dx, dy, dz, gridMin, voxelSize };
}
