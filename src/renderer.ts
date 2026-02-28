// WebGPU render pipeline for two-level DDA voxel raymarcher.

import raytraceWgsl from './shaders/raytrace.wgsl?raw';
import type { ChunkBuffers } from './pack-chunks';
import type { VoxelGrid } from './voxelizer';

export class VoxelRenderer {
  private device:       GPUDevice;
  private context:      GPUCanvasContext;
  private pipeline:     GPURenderPipeline | null = null;
  private bindGroup:    GPUBindGroup      | null = null;
  private uniformBuf:   GPUBuffer         | null = null;
  private depthTex:     GPUTexture        | null = null;
  private depthView:    GPUTextureView    | null = null;
  private width  = 0;
  private height = 0;

  // Cached scene info for uniform updates
  private gridMin    = new Float32Array(3);
  private gridDims   = new Float32Array(3);
  private chunkWorldSize    = 1;
  private chunkWorldSizeInv = 1;

  constructor(device: GPUDevice, context: GPUCanvasContext) {
    this.device  = device;
    this.context = context;
  }

  /** Upload new voxel data and build the render pipeline. */
  async loadScene(grid: VoxelGrid, chunks: ChunkBuffers): Promise<void> {
    const { device } = this;

    // Cleanup old buffers
    this.uniformBuf?.destroy();
    this.pipeline   = null;
    this.bindGroup  = null;

    const { gridMin, voxelSize, dx } = grid;
    const { cdx, cdy, cdz, chunkGrid, voxelBuffer } = chunks;

    const chunkWorldSize    = voxelSize * 8;
    const chunkWorldSizeInv = 1.0 / chunkWorldSize;
    this.gridMin    = new Float32Array(gridMin);
    this.gridDims   = new Float32Array([cdx, cdy, cdz]);
    this.chunkWorldSize    = chunkWorldSize;
    this.chunkWorldSizeInv = chunkWorldSizeInv;

    // ── Uniform buffer (112 bytes) ────────────────────────────────────────
    // Layout (std140, all vec3 padded to vec4):
    //   mat4x4f mvp          (64 bytes, offset 0)
    //   vec4f   gridMin      (16 bytes, offset 64)  xyz + w=chunkWorldSize
    //   vec4f   gridDims     (16 bytes, offset 80)  xyz + w=chunkWorldSizeInv
    //   vec4f   cameraPos    (16 bytes, offset 96)  xyz + w=0
    const uBuf = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uniformBuf = uBuf;

    // ── Storage buffers ───────────────────────────────────────────────────
    const chunkBuf = device.createBuffer({
      size: chunkGrid.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(chunkBuf, 0, chunkGrid);

    const voxBuf = device.createBuffer({
      size: Math.max(voxelBuffer.byteLength, 4),  // never 0-byte
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(voxBuf, 0, voxelBuffer);

    // ── Shader + pipeline ─────────────────────────────────────────────────
    const module = device.createShaderModule({ code: raytraceWgsl });

    // Surface actual WGSL compile errors (if any) before pipeline creation
    const info = await module.getCompilationInfo();
    for (const msg of info.messages) {
      const tag = msg.type === 'error' ? '🔴 WGSL' : '⚠️ WGSL';
      console.error(`${tag} line ${msg.lineNum}:${msg.linePos} — ${msg.message}`);
    }
    if (info.messages.some(m => m.type === 'error')) {
      throw new Error('WGSL compilation failed — see console for details');
    }

    this.pipeline = await device.createRenderPipelineAsync({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uBuf } },
        { binding: 1, resource: { buffer: chunkBuf } },
        { binding: 2, resource: { buffer: voxBuf } },
      ],
    });
  }

  /** Call every frame. `mvp` = column-major 4×4, `camPos` = world position. */
  render(mvp: Float32Array, camPos: [number, number, number]): void {
    const { device, pipeline, bindGroup, uniformBuf } = this;
    if (!pipeline || !bindGroup || !uniformBuf) return;

    const canvas = this.context.canvas as HTMLCanvasElement;
    if (canvas.width !== this.width || canvas.height !== this.height) this.resizeDepth(canvas.width, canvas.height);
    if (!this.depthView) return;

    // Write uniform buffer
    const u = new Float32Array(28);           // 112 bytes
    u.set(mvp, 0);                            // mvp [0..15]
    u[16] = this.gridMin[0]; u[17] = this.gridMin[1]; u[18] = this.gridMin[2];
    u[19] = this.chunkWorldSize;              // gridMin.w
    u[20] = this.gridDims[0]; u[21] = this.gridDims[1]; u[22] = this.gridDims[2];
    u[23] = this.chunkWorldSizeInv;           // gridDims.w
    u[24] = camPos[0]; u[25] = camPos[1]; u[26] = camPos[2];
    u[27] = 0;
    device.queue.writeBuffer(uniformBuf, 0, u);

    const colorView = this.context.getCurrentTexture().createView();
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: colorView,
        loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0.08, g: 0.08, b: 0.10, a: 1 },
      }],
      depthStencilAttachment: {
        view: this.depthView!,
        depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
      },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(36);    // unit cube: 12 triangles × 3 verts from const arrays in shader
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  private resizeDepth(w: number, h: number): void {
    this.depthTex?.destroy();
    this.width = w; this.height = h;
    this.depthTex = this.device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthView = this.depthTex.createView();
  }
}
