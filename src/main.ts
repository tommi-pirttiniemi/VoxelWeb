import { parseGlb }       from './glb-parse';
import { voxelizeMesh }   from './voxelizer';
import { packChunks }     from './pack-chunks';
import { VoxelRenderer }  from './renderer';
import { OrbitCamera }    from './camera';

// ── WebGPU init ───────────────────────────────────────────────────────────────

async function initWebGPU(canvas: HTMLCanvasElement): Promise<{ device: GPUDevice; context: GPUCanvasContext }> {
  if (!navigator.gpu) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter');
  const device  = await adapter.requestDevice();
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'premultiplied' });
  return { device, context };
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const canvas    = document.getElementById('c') as HTMLCanvasElement;
  const loadBtn   = document.getElementById('load-btn') as HTMLButtonElement;
  const fileInput = document.getElementById('file-input') as HTMLInputElement;
  const statusEl  = document.getElementById('status') as HTMLDivElement;
  const noWebGpu  = document.getElementById('no-webgpu') as HTMLDivElement;

  // Size canvas to window
  const resize = () => {
    canvas.width  = canvas.clientWidth  * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;
  };
  resize();
  window.addEventListener('resize', resize);

  let device: GPUDevice;
  let context: GPUCanvasContext;
  try {
    ({ device, context } = await initWebGPU(canvas));
  } catch {
    noWebGpu.style.display = 'flex';
    return;
  }

  const renderer = new VoxelRenderer(device, context);
  const camera   = new OrbitCamera(canvas);
  let   hasScene = false;

  // ── File loading ──────────────────────────────────────────────────────────
  async function loadGlb(buffer: ArrayBuffer) {
    loadBtn.disabled = true;
    statusEl.textContent = 'Parsing GLB…';
    try {
      const mesh = parseGlb(buffer);
      statusEl.textContent = `Voxelizing (${mesh.indices.length / 3 | 0} triangles)…`;
      const grid   = await voxelizeMesh(device, mesh, 64);
      statusEl.textContent = 'Packing chunks…';
      const chunks = packChunks(grid.data, grid.dx, grid.dy, grid.dz);
      statusEl.textContent = 'Building pipeline…';
      await renderer.loadScene(grid, chunks);

      // Auto-frame camera
      const span = grid.voxelSize * grid.dx;
      camera.dist   = span * 1.8;
      camera.target = [
        grid.gridMin[0] + span / 2,
        grid.gridMin[1] + span / 2,
        grid.gridMin[2] + span / 2,
      ];
      hasScene = true;
      const filled = grid.data.filter(v => v > 0).length;
      statusEl.textContent = `${filled.toLocaleString()} voxels — drag to orbit, scroll to zoom`;
    } catch (e) {
      statusEl.textContent = `Error: ${e}`;
    }
    loadBtn.disabled = false;
  }

  loadBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    file.arrayBuffer().then(loadGlb);
  });

  // Drag-and-drop onto canvas
  canvas.addEventListener('dragover', e => e.preventDefault());
  canvas.addEventListener('drop', e => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0];
    if (file) file.arrayBuffer().then(loadGlb);
  });

  // ── Render loop ───────────────────────────────────────────────────────────
  function frame() {
    if (hasScene) {
      const aspect = canvas.width / canvas.height;
      const mvp    = camera.getMVP(aspect);
      const pos    = camera.getPosition();
      renderer.render(mvp, pos);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();
