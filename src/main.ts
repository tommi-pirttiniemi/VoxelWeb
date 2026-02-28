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
  const canvas         = document.getElementById('c')          as HTMLCanvasElement;
  const loadBtn        = document.getElementById('load-btn')   as HTMLButtonElement;
  const fileInput      = document.getElementById('file-input') as HTMLInputElement;
  const statusEl       = document.getElementById('status')     as HTMLDivElement;
  const noWebGpu       = document.getElementById('no-webgpu')  as HTMLDivElement;
  const resolutionEl   = document.getElementById('resolution') as HTMLSelectElement;
  const paletteEl      = document.getElementById('palette')    as HTMLSelectElement;
  const metallicEl     = document.getElementById('metallic')   as HTMLInputElement;
  const metallicVal    = document.getElementById('metallic-val')   as HTMLSpanElement;
  const smoothnessEl   = document.getElementById('smoothness') as HTMLInputElement;
  const smoothnessVal  = document.getElementById('smoothness-val') as HTMLSpanElement;
  const ambientEl      = document.getElementById('ambient')    as HTMLInputElement;
  const ambientVal     = document.getElementById('ambient-val')    as HTMLSpanElement;
  const emissiveEl     = document.getElementById('emissive')   as HTMLInputElement;

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
  let   lastBuffer: ArrayBuffer | null = null;

  // ── File loading ──────────────────────────────────────────────────────────
  async function loadGlb(buffer: ArrayBuffer) {
    loadBtn.disabled = true;
    lastBuffer = buffer;
    const resolution = parseInt(resolutionEl.value, 10);
    statusEl.textContent = 'Parsing GLB…';
    try {
      const primitives = parseGlb(buffer);
      const totalTris  = primitives.reduce((s, p) => s + p.indices.length / 3, 0);
      statusEl.textContent = `Voxelizing (${primitives.length} primitive(s), ${totalTris | 0} triangles, res=${resolution})…`;
      const grid   = await voxelizeMesh(device, primitives, resolution);
      statusEl.textContent = 'Packing chunks…';
      const chunks = packChunks(grid.data, grid.dx, grid.dy, grid.dz);
      statusEl.textContent = 'Building pipeline…';
      await renderer.loadScene(grid, chunks);

      // Auto-frame camera on the grid centre
      const maxSpan = Math.max(grid.dx, grid.dy, grid.dz) * grid.voxelSize;
      camera.dist   = maxSpan * 2.0;
      camera.target = [
        grid.gridMin[0] + (grid.dx * grid.voxelSize) / 2,
        grid.gridMin[1] + (grid.dy * grid.voxelSize) / 2,
        grid.gridMin[2] + (grid.dz * grid.voxelSize) / 2,
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

  // ── Rendering controls ────────────────────────────────────────────────────

  // Resolution: re-voxelize with new resolution
  resolutionEl.addEventListener('change', () => {
    if (lastBuffer) loadGlb(lastBuffer);
  });

  // Palette
  paletteEl.addEventListener('change', () => {
    renderer.setRenderParams({ paletteMode: parseInt(paletteEl.value, 10) });
  });

  // Metallic
  metallicEl.addEventListener('input', () => {
    const v = parseFloat(metallicEl.value);
    metallicVal.textContent = v.toFixed(2);
    renderer.setRenderParams({ metallic: v });
  });

  // Smoothness
  smoothnessEl.addEventListener('input', () => {
    const v = parseFloat(smoothnessEl.value);
    smoothnessVal.textContent = v.toFixed(2);
    renderer.setRenderParams({ smoothness: v });
  });

  // Ambient
  ambientEl.addEventListener('input', () => {
    const v = parseFloat(ambientEl.value);
    ambientVal.textContent = v.toFixed(2);
    renderer.setRenderParams({ ambient: v });
  });

  // Emissive
  emissiveEl.addEventListener('change', () => {
    renderer.setRenderParams({ emissive: emissiveEl.checked });
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
