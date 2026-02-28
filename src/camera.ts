// Orbit camera with mouse drag + scroll zoom.
// Outputs a view-projection matrix (column-major Float32Array, 16 floats)
// and the world-space camera position.

export class OrbitCamera {
  yaw   = 0.3;
  pitch = 0.4;
  dist  = 4.0;
  target= [0, 0, 0] as [number, number, number];

  private dragging = false;
  private lastX = 0;
  private lastY = 0;

  constructor(canvas: HTMLCanvasElement) {
    canvas.addEventListener('mousedown', e => { this.dragging = true; this.lastX = e.clientX; this.lastY = e.clientY; });
    window.addEventListener('mouseup',   () => { this.dragging = false; });
    window.addEventListener('mousemove', e => {
      if (!this.dragging) return;
      this.yaw   += (e.clientX - this.lastX) * 0.005;
      this.pitch  = Math.max(-1.4, Math.min(1.4, this.pitch + (e.clientY - this.lastY) * 0.005));
      this.lastX = e.clientX; this.lastY = e.clientY;
    });
    canvas.addEventListener('wheel', e => {
      this.dist = Math.max(0.5, Math.min(50, this.dist * (1 + e.deltaY * 0.001)));
    }, { passive: true });
    // Touch
    let lastDist = 0;
    canvas.addEventListener('touchstart', e => {
      if (e.touches.length === 1) { this.dragging = true; this.lastX = e.touches[0].clientX; this.lastY = e.touches[0].clientY; }
      if (e.touches.length === 2) { lastDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); }
    });
    canvas.addEventListener('touchend',   () => { this.dragging = false; });
    canvas.addEventListener('touchmove',  e => {
      if (e.touches.length === 1 && this.dragging) {
        this.yaw   += (e.touches[0].clientX - this.lastX) * 0.005;
        this.pitch  = Math.max(-1.4, Math.min(1.4, this.pitch + (e.touches[0].clientY - this.lastY) * 0.005));
        this.lastX = e.touches[0].clientX; this.lastY = e.touches[0].clientY;
      }
      if (e.touches.length === 2) {
        const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
        this.dist = Math.max(0.5, Math.min(50, this.dist * (lastDist / d)));
        lastDist = d;
      }
    }, { passive: true });
  }

  /** Returns camera world position */
  getPosition(): [number, number, number] {
    const x = Math.cos(this.pitch) * Math.sin(this.yaw) * this.dist + this.target[0];
    const y = Math.sin(this.pitch)                       * this.dist + this.target[1];
    const z = Math.cos(this.pitch) * Math.cos(this.yaw) * this.dist + this.target[2];
    return [x, y, z];
  }

  /** Column-major 4×4 view-projection matrix suitable for WebGPU (z in [0,1]) */
  getMVP(aspect: number): Float32Array {
    const pos = this.getPosition();
    const view = lookAt(pos, this.target, [0, 1, 0]);
    const proj = perspective(Math.PI / 4, aspect, 0.01, 1000);
    return mat4mul(proj, view);
  }
}

// ── Math helpers ─────────────────────────────────────────────────────────────

function lookAt(eye: [number,number,number], center: [number,number,number], up: [number,number,number]): Float32Array {
  const f = norm3(sub3(center, eye));
  const s = norm3(cross3(f, up));
  const u = cross3(s, f);
  const m = new Float32Array(16);
  m[0]=s[0]; m[4]=s[1]; m[8]=s[2];   m[12]=-dot3(s,eye);
  m[1]=u[0]; m[5]=u[1]; m[9]=u[2];   m[13]=-dot3(u,eye);
  m[2]=-f[0];m[6]=-f[1];m[10]=-f[2]; m[14]= dot3(f,eye);
  m[3]=0;    m[7]=0;    m[11]=0;      m[15]=1;
  return m;
}

/** WebGPU perspective: right-handed, z in [0,1] */
function perspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovy / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far / (near - far);
  m[11] = -1;
  m[14] = (near * far) / (near - far);
  return m;
}

function mat4mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
    for (let k = 0; k < 4; k++) o[j*4+i] += a[k*4+i] * b[j*4+k];
  return o;
}

type V3 = [number,number,number];
function sub3(a: V3, b: V3): V3  { return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function dot3(a: V3, b: V3): number { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function cross3(a: V3, b: V3): V3 { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function norm3(a: V3): V3 { const l=Math.sqrt(dot3(a,a)); return [a[0]/l,a[1]/l,a[2]/l]; }
