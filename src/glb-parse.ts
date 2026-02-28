// Minimal GLB parser — extracts positions, indices, and optional vertex colors.
// Only reads the first mesh primitive. Sufficient for Meshy output.

export interface MeshGeometry {
  positions: Float32Array;   // flat xyz
  indices:   Uint32Array;
  color:     [number, number, number];  // base material color [0..1]
}

const GLTF_COMPONENT = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 } as Record<number,number>;
const GLTF_TYPE_COUNT = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 } as Record<string,number>;

export function parseGlb(buffer: ArrayBuffer): MeshGeometry {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== 0x46546C67) throw new Error('Not a GLB file');

  let jsonChunk: string | null = null;
  let binChunk: ArrayBuffer | null = null;
  let offset = 12;

  while (offset < buffer.byteLength) {
    const len  = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const data = buffer.slice(offset + 8, offset + 8 + len);
    if (type === 0x4E4F534A) jsonChunk = new TextDecoder().decode(data);
    if (type === 0x004E4942) binChunk  = data;
    offset += 8 + len;
  }

  if (!jsonChunk) throw new Error('No JSON chunk');
  const gltf = JSON.parse(jsonChunk);
  const bin  = binChunk ?? new ArrayBuffer(0);

  function accessor(idx: number): { data: ArrayBuffer; componentType: number; count: number; type: string } {
    const acc  = gltf.accessors[idx];
    const bv   = gltf.bufferViews[acc.bufferView ?? 0];
    const byteOffset = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const componentBytes = GLTF_COMPONENT[acc.componentType] ?? 4;
    const components     = GLTF_TYPE_COUNT[acc.type] ?? 1;
    const stride = bv.byteStride ?? (componentBytes * components);
    let data: ArrayBuffer;
    if (stride === componentBytes * components) {
      data = bin.slice(byteOffset, byteOffset + acc.count * stride);
    } else {
      // Interleaved — copy out
      const out = new ArrayBuffer(acc.count * componentBytes * components);
      const dst = new Uint8Array(out);
      const src = new Uint8Array(bin);
      for (let i = 0; i < acc.count; i++) {
        dst.set(src.slice(byteOffset + i * stride, byteOffset + i * stride + componentBytes * components), i * componentBytes * components);
      }
      data = out;
    }
    return { data, componentType: acc.componentType, count: acc.count, type: acc.type };
  }

  const prim = gltf.meshes[0].primitives[0];

  // Positions
  const posAcc = accessor(prim.attributes.POSITION);
  const positions = new Float32Array(posAcc.data);

  // Indices — normalise to Uint32Array
  let indices: Uint32Array;
  if (prim.indices !== undefined) {
    const idxAcc = accessor(prim.indices);
    if (idxAcc.componentType === 5125) {
      indices = new Uint32Array(idxAcc.data);
    } else if (idxAcc.componentType === 5123) {
      const u16 = new Uint16Array(idxAcc.data);
      indices = new Uint32Array(u16.length);
      for (let i = 0; i < u16.length; i++) indices[i] = u16[i];
    } else {
      const u8 = new Uint8Array(idxAcc.data);
      indices = new Uint32Array(u8.length);
      for (let i = 0; i < u8.length; i++) indices[i] = u8[i];
    }
  } else {
    // Non-indexed — generate sequential indices
    indices = new Uint32Array(positions.length / 3);
    for (let i = 0; i < indices.length; i++) indices[i] = i;
  }

  // Base color from material
  let color: [number, number, number] = [0.7, 0.7, 0.7];
  try {
    const matIdx = prim.material ?? 0;
    const mat    = gltf.materials?.[matIdx];
    const bcf    = mat?.pbrMetallicRoughness?.baseColorFactor;
    if (Array.isArray(bcf) && bcf.length >= 3) color = [bcf[0], bcf[1], bcf[2]];
  } catch { /* ignore */ }

  return { positions, indices, color };
}
