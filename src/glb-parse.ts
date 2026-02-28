// Minimal GLB/glTF parser.
// Extracts all mesh primitives with positions, indices, UVs, and embedded texture bytes.
// Handles interleaved vertex buffers and Uint16/Uint32 index buffers.

export interface MeshPrimitive {
  positions:   Float32Array;          // flat xyz, length = vertexCount * 3
  indices:     Uint32Array;
  uvs:         Float32Array;          // flat uv,  length = vertexCount * 2
  textureData: Uint8Array | null;     // raw JPEG/PNG bytes from GLB binary chunk
  mimeType:    string;                // 'image/jpeg' | 'image/png' | ''
  color:       [number, number, number]; // baseColorFactor fallback
}

const COMPONENT_BYTES: Record<number, number> = {
  5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4,
};
const TYPE_COMPONENTS: Record<string, number> = {
  SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16,
};

export function parseGlb(buffer: ArrayBuffer): MeshPrimitive[] {
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

  if (!jsonChunk) throw new Error('No JSON chunk in GLB');
  const gltf = JSON.parse(jsonChunk);
  const bin  = binChunk ?? new ArrayBuffer(0);

  // ── Accessor reader ──────────────────────────────────────────────────────
  function readAccessor(idx: number): { data: ArrayBuffer; componentType: number; count: number; type: string } {
    const acc  = gltf.accessors[idx];
    const bv   = gltf.bufferViews[acc.bufferView ?? 0];
    const byteOffset     = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const componentBytes = COMPONENT_BYTES[acc.componentType] ?? 4;
    const numComponents  = TYPE_COMPONENTS[acc.type] ?? 1;
    const stride         = bv.byteStride ?? (componentBytes * numComponents);
    let data: ArrayBuffer;
    if (stride === componentBytes * numComponents) {
      data = bin.slice(byteOffset, byteOffset + acc.count * stride);
    } else {
      // Interleaved — deinterleave into a packed buffer
      const elemSize = componentBytes * numComponents;
      const out = new ArrayBuffer(acc.count * elemSize);
      const dst = new Uint8Array(out);
      const src = new Uint8Array(bin);
      for (let i = 0; i < acc.count; i++) {
        dst.set(
          src.subarray(byteOffset + i * stride, byteOffset + i * stride + elemSize),
          i * elemSize,
        );
      }
      data = out;
    }
    return { data, componentType: acc.componentType, count: acc.count, type: acc.type };
  }

  // ── Image extractor ──────────────────────────────────────────────────────
  function extractImage(imgIdx: number): { data: Uint8Array; mimeType: string } | null {
    const img = gltf.images?.[imgIdx];
    if (!img) return null;
    if (img.bufferView !== undefined) {
      const bv   = gltf.bufferViews[img.bufferView];
      const data = new Uint8Array(bin, bv.byteOffset ?? 0, bv.byteLength);
      const mime = img.mimeType ?? detectMimeType(data);
      return { data: new Uint8Array(data), mimeType: mime };
    }
    return null;  // external URI not supported
  }

  function detectMimeType(data: Uint8Array): string {
    if (data[0] === 0xFF && data[1] === 0xD8) return 'image/jpeg';
    if (data[0] === 0x89 && data[1] === 0x50) return 'image/png';
    return 'image/png';
  }

  // ── Parse each mesh primitive ─────────────────────────────────────────────
  const primitives: MeshPrimitive[] = [];

  for (const mesh of (gltf.meshes ?? [])) {
    for (const prim of (mesh.primitives ?? [])) {
      // Positions
      const posAcc   = readAccessor(prim.attributes.POSITION);
      const positions = new Float32Array(posAcc.data);

      // Indices
      let indices: Uint32Array;
      if (prim.indices !== undefined) {
        const idxAcc = readAccessor(prim.indices);
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
        indices = new Uint32Array(positions.length / 3);
        for (let i = 0; i < indices.length; i++) indices[i] = i;
      }

      // UVs (TEXCOORD_0)
      let uvs: Float32Array;
      if (prim.attributes.TEXCOORD_0 !== undefined) {
        const uvAcc = readAccessor(prim.attributes.TEXCOORD_0);
        if (uvAcc.componentType === 5126) {
          uvs = new Float32Array(uvAcc.data);
        } else {
          // Normalised unsigned short UVs → float
          const u16 = new Uint16Array(uvAcc.data);
          uvs = new Float32Array(u16.length);
          const scale = uvAcc.componentType === 5123 ? 1 / 65535 : 1 / 255;
          for (let i = 0; i < u16.length; i++) uvs[i] = u16[i] * scale;
        }
      } else {
        // No UVs — fill with zeros
        uvs = new Float32Array((positions.length / 3) * 2);
      }

      // Material / texture
      let textureData: Uint8Array | null = null;
      let mimeType = '';
      let color: [number, number, number] = [0.7, 0.7, 0.7];

      try {
        const matIdx = prim.material ?? 0;
        const mat    = gltf.materials?.[matIdx];
        const pbr    = mat?.pbrMetallicRoughness;

        const texRef = pbr?.baseColorTexture;
        if (texRef !== undefined) {
          const texInfo = gltf.textures?.[texRef.index];
          if (texInfo?.source !== undefined) {
            const img = extractImage(texInfo.source);
            if (img) { textureData = img.data; mimeType = img.mimeType; }
          }
        }

        const bcf = pbr?.baseColorFactor;
        if (Array.isArray(bcf) && bcf.length >= 3) {
          color = [bcf[0], bcf[1], bcf[2]];
        }
      } catch { /* ignore missing material */ }

      primitives.push({ positions, indices, uvs, textureData, mimeType, color });
    }
  }

  if (primitives.length === 0) throw new Error('No mesh primitives found in GLB');
  return primitives;
}
