// Converts a flat voxel grid (readback from GPU) into the two-buffer chunk format
// that the raytracer expects:
//   chunkGrid:   Uint32Array  — 2 u32 per chunk, [offset, packedAABB]
//   voxelBuffer: Uint32Array  — 512 u32 per chunk (8×8×8 voxels)
//
// Chunk index:  cx + cy*cdx + cz*cdx*cdy
// Voxel index within chunk: lx + (ly<<3) + (lz<<6)
// Packed AABB: minX|(minY<<3)|(minZ<<6)|(maxX<<9)|(maxY<<12)|(maxZ<<15)  (3 bits each)

export interface ChunkBuffers {
  chunkGrid:    Uint32Array;   // 2*totalChunks u32s
  voxelBuffer:  Uint32Array;   // 512*filledChunks u32s
  cdx: number; cdy: number; cdz: number;
}

export function packChunks(
  flatGrid: Uint32Array,
  dx: number, dy: number, dz: number
): ChunkBuffers {
  const cdx = Math.ceil(dx / 8);
  const cdy = Math.ceil(dy / 8);
  const cdz = Math.ceil(dz / 8);
  const totalChunks = cdx * cdy * cdz;

  // First pass — which chunks are non-empty?
  const filledFlags = new Uint8Array(totalChunks);
  let filledCount = 0;

  for (let cz = 0; cz < cdz; cz++) {
    for (let cy = 0; cy < cdy; cy++) {
      for (let cx = 0; cx < cdx; cx++) {
        let filled = false;
        outer: for (let lz = 0; lz < 8; lz++) {
          for (let ly = 0; ly < 8; ly++) {
            for (let lx = 0; lx < 8; lx++) {
              const gx = cx * 8 + lx, gy = cy * 8 + ly, gz = cz * 8 + lz;
              if (gx < dx && gy < dy && gz < dz && flatGrid[gx + gy * dx + gz * dx * dy] > 0) {
                filled = true;
                break outer;
              }
            }
          }
        }
        const ci = cx + cy * cdx + cz * cdx * cdy;
        if (filled) { filledFlags[ci] = 1; filledCount++; }
      }
    }
  }

  // Allocate output buffers (all 0xFFFFFFFF = empty sentinel for chunkGrid.x)
  const chunkGrid   = new Uint32Array(totalChunks * 2).fill(0xFFFFFFFF);
  const voxelBuffer = new Uint32Array(filledCount * 512);

  let offset = 0;
  for (let cz = 0; cz < cdz; cz++) {
    for (let cy = 0; cy < cdy; cy++) {
      for (let cx = 0; cx < cdx; cx++) {
        const ci = cx + cy * cdx + cz * cdx * cdy;
        if (!filledFlags[ci]) continue;

        chunkGrid[ci * 2] = offset;   // chunk data offset

        let mnx = 7, mny = 7, mnz = 7, mxx = 0, mxy = 0, mxz = 0;

        for (let lz = 0; lz < 8; lz++) {
          for (let ly = 0; ly < 8; ly++) {
            for (let lx = 0; lx < 8; lx++) {
              const gx = cx * 8 + lx, gy = cy * 8 + ly, gz = cz * 8 + lz;
              let v = 0;
              if (gx < dx && gy < dy && gz < dz) v = flatGrid[gx + gy * dx + gz * dx * dy];
              voxelBuffer[offset * 512 + lx + (ly << 3) + (lz << 6)] = v;
              if (v > 0) {
                if (lx < mnx) mnx = lx; if (lx > mxx) mxx = lx;
                if (ly < mny) mny = ly; if (ly > mxy) mxy = ly;
                if (lz < mnz) mnz = lz; if (lz > mxz) mxz = lz;
              }
            }
          }
        }

        const aabb = mnx | (mny << 3) | (mnz << 6) | (mxx << 9) | (mxy << 12) | (mxz << 15);
        chunkGrid[ci * 2 + 1] = aabb;
        offset++;
      }
    }
  }

  return { chunkGrid, voxelBuffer, cdx, cdy, cdz };
}
