// Vectorized Q8_0 GEMM compute shaders (SM 5.1, compiled via inbox d3dcompiler_47.dll).
// Computes Y[M, N] = X[M, K] * dequant(W[N, K])^T (+ bias) with optional fused epilogue.
// Two tile sizes; ParakeetGpu picks by M, using the small one until the large one has compiled. Both compile to the same chain of mads in k order, so they give
// the same bits on the GPUs tested (a driver may still contract mads differently in the two programs):
//
// Q8GemmMain: 64 x 64 outputs, BK = 32 (exact Q8_0 block size), 16 x 8 = 128 threads.
//   Each thread computes 8 rows x 4 columns: rows ty*4..+3 and 32+ty*4..+3, columns tx*4..+3.
//   Shared memory: sX4[32][16] + sW4[32][16] float4 = 16 KB.
// Q8GemmSmallMain: 32 x 32 outputs, 8 x 8 = 64 threads, 4 x 4 outputs each, 8 KB shared memory.
//   Short utterances: twice the thread groups keep a large GPU busy.
//
// Measured 2026-11-22 against the single 32 x 32 kernel used before (Spike --gemm-bench, encoder
// shapes per layer; frames = rows of most of the layer's multiplications):
//   frames                 25     64     96     128    250    1500
//   RTX 5070 Ti     small  1.09x  1.03x  1.01x  1.01x  1.02x  1.00x
//                   large  0.82x  0.86x  0.95x  1.03x  1.38x  1.54x
//   Intel Graphics  small  1.05x  1.10x  1.11x  1.09x  1.11x  1.12x
//                   large  0.88x  1.54x  1.29x  1.65x  1.66x  1.61x
// Both load four rows under one bounds check, as the old kernel did. A check per row was faster on the
// RTX (small kernel 1.26x at 25 frames instead of 1.09x) and slower on the Intel GPU (0.94x, not 1.05x).
// FXC's [unroll(n)] with n below the trip count runs only n iterations, so the k loops unroll fully.

cbuffer Params : register(b0)
{
    uint M;
    uint N;
    uint K;
    uint hasBias;
    uint epilogue; // 0 = store, 1 = SiLU, 2 = Y += out, 3 = Y += 0.5 * out
    uint xByteOffset;
    uint yByteOffset;
    uint pad0;
};

ByteAddressBuffer X    : register(t0);
ByteAddressBuffer Wq   : register(t1);
ByteAddressBuffer Wd   : register(t2);
ByteAddressBuffer Bias : register(t3);

RWByteAddressBuffer Y  : register(u0);

// [k][g]: k within the block, g = group of 4 consecutive rows (of X) or columns (of W).
groupshared float4 sX4[32][16];
groupshared float4 sW4[32][16];

float4 UnpackS8x4(uint packed, float scale)
{
    int wi = asint(packed);
    return float4(
        float((wi << 24) >> 24),
        float((wi << 16) >> 24),
        float((wi <<  8) >> 24),
        float( wi        >> 24)) * scale;
}

float LoadScale(uint gn, uint kBlocks, uint kb)
{
    uint dIdx = gn * kBlocks + kb;
    uint dWord = Wd.Load((dIdx >> 1) << 2);
    return f16tof32((dIdx & 1) != 0 ? (dWord >> 16) : (dWord & 0xFFFF));
}

// Real [branch] guards prevent FXC from hoisting out-of-bounds root SRV loads.
float4 LoadX4(uint gm, uint kStart)
{
    float4 r = 0;
    [branch] if (gm < M) r = asfloat(X.Load4(xByteOffset + (gm * K + kStart) * 4));
    return r;
}

float4 LoadW4(uint gn, uint kStart, uint kBlocks, uint kb)
{
    float4 r = 0;
    [branch] if (gn < N) r = UnpackS8x4(Wq.Load(gn * K + kStart), LoadScale(gn, kBlocks, kb));
    return r;
}

// Four consecutive rows with one bounds check while all four are inside.
void LoadX4Rows(uint row, uint kStart, out float4 r0, out float4 r1, out float4 r2, out float4 r3)
{
    [branch]
    if (row + 3 < M)
    {
        r0 = asfloat(X.Load4(xByteOffset + ((row + 0) * K + kStart) * 4));
        r1 = asfloat(X.Load4(xByteOffset + ((row + 1) * K + kStart) * 4));
        r2 = asfloat(X.Load4(xByteOffset + ((row + 2) * K + kStart) * 4));
        r3 = asfloat(X.Load4(xByteOffset + ((row + 3) * K + kStart) * 4));
    }
    else
    {
        r0 = LoadX4(row + 0, kStart); r1 = LoadX4(row + 1, kStart); r2 = LoadX4(row + 2, kStart); r3 = 0;
    }
}

void LoadW4Rows(uint row, uint kStart, uint kBlocks, uint kb, out float4 r0, out float4 r1, out float4 r2, out float4 r3)
{
    [branch]
    if (row + 3 < N)
    {
        r0 = UnpackS8x4(Wq.Load((row + 0) * K + kStart), LoadScale(row + 0, kBlocks, kb));
        r1 = UnpackS8x4(Wq.Load((row + 1) * K + kStart), LoadScale(row + 1, kBlocks, kb));
        r2 = UnpackS8x4(Wq.Load((row + 2) * K + kStart), LoadScale(row + 2, kBlocks, kb));
        r3 = UnpackS8x4(Wq.Load((row + 3) * K + kStart), LoadScale(row + 3, kBlocks, kb));
    }
    else
    {
        r0 = LoadW4(row + 0, kStart, kBlocks, kb); r1 = LoadW4(row + 1, kStart, kBlocks, kb); r2 = LoadW4(row + 2, kStart, kBlocks, kb); r3 = 0;
    }
}

float4 LoadBias4(uint gn)
{
    float4 bias = 0;
    [branch]
    if (hasBias != 0)
    {
        [branch]
        if (gn + 3 < N) bias = asfloat(Bias.Load4(gn << 2));
        else
        {
            [branch] if (gn + 0 < N) bias.x = asfloat(Bias.Load((gn + 0) << 2));
            [branch] if (gn + 1 < N) bias.y = asfloat(Bias.Load((gn + 1) << 2));
            [branch] if (gn + 2 < N) bias.z = asfloat(Bias.Load((gn + 2) << 2));
        }
    }
    return bias;
}

void StoreRow(uint gm, uint gn, float4 v)
{
    if (gm >= M) return;
    if (gn + 3 < N)
    {
        uint outByte = yByteOffset + ((gm * N + gn) << 2);
        if (epilogue == 1) v = v / (1.0f + exp(-v));
        else if (epilogue == 2) v = asfloat(Y.Load4(outByte)) + v;
        else if (epilogue == 3) v = asfloat(Y.Load4(outByte)) + 0.5f * v;
        Y.Store4(outByte, asuint(v));
    }
    else
    {
        [unroll]
        for (uint c = 0; c < 4; c++)
        {
            if (gn + c >= N) break;
            uint outByte = yByteOffset + ((gm * N + gn + c) << 2);
            float val = v[c];
            if (epilogue == 1) val = val / (1.0f + exp(-val));
            else if (epilogue == 2) val = asfloat(Y.Load(outByte)) + val;
            else if (epilogue == 3) val = asfloat(Y.Load(outByte)) + 0.5f * val;
            Y.Store(outByte, asuint(val));
        }
    }
}

[numthreads(16, 8, 1)]
void Q8GemmMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tx = gtid.x; // 0..15
    const uint ty = gtid.y; // 0..7
    const uint tid = ty * 16 + tx;

    const uint groupN = gid.x * 64;
    const uint groupM = gid.y * 64;
    const uint kBlocks = K >> 5;

    // Each thread stages 4 rows x 4 k of X and the same of W, transposed into one float4 per k:
    //   g    = tid >> 3 (0..15, group of 4 rows)
    //   kCol = 4 * (tid & 7) (0..28)
    const uint g = tid >> 3;
    const uint kCol = (tid & 7) << 2;
    const uint xRow = groupM + (g << 2);
    const uint wRow = groupN + (g << 2);

    float4 top0 = 0, top1 = 0, top2 = 0, top3 = 0;
    float4 bottom0 = 0, bottom1 = 0, bottom2 = 0, bottom3 = 0;
    for (uint kb = 0; kb < kBlocks; kb++)
    {
        const uint kStart = (kb << 5) + kCol;

        float4 rx0, rx1, rx2, rx3;
        LoadX4Rows(xRow, kStart, rx0, rx1, rx2, rx3);
        sX4[kCol + 0][g] = float4(rx0.x, rx1.x, rx2.x, rx3.x);
        sX4[kCol + 1][g] = float4(rx0.y, rx1.y, rx2.y, rx3.y);
        sX4[kCol + 2][g] = float4(rx0.z, rx1.z, rx2.z, rx3.z);
        sX4[kCol + 3][g] = float4(rx0.w, rx1.w, rx2.w, rx3.w);

        float4 rw0, rw1, rw2, rw3;
        LoadW4Rows(wRow, kStart, kBlocks, kb, rw0, rw1, rw2, rw3);
        sW4[kCol + 0][g] = float4(rw0.x, rw1.x, rw2.x, rw3.x);
        sW4[kCol + 1][g] = float4(rw0.y, rw1.y, rw2.y, rw3.y);
        sW4[kCol + 2][g] = float4(rw0.z, rw1.z, rw2.z, rw3.z);
        sW4[kCol + 3][g] = float4(rw0.w, rw1.w, rw2.w, rw3.w);

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < 32; k++)
        {
            float4 top = sX4[k][ty], bottom = sX4[k][ty + 8], w = sW4[k][tx];
            top0 += top.x * w; top1 += top.y * w; top2 += top.z * w; top3 += top.w * w;
            bottom0 += bottom.x * w; bottom1 += bottom.y * w; bottom2 += bottom.z * w; bottom3 += bottom.w * w;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    const uint gn = groupN + (tx << 2);
    const float4 bias = LoadBias4(gn);
    const uint gmTop = groupM + (ty << 2), gmBottom = gmTop + 32;
    StoreRow(gmTop + 0, gn, top0 + bias);
    StoreRow(gmTop + 1, gn, top1 + bias);
    StoreRow(gmTop + 2, gn, top2 + bias);
    StoreRow(gmTop + 3, gn, top3 + bias);
    StoreRow(gmBottom + 0, gn, bottom0 + bias);
    StoreRow(gmBottom + 1, gn, bottom1 + bias);
    StoreRow(gmBottom + 2, gn, bottom2 + bias);
    StoreRow(gmBottom + 3, gn, bottom3 + bias);
}

groupshared float4 sX4Small[32][8];
groupshared float4 sW4Small[32][8];

[numthreads(8, 8, 1)]
void Q8GemmSmallMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tx = gtid.x; // 0..7
    const uint ty = gtid.y; // 0..7
    const uint tid = ty * 8 + tx; // 0..63

    const uint groupN = gid.x * 32;
    const uint groupM = gid.y * 32;
    const uint kBlocks = K >> 5;

    float4 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    // 64 threads load 32x32 tiles: g = tid >> 3 (group of 4 rows), kCol = 4 * (tid & 7).
    const uint g = tid >> 3;
    const uint kCol = (tid & 7) << 2;
    const uint xRow = groupM + (g << 2);
    const uint wRow = groupN + (g << 2);

    for (uint kb = 0; kb < kBlocks; kb++)
    {
        const uint kStart = (kb << 5) + kCol;

        float4 rx0, rx1, rx2, rx3;
        LoadX4Rows(xRow, kStart, rx0, rx1, rx2, rx3);
        sX4Small[kCol + 0][g] = float4(rx0.x, rx1.x, rx2.x, rx3.x);
        sX4Small[kCol + 1][g] = float4(rx0.y, rx1.y, rx2.y, rx3.y);
        sX4Small[kCol + 2][g] = float4(rx0.z, rx1.z, rx2.z, rx3.z);
        sX4Small[kCol + 3][g] = float4(rx0.w, rx1.w, rx2.w, rx3.w);

        float4 rw0, rw1, rw2, rw3;
        LoadW4Rows(wRow, kStart, kBlocks, kb, rw0, rw1, rw2, rw3);
        sW4Small[kCol + 0][g] = float4(rw0.x, rw1.x, rw2.x, rw3.x);
        sW4Small[kCol + 1][g] = float4(rw0.y, rw1.y, rw2.y, rw3.y);
        sW4Small[kCol + 2][g] = float4(rw0.z, rw1.z, rw2.z, rw3.z);
        sW4Small[kCol + 3][g] = float4(rw0.w, rw1.w, rw2.w, rw3.w);

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < 32; k++)
        {
            float4 vx = sX4Small[k][ty], vw = sW4Small[k][tx];
            acc0 += vx.x * vw; acc1 += vx.y * vw; acc2 += vx.z * vw; acc3 += vx.w * vw;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    const uint gn = groupN + (tx << 2);
    const float4 bias = LoadBias4(gn);
    const uint gm = groupM + (ty << 2);
    StoreRow(gm + 0, gn, acc0 + bias);
    StoreRow(gm + 1, gn, acc1 + bias);
    StoreRow(gm + 2, gn, acc2 + bias);
    StoreRow(gm + 3, gn, acc3 + bias);
}
