cbuffer Params : register(b0)
{
    uint P0;
    uint P1;
    uint P2;
    uint P3;
    uint P4;
    uint P5;
    uint P6;
    uint P7;
};

ByteAddressBuffer In0 : register(t0);
ByteAddressBuffer In1 : register(t1);
ByteAddressBuffer In2 : register(t2);
ByteAddressBuffer In3 : register(t3);

RWByteAddressBuffer Out0 : register(u0);

static const float LayerNormEpsilon = 1e-5f;
static const uint MaxGroupsPerDispatchDimension = 65535u;

static const uint AttnKTileStart = 256u;
static const uint AttnPosTileStart = 544u;
static const uint AttnPaddedRowFloat4s = 9u;

groupshared float sLnRed[256];

[numthreads(256, 1, 1)]
void LayerNormMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint row = gid.x;
    const uint D = P0;
    const uint rowBase = row * D;

    float sum = 0.0f;
    for (uint i = tid; i < D; i += 256)
        sum += asfloat(In0.Load((rowBase + i) << 2));
    sLnRed[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid < s) sLnRed[tid] += sLnRed[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = sLnRed[0] / (float)D;
    GroupMemoryBarrierWithGroupSync();

    float varSum = 0.0f;
    for (uint j = tid; j < D; j += 256)
    {
        float diff = asfloat(In0.Load((rowBase + j) << 2)) - mean;
        varSum += diff * diff;
    }
    sLnRed[tid] = varSum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s2 = 128; s2 > 0; s2 >>= 1)
    {
        if (tid < s2) sLnRed[tid] += sLnRed[tid + s2];
        GroupMemoryBarrierWithGroupSync();
    }
    float invStd = 1.0f / sqrt(sLnRed[0] / (float)D + LayerNormEpsilon);

    for (uint k = tid; k < D; k += 256)
    {
        float x = asfloat(In0.Load((rowBase + k) << 2));
        float g = asfloat(In1.Load(k << 2));
        float b = asfloat(In2.Load(k << 2));
        float y = (x - mean) * invStd * g + b;
        Out0.Store((rowBase + k) << 2, asuint(y));
    }
}

float ExpTreatingNaNAsZero(float nonPositive)
{
    return exp(min(nonPositive, 0.0f));
}

groupshared float4 sTile[1120];
groupshared float sS[32][33];
groupshared float4 sU[32];
groupshared float4 sVb[32];
groupshared float sUK[32];
groupshared float sVP[64];
groupshared float sM[32];
groupshared float sL[32];
groupshared float sAlpha[32];

[numthreads(16, 16, 1)]
void RelPosAttentionMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tx = gtid.x;
    const uint ty = gtid.y;
    const uint tid = ty * 16u + tx;
    const uint i0 = gid.x * 32u;
    const uint hd = gid.y;
    const uint T = P0;
    const uint D = P1;
    const uint dk = P2;
    const float scale = asfloat(P3);
    const uint ho = hd * dk;
    const uint dk4 = dk >> 2;
    const uint posRows = 2u * T - 1u;
    const uint stride3D = 3u * D;

    if (tid < 32u)
    {
        float4 uVal = 0, vVal = 0;
        [branch]
        if (tid < dk4)
        {
            uint bOff = (ho + (tid << 2)) << 2;
            uVal = asfloat(In2.Load4(bOff));
            vVal = asfloat(In3.Load4(bOff));
        }
        sU[tid] = uVal;
        sVb[tid] = vVal;
        sM[tid] = 0.0f;
        sL[tid] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint r0 = ty;
    const uint r1 = ty + 16u;
    const uint c0 = tx;
    const uint c1 = tx + 16u;
    const uint p00 = (31u - r0) + c0;
    const uint qBase0 = r0 << 3;
    const uint qBase1 = r1 << 3;
    const uint kBase0 = AttnKTileStart + c0 * AttnPaddedRowFloat4s;
    const uint kBase1 = AttnKTileStart + c1 * AttnPaddedRowFloat4s;
    const uint pBaseA = AttnPosTileStart + p00 * AttnPaddedRowFloat4s;
    const uint pBaseB = AttnPosTileStart + (p00 + 16u) * AttnPaddedRowFloat4s;
    const uint pBaseC = AttnPosTileStart + (p00 - 16u) * AttnPaddedRowFloat4s;

    const uint loadRow = tid >> 3;
    const uint loadCol = tid & 7u;
    const uint d0 = tx;
    const uint d1 = tx + 16u;

    float4 o0a = 0, o0b = 0;
    float4 o1a = 0, o1b = 0;

    for (uint j0 = 0; j0 < T; j0 += 32u)
    {
        float s00 = 0.0f, s01 = 0.0f, s10 = 0.0f, s11 = 0.0f;
        float uKAcc = 0.0f;
        float vPAcc = 0.0f;
        const int pBase = (int)(T - 1u) - (int)(i0 + 31u) + (int)j0;

        for (uint k4 = 0; k4 < dk4; k4 += 8u)
        {
            const uint kIdx = k4 + loadCol;
            const uint kElem = ho + (kIdx << 2);

            float4 qv = 0;
            [branch]
            if (i0 + loadRow < T && kIdx < dk4)
                qv = asfloat(In0.Load4(((i0 + loadRow) * stride3D + kElem) << 2));
            sTile[tid] = qv;

            float4 kv = 0;
            [branch]
            if (j0 + loadRow < T && kIdx < dk4)
                kv = asfloat(In0.Load4(((j0 + loadRow) * stride3D + D + kElem) << 2));
            sTile[AttnKTileStart + loadRow * AttnPaddedRowFloat4s + loadCol] = kv;

            int pRow0 = pBase + (int)loadRow;
            float4 pv0 = 0;
            [branch]
            if (pRow0 >= 0 && (uint)pRow0 < posRows && kIdx < dk4)
                pv0 = asfloat(In1.Load4(((uint)pRow0 * D + kElem) << 2));
            sTile[AttnPosTileStart + loadRow * AttnPaddedRowFloat4s + loadCol] = pv0;

            if (loadRow < 31u)
            {
                uint pSlot1 = loadRow + 32u;
                int pRow1 = pBase + (int)pSlot1;
                float4 pv1 = 0;
                [branch]
                if (pRow1 >= 0 && (uint)pRow1 < posRows && kIdx < dk4)
                    pv1 = asfloat(In1.Load4(((uint)pRow1 * D + kElem) << 2));
                sTile[AttnPosTileStart + pSlot1 * AttnPaddedRowFloat4s + loadCol] = pv1;
            }

            GroupMemoryBarrierWithGroupSync();

            [unroll]
            for (uint kk = 0; kk < 8u; kk++)
            {
                float4 q0 = sTile[qBase0 + kk];
                float4 q1 = sTile[qBase1 + kk];
                float4 k0 = sTile[kBase0 + kk];
                float4 k1 = sTile[kBase1 + kk];
                s00 += dot(q0, k0);
                s01 += dot(q0, k1);
                s10 += dot(q1, k0);
                s11 += dot(q1, k1);

                float4 pa = sTile[pBaseA + kk];
                float4 pb = sTile[pBaseB + kk];
                float4 pc = sTile[pBaseC + kk];
                s00 += dot(q0, pa);
                s01 += dot(q0, pb);
                s10 += dot(q1, pc);
                s11 += dot(q1, pa);
            }

            if (tid < 32u)
            {
                uint kRowBase = AttnKTileStart + tid * AttnPaddedRowFloat4s;
                [unroll]
                for (uint kk2 = 0; kk2 < 8u; kk2++)
                    uKAcc += dot(sU[k4 + kk2], sTile[kRowBase + kk2]);
            }
            else if (tid < 95u)
            {
                uint pRowBase = AttnPosTileStart + (tid - 32u) * AttnPaddedRowFloat4s;
                [unroll]
                for (uint kk3 = 0; kk3 < 8u; kk3++)
                    vPAcc += dot(sVb[k4 + kk3], sTile[pRowBase + kk3]);
            }

            GroupMemoryBarrierWithGroupSync();
        }

        if (tid < 32u) sUK[tid] = uKAcc;
        else if (tid < 95u) sVP[tid - 32u] = vPAcc;
        GroupMemoryBarrierWithGroupSync();

        sS[r0][c0] = (s00 + sUK[c0] + sVP[p00])       * scale;
        sS[r0][c1] = (s01 + sUK[c1] + sVP[p00 + 16u]) * scale;
        sS[r1][c0] = (s10 + sUK[c0] + sVP[p00 - 16u]) * scale;
        sS[r1][c1] = (s11 + sUK[c1] + sVP[p00])       * scale;

        [unroll]
        for (uint vp = 0; vp < 4u; vp++)
        {
            uint idx = vp * 256u + tid;
            uint vRow = idx >> 5;
            uint vCol = idx & 31u;
            float4 vv = 0;
            [branch]
            if (j0 + vRow < T && vCol < dk4)
                vv = asfloat(In0.Load4(((j0 + vRow) * stride3D + 2u * D + ho + (vCol << 2)) << 2));
            sTile[idx] = vv;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < 32u)
        {
            if (i0 + tid < T)
            {
                uint validC = min(32u, T - j0);
                float mTile = sS[tid][0];
                for (uint c = 1u; c < validC; c++)
                    mTile = max(mTile, sS[tid][c]);

                float mOld = sM[tid];
                float mNew = (j0 == 0u) ? mTile : max(mOld, mTile);
                float alpha = (j0 == 0u) ? 0.0f : ExpTreatingNaNAsZero(mOld - mNew);

                float lTile = 0.0f;
                for (uint c2 = 0u; c2 < 32u; c2++)
                {
                    float w = 0.0f;
                    if (c2 < validC)
                    {
                        w = ExpTreatingNaNAsZero(sS[tid][c2] - mNew);
                        lTile += w;
                    }
                    sS[tid][c2] = w;
                }

                sM[tid] = mNew;
                sL[tid] = sL[tid] * alpha + lTile;
                sAlpha[tid] = alpha;
            }
            else
            {
                sAlpha[tid] = 0.0f;
                sL[tid] = 1.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        float a0 = sAlpha[r0];
        float a1 = sAlpha[r1];
        o0a *= a0; o0b *= a0;
        o1a *= a1; o1b *= a1;

        [loop]
        for (uint c3 = 0; c3 < 32u; c3 += 4u)
        {
            float w00 = sS[r0][c3 + 0u], w10 = sS[r1][c3 + 0u];
            float4 va0 = sTile[((c3 + 0u) << 5) + d0], vb0 = sTile[((c3 + 0u) << 5) + d1];
            o0a += w00 * va0; o0b += w00 * vb0;
            o1a += w10 * va0; o1b += w10 * vb0;

            float w01 = sS[r0][c3 + 1u], w11 = sS[r1][c3 + 1u];
            float4 va1 = sTile[((c3 + 1u) << 5) + d0], vb1 = sTile[((c3 + 1u) << 5) + d1];
            o0a += w01 * va1; o0b += w01 * vb1;
            o1a += w11 * va1; o1b += w11 * vb1;

            float w02 = sS[r0][c3 + 2u], w12 = sS[r1][c3 + 2u];
            float4 va2 = sTile[((c3 + 2u) << 5) + d0], vb2 = sTile[((c3 + 2u) << 5) + d1];
            o0a += w02 * va2; o0b += w02 * vb2;
            o1a += w12 * va2; o1b += w12 * vb2;

            float w03 = sS[r0][c3 + 3u], w13 = sS[r1][c3 + 3u];
            float4 va3 = sTile[((c3 + 3u) << 5) + d0], vb3 = sTile[((c3 + 3u) << 5) + d1];
            o0a += w03 * va3; o0b += w03 * vb3;
            o1a += w13 * va3; o1b += w13 * vb3;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    const float invL0 = 1.0f / sL[r0];
    const float invL1 = 1.0f / sL[r1];
    [branch]
    if (i0 + r0 < T)
    {
        uint outBase0 = (i0 + r0) * D + ho;
        [branch] if (d0 < dk4) Out0.Store4((outBase0 + (d0 << 2)) << 2, asuint(o0a * invL0));
        [branch] if (d1 < dk4) Out0.Store4((outBase0 + (d1 << 2)) << 2, asuint(o0b * invL0));
    }
    [branch]
    if (i0 + r1 < T)
    {
        uint outBase1 = (i0 + r1) * D + ho;
        [branch] if (d0 < dk4) Out0.Store4((outBase1 + (d0 << 2)) << 2, asuint(o1a * invL1));
        [branch] if (d1 < dk4) Out0.Store4((outBase1 + (d1 << 2)) << 2, asuint(o1b * invL1));
    }
}

[numthreads(256, 1, 1)]
void GluMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint idx = (gid.y * MaxGroupsPerDispatchDimension + gid.x) * 256u + gtid.x;
    const uint total = P0;
    if (idx >= total) return;
    const uint D = P1;
    const uint t = idx / D;
    const uint ch = idx - t * D;
    const uint srcBase = t * 2 * D + ch;
    float a = asfloat(In0.Load(srcBase << 2));
    float b = asfloat(In0.Load((srcBase + D) << 2));
    float y = a / (1.0f + exp(-b));
    Out0.Store(idx << 2, asuint(y));
}

[numthreads(256, 1, 1)]
void DepthwiseConvBnSiluMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint idx = (gid.y * MaxGroupsPerDispatchDimension + gid.x) * 256u + gtid.x;
    const uint T = P0;
    const uint D = P1;
    if (idx >= T * D) return;
    const uint kernel = P2;
    const int pad = (int)(kernel - 1) >> 1;
    const uint t = idx / D;
    const uint ch = idx - t * D;

    float sum = 0.0f;
    const uint dwBase = ch * kernel;
    for (uint k = 0; k < kernel; k++)
    {
        int ti = (int)t + (int)k - pad;
        if (ti >= 0 && ti < (int)T)
        {
            float x = asfloat(In0.Load(((uint)ti * D + ch) << 2));
            float w = asfloat(In1.Load((dwBase + k) << 2));
            sum += x * w;
        }
    }
    float v = sum * asfloat(In2.Load(ch << 2)) + asfloat(In3.Load(ch << 2));
    float y = v / (1.0f + exp(-v));
    Out0.Store(idx << 2, asuint(y));
}
