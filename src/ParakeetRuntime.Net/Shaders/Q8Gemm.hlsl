cbuffer Params : register(b0)
{
    uint M;
    uint N;
    uint K;
    uint hasBias;
    uint epilogue;
    uint xByteOffset;
    uint yByteOffset;
    uint pad0;
};

ByteAddressBuffer X    : register(t0);
ByteAddressBuffer Wq   : register(t1);
ByteAddressBuffer Wd   : register(t2);
ByteAddressBuffer Bias : register(t3);

RWByteAddressBuffer Y  : register(u0);

static const uint EpilogueSilu = 1;
static const uint EpilogueAddResidual = 2;
static const uint EpilogueAddHalfResidual = 3;

groupshared float4 sXByKAndRowQuad[32][16];
groupshared float4 sWByKAndColumnQuad[32][16];

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
        if (epilogue == EpilogueSilu) v = v / (1.0f + exp(-v));
        else if (epilogue == EpilogueAddResidual) v = asfloat(Y.Load4(outByte)) + v;
        else if (epilogue == EpilogueAddHalfResidual) v = asfloat(Y.Load4(outByte)) + 0.5f * v;
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
            if (epilogue == EpilogueSilu) val = val / (1.0f + exp(-val));
            else if (epilogue == EpilogueAddResidual) val = asfloat(Y.Load(outByte)) + val;
            else if (epilogue == EpilogueAddHalfResidual) val = asfloat(Y.Load(outByte)) + 0.5f * val;
            Y.Store(outByte, asuint(val));
        }
    }
}

[numthreads(16, 8, 1)]
void Q8GemmMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tx = gtid.x;
    const uint ty = gtid.y;
    const uint tid = ty * 16 + tx;

    const uint groupN = gid.x * 64;
    const uint groupM = gid.y * 64;
    const uint kBlocks = K >> 5;

    const uint rowQuad = tid >> 3;
    const uint kCol = (tid & 7) << 2;
    const uint xRow = groupM + (rowQuad << 2);
    const uint wRow = groupN + (rowQuad << 2);

    float4 top0 = 0, top1 = 0, top2 = 0, top3 = 0;
    float4 bottom0 = 0, bottom1 = 0, bottom2 = 0, bottom3 = 0;
    for (uint kb = 0; kb < kBlocks; kb++)
    {
        const uint kStart = (kb << 5) + kCol;

        float4 rx0, rx1, rx2, rx3;
        LoadX4Rows(xRow, kStart, rx0, rx1, rx2, rx3);
        sXByKAndRowQuad[kCol + 0][rowQuad] = float4(rx0.x, rx1.x, rx2.x, rx3.x);
        sXByKAndRowQuad[kCol + 1][rowQuad] = float4(rx0.y, rx1.y, rx2.y, rx3.y);
        sXByKAndRowQuad[kCol + 2][rowQuad] = float4(rx0.z, rx1.z, rx2.z, rx3.z);
        sXByKAndRowQuad[kCol + 3][rowQuad] = float4(rx0.w, rx1.w, rx2.w, rx3.w);

        float4 rw0, rw1, rw2, rw3;
        LoadW4Rows(wRow, kStart, kBlocks, kb, rw0, rw1, rw2, rw3);
        sWByKAndColumnQuad[kCol + 0][rowQuad] = float4(rw0.x, rw1.x, rw2.x, rw3.x);
        sWByKAndColumnQuad[kCol + 1][rowQuad] = float4(rw0.y, rw1.y, rw2.y, rw3.y);
        sWByKAndColumnQuad[kCol + 2][rowQuad] = float4(rw0.z, rw1.z, rw2.z, rw3.z);
        sWByKAndColumnQuad[kCol + 3][rowQuad] = float4(rw0.w, rw1.w, rw2.w, rw3.w);

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < 32; k++)
        {
            float4 top = sXByKAndRowQuad[k][ty], bottom = sXByKAndRowQuad[k][ty + 8], w = sWByKAndColumnQuad[k][tx];
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

groupshared float4 sXByKAndRowQuadSmall[32][8];
groupshared float4 sWByKAndColumnQuadSmall[32][8];

[numthreads(8, 8, 1)]
void Q8GemmSmallMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tx = gtid.x;
    const uint ty = gtid.y;
    const uint tid = ty * 8 + tx;

    const uint groupN = gid.x * 32;
    const uint groupM = gid.y * 32;
    const uint kBlocks = K >> 5;

    float4 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    const uint rowQuad = tid >> 3;
    const uint kCol = (tid & 7) << 2;
    const uint xRow = groupM + (rowQuad << 2);
    const uint wRow = groupN + (rowQuad << 2);

    for (uint kb = 0; kb < kBlocks; kb++)
    {
        const uint kStart = (kb << 5) + kCol;

        float4 rx0, rx1, rx2, rx3;
        LoadX4Rows(xRow, kStart, rx0, rx1, rx2, rx3);
        sXByKAndRowQuadSmall[kCol + 0][rowQuad] = float4(rx0.x, rx1.x, rx2.x, rx3.x);
        sXByKAndRowQuadSmall[kCol + 1][rowQuad] = float4(rx0.y, rx1.y, rx2.y, rx3.y);
        sXByKAndRowQuadSmall[kCol + 2][rowQuad] = float4(rx0.z, rx1.z, rx2.z, rx3.z);
        sXByKAndRowQuadSmall[kCol + 3][rowQuad] = float4(rx0.w, rx1.w, rx2.w, rx3.w);

        float4 rw0, rw1, rw2, rw3;
        LoadW4Rows(wRow, kStart, kBlocks, kb, rw0, rw1, rw2, rw3);
        sWByKAndColumnQuadSmall[kCol + 0][rowQuad] = float4(rw0.x, rw1.x, rw2.x, rw3.x);
        sWByKAndColumnQuadSmall[kCol + 1][rowQuad] = float4(rw0.y, rw1.y, rw2.y, rw3.y);
        sWByKAndColumnQuadSmall[kCol + 2][rowQuad] = float4(rw0.z, rw1.z, rw2.z, rw3.z);
        sWByKAndColumnQuadSmall[kCol + 3][rowQuad] = float4(rw0.w, rw1.w, rw2.w, rw3.w);

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < 32; k++)
        {
            float4 vx = sXByKAndRowQuadSmall[k][ty], vw = sWByKAndColumnQuadSmall[k][tx];
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
