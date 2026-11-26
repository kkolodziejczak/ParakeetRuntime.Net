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

static const uint MaxGroupsPerDispatchDimension = 65535u;
static const int FrameCentreOffset = 256;
static const uint TwiddleSinStart = 512u;
static const uint PowerBins = 257u;
static const int NyquistBin = 256;
static const int FftStages = 9;
static const float LogGuard = 5.9604645e-8f;
static const float MelStdEpsilon = 1e-5f;

groupshared float sRe[512];
groupshared float sIm[512];
groupshared float sPow[PowerBins];

float ReadPreemph(uint nSamples, int aIdx, float preemph)
{
    if (aIdx < 0 || aIdx >= (int)nSamples)
        return 0.0f;
    float cur = asfloat(In0.Load((uint)aIdx << 2));
    if (aIdx == 0)
        return cur;
    float prev = asfloat(In0.Load((uint)(aIdx - 1) << 2));
    return cur - preemph * prev;
}

uint BitReversedFftIndex(uint i)
{
    return reversebits(i) >> 23;
}

[numthreads(256, 1, 1)]
void MelFftLogMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint f = gid.y * MaxGroupsPerDispatchDimension + gid.x;
    const uint validFrames = P6;
    if (validFrames != 0u && f >= validFrames) return;
    const uint tid = gtid.x;
    const uint nSamples = P0, hop = P1, win = P2, woff = P3, nMels = P4;
    const float preemph = asfloat(P5);

    [unroll]
    for (uint p = 0; p < 2; p++)
    {
        uint idx = tid + p * 256u;
        int wi = (int)idx - (int)woff;
        float val = 0.0f;
        if (wi >= 0 && wi < (int)win)
        {
            int aIdx = (int)(f * hop + idx) - FrameCentreOffset;
            val = ReadPreemph(nSamples, aIdx, preemph) * asfloat(In1.Load((uint)wi << 2));
        }
        uint rev = BitReversedFftIndex(idx);
        sRe[rev] = val;
        sIm[rev] = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stage = 0; stage < FftStages; stage++)
    {
        uint halfLen = 1u << stage;
        uint len = halfLen << 1;
        uint grp = tid >> stage;
        uint k = tid & (halfLen - 1u);
        uint a = grp * len + k;
        uint b = a + halfLen;

        float wr = asfloat(In3.Load((halfLen + k) << 2));
        float wi = asfloat(In3.Load((TwiddleSinStart + halfLen + k) << 2));

        float ra = sRe[a], ia = sIm[a];
        float rb = sRe[b], ib = sIm[b];
        float xr = rb * wr - ib * wi;
        float xi = rb * wi + ib * wr;

        sRe[a] = ra + xr; sIm[a] = ia + xi;
        sRe[b] = ra - xr; sIm[b] = ia - xi;
        GroupMemoryBarrierWithGroupSync();
    }

    sPow[tid] = sRe[tid] * sRe[tid] + sIm[tid] * sIm[tid];
    if (tid == 0)
        sPow[NyquistBin] = sRe[NyquistBin] * sRe[NyquistBin] + sIm[NyquistBin] * sIm[NyquistBin];
    GroupMemoryBarrierWithGroupSync();

    if (tid < nMels)
    {
        uint fbBase = tid * PowerBins;
        float acc = 0.0f;
        for (uint k = 0; k < PowerBins; k++)
            acc += asfloat(In2.Load((fbBase + k) << 2)) * sPow[k];
        Out0.Store((f * nMels + tid) << 2, asuint(log(acc + LogGuard)));
    }
}

groupshared float sMelRed[256];

[numthreads(256, 1, 1)]
void MelNormalizeMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint m = gid.x;
    const uint frames = P0, valid = P1, nMels = P2;

    float sum = 0.0f;
    for (uint f = tid; f < valid; f += 256u)
        sum += asfloat(Out0.Load((f * nMels + m) << 2));
    sMelRed[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid < s) sMelRed[tid] += sMelRed[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = valid > 0u ? sMelRed[0] / (float)valid : 0.0f;
    GroupMemoryBarrierWithGroupSync();

    float varSum = 0.0f;
    for (uint f2 = tid; f2 < valid; f2 += 256u)
    {
        float diff = asfloat(Out0.Load((f2 * nMels + m) << 2)) - mean;
        varSum += diff * diff;
    }
    sMelRed[tid] = varSum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s2 = 128; s2 > 0; s2 >>= 1)
    {
        if (tid < s2) sMelRed[tid] += sMelRed[tid + s2];
        GroupMemoryBarrierWithGroupSync();
    }
    float variance = valid > 1u ? sMelRed[0] / (float)(valid - 1u) : sMelRed[0];
    float inv = 1.0f / (sqrt(variance) + MelStdEpsilon);

    for (uint f3 = tid; f3 < frames; f3 += 256u)
    {
        float val = 0.0f;
        [branch] if (f3 < valid) val = (asfloat(Out0.Load((f3 * nMels + m) << 2)) - mean) * inv;
        Out0.Store((f3 * nMels + m) << 2, asuint(val));
    }
}
