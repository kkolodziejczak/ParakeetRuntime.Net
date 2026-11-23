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
ByteAddressBuffer StateIn : register(t4);
ByteAddressBuffer In5 : register(t5);

RWByteAddressBuffer Out0 : register(u0);
RWByteAddressBuffer Out1 : register(u1);

static const int DecStateFrameByte = 0;
static const int DecStateSymbolsByte = 4;
static const int DecStateTokenCountByte = 8;
static const int DecStateLastTokenByte = 12;
static const int DecStateNeedPredictByte = 16;
static const int DecStateLstmSlotByte = 20;
static const uint NoToken = 0xFFFFFFFFu;

static const uint GemvUngated = 0u;
static const uint GemvWhileFramesLeft = 1u;
static const uint GemvWhilePredictPending = 2u;
static const uint GemvJointPredFromLstmOutput = 3u;
static const uint GemvJointOutFromEncoderPlusPred = 4u;

static const float LowestFloat = -3.402823466e+38f;

float DotS8x4(float4 x, uint packed)
{
    int wi = asint(packed);
    float4 w = float4(
        float((wi << 24) >> 24),
        float((wi << 16) >> 24),
        float((wi <<  8) >> 24),
        float( wi        >> 24));
    return dot(x, w);
}

[numthreads(1, 1, 1)]
void InitDecStateMain()
{
    Out0.Store(DecStateFrameByte, 0u);
    Out0.Store(DecStateSymbolsByte, 0u);
    Out0.Store(DecStateTokenCountByte, 0u);
    Out0.Store(DecStateLastTokenByte, NoToken);
    Out0.Store(DecStateNeedPredictByte, 1u);
    Out0.Store(DecStateLstmSlotByte, 0u);
}

[numthreads(256, 1, 1)]
void ZeroBufferMain(uint3 dtid : SV_DispatchThreadID)
{
    const uint floatCount = P0;
    if (dtid.x < floatCount)
        Out0.Store(dtid.x << 2, 0u);
}

bool LstmInputIsEmbedding()
{
    return P2 != 0u;
}

groupshared float4 sLstmX4[256];
groupshared float4 sLstmH4[256];
groupshared float sLstmRed[32][4];

[numthreads(128, 1, 1)]
void LstmLayerCondMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint T = P1;
    if (StateIn.Load(DecStateFrameByte) >= T || StateIn.Load(DecStateNeedPredictByte) == 0u) return;

    const uint tid = gtid.x;
    const uint H = P0;
    const uint hhQByteOffset = P3;
    const uint hhDByteOffset = P4;
    const uint h4 = H >> 2;
    const uint kBlocks = H >> 5;
    const uint srcSlot = StateIn.Load(DecStateLstmSlotByte);
    const uint dstSlot = 1u - srcSlot;
    const uint label = StateIn.Load(DecStateLastTokenByte);

    for (uint i = tid; i < h4; i += 128u)
    {
        float4 xv = 0;
        [branch]
        if (LstmInputIsEmbedding())
        {
            [branch]
            if (label != NoToken)
                xv = asfloat(In0.Load4((label * H + (i << 2)) << 2));
        }
        else
        {
            xv = asfloat(In0.Load4((dstSlot * H + (i << 2)) << 2));
        }
        sLstmX4[i] = xv;
        sLstmH4[i] = asfloat(Out0.Load4((srcSlot * H + (i << 2)) << 2));
    }
    GroupMemoryBarrierWithGroupSync();

    const uint gateRow = tid >> 2;
    const uint lane = tid & 3u;
    const uint gate = gateRow >> 3;
    const uint unit = gateRow & 7u;
    const uint k = gid.x * 8u + unit;

    float acc = 0.0f;
    [branch]
    if (k < H)
    {
        const uint gRow = gate * H + k;
        for (uint kb = lane; kb < kBlocks; kb += 4u)
        {
            uint dIdx = gRow * kBlocks + kb;
            uint dOff = (dIdx >> 1) << 2;
            uint shift = (dIdx & 1u) << 4;

            float scaleI = f16tof32((In2.Load(dOff) >> shift) & 0xFFFFu);
            uint qByteI = gRow * H + (kb << 5);
            uint4 qaI = In1.Load4(qByteI);
            uint4 qbI = In1.Load4(qByteI + 16u);

            float scaleH = f16tof32((In2.Load(hhDByteOffset + dOff) >> shift) & 0xFFFFu);
            uint qByteH = hhQByteOffset + qByteI;
            uint4 qaH = In1.Load4(qByteH);
            uint4 qbH = In1.Load4(qByteH + 16u);

            uint k4Base = kb << 3;
            float bsumI =
                DotS8x4(sLstmX4[k4Base + 0], qaI.x) + DotS8x4(sLstmX4[k4Base + 1], qaI.y) +
                DotS8x4(sLstmX4[k4Base + 2], qaI.z) + DotS8x4(sLstmX4[k4Base + 3], qaI.w) +
                DotS8x4(sLstmX4[k4Base + 4], qbI.x) + DotS8x4(sLstmX4[k4Base + 5], qbI.y) +
                DotS8x4(sLstmX4[k4Base + 6], qbI.z) + DotS8x4(sLstmX4[k4Base + 7], qbI.w);

            float bsumH =
                DotS8x4(sLstmH4[k4Base + 0], qaH.x) + DotS8x4(sLstmH4[k4Base + 1], qaH.y) +
                DotS8x4(sLstmH4[k4Base + 2], qaH.z) + DotS8x4(sLstmH4[k4Base + 3], qaH.w) +
                DotS8x4(sLstmH4[k4Base + 4], qbH.x) + DotS8x4(sLstmH4[k4Base + 5], qbH.y) +
                DotS8x4(sLstmH4[k4Base + 6], qbH.z) + DotS8x4(sLstmH4[k4Base + 7], qbH.w);

            acc += bsumI * scaleI + bsumH * scaleH;
        }
    }

    sLstmRed[gateRow][lane] = acc;
    GroupMemoryBarrierWithGroupSync();

    [branch]
    if (tid < 8u)
    {
        uint ku = gid.x * 8u + tid;
        [branch]
        if (ku < H)
        {
            uint rI = tid;
            uint rF = 8u + tid;
            uint rG = 16u + tid;
            uint rO = 24u + tid;

            float iPre = sLstmRed[rI][0] + sLstmRed[rI][1] + sLstmRed[rI][2] + sLstmRed[rI][3] + asfloat(In3.Load(ku << 2));
            float fPre = sLstmRed[rF][0] + sLstmRed[rF][1] + sLstmRed[rF][2] + sLstmRed[rF][3] + asfloat(In3.Load((H + ku) << 2));
            float gPre = sLstmRed[rG][0] + sLstmRed[rG][1] + sLstmRed[rG][2] + sLstmRed[rG][3] + asfloat(In3.Load((2u * H + ku) << 2));
            float oPre = sLstmRed[rO][0] + sLstmRed[rO][1] + sLstmRed[rO][2] + sLstmRed[rO][3] + asfloat(In3.Load((3u * H + ku) << 2));

            float ig = 1.0f / (1.0f + exp(-iPre));
            float fg = 1.0f / (1.0f + exp(-fPre));
            float gg = tanh(gPre);
            float og = 1.0f / (1.0f + exp(-oPre));

            float cPrev = asfloat(Out1.Load(ku << 2));
            float cn = fg * cPrev + ig * gg;
            float hn = og * tanh(cn);

            Out0.Store((dstSlot * H + ku) << 2, asuint(hn));
            Out1.Store(ku << 2, asuint(cn));
        }
    }
}

groupshared float4 sGemv1X4[256];
groupshared float sGemv1Red[16][4];

[numthreads(64, 1, 1)]
void Q8GemvCondMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint T = P3;
    const uint mode = P4;
    uint tCur = 0u;
    if (mode != GemvUngated)
    {
        tCur = StateIn.Load(DecStateFrameByte);
        if (tCur >= T) return;
        if ((mode == GemvWhilePredictPending || mode == GemvJointPredFromLstmOutput) && StateIn.Load(DecStateNeedPredictByte) == 0u) return;
    }

    const uint tid = gtid.x;
    const uint N = P0, K = P1, hasBias = P2;
    const uint k4 = K >> 2;
    const uint kBlocks = K >> 5;

    [branch]
    if (mode == GemvJointOutFromEncoderPlusPred)
    {
        const uint encBase = tCur * K;
        for (uint i = tid; i < k4; i += 64u)
        {
            float4 e = asfloat(In0.Load4((encBase + (i << 2)) << 2));
            float4 p = asfloat(In5.Load4((i << 2) << 2));
            sGemv1X4[i] = max(0.0f, e + p);
        }
    }
    else
    {
        const uint xBase = (mode == GemvJointPredFromLstmOutput) ? ((1u - StateIn.Load(DecStateLstmSlotByte)) * K) : 0u;
        for (uint i = tid; i < k4; i += 64u)
            sGemv1X4[i] = asfloat(In0.Load4((xBase + (i << 2)) << 2));
    }
    GroupMemoryBarrierWithGroupSync();

    const uint row = tid >> 2;
    const uint lane = tid & 3u;
    const uint gn = gid.x * 16u + row;

    float acc = 0.0f;
    [branch]
    if (gn < N)
    {
        for (uint kb = lane; kb < kBlocks; kb += 4u)
        {
            uint dIdx = gn * kBlocks + kb;
            uint dWord = In2.Load((dIdx >> 1) << 2);
            float scale = f16tof32((dIdx & 1u) != 0u ? (dWord >> 16) : (dWord & 0xFFFFu));

            uint qByte = gn * K + (kb << 5);
            uint4 qa = In1.Load4(qByte);
            uint4 qb = In1.Load4(qByte + 16u);
            uint k4Base = kb << 3;

            float bsum =
                DotS8x4(sGemv1X4[k4Base + 0], qa.x) + DotS8x4(sGemv1X4[k4Base + 1], qa.y) +
                DotS8x4(sGemv1X4[k4Base + 2], qa.z) + DotS8x4(sGemv1X4[k4Base + 3], qa.w) +
                DotS8x4(sGemv1X4[k4Base + 4], qb.x) + DotS8x4(sGemv1X4[k4Base + 5], qb.y) +
                DotS8x4(sGemv1X4[k4Base + 6], qb.z) + DotS8x4(sGemv1X4[k4Base + 7], qb.w);

            acc += bsum * scale;
        }
    }

    sGemv1Red[row][lane] = acc;
    GroupMemoryBarrierWithGroupSync();

    [branch]
    if (lane == 0u && gn < N)
    {
        float bias = 0.0f;
        [branch] if (hasBias != 0u) bias = asfloat(In3.Load(gn << 2));
        float total = sGemv1Red[row][0] + sGemv1Red[row][1] + sGemv1Red[row][2] + sGemv1Red[row][3] + bias;
        Out0.Store(gn << 2, asuint(total));
    }
}

groupshared float sMaxVal[256];
groupshared uint sMaxIdx[256];

[numthreads(256, 1, 1)]
void ArgMaxAndStepMain(uint3 gtid : SV_GroupThreadID)
{
    const uint vocab = P0;
    const uint blank = P1;
    const uint numDur = P2;
    const uint T = P3;
    const uint maxSymbolsPerFrame = P4;
    const uint maxTokens = P5;
    const uint durationsPackedLo = P6;
    const uint durationsPackedHi = P7;

    const uint t = Out0.Load(DecStateFrameByte);
    const bool active = (t < T);

    const uint tid = gtid.x;

    float bestVal = LowestFloat;
    uint bestIdx = 0;
    if (active)
    {
        for (uint i = tid; i < vocab; i += 256u)
        {
            float v = asfloat(In0.Load(i << 2));
            if (v > bestVal || (v == bestVal && i < bestIdx))
            {
                bestVal = v;
                bestIdx = i;
            }
        }
    }
    sMaxVal[tid] = bestVal;
    sMaxIdx[tid] = bestIdx;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1)
    {
        if (tid < s)
        {
            float ov = sMaxVal[tid + s];
            uint oi = sMaxIdx[tid + s];
            if (ov > sMaxVal[tid] || (ov == sMaxVal[tid] && oi < sMaxIdx[tid]))
            {
                sMaxVal[tid] = ov;
                sMaxIdx[tid] = oi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (active && tid == 0u)
    {
        uint tok = sMaxIdx[0];
        uint bestDur = 0;
        float bestDurVal = asfloat(In0.Load(vocab << 2));
        for (uint d = 1u; d < numDur; d++)
        {
            float dv = asfloat(In0.Load((vocab + d) << 2));
            if (dv > bestDurVal)
            {
                bestDurVal = dv;
                bestDur = d;
            }
        }

        uint durWord = (bestDur < 4u) ? durationsPackedLo : durationsPackedHi;
        uint skip = (durWord >> ((bestDur & 3u) * 8u)) & 0xFFu;
        if (tok == blank && skip == 0u) skip = 1u;

        if (Out0.Load(DecStateNeedPredictByte) != 0u)
            Out0.Store(DecStateLstmSlotByte, 1u - Out0.Load(DecStateLstmSlotByte));

        uint needPred = 0u;
        if (tok != blank)
        {
            uint tc = Out0.Load(DecStateTokenCountByte);
            if (tc < maxTokens)
                Out1.Store(tc << 2, tok);
            Out0.Store(DecStateTokenCountByte, tc + 1u);
            Out0.Store(DecStateLastTokenByte, tok);
            needPred = 1u;
        }

        uint syms = Out0.Load(DecStateSymbolsByte) + 1u;
        uint tNew = t + skip;
        if (skip > 0u)
        {
            syms = 0u;
        }
        else if (syms >= maxSymbolsPerFrame)
        {
            tNew += 1u;
            syms = 0u;
        }

        Out0.Store(DecStateFrameByte, tNew);
        Out0.Store(DecStateSymbolsByte, syms);
        Out0.Store(DecStateNeedPredictByte, needPred);
    }
}
