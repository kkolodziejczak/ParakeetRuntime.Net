// GPU-loop TDT greedy decoder shaders (SM 5.1).
// All conditional shaders check decState[0] (current frame t) against T (and optionally
// decState[4] needPredict) with a uniform early return before any shared-memory barrier,
// so hundreds of decode steps run in a single D3D12 command list with minimal overhead.
//
// decState layout (uints):
//   [0] t           (current encoder frame, 0..T)
//   [1] symbols     (emitted symbols at current frame t)
//   [2] tokenCount  (number of tokens written to Tokens[])
//   [3] lastTok     (last emitted token id, or 0xFFFFFFFF initially)
//   [4] needPredict (1 if LSTM + jointPred must run before jointOut, else 0)
//   [5] lstmSlot    (0 or 1: ping-pong slot in decH[i] holding current h[i])

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

// 1. InitDecStateMain: initializes decState = [0, 0, 0, 0xFFFFFFFF, 1, 0].
[numthreads(1, 1, 1)]
void InitDecStateMain()
{
    Out0.Store(0, 0u);          // t = 0
    Out0.Store(4, 0u);          // symbols = 0
    Out0.Store(8, 0u);          // tokenCount = 0
    Out0.Store(12, 0xFFFFFFFFu);// lastTok = -1
    Out0.Store(16, 1u);         // needPredict = 1
    Out0.Store(20, 0u);         // lstmSlot = 0
}

// 2. ZeroBufferMain: P0 = floatCount
[numthreads(256, 1, 1)]
void ZeroBufferMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x < P0)
        Out0.Store(dtid.x << 2, 0u);
}

// 3. LstmLayerCondMain: fused (embed/prev-h load + ih*x + hh*h + bias + LSTM cell update).
// Reads h from Out0[srcSlot*H..] and writes new h to Out0[dstSlot*H..] (disjoint ping-pong slots),
// and updates c in-place in Out1[0..H-1].
// In0 = embed[vocab, H] (if P2 != 0) or prevLayer decH[2*H] (reads dstSlot*H, if P2 == 0)
// In1 = wqMega + ih.QByteOffset (hh at +P3), In2 = wdMega + ih.DByteOffset (hh at +P4)
// In3 = pre-summed (ihBias + hhBias)[4*H], StateIn = decState
// Out0 = decH[i] (2*H floats), Out1 = decC[i] (H floats)
// Dispatch: ((H + 7) / 8, 1, 1)
// P0 = H (<= 1024, multiple of 32), P1 = T, P2 = useEmbed (0 or 1), P3 = hhQByteOffset, P4 = hhDByteOffset
groupshared float4 sLstmX4[256];
groupshared float4 sLstmH4[256];
groupshared float sLstmRed[32][4];

[numthreads(128, 1, 1)]
void LstmLayerCondMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    if (StateIn.Load(0) >= P1 || StateIn.Load(16) == 0u) return;

    const uint tid = gtid.x;
    const uint H = P0;
    const uint h4 = H >> 2;
    const uint kBlocks = H >> 5;
    const uint srcSlot = StateIn.Load(20);
    const uint dstSlot = 1u - srcSlot;
    const uint label = StateIn.Load(12);

    for (uint i = tid; i < h4; i += 128u)
    {
        float4 xv = 0;
        [branch]
        if (P2 != 0u)
        {
            [branch]
            if (label != 0xFFFFFFFFu)
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

    // 128 threads = 32 gate rows (4 gates x 8 hidden units) x 4 lanes along K.
    const uint row = tid >> 2;   // 0..31
    const uint lane = tid & 3u;  // 0..3
    const uint gate = row >> 3;  // 0..3 (i, f, g, o)
    const uint unit = row & 7u;  // 0..7
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

            float scaleH = f16tof32((In2.Load(P4 + dOff) >> shift) & 0xFFFFu);
            uint qByteH = P3 + qByteI;
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

    sLstmRed[row][lane] = acc;
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

// 4. Q8GemvCondMain: M = 1 vectorized Q8_0 GEMV with optional GPU state gating and fused input modes.
// 64 threads = 16 output rows x 4 lanes along K -> Dispatch: ((N + 15) / 16, 1, 1).
// In0 = X[K] (or encProj[T, K] when mode == 4), In1 = Wq[N, K], In2 = Wd[N, K/32], In3 = Bias[N],
// StateIn = decState (when P4 != 0), In5 = predProj[K] (when mode == 4). Out0 = Y[N].
// P0 = N, P1 = K (<= 1024, multiple of 32), P2 = hasBias, P3 = T,
// P4 = mode:
//   0 = unconditional, plain In0[0..K-1]
//   1 = gate t < T, plain In0[0..K-1]
//   2 = gate t < T && needPredict, plain In0[0..K-1]
//   3 = gate t < T && needPredict, ping-pong In0[(1 - lstmSlot)*K..] (jointPred reading decH)
//   4 = gate t < T, fused JointRelu: max(0, In0[t*K + i] + In5[i]) (jointOut)
groupshared float4 sGemv1X4[256];
groupshared float sGemv1Red[16][4];

[numthreads(64, 1, 1)]
void Q8GemvCondMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint tCur = 0u;
    if (P4 != 0u)
    {
        tCur = StateIn.Load(0);
        if (tCur >= P3) return;
        if ((P4 == 2u || P4 == 3u) && StateIn.Load(16) == 0u) return;
    }

    const uint tid = gtid.x;
    const uint N = P0, K = P1, hasBias = P2;
    const uint k4 = K >> 2;
    const uint kBlocks = K >> 5;

    [branch]
    if (P4 == 4u)
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
        const uint xBase = (P4 == 3u) ? ((1u - StateIn.Load(20)) * K) : 0u;
        for (uint i = tid; i < k4; i += 64u)
            sGemv1X4[i] = asfloat(In0.Load4((xBase + (i << 2)) << 2));
    }
    GroupMemoryBarrierWithGroupSync();

    const uint row = tid >> 2;  // 0..15
    const uint lane = tid & 3u; // 0..3
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

// 5. ArgMaxAndStepMain: finds argmax(logits[0..vocab-1]) and argmax(logits[vocab..vocab+numDur-1]),
// advances decState (Out0), flips lstmSlot if LSTM ran this step, and appends non-blank token to Out1.
// In0 = logits[vocab + numDur], Out0 = decState, Out1 = Tokens[maxTokens]
// P0 = vocab, P1 = blank, P2 = numDur, P3 = T, P4 = maxSymbols, P5 = maxTokens, P6 = durPackedLo, P7 = durPackedHi
groupshared float sMaxVal[256];
groupshared uint sMaxIdx[256];

[numthreads(256, 1, 1)]
void ArgMaxAndStepMain(uint3 gtid : SV_GroupThreadID)
{
    const uint t = Out0.Load(0);
    const bool active = (t < P3);

    const uint tid = gtid.x;
    const uint vocab = P0;
    const uint blank = P1;
    const uint numDur = P2;

    float bestVal = -3.402823466e+38f;
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

        uint durWord = (bestDur < 4u) ? P6 : P7;
        uint skip = (durWord >> ((bestDur & 3u) * 8u)) & 0xFFu;
        if (tok == blank && skip == 0u) skip = 1u;

        if (Out0.Load(16) != 0u)
            Out0.Store(20, 1u - Out0.Load(20));

        uint needPred = 0u;
        if (tok != blank)
        {
            uint tc = Out0.Load(8);
            if (tc < P5)
                Out1.Store(tc << 2, tok);
            Out0.Store(8, tc + 1u);
            Out0.Store(12, tok);
            needPred = 1u;
        }

        uint syms = Out0.Load(4) + 1u;
        uint tNew = t + skip;
        if (skip > 0u)
        {
            syms = 0u;
        }
        else if (syms >= P4)
        {
            tNew += 1u;
            syms = 0u;
        }

        Out0.Store(0, tNew);
        Out0.Store(4, syms);
        Out0.Store(16, needPred);
    }
}
