// FastConformer dw_striding subsampling shaders (SM 5.1):
//   1. SubConv0ReluMain: 3x3 stride-2 conv (1 -> C) + bias + ReLU
//   2. SubDwConvMain:    3x3 stride-2 depthwise conv (C -> C) + bias
//   3. F32GemmReluMain:  tiled float32 GEMM (1x1 pointwise conv) + bias + ReLU
//   4. SubTransposeMain: [T, W, C] -> [T, C, W]

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

RWByteAddressBuffer Out0 : register(u0);

// 1. Conv2D 3x3 stride 2 (1 -> C) + bias + ReLU.
// In0 = feats[H, W], In1 = weight[C, 9], In2 = bias[C], Out0 = out[H1, W1, C]
// P0 = H, P1 = W, P2 = H1, P3 = W1, P4 = C
[numthreads(256, 1, 1)]
void SubConv0ReluMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint idx = (gid.y * 65535u + gid.x) * 256u + gtid.x;
    const uint H = P0, W = P1, H1 = P2, W1 = P3, C = P4;
    const uint total = H1 * W1 * C;
    if (idx >= total) return;

    const uint c = idx % C;
    const uint hw = idx / C;
    const uint h = hw / W1;
    const uint w = hw - h * W1;

    float s = asfloat(In2.Load(c << 2));
    const uint wBase = c * 9;
    [unroll]
    for (uint kh = 0; kh < 3; kh++)
    {
        int ih = (int)(2 * h) - 1 + (int)kh;
        if (ih < 0 || ih >= (int)H) continue;
        [unroll]
        for (uint kw = 0; kw < 3; kw++)
        {
            int iw = (int)(2 * w) - 1 + (int)kw;
            if (iw < 0 || iw >= (int)W) continue;
            float xv = asfloat(In0.Load(((uint)ih * W + (uint)iw) << 2));
            float wv = asfloat(In1.Load((wBase + kh * 3 + kw) << 2));
            s += xv * wv;
        }
    }
    Out0.Store(idx << 2, asuint(max(0.0f, s)));
}

// 2. Depthwise Conv2D 3x3 stride 2 (C -> C) + bias.
// In0 = x[H0, W0, C], In1 = dww[C, 9], In2 = dwb[C], Out0 = y[H2, W2, C]
// P0 = H0, P1 = W0, P2 = H2, P3 = W2, P4 = C
[numthreads(256, 1, 1)]
void SubDwConvMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint idx = (gid.y * 65535u + gid.x) * 256u + gtid.x;
    const uint H0 = P0, W0 = P1, H2 = P2, W2 = P3, C = P4;
    const uint total = H2 * W2 * C;
    if (idx >= total) return;

    const uint c = idx % C;
    const uint hw = idx / C;
    const uint h = hw / W2;
    const uint w = hw - h * W2;

    float s = asfloat(In2.Load(c << 2));
    const uint wBase = c * 9;
    [unroll]
    for (uint kh = 0; kh < 3; kh++)
    {
        int ih = (int)(2 * h) - 1 + (int)kh;
        if (ih < 0 || ih >= (int)H0) continue;
        [unroll]
        for (uint kw = 0; kw < 3; kw++)
        {
            int iw = (int)(2 * w) - 1 + (int)kw;
            if (iw < 0 || iw >= (int)W0) continue;
            float xv = asfloat(In0.Load((((uint)ih * W0 + (uint)iw) * C + c) << 2));
            float wv = asfloat(In1.Load((wBase + kh * 3 + kw) << 2));
            s += xv * wv;
        }
    }
    Out0.Store(idx << 2, asuint(s));
}

// 3. Tiled FP32 GEMM + bias + ReLU: Out0[M, N] = max(0, In0[M, K] * In1[N, K]^T + In2[N])
// K must be a multiple of 32 (subsampling C = 256).
// P0 = M, P1 = N, P2 = K
groupshared float sFX[32][33];
groupshared float sFW[32][65];

[numthreads(16, 8, 1)]
void F32GemmReluMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tx = gtid.x, ty = gtid.y;
    const uint tid = ty * 16 + tx;
    const uint M = P0, N = P1, K = P2;
    const uint groupN = gid.x * 64;
    const uint groupM = gid.y * 32;

    float acc[4][4];
    [unroll] for (uint r = 0; r < 4; r++)
        [unroll] for (uint c = 0; c < 4; c++)
            acc[r][c] = 0.0f;

    const uint xRow = tid >> 2;
    const uint xCol = (tid & 3) << 3;
    const uint gmLoad = groupM + xRow;

    const uint wRow = tid >> 1;
    const uint wHalf = tid & 1;
    const uint gnLoad = groupN + wRow;

    for (uint k0 = 0; k0 < K; k0 += 32)
    {
        [branch]
        if (gmLoad < M)
        {
            uint baseByte = (gmLoad * K + k0 + xCol) << 2;
            float4 v0 = asfloat(In0.Load4(baseByte));
            float4 v1 = asfloat(In0.Load4(baseByte + 16));
            sFX[xCol + 0][xRow] = v0.x; sFX[xCol + 1][xRow] = v0.y;
            sFX[xCol + 2][xRow] = v0.z; sFX[xCol + 3][xRow] = v0.w;
            sFX[xCol + 4][xRow] = v1.x; sFX[xCol + 5][xRow] = v1.y;
            sFX[xCol + 6][xRow] = v1.z; sFX[xCol + 7][xRow] = v1.w;
        }
        else
        {
            [unroll] for (uint i = 0; i < 8; i++) sFX[xCol + i][xRow] = 0.0f;
        }

        [branch]
        if (gnLoad < N)
        {
            uint kBase = wHalf << 4;
            uint baseByte = (gnLoad * K + k0 + kBase) << 2;
            float4 w0 = asfloat(In1.Load4(baseByte));
            float4 w1 = asfloat(In1.Load4(baseByte + 16));
            float4 w2 = asfloat(In1.Load4(baseByte + 32));
            float4 w3 = asfloat(In1.Load4(baseByte + 48));
            sFW[kBase +  0][wRow] = w0.x; sFW[kBase +  1][wRow] = w0.y; sFW[kBase +  2][wRow] = w0.z; sFW[kBase +  3][wRow] = w0.w;
            sFW[kBase +  4][wRow] = w1.x; sFW[kBase +  5][wRow] = w1.y; sFW[kBase +  6][wRow] = w1.z; sFW[kBase +  7][wRow] = w1.w;
            sFW[kBase +  8][wRow] = w2.x; sFW[kBase +  9][wRow] = w2.y; sFW[kBase + 10][wRow] = w2.z; sFW[kBase + 11][wRow] = w2.w;
            sFW[kBase + 12][wRow] = w3.x; sFW[kBase + 13][wRow] = w3.y; sFW[kBase + 14][wRow] = w3.z; sFW[kBase + 15][wRow] = w3.w;
        }
        else
        {
            uint kBase = wHalf << 4;
            [unroll] for (uint i = 0; i < 16; i++) sFW[kBase + i][wRow] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < 32; k++)
        {
            float vx0 = sFX[k][ty +  0], vx1 = sFX[k][ty +  8], vx2 = sFX[k][ty + 16], vx3 = sFX[k][ty + 24];
            float vw0 = sFW[k][tx +  0], vw1 = sFW[k][tx + 16], vw2 = sFW[k][tx + 32], vw3 = sFW[k][tx + 48];
            acc[0][0] += vx0 * vw0; acc[0][1] += vx0 * vw1; acc[0][2] += vx0 * vw2; acc[0][3] += vx0 * vw3;
            acc[1][0] += vx1 * vw0; acc[1][1] += vx1 * vw1; acc[1][2] += vx1 * vw2; acc[1][3] += vx1 * vw3;
            acc[2][0] += vx2 * vw0; acc[2][1] += vx2 * vw1; acc[2][2] += vx2 * vw2; acc[2][3] += vx2 * vw3;
            acc[3][0] += vx3 * vw0; acc[3][1] += vx3 * vw1; acc[3][2] += vx3 * vw2; acc[3][3] += vx3 * vw3;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    float biases[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    [branch] if ((groupN + tx +  0) < N) biases[0] = asfloat(In2.Load((groupN + tx +  0) << 2));
    [branch] if ((groupN + tx + 16) < N) biases[1] = asfloat(In2.Load((groupN + tx + 16) << 2));
    [branch] if ((groupN + tx + 32) < N) biases[2] = asfloat(In2.Load((groupN + tx + 32) << 2));
    [branch] if ((groupN + tx + 48) < N) biases[3] = asfloat(In2.Load((groupN + tx + 48) << 2));

    [unroll]
    for (uint r = 0; r < 4; r++)
    {
        uint gm = groupM + ty + r * 8;
        if (gm >= M) continue;
        [unroll]
        for (uint c = 0; c < 4; c++)
        {
            uint gn = groupN + tx + c * 16;
            if (gn >= N) continue;
            float val = max(0.0f, acc[r][c] + biases[c]);
            Out0.Store((gm * N + gn) << 2, asuint(val));
        }
    }
}

// 4. Transpose [T, W, C] -> [T, C, W].
// P0 = T, P1 = W, P2 = C
[numthreads(256, 1, 1)]
void SubTransposeMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint idx = (gid.y * 65535u + gid.x) * 256u + gtid.x;
    const uint T = P0, W = P1, C = P2;
    const uint total = T * C * W;
    if (idx >= total) return;

    const uint w = idx % W;
    const uint tc = idx / W;
    const uint c = tc % C;
    const uint t = tc / C;
    uint srcIdx = (t * W + w) * C + c;
    Out0.Store(idx << 2, In0.Load(srcIdx << 2));
}
