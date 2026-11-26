using static ParakeetRuntime.Net.D3D12Native;

namespace ParakeetRuntime.Net;

internal enum GpuEpilogue : uint
{
    Store = 0,
    Silu = 1,
    AddResidual = 2,
    AddHalfResidual = 3,
}

internal readonly record struct GpuQ8Weight(int Rows, int K, ulong QByteOffset, ulong DByteOffset, ulong BiasByteOffset, bool HasBias);

internal sealed unsafe class ParakeetGpu : IDisposable
{
    internal const int MinDecodeBatchSteps = 4;
    private const int UncalibratedDecodeBatchSteps = 64;
    private const int MaxDecodeBatchSteps = 512;
    private const int MinDecodedTokensCapacity = 4096;
    private const int InlineTokenCopyCount = 1024;
    private const double SubmitBudgetMsUnderGpuTimeout = 350.0;
    private const uint MaxDispatchX = 65535u;
    private const int LstmHiddenPingPongSlots = 2;
    private const uint GemvModeJointPredFromPingPongSlot = 3u;
    private const uint GemvModeJointOutWithFusedRelu = 4u;

    private readonly record struct LstmLayerGpu(ulong QByteOffset, ulong DByteOffset, uint HhQByteOffset, uint HhDByteOffset, ulong BiasByteOffset);

    private sealed class LayerGpu
    {
        public GpuQ8Weight Ff1a, Ff1b, Ff2a, Ff2b, Pw1, Pw2, Qkv, Outp, Pos;
        public ulong NFf1g, NFf1b, NAttg, NAttb, NConvg, NConvb, NFf2g, NFf2b, NOutg, NOutb;
        public ulong Dw, BnScale, BnShift, U, V;
    }

    private readonly D3D12Device gpu;
    private readonly int d, heads, dk, ff, kernel, peRows, jointDim, melBins, subCh;
    private readonly int melNFft, melHop, melWin;
    private readonly float melPreemph;
    private readonly int predH, predLayers, vocab, blank, maxSymbols, durationCount;
    private readonly uint durPackedLo, durPackedHi;
    private readonly GpuBuffer wqMega, wdMega, floatMega;
    private readonly ulong peOffset, embedOffset, melWinOff, melFbOff, melTwidOff;
    private readonly ulong c0wOff, c0bOff, c2wOff, c2bOff, c3wOff, c3bOff, c5wOff, c5bOff, c6wOff, c6bOff;
    private readonly GpuQ8Weight subOutWeight, jointEncWeight, jointPredWeight, jointOutWeight;
    private readonly LstmLayerGpu[] lstmLayers;
    private readonly LayerGpu[] layers;

    internal static readonly (string File, string Entry)[] Pipelines =
    [
        ("EncoderOps.hlsl", "LayerNormMain"),
        ("EncoderOps.hlsl", "RelPosAttentionMain"),
        ("EncoderOps.hlsl", "GluMain"),
        ("EncoderOps.hlsl", "DepthwiseConvBnSiluMain"),
        ("SubsampleOps.hlsl", "SubConv0ReluMain"),
        ("SubsampleOps.hlsl", "SubDwConvMain"),
        ("SubsampleOps.hlsl", "F32GemmReluMain"),
        ("SubsampleOps.hlsl", "SubTransposeMain"),
        ("DecoderOps.hlsl", "InitDecStateMain"),
        ("DecoderOps.hlsl", "ZeroBufferMain"),
        ("DecoderOps.hlsl", "LstmLayerCondMain"),
        ("DecoderOps.hlsl", "Q8GemvCondMain"),
        ("DecoderOps.hlsl", "ArgMaxAndStepMain"),
        ("MelOps.hlsl", "MelFftLogMain"),
        ("MelOps.hlsl", "MelNormalizeMain"),
        ("Q8Gemm.hlsl", "Q8GemmSmallMain"),
    ];

    internal static readonly (string File, string Entry) LargeGemmPipeline = ("Q8Gemm.hlsl", "Q8GemmMain");

    private readonly Task<nint> largeGemmCompile;
    private nint psoQ8GemmSmall, psoLn, psoAtt, psoGlu, psoDw;
    private nint psoSubConv0, psoSubDw, psoF32GemmRelu, psoSubTranspose;
    private nint psoInitDec, psoZero, psoLstmLayer, psoQ8GemvCond, psoArgMaxStep;
    private nint psoMelFftLog, psoMelNorm;
    private double msPerLayerFrame;
    private double msPerDecodeStep;

    private int scratchSamples, scratchFrames, scratchT, decTokensCap;
    private GpuBuffer? bufAudio, bufAudioUpload;
    private GpuBuffer? bufFeats, bufSubA, bufSubB, bufFeatsUpload;
    private GpuBuffer? bufX, bufA, bufB, bufWide, bufEncProj;
    private GpuBuffer? decTokens, decTokensRb;
    private const int MostFramesReadingTheCachedPositions = 251;
    private const ulong SmallStagingQBytes = 16UL * 1024 * 1024;
    private const int QkvTensorsStagedTogether = 3;
    private GpuBuffer? projectedPositionsUpTo20s;
    private bool positionsCanBeCached;
    private ulong ProjectedPositionsBytesPerLayer => (ulong)(2 * MostFramesReadingTheCachedPositions - 1) * (ulong)d * 4;

    private readonly GpuBuffer[] decH, decC;
    private readonly GpuBuffer decPredProj, decLogits;
    private readonly GpuBuffer decState, decStateRb, decTsRb;
    private bool disposed;

    public string AdapterName => gpu.AdapterName;
    public int PeRows => peRows;
    public long GpuMemoryUsedBytes => gpu.QueryVideoMemoryUsageBytes();
    public bool CanRunGpuMel => melNFft == 512 && melBins <= 256;

    private ParakeetGpu(
        D3D12Device gpu, int d, int heads, int ff, int kernel, int peRows, int jointDim, int melBins, int subCh,
        int melNFft, int melHop, int melWin, float melPreemph,
        int predH, int predLayers, int vocab, int blank, int maxSymbols, int[] durations,
        GpuBuffer wqMega, GpuBuffer wdMega, GpuBuffer floatMega, ulong peOffset, ulong embedOffset,
        ulong melWinOff, ulong melFbOff, ulong melTwidOff,
        ulong c0wOff, ulong c0bOff, ulong c2wOff, ulong c2bOff, ulong c3wOff, ulong c3bOff, ulong c5wOff, ulong c5bOff, ulong c6wOff, ulong c6bOff,
        GpuQ8Weight subOutWeight, GpuQ8Weight jointEncWeight, GpuQ8Weight jointPredWeight, GpuQ8Weight jointOutWeight,
        LstmLayerGpu[] lstmLayers, LayerGpu[] layers,
        nint[] psos, Task<nint> largeGemmCompile)
    {
        this.gpu = gpu;
        this.d = d;
        this.heads = heads;
        dk = d / heads;
        this.ff = ff;
        this.kernel = kernel;
        this.peRows = peRows;
        positionsCanBeCached = (peRows + 1) / 2 >= MostFramesReadingTheCachedPositions;
        this.jointDim = jointDim;
        this.melBins = melBins;
        this.subCh = subCh;
        this.melNFft = melNFft;
        this.melHop = melHop;
        this.melWin = melWin;
        this.melPreemph = melPreemph;
        this.predH = predH;
        this.predLayers = predLayers;
        this.vocab = vocab;
        this.blank = blank;
        this.maxSymbols = maxSymbols;
        durationCount = durations.Length;
        for (int i = 0; i < durations.Length && i < 8; i++)
        {
            if (i < 4) durPackedLo |= ((uint)durations[i] & 0xFFu) << (i * 8);
            else durPackedHi |= ((uint)durations[i] & 0xFFu) << ((i - 4) * 8);
        }
        this.wqMega = wqMega;
        this.wdMega = wdMega;
        this.floatMega = floatMega;
        this.peOffset = peOffset;
        this.embedOffset = embedOffset;
        this.melWinOff = melWinOff;
        this.melFbOff = melFbOff;
        this.melTwidOff = melTwidOff;
        this.c0wOff = c0wOff; this.c0bOff = c0bOff;
        this.c2wOff = c2wOff; this.c2bOff = c2bOff;
        this.c3wOff = c3wOff; this.c3bOff = c3bOff;
        this.c5wOff = c5wOff; this.c5bOff = c5bOff;
        this.c6wOff = c6wOff; this.c6bOff = c6bOff;
        this.subOutWeight = subOutWeight;
        this.jointEncWeight = jointEncWeight;
        this.jointPredWeight = jointPredWeight;
        this.jointOutWeight = jointOutWeight;
        this.lstmLayers = lstmLayers;
        this.layers = layers;
        this.largeGemmCompile = largeGemmCompile;
        psoLn = psos[0]; psoAtt = psos[1];
        psoGlu = psos[2]; psoDw = psos[3];
        psoSubConv0 = psos[4]; psoSubDw = psos[5];
        psoF32GemmRelu = psos[6]; psoSubTranspose = psos[7];
        psoInitDec = psos[8]; psoZero = psos[9]; psoLstmLayer = psos[10];
        psoQ8GemvCond = psos[11]; psoArgMaxStep = psos[12];
        psoMelFftLog = psos[13]; psoMelNorm = psos[14];
        psoQ8GemmSmall = psos[15];

        decH = new GpuBuffer[predLayers];
        decC = new GpuBuffer[predLayers];
        ulong hBytes = (ulong)predH * 4;
        try
        {
            for (int i = 0; i < predLayers; i++)
            {
                decH[i] = gpu.CreateDefaultBuffer(hBytes * LstmHiddenPingPongSlots, allowUav: true);
                decC[i] = gpu.CreateDefaultBuffer(hBytes, allowUav: true);
            }
            decPredProj = gpu.CreateDefaultBuffer((ulong)jointDim * 4, allowUav: true);
            decLogits = gpu.CreateDefaultBuffer((ulong)(vocab + durationCount) * 4, allowUav: true);
            decState = gpu.CreateDefaultBuffer(256, allowUav: true);
            decStateRb = gpu.CreateReadbackBuffer(256);
            decTsRb = gpu.CreateReadbackBuffer(256);
        }
        catch
        {
            foreach (var b in decH) b?.Dispose();
            foreach (var b in decC) b?.Dispose();
            decPredProj?.Dispose();
            decLogits?.Dispose(); decState?.Dispose(); decStateRb?.Dispose(); decTsRb?.Dispose();
            throw;
        }
    }

    public static ParakeetGpu? TryCreate(
        Gguf g, Mel mel, int layersN, int d, int heads, int ff, int kernel, int subCh,
        int predH, int predLayers, int vocab, int blank, int maxSymbols, int[] durations,
        string? adapterNameFilter = null)
    {
        if (d <= 0 || heads <= 0 || d % heads != 0 || (d / heads) > 128 || (d / heads) % 4 != 0 ||
            d % 32 != 0 || ff <= 0 || ff % 32 != 0 || subCh <= 0 || subCh % 32 != 0 ||
            kernel <= 0 || (kernel & 1) == 0 || layersN <= 0 ||
            predH <= 0 || predH > 1024 || predH % 32 != 0 || predLayers <= 0 ||
            vocab <= 0 || blank < 0 || blank >= vocab || durations.Length is 0 or > 8 || maxSymbols <= 0)
            return null;
        var gpu = D3D12Device.TryCreate(adapterNameFilter: adapterNameFilter);
        if (gpu is null)
            return null;

        int melBins = mel.Mels;
        GpuBuffer? wqMega = null, wdMega = null, floatMega = null;
        var compiledPsos = new nint[Pipelines.Length];
        Task? compiling = null;
        Task<nint>? largeGemm = null;
        try
        {
            var melTwiddles = mel.BuildTwiddlesF32();
            var pe = g.Floats("encoder.pos_enc.pe");
            var c0w = g.Floats("encoder.pre_encode.conv.0.weight"); var c0b = g.Floats("encoder.pre_encode.conv.0.bias");
            var c2w = g.Floats("encoder.pre_encode.conv.2.weight"); var c2b = g.Floats("encoder.pre_encode.conv.2.bias");
            var c3w = g.Floats("encoder.pre_encode.conv.3.weight"); var c3b = g.Floats("encoder.pre_encode.conv.3.bias");
            var c5w = g.Floats("encoder.pre_encode.conv.5.weight"); var c5b = g.Floats("encoder.pre_encode.conv.5.bias");
            var c6w = g.Floats("encoder.pre_encode.conv.6.weight"); var c6b = g.Floats("encoder.pre_encode.conv.6.bias");
            var subOutShape = g.Q8Shape("encoder.pre_encode.out.weight"); var subOutBias = g.Floats("encoder.pre_encode.out.bias");
            var jointEncShape = g.Q8Shape("joint.enc.weight"); var jointEncBias = g.Floats("joint.enc.bias");
            var jointPredShape = g.Q8Shape("joint.pred.weight"); var jointPredBias = g.Floats("joint.pred.bias");
            var jointOutShape = g.Q8Shape("joint.joint_net.2.weight"); var jointOutBias = g.Floats("joint.joint_net.2.bias");
            var embed = g.Floats("decoder.prediction.embed.weight");
            var ihShapes = new (int Rows, int K)[predLayers];
            var hhShapes = new (int Rows, int K)[predLayers];
            var ihBiases = new float[predLayers][];
            var hhBiases = new float[predLayers][];
            for (int i = 0; i < predLayers; i++)
            {
                ihShapes[i] = g.Q8Shape($"decoder.prediction.dec_rnn.lstm.ih_l{i}.weight");
                ihBiases[i] = g.Floats($"decoder.prediction.dec_rnn.lstm.ih_l{i}.bias");
                hhShapes[i] = g.Q8Shape($"decoder.prediction.dec_rnn.lstm.hh_l{i}.weight");
                hhBiases[i] = g.Floats($"decoder.prediction.dec_rnn.lstm.hh_l{i}.bias");
            }

            int subW3 = (((melBins - 1) / 2 + 1 - 1) / 2 + 1 - 1) / 2 + 1;
            static void Require(bool cond, string message)
            {
                if (!cond) throw new InvalidDataException(message);
            }
            Require(pe.Length >= d && pe.Length % d == 0, "Invalid pos_enc.pe shape.");
            Require(embed.Length == vocab * predH, "Invalid decoder embed shape.");
            Require(c0w.Length == subCh * 9 && c0b.Length == subCh, "Invalid subsampling conv.0 shape.");
            Require(c2w.Length == subCh * 9 && c2b.Length == subCh, "Invalid subsampling conv.2 shape.");
            Require(c3w.Length == subCh * subCh && c3b.Length == subCh, "Invalid subsampling conv.3 shape.");
            Require(c5w.Length == subCh * 9 && c5b.Length == subCh, "Invalid subsampling conv.5 shape.");
            Require(c6w.Length == subCh * subCh && c6b.Length == subCh, "Invalid subsampling conv.6 shape.");
            Require(subOutShape.Rows == d && subOutShape.K == subCh * subW3 && subOutShape.K % 32 == 0 && subOutBias.Length == d, "Invalid subsampling out shape.");
            Require(jointEncShape.Rows is > 0 and <= 1024 && jointEncShape.Rows % 32 == 0 && jointEncShape.K == d && jointEncBias.Length == jointEncShape.Rows, "Invalid joint.enc shape.");
            Require(jointPredShape.Rows == jointEncShape.Rows && jointPredShape.K == predH && jointPredBias.Length == jointEncShape.Rows, "Invalid joint.pred shape.");
            Require(jointOutShape.Rows == vocab + durations.Length && jointOutShape.K == jointEncShape.Rows && jointOutBias.Length == jointOutShape.Rows, "Invalid joint.joint_net.2 shape.");
            for (int i = 0; i < predLayers; i++)
            {
                Require(ihShapes[i].Rows == 4 * predH && ihShapes[i].K == predH && ihBiases[i].Length == 4 * predH, $"Invalid LSTM ih_l{i} shape.");
                Require(hhShapes[i].Rows == 4 * predH && hhShapes[i].K == predH && hhBiases[i].Length == 4 * predH, $"Invalid LSTM hh_l{i} shape.");
            }
            for (int i = 0; i < layersN; i++)
            {
                string p = $"encoder.layers.{i}.";
                Require(g.Q8Shape(p + "feed_forward1.linear1.weight") == (ff, d), $"Invalid {p}ff1.linear1 shape.");
                Require(g.Q8Shape(p + "feed_forward1.linear2.weight") == (d, ff), $"Invalid {p}ff1.linear2 shape.");
                Require(g.Q8Shape(p + "feed_forward2.linear1.weight") == (ff, d), $"Invalid {p}ff2.linear1 shape.");
                Require(g.Q8Shape(p + "feed_forward2.linear2.weight") == (d, ff), $"Invalid {p}ff2.linear2 shape.");
                Require(g.Q8Shape(p + "conv.pointwise_conv1.weight") == (2 * d, d), $"Invalid {p}pw1 shape.");
                Require(g.Q8Shape(p + "conv.pointwise_conv2.weight") == (d, d), $"Invalid {p}pw2 shape.");
                Require(g.Q8Shape(p + "self_attn.linear_q.weight") == (d, d), $"Invalid {p}linear_q shape.");
                Require(g.Q8Shape(p + "self_attn.linear_k.weight") == (d, d), $"Invalid {p}linear_k shape.");
                Require(g.Q8Shape(p + "self_attn.linear_v.weight") == (d, d), $"Invalid {p}linear_v shape.");
                Require(g.Q8Shape(p + "self_attn.linear_out.weight") == (d, d), $"Invalid {p}linear_out shape.");
                Require(g.Q8Shape(p + "self_attn.linear_pos.weight") == (d, d), $"Invalid {p}linear_pos shape.");
            }

            largeGemm = Task.Run(() => gpu.EmbeddedPipeline(LargeGemmPipeline.File, LargeGemmPipeline.Entry));
            compiling = Task.Run(() => Parallel.For(0, Pipelines.Length, idx =>
                compiledPsos[idx] = gpu.EmbeddedPipeline(Pipelines[idx].File, Pipelines[idx].Entry)));

            (int Rows, int K)[] perLayerQ8 =
            [
                (ff, d), (d, ff), (ff, d), (d, ff),
                (2 * d, d), (d, d), (3 * d, d), (d, d), (d, d),
            ];
            ulong totalQ = 0, totalD = 0;
            void AddQ8Size(int rows, int k)
            {
                totalQ += D3D12Device.Align256((ulong)rows * (ulong)k);
                ulong halves = (ulong)rows * (ulong)(k / 32);
                totalD += D3D12Device.Align256(((halves + 1) / 2) * 4);
            }
            AddQ8Size(subOutShape.Rows, subOutShape.K);
            for (int i = 0; i < layersN; i++)
                foreach (var (r, k) in perLayerQ8) AddQ8Size(r, k);
            AddQ8Size(jointEncShape.Rows, jointEncShape.K);
            AddQ8Size(jointPredShape.Rows, jointPredShape.K);
            AddQ8Size(jointOutShape.Rows, jointOutShape.K);
            for (int i = 0; i < predLayers; i++)
            {
                AddQ8Size(ihShapes[i].Rows, ihShapes[i].K);
                AddQ8Size(hhShapes[i].Rows, hhShapes[i].K);
            }

            ulong totalF = 256;
            void AddFSize(int len) => totalF += D3D12Device.Align256((ulong)len * 4);
            AddFSize(pe.Length);
            AddFSize(embed.Length);
            AddFSize(mel.Window.Length);
            AddFSize(mel.Filterbank.Length);
            AddFSize(melTwiddles.Length);
            AddFSize(c0w.Length); AddFSize(c0b.Length);
            AddFSize(c2w.Length); AddFSize(c2b.Length);
            AddFSize(c3w.Length); AddFSize(c3b.Length);
            AddFSize(c5w.Length); AddFSize(c5b.Length);
            AddFSize(c6w.Length); AddFSize(c6b.Length);
            AddFSize(subOutBias.Length);
            AddFSize(jointEncBias.Length);
            AddFSize(jointPredBias.Length);
            AddFSize(jointOutBias.Length);
            for (int i = 0; i < predLayers; i++)
                AddFSize(ihBiases[i].Length);
            ulong perLayerFloats =
                14 * D3D12Device.Align256((ulong)d * 4) +
                D3D12Device.Align256((ulong)(d * kernel) * 4);
            totalF += (ulong)layersN * perLayerFloats;

            wqMega = gpu.CreateDefaultBuffer(totalQ, allowUav: false);
            wdMega = gpu.CreateDefaultBuffer(totalD, allowUav: false);
            floatMega = gpu.CreateDefaultBuffer(totalF, allowUav: false);

            ulong stageQCap = Math.Max(SmallStagingQBytes, D3D12Device.Align256((ulong)(QkvTensorsStagedTogether * g.LargestQ8TensorValues)));
            ulong stageDCap = stageQCap / 8;
            using var firstStageQ = gpu.CreateUploadBuffer(stageQCap);
            using var firstStageD = gpu.CreateUploadBuffer(stageDCap);
            using var secondStageQ = gpu.CreateUploadBuffer(stageQCap);
            using var secondStageD = gpu.CreateUploadBuffer(stageDCap);
            using var stageF = gpu.CreateUploadBuffer(totalF);
            using var stagingIsReleasedOnlyWhenTheGpuIsIdle = new WaitsForTheGpuOnDispose(gpu);
            var stageQ = firstStageQ;
            var stageD = firstStageD;
            var otherStageQ = secondStageQ;
            var otherStageD = secondStageD;
            ulong otherStageFreeAt = 0;

            ulong qCursor = 0, dCursor = 0, fCursor = 256;
            ulong batchQStart = 0, batchQUsed = 0;
            ulong batchDStart = 0, batchDUsed = 0;

            var pStageQ = (byte*)stageQ.Map();
            var pStageD = (byte*)stageD.Map();
            var pOtherStageQ = (byte*)otherStageQ.Map();
            var pOtherStageD = (byte*)otherStageD.Map();
            var pStageF = (byte*)stageF.Map();
            new Span<byte>(pStageF, 256).Clear();

            void FlushQ8Batch()
            {
                if (batchQUsed == 0 && batchDUsed == 0)
                    return;
                gpu.BeginCommands();
                gpu.Transition(wqMega, D3D12_RESOURCE_STATE_COPY_DEST);
                gpu.Transition(wdMega, D3D12_RESOURCE_STATE_COPY_DEST);
                if (batchQUsed > 0)
                    CopyBufferRegion(gpu.CommandList, wqMega.Resource, batchQStart, stageQ.Resource, 0, batchQUsed);
                if (batchDUsed > 0)
                    CopyBufferRegion(gpu.CommandList, wdMega.Resource, batchDStart, stageD.Resource, 0, batchDUsed);
                var thisStageFreeAt = gpu.Submit();
                gpu.WaitFor(otherStageFreeAt);
                (stageQ, otherStageQ) = (otherStageQ, stageQ);
                (stageD, otherStageD) = (otherStageD, stageD);
                var filledQ = pStageQ; pStageQ = pOtherStageQ; pOtherStageQ = filledQ;
                var filledD = pStageD; pStageD = pOtherStageD; pOtherStageD = filledD;
                otherStageFreeAt = thisStageFreeAt;
                batchQStart = qCursor;
                batchDStart = dCursor;
                batchQUsed = 0;
                batchDUsed = 0;
            }

            ulong AppendFloats(float[] arr)
            {
                ulong off = fCursor;
                ulong bytes = (ulong)arr.Length * 4;
                ulong aligned = D3D12Device.Align256(bytes);
                if (off + aligned > totalF)
                    throw new InvalidDataException("Float tensor exceeds preallocated GPU buffer.");
                arr.AsSpan().CopyTo(new Span<float>(pStageF + off, arr.Length));
                fCursor += aligned;
                return off;
            }

            GpuQ8Weight StreamQ8(string tensorName, float[]? bias = null)
            {
                var (rows, k) = g.Q8Shape(tensorName);
                ulong qBytes = (ulong)rows * (ulong)k;
                ulong dCount = (ulong)rows * (ulong)(k / 32);
                ulong qAligned = D3D12Device.Align256(qBytes);
                ulong dAligned = D3D12Device.Align256(((dCount + 1) / 2) * 4);
                if (qAligned > stageQCap || dAligned > stageDCap)
                    throw new InvalidDataException($"Q8 tensor {tensorName} exceeds GPU staging buffer capacity.");
                if (qCursor + qAligned > totalQ || dCursor + dAligned > totalD)
                    throw new InvalidDataException($"Q8 tensor {tensorName} exceeds preallocated GPU buffer.");
                if (batchQUsed + qAligned > stageQCap || batchDUsed + dAligned > stageDCap)
                    FlushQ8Batch();

                ulong qOff = qCursor;
                ulong dOff = dCursor;
                var pD = (ushort*)(pStageD + batchDUsed);
                g.CopyQ8Raw(tensorName, pStageQ + batchQUsed, pD);
                if ((dCount & 1) != 0)
                    pD[dCount] = 0;

                batchQUsed += qAligned;
                batchDUsed += dAligned;
                qCursor += qAligned;
                dCursor += dAligned;
                ulong bOff = bias is not null ? AppendFloats(bias) : 0;
                return new GpuQ8Weight(rows, k, qOff, dOff, bOff, bias is not null);
            }

            GpuQ8Weight StreamQ8Concat(params string[] names)
            {
                int totalRows = 0, k = -1;
                foreach (var n in names)
                {
                    var s = g.Q8Shape(n);
                    if (k < 0) k = s.K;
                    else if (s.K != k) throw new InvalidDataException($"Mismatched K across concatenated tensors ({n}).");
                    totalRows += s.Rows;
                }
                ulong qBytes = (ulong)totalRows * (ulong)k;
                ulong dCount = (ulong)totalRows * (ulong)(k / 32);
                ulong qAligned = D3D12Device.Align256(qBytes);
                ulong dAligned = D3D12Device.Align256(((dCount + 1) / 2) * 4);
                if (qAligned > stageQCap || dAligned > stageDCap)
                    throw new InvalidDataException("Concatenated Q8 tensor exceeds GPU staging buffer capacity.");
                if (qCursor + qAligned > totalQ || dCursor + dAligned > totalD)
                    throw new InvalidDataException("Concatenated Q8 tensor exceeds preallocated GPU buffer.");
                if (batchQUsed + qAligned > stageQCap || batchDUsed + dAligned > stageDCap)
                    FlushQ8Batch();

                ulong qOff = qCursor;
                ulong dOff = dCursor;
                ulong qSub = 0, dSub = 0;
                var pD = (ushort*)(pStageD + batchDUsed);
                foreach (var n in names)
                {
                    var (r, rk) = g.CopyQ8Raw(n, pStageQ + batchQUsed + qSub, pD + dSub);
                    qSub += (ulong)r * (ulong)rk;
                    dSub += (ulong)r * (ulong)(rk / 32);
                }
                if ((dCount & 1) != 0)
                    pD[dCount] = 0;

                batchQUsed += qAligned;
                batchDUsed += dAligned;
                qCursor += qAligned;
                dCursor += dAligned;
                return new GpuQ8Weight(totalRows, k, qOff, dOff, 0, false);
            }

            ulong peOffset = AppendFloats(pe);
            ulong embedOffset = AppendFloats(embed);
            ulong melWinOff = AppendFloats(mel.Window);
            ulong melFbOff = AppendFloats(mel.Filterbank);
            ulong melTwidOff = AppendFloats(melTwiddles);
            ulong c0wOff = AppendFloats(c0w), c0bOff = AppendFloats(c0b);
            ulong c2wOff = AppendFloats(c2w), c2bOff = AppendFloats(c2b);
            ulong c3wOff = AppendFloats(c3w), c3bOff = AppendFloats(c3b);
            ulong c5wOff = AppendFloats(c5w), c5bOff = AppendFloats(c5b);
            ulong c6wOff = AppendFloats(c6w), c6bOff = AppendFloats(c6b);
            var subOutWeight = StreamQ8("encoder.pre_encode.out.weight", subOutBias);
            var jointEncWeight = StreamQ8("joint.enc.weight", jointEncBias);
            var jointPredWeight = StreamQ8("joint.pred.weight", jointPredBias);
            var jointOutWeight = StreamQ8("joint.joint_net.2.weight", jointOutBias);
            var lstmLayers = new LstmLayerGpu[predLayers];
            for (int i = 0; i < predLayers; i++)
            {
                var ihW = StreamQ8($"decoder.prediction.dec_rnn.lstm.ih_l{i}.weight");
                var hhW = StreamQ8($"decoder.prediction.dec_rnn.lstm.hh_l{i}.weight");
                var combinedBias = new float[ihBiases[i].Length];
                for (int k = 0; k < combinedBias.Length; k++)
                    combinedBias[k] = ihBiases[i][k] + hhBiases[i][k];
                ulong biasOff = AppendFloats(combinedBias);
                ulong hhQDelta = hhW.QByteOffset - ihW.QByteOffset;
                ulong hhDDelta = hhW.DByteOffset - ihW.DByteOffset;
                Require(hhW.QByteOffset > ihW.QByteOffset && hhQDelta <= uint.MaxValue &&
                        hhW.DByteOffset > ihW.DByteOffset && hhDDelta <= uint.MaxValue,
                    $"Invalid LSTM weight layout for layer {i}.");
                lstmLayers[i] = new LstmLayerGpu(ihW.QByteOffset, ihW.DByteOffset, (uint)hhQDelta, (uint)hhDDelta, biasOff);
            }

            var layers = new LayerGpu[layersN];
            for (int i = 0; i < layersN; i++)
            {
                string p = $"encoder.layers.{i}.";
                var w = g.Floats(p + "conv.batch_norm.weight");
                var b = g.Floats(p + "conv.batch_norm.bias");
                var mean = g.Floats(p + "conv.batch_norm.running_mean");
                var variance = g.Floats(p + "conv.batch_norm.running_var");
                Require(w.Length == d && b.Length == d && mean.Length == d && variance.Length == d, $"Invalid {p}batch_norm shape.");
                var bnScale = new float[d];
                var bnShift = new float[d];
                for (int c = 0; c < d; c++)
                {
                    bnScale[c] = w[c] / MathF.Sqrt(variance[c] + 1e-5f);
                    bnShift[c] = b[c] - mean[c] * bnScale[c];
                }

                layers[i] = new LayerGpu
                {
                    NFf1g = AppendFloats(g.Floats(p + "norm_feed_forward1.weight")),
                    NFf1b = AppendFloats(g.Floats(p + "norm_feed_forward1.bias")),
                    NAttg = AppendFloats(g.Floats(p + "norm_self_att.weight")),
                    NAttb = AppendFloats(g.Floats(p + "norm_self_att.bias")),
                    NConvg = AppendFloats(g.Floats(p + "norm_conv.weight")),
                    NConvb = AppendFloats(g.Floats(p + "norm_conv.bias")),
                    NFf2g = AppendFloats(g.Floats(p + "norm_feed_forward2.weight")),
                    NFf2b = AppendFloats(g.Floats(p + "norm_feed_forward2.bias")),
                    NOutg = AppendFloats(g.Floats(p + "norm_out.weight")),
                    NOutb = AppendFloats(g.Floats(p + "norm_out.bias")),
                    Dw = AppendFloats(g.Floats(p + "conv.depthwise_conv.weight")),
                    BnScale = AppendFloats(bnScale),
                    BnShift = AppendFloats(bnShift),
                    U = AppendFloats(g.Floats(p + "self_attn.pos_bias_u")),
                    V = AppendFloats(g.Floats(p + "self_attn.pos_bias_v")),
                    Ff1a = StreamQ8(p + "feed_forward1.linear1.weight"),
                    Ff1b = StreamQ8(p + "feed_forward1.linear2.weight"),
                    Ff2a = StreamQ8(p + "feed_forward2.linear1.weight"),
                    Ff2b = StreamQ8(p + "feed_forward2.linear2.weight"),
                    Pw1 = StreamQ8(p + "conv.pointwise_conv1.weight"),
                    Pw2 = StreamQ8(p + "conv.pointwise_conv2.weight"),
                    Qkv = StreamQ8Concat(p + "self_attn.linear_q.weight", p + "self_attn.linear_k.weight", p + "self_attn.linear_v.weight"),
                    Outp = StreamQ8(p + "self_attn.linear_out.weight"),
                    Pos = StreamQ8(p + "self_attn.linear_pos.weight"),
                };
            }

            FlushQ8Batch();
            stageF.Unmap();

            gpu.BeginCommands();
            gpu.Transition(floatMega, D3D12_RESOURCE_STATE_COPY_DEST);
            CopyBufferRegion(gpu.CommandList, floatMega.Resource, 0, stageF.Resource, 0, fCursor);
            gpu.Transition(wqMega, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(wdMega, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.Transition(floatMega, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            gpu.ExecuteAndWait();
            compiling.GetAwaiter().GetResult();

            return new ParakeetGpu(
                gpu, d, heads, ff, kernel, pe.Length / d, jointEncShape.Rows, melBins, subCh,
                mel.NFft, mel.Hop, mel.Win, mel.Preemph,
                predH, predLayers, vocab, blank, maxSymbols, durations,
                wqMega, wdMega, floatMega, peOffset, embedOffset, melWinOff, melFbOff, melTwidOff,
                c0wOff, c0bOff, c2wOff, c2bOff, c3wOff, c3bOff, c5wOff, c5bOff, c6wOff, c6bOff,
                subOutWeight, jointEncWeight, jointPredWeight, jointOutWeight,
                lstmLayers, layers, compiledPsos, largeGemm);
        }
        catch
        {
            WaitUntilNoCompileCanStoreAPipeline(compiling);
            for (int i = 0; i < compiledPsos.Length; i++)
                SafeRelease(ref compiledPsos[i]);
            var large = WaitForLargeGemmPipelineOrZero(largeGemm);
            SafeRelease(ref large);
            floatMega?.Dispose();
            wdMega?.Dispose();
            wqMega?.Dispose();
            gpu.Dispose();
            return null;
        }
    }

    private sealed class WaitsForTheGpuOnDispose(D3D12Device gpu) : IDisposable
    {
        public void Dispose() => gpu.WaitUntilIdle();
    }

    private static void WaitUntilNoCompileCanStoreAPipeline(Task? compiling)
    {
        try { compiling?.Wait(); } catch (AggregateException) { }
    }

    private void EnsureScratch(int samples, int frames, int T)
    {
        if (samples > scratchSamples)
        {
            bufAudio?.Dispose();
            bufAudioUpload?.Dispose();
            scratchSamples = Math.Max(samples, Math.Max(160000, scratchSamples * 2));
            ulong audioBytes = (ulong)scratchSamples * 4;
            bufAudio = gpu.CreateDefaultBuffer(audioBytes, allowUav: false);
            bufAudioUpload = gpu.CreateUploadBuffer(audioBytes);
        }

        if (frames > scratchFrames)
        {
            bufFeats?.Dispose();
            bufSubA?.Dispose();
            bufSubB?.Dispose();
            bufFeatsUpload?.Dispose();

            scratchFrames = Math.Max(frames, Math.Max(1024, scratchFrames * 2));
            int h1 = (scratchFrames - 1) / 2 + 1, w1 = (melBins - 1) / 2 + 1;
            int h2 = (h1 - 1) / 2 + 1, w2 = (w1 - 1) / 2 + 1;
            ulong featsBytes = (ulong)scratchFrames * (ulong)melBins * 4;
            ulong subABytes = (ulong)h1 * (ulong)w1 * (ulong)subCh * 4;
            ulong subBBytes = (ulong)h2 * (ulong)w2 * (ulong)subCh * 4;

            bufFeats = gpu.CreateDefaultBuffer(featsBytes, allowUav: true);
            bufFeatsUpload = gpu.CreateUploadBuffer(featsBytes);
            bufSubA = gpu.CreateDefaultBuffer(subABytes, allowUav: true);
            bufSubB = gpu.CreateDefaultBuffer(subBBytes, allowUav: true);
        }

        if (T > scratchT)
        {
            bufX?.Dispose();
            bufA?.Dispose();
            bufB?.Dispose();
            bufWide?.Dispose();
            bufEncProj?.Dispose();

            scratchT = Math.Max(T, Math.Max(256, scratchT * 2));
            ulong tdBytes = (ulong)scratchT * (ulong)d * 4;
            ulong posBytes = (ulong)(2 * scratchT - 1) * (ulong)d * 4;
            ulong wideBytes = (ulong)scratchT * (ulong)Math.Max(Math.Max(ff, 3 * d), subOutWeight.K) * 4;
            ulong outBytes = (ulong)scratchT * (ulong)jointDim * 4;

            bufX = gpu.CreateDefaultBuffer(tdBytes, allowUav: true);
            bufA = gpu.CreateDefaultBuffer(tdBytes, allowUav: true);
            bufB = gpu.CreateDefaultBuffer(Math.Max(tdBytes, posBytes), allowUav: true);
            bufWide = gpu.CreateDefaultBuffer(wideBytes, allowUav: true);
            bufEncProj = gpu.CreateDefaultBuffer(outBytes, allowUav: true);
        }

        int neededTokens = Math.Max(MinDecodedTokensCapacity, checked(T * Math.Max(maxSymbols, 1)));
        if (neededTokens > decTokensCap)
        {
            decTokens?.Dispose();
            decTokensRb?.Dispose();
            decTokensCap = neededTokens;
            ulong tokBytes = (ulong)decTokensCap * 4;
            decTokens = gpu.CreateDefaultBuffer(tokBytes, allowUav: true);
            decTokensRb = gpu.CreateReadbackBuffer(tokBytes);
        }
    }

    public List<int> TranscribeAudio(float[] audio, out double encMs, out double decMs)
    {
        var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        int n = audio.Length;
        int frames = n / melHop + 1;
        int valid = Math.Min(n / melHop, frames);
        int H0 = frames, W0 = melBins;
        int H1 = (H0 - 1) / 2 + 1, W1 = (W0 - 1) / 2 + 1;
        int H2 = (H1 - 1) / 2 + 1, W2 = (W1 - 1) / 2 + 1;
        int H3 = (H2 - 1) / 2 + 1, W3 = (W2 - 1) / 2 + 1;
        int T = ParakeetModel.EncoderFramesOfValidMelFrames(H3, valid);
        if (T <= 0)
        {
            encMs = decMs = 0;
            return [];
        }

        EnsureScratch(n, frames, T);
        ulong audioBytes = (ulong)n * 4;
        audio.AsSpan().CopyTo(new Span<float>((float*)bufAudioUpload!.Map(), n));
        bufAudioUpload.Unmap();

        gpu.BeginCommands();
        gpu.Transition(bufAudio!, D3D12_RESOURCE_STATE_COPY_DEST);
        CopyBufferRegion(gpu.CommandList, bufAudio!.Resource, 0, bufAudioUpload.Resource, 0, audioBytes);
        gpu.Transition(bufAudio, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        RecordLogMelSpectrogram(n, frames, valid);

        return RunFromFeatsInOpenCommandList(t0, H0, W0, H1, W1, H2, W2, H3, W3, T, out encMs, out decMs);
    }

    private void RecordLogMelSpectrogram(int n, int frames, int valid)
    {
        if (bufFeats!.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(bufFeats);
        else gpu.Transition(bufFeats, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (valid > 0)
        {
            SetPipelineState(gpu.CommandList, psoMelFftLog);
            int woff = (melNFft - melWin) / 2;
            Span<uint> cm = [(uint)n, (uint)melHop, (uint)melWin, (uint)woff, (uint)melBins, BitConverter.SingleToUInt32Bits(melPreemph), (uint)valid];
            gpu.SetRootConstants(cm);
            gpu.SetSrv(0, bufAudio!);
            gpu.SetSrv(1, floatMega, melWinOff);
            gpu.SetSrv(2, floatMega, melFbOff);
            gpu.SetSrv(3, floatMega, melTwidOff);
            gpu.SetUav(0, bufFeats);
            Dispatch1D((ulong)valid);
            gpu.UavBarrier(bufFeats);
        }
        SetPipelineState(gpu.CommandList, psoMelNorm);
        Span<uint> cn = [(uint)frames, (uint)valid, (uint)melBins];
        gpu.SetRootConstants(cn);
        gpu.SetUav(0, bufFeats);
        Dispatch(gpu.CommandList, (uint)melBins, 1, 1);
    }

    public List<int> TranscribeFromMel(float[] feats, int frames, out double encMs, out double decMs)
    {
        var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        int H0 = frames, W0 = melBins;
        int H1 = (H0 - 1) / 2 + 1, W1 = (W0 - 1) / 2 + 1;
        int H2 = (H1 - 1) / 2 + 1, W2 = (W1 - 1) / 2 + 1;
        int H3 = (H2 - 1) / 2 + 1, W3 = (W2 - 1) / 2 + 1;
        int T = ParakeetModel.EncoderFramesOfValidMelFrames(H3, frames - Mel.ZeroedLastFrames);
        if (T <= 0)
        {
            encMs = decMs = 0;
            return [];
        }

        EnsureScratch(0, frames, T);
        ulong featsBytes = (ulong)(frames * melBins) * 4;
        feats.AsSpan(0, frames * melBins).CopyTo(new Span<float>((float*)bufFeatsUpload!.Map(), frames * melBins));
        bufFeatsUpload.Unmap();

        gpu.BeginCommands();
        gpu.Transition(bufFeats!, D3D12_RESOURCE_STATE_COPY_DEST);
        CopyBufferRegion(gpu.CommandList, bufFeats!.Resource, 0, bufFeatsUpload.Resource, 0, featsBytes);
        return RunFromFeatsInOpenCommandList(t0, H0, W0, H1, W1, H2, W2, H3, W3, T, out encMs, out decMs);
    }

    private List<int> RunFromFeatsInOpenCommandList(long t0, int H0, int W0, int H1, int W1, int H2, int W2, int H3, int W3, int T, out double encMs, out double decMs)
    {
        var curX = bufX!;
        var curA = bufA!;
        var curB = bufB!;
        var curWide = bufWide!;

        int center = (peRows + 1) / 2;
        int peStart = center - T;
        int P = 2 * T - 1;
        uint peSliceByteOffset = (uint)((long)peStart * d * 4);
        uint scaleBits = BitConverter.SingleToUInt32Bits(1f / MathF.Sqrt(dk));

        RecordSubsampling(H0, W0, H1, W1, H2, W2, H3, W3, T, curWide, curX);
        if (T <= MostFramesReadingTheCachedPositions && projectedPositionsUpTo20s is null && positionsCanBeCached)
            ProjectPositionsUpTo20sInTheirOwnSubmit();

        double toMs = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        int startLayer = 0;
        bool layerTimeNotYetCalibrated = msPerLayerFrame <= 0 && layers.Length > 0;
        if (layerTimeNotYetCalibrated)
        {
            gpu.ExecuteAndWait();
            gpu.BeginCommands();
            RecordConformerLayer(0, layers[0], ref curX, ref curA, curB, curWide, T, P, peSliceByteOffset, scaleBits);
            long layer0Start = System.Diagnostics.Stopwatch.GetTimestamp();
            gpu.ExecuteAndWait();
            double layer0Ms = (System.Diagnostics.Stopwatch.GetTimestamp() - layer0Start) * toMs;
            msPerLayerFrame = Math.Max(layer0Ms / Math.Max(T, 32), 1e-5);
            startLayer = 1;
            gpu.BeginCommands();
        }

        int layersPerSubmit = msPerLayerFrame > 0
            ? Math.Clamp((int)(SubmitBudgetMsUnderGpuTimeout / (msPerLayerFrame * Math.Max(T, 1))), 1, layers.Length)
            : layers.Length;
        int layersInBatch = 0;
        for (int i = startLayer; i < layers.Length; i++)
        {
            RecordConformerLayer(i, layers[i], ref curX, ref curA, curB, curWide, T, P, peSliceByteOffset, scaleBits);
            layersInBatch++;
            if (layersPerSubmit < layers.Length && layersInBatch >= layersPerSubmit && i + 1 < layers.Length)
            {
                long chunkStart = System.Diagnostics.Stopwatch.GetTimestamp();
                gpu.ExecuteAndWait();
                double chunkMs = (System.Diagnostics.Stopwatch.GetTimestamp() - chunkStart) * toMs;
                double sample = chunkMs / Math.Max(layersInBatch * T, 1);
                msPerLayerFrame = TrackSlowerAtOnceFasterGradually(msPerLayerFrame, sample);
                layersPerSubmit = Math.Clamp((int)(SubmitBudgetMsUnderGpuTimeout / (msPerLayerFrame * Math.Max(T, 1))), 1, layers.Length);
                layersInBatch = 0;
                gpu.BeginCommands();
            }
        }

        RecordQ8Gemm(curX, 0, bufEncProj!, T, jointEncWeight, GpuEpilogue.Store);
        gpu.Transition(bufEncProj!, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        bool splitSubmits = layersPerSubmit < layers.Length;
        long tEncEnd = 0;
        if (splitSubmits)
        {
            gpu.ExecuteAndWait();
            tEncEnd = System.Diagnostics.Stopwatch.GetTimestamp();
            gpu.BeginCommands();
        }
        bool useGpuTimestamps = !splitSubmits && gpu.HasTimestamps;
        if (useGpuTimestamps)
            gpu.WriteTimestamp(0);

        RecordInitDecoderState();
        double firstBatchBudgetMsLeftByEncoder = splitSubmits ? SubmitBudgetMsUnderGpuTimeout : Math.Max(SubmitBudgetMsUnderGpuTimeout - msPerLayerFrame * T * layers.Length, 0);
        int firstBatchSteps = DecodeBatchSteps(T, firstBatchBudgetMsLeftByEncoder, msPerDecodeStep);
        RecordDecodeSteps(T, firstBatchSteps);
        if (useGpuTimestamps)
            gpu.WriteTimestamp(1);
        CopyStateToReadback(resolveTimestamps: useGpuTimestamps);
        gpu.ExecuteAndWait();

        long tFirstDone = System.Diagnostics.Stopwatch.GetTimestamp();
        if (splitSubmits)
            LearnDecodeStepTime((tFirstDone - tEncEnd) * toMs, firstBatchSteps);

        ReadDecState(out uint curT, out uint tokenCount);
        int maxTotalSteps = Math.Max(T * (maxSymbols + 1) + 256, 512);
        int stepsExecuted = firstBatchSteps;
        while (curT < (uint)T)
        {
            if (stepsExecuted >= maxTotalSteps)
                throw new InvalidOperationException("GPU decoder exceeded maximum step limit without advancing.");
            int batchSteps = DecodeBatchSteps(T - (int)curT, SubmitBudgetMsUnderGpuTimeout, msPerDecodeStep);
            long batchStart = System.Diagnostics.Stopwatch.GetTimestamp();
            gpu.BeginCommands();
            RecordDecodeSteps(T, batchSteps);
            CopyStateToReadback(resolveTimestamps: false);
            gpu.ExecuteAndWait();
            LearnDecodeStepTime((System.Diagnostics.Stopwatch.GetTimestamp() - batchStart) * toMs, batchSteps);
            stepsExecuted += batchSteps;
            ReadDecState(out curT, out tokenCount);
        }

        if (tokenCount > (uint)decTokensCap)
            throw new InvalidOperationException($"GPU decoded {tokenCount} tokens, exceeding buffer capacity {decTokensCap}.");
        int count = (int)tokenCount;
        var tokens = new List<int>(count);
        if (count > 0)
        {
            if (count > InlineTokenCopyCount)
            {
                gpu.BeginCommands();
                gpu.Transition(decTokens!, D3D12_RESOURCE_STATE_COPY_SOURCE);
                CopyBufferRegion(gpu.CommandList, decTokensRb!.Resource, 0, decTokens!.Resource, 0, (ulong)count * 4);
                gpu.ExecuteAndWait();
            }

            var pTok = (uint*)decTokensRb!.Map(forRead: true);
            for (int i = 0; i < count; i++)
                tokens.Add((int)pTok[i]);
            decTokensRb.Unmap(wasWritten: false);
        }

        long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
        if (splitSubmits)
        {
            encMs = (tEncEnd - t0) * toMs;
            decMs = (t2 - tEncEnd) * toMs;
        }
        else
        {
            double firstSubmitMs = (tFirstDone - t0) * toMs;
            double firstDecMs = 0.0;
            if (useGpuTimestamps)
            {
                var pTs = (ulong*)decTsRb.Map(forRead: true);
                ulong ts0 = pTs[0], ts1 = pTs[1];
                decTsRb.Unmap(wasWritten: false);
                if (ts1 > ts0)
                    firstDecMs = Math.Clamp((ts1 - ts0) * 1000.0 / gpu.TimestampFrequency, 0.0, firstSubmitMs);
            }
            encMs = firstSubmitMs - firstDecMs;
            decMs = firstDecMs + (t2 - tFirstDone) * toMs;
        }
        return tokens;
    }

    private void RecordSubsampling(int H0, int W0, int H1, int W1, int H2, int W2, int H3, int W3, int T, GpuBuffer subsampledWide, GpuBuffer encoderInput)
    {
        RecordSubConv0Relu(bufFeats!, bufSubA!, H0, W0, H1, W1);
        RecordSubDwConv(bufSubA!, bufSubB!, c2wOff, c2bOff, H1, W1, H2, W2);
        RecordF32GemmRelu(bufSubB!, bufSubA!, c3wOff, c3bOff, H2 * W2, subCh, subCh);
        RecordSubDwConv(bufSubA!, bufSubB!, c5wOff, c5bOff, H2, W2, H3, W3);
        RecordF32GemmRelu(bufSubB!, bufSubA!, c6wOff, c6bOff, H3 * W3, subCh, subCh);
        RecordSubTranspose(bufSubA!, subsampledWide, T, W3, subCh);
        RecordQ8Gemm(subsampledWide, 0, encoderInput, T, subOutWeight, GpuEpilogue.Store);
    }

    internal static int DecodeBatchSteps(int remainingFrames, double budgetMs, double msPerStep)
    {
        int stepsForFramesLeft = remainingFrames * 3 / 5 + 16;
        int stepsFittingBudget = msPerStep > 0 ? (int)Math.Min(budgetMs / msPerStep, MaxDecodeBatchSteps) : UncalibratedDecodeBatchSteps;
        return Math.Clamp(Math.Min(stepsForFramesLeft, stepsFittingBudget), MinDecodeBatchSteps, MaxDecodeBatchSteps);
    }

    private void LearnDecodeStepTime(double decoderSubmitWallMs, int steps)
    {
        double sample = decoderSubmitWallMs / steps;
        msPerDecodeStep = msPerDecodeStep <= 0 ? sample : TrackSlowerAtOnceFasterGradually(msPerDecodeStep, sample);
    }

    private static double TrackSlowerAtOnceFasterGradually(double previous, double sample) =>
        Math.Max(previous * 0.7 + sample * 0.3, sample * 0.85);

    private void RecordConformerLayer(int layerIndex, LayerGpu l, ref GpuBuffer curX, ref GpuBuffer curA, GpuBuffer curB, GpuBuffer curWide, int T, int P, uint peSliceByteOffset, uint scaleBits)
    {
        RecordLayerNorm(curX, curA, l.NFf1g, l.NFf1b, T);
        RecordQ8Gemm(curA, 0, curWide, T, l.Ff1a, GpuEpilogue.Silu);
        RecordQ8Gemm(curWide, 0, curX, T, l.Ff1b, GpuEpilogue.AddHalfResidual);

        RecordLayerNorm(curX, curA, l.NAttg, l.NAttb, T);
        RecordQ8Gemm(curA, 0, curWide, T, l.Qkv, GpuEpilogue.Store);
        if (T <= MostFramesReadingTheCachedPositions && projectedPositionsUpTo20s is not null)
        {
            ulong rowOfPositionTMinus1 = (ulong)(MostFramesReadingTheCachedPositions - T);
            RecordAttention(curWide, projectedPositionsUpTo20s!, (ulong)layerIndex * ProjectedPositionsBytesPerLayer + rowOfPositionTMinus1 * (ulong)d * 4, curA, l.U, l.V, T, scaleBits);
        }
        else
        {
            RecordQ8GemmFromStatic(floatMega, peOffset, peSliceByteOffset, curB, P, l.Pos, GpuEpilogue.Store);
            RecordAttention(curWide, curB, 0, curA, l.U, l.V, T, scaleBits);
        }
        RecordQ8Gemm(curA, 0, curX, T, l.Outp, GpuEpilogue.AddResidual);

        RecordLayerNorm(curX, curA, l.NConvg, l.NConvb, T);
        RecordQ8Gemm(curA, 0, curWide, T, l.Pw1, GpuEpilogue.Store);
        RecordGlu(curWide, curA, T);
        RecordDepthwiseConvBnSilu(curA, curB, l.Dw, l.BnScale, l.BnShift, T);
        RecordQ8Gemm(curB, 0, curX, T, l.Pw2, GpuEpilogue.AddResidual);

        RecordLayerNorm(curX, curA, l.NFf2g, l.NFf2b, T);
        RecordQ8Gemm(curA, 0, curWide, T, l.Ff2a, GpuEpilogue.Silu);
        RecordQ8Gemm(curWide, 0, curX, T, l.Ff2b, GpuEpilogue.AddHalfResidual);

        RecordLayerNorm(curX, curA, l.NOutg, l.NOutb, T);
        (curX, curA) = (curA, curX);
    }

    private void RecordInitDecoderState()
    {
        if (decState.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(decState);
        else gpu.Transition(decState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoInitDec);
        gpu.SetUav(0, decState);
        Dispatch(gpu.CommandList, 1, 1, 1);

        for (int i = 0; i < predLayers; i++)
        {
            RecordZeroBuffer(decH[i], predH * LstmHiddenPingPongSlots);
            RecordZeroBuffer(decC[i], predH);
        }
    }

    private void RecordDecodeSteps(int T, int steps)
    {
        uint tLim = (uint)T;
        uint lstmGroups = (uint)((predH + 7) / 8);
        uint predGroups = (uint)((jointPredWeight.Rows + 15) / 16);
        uint outGroups = (uint)((jointOutWeight.Rows + 15) / 16);
        Span<D3D12_RESOURCE_BARRIER> barriers = stackalloc D3D12_RESOURCE_BARRIER[4];

        for (int step = 0; step < steps; step++)
        {
            RecordLstmLayers(tLim, lstmGroups, barriers);
            RecordJointPred(tLim, predGroups, barriers);
            RecordJointOutWithFusedRelu(tLim, outGroups, barriers);
            RecordArgMaxAndStep(tLim, barriers);
        }
    }

    private void RecordLstmLayers(uint tLim, uint lstmGroups, Span<D3D12_RESOURCE_BARRIER> barriers)
    {
        for (int i = 0; i < predLayers; i++)
        {
            int nb = 0;
            if (i == 0 && D3D12Device.TryTransition(decState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, out var bSt))
                barriers[nb++] = bSt;
            if (i > 0 && D3D12Device.TryTransition(decH[i - 1], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, out var bPrev))
                barriers[nb++] = bPrev;
            barriers[nb++] = D3D12Device.MakeUavBarrierOrTransition(decH[i]);
            barriers[nb++] = D3D12Device.MakeUavBarrierOrTransition(decC[i]);
            gpu.Barriers(barriers[..nb]);

            var ll = lstmLayers[i];
            SetPipelineState(gpu.CommandList, psoLstmLayer);
            Span<uint> cl = [(uint)predH, tLim, i == 0 ? 1u : 0u, ll.HhQByteOffset, ll.HhDByteOffset];
            gpu.SetRootConstants(cl);
            if (i == 0) gpu.SetSrv(0, floatMega, embedOffset);
            else gpu.SetSrv(0, decH[i - 1]);
            gpu.SetSrv(1, wqMega, ll.QByteOffset);
            gpu.SetSrv(2, wdMega, ll.DByteOffset);
            gpu.SetSrv(3, floatMega, ll.BiasByteOffset);
            gpu.SetSrv(4, decState);
            gpu.SetUav(0, decH[i]);
            gpu.SetUav(1, decC[i]);
            Dispatch(gpu.CommandList, lstmGroups, 1, 1);
        }
    }

    private void RecordJointPred(uint tLim, uint predGroups, Span<D3D12_RESOURCE_BARRIER> barriers)
    {
        int nb = 0;
        if (D3D12Device.TryTransition(decH[predLayers - 1], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, out var bH))
            barriers[nb++] = bH;
        barriers[nb++] = D3D12Device.MakeUavBarrierOrTransition(decPredProj);
        gpu.Barriers(barriers[..nb]);

        SetPipelineState(gpu.CommandList, psoQ8GemvCond);
        Span<uint> cp = [(uint)jointPredWeight.Rows, (uint)jointPredWeight.K, jointPredWeight.HasBias ? 1u : 0u, tLim, GemvModeJointPredFromPingPongSlot];
        gpu.SetRootConstants(cp);
        gpu.SetSrv(0, decH[predLayers - 1]);
        gpu.SetSrv(1, wqMega, jointPredWeight.QByteOffset);
        gpu.SetSrv(2, wdMega, jointPredWeight.DByteOffset);
        gpu.SetSrv(3, floatMega, jointPredWeight.BiasByteOffset);
        gpu.SetSrv(4, decState);
        gpu.SetUav(0, decPredProj);
        Dispatch(gpu.CommandList, predGroups, 1, 1);
    }

    private void RecordJointOutWithFusedRelu(uint tLim, uint outGroups, Span<D3D12_RESOURCE_BARRIER> barriers)
    {
        int nb = 0;
        if (D3D12Device.TryTransition(decPredProj, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, out var bP))
            barriers[nb++] = bP;
        barriers[nb++] = D3D12Device.MakeUavBarrierOrTransition(decLogits);
        gpu.Barriers(barriers[..nb]);

        SetPipelineState(gpu.CommandList, psoQ8GemvCond);
        Span<uint> co = [(uint)jointOutWeight.Rows, (uint)jointOutWeight.K, jointOutWeight.HasBias ? 1u : 0u, tLim, GemvModeJointOutWithFusedRelu];
        gpu.SetRootConstants(co);
        gpu.SetSrv(0, bufEncProj!);
        gpu.SetSrv(1, wqMega, jointOutWeight.QByteOffset);
        gpu.SetSrv(2, wdMega, jointOutWeight.DByteOffset);
        gpu.SetSrv(3, floatMega, jointOutWeight.BiasByteOffset);
        gpu.SetSrv(4, decState);
        gpu.SetSrv(5, decPredProj);
        gpu.SetUav(0, decLogits);
        Dispatch(gpu.CommandList, outGroups, 1, 1);
    }

    private void RecordArgMaxAndStep(uint tLim, Span<D3D12_RESOURCE_BARRIER> barriers)
    {
        int nb = 0;
        if (D3D12Device.TryTransition(decLogits, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, out var bL))
            barriers[nb++] = bL;
        barriers[nb++] = D3D12Device.MakeUavBarrierOrTransition(decState);
        barriers[nb++] = D3D12Device.MakeUavBarrierOrTransition(decTokens!);
        gpu.Barriers(barriers[..nb]);

        SetPipelineState(gpu.CommandList, psoArgMaxStep);
        Span<uint> ca = [(uint)vocab, (uint)blank, (uint)durationCount, tLim, (uint)maxSymbols, (uint)decTokensCap, durPackedLo, durPackedHi];
        gpu.SetRootConstants(ca);
        gpu.SetSrv(0, decLogits);
        gpu.SetUav(0, decState);
        gpu.SetUav(1, decTokens!);
        Dispatch(gpu.CommandList, 1, 1, 1);
    }

    private void CopyStateToReadback(bool resolveTimestamps)
    {
        Span<D3D12_RESOURCE_BARRIER> barriers = stackalloc D3D12_RESOURCE_BARRIER[2];
        int nb = 0;
        if (D3D12Device.TryTransition(decState, D3D12_RESOURCE_STATE_COPY_SOURCE, out var bSt))
            barriers[nb++] = bSt;
        if (D3D12Device.TryTransition(decTokens!, D3D12_RESOURCE_STATE_COPY_SOURCE, out var bTok))
            barriers[nb++] = bTok;
        gpu.Barriers(barriers[..nb]);
        CopyBufferRegion(gpu.CommandList, decStateRb.Resource, 0, decState.Resource, 0, 32);
        CopyBufferRegion(gpu.CommandList, decTokensRb!.Resource, 0, decTokens!.Resource, 0, (ulong)InlineTokenCopyCount * 4);
        if (resolveTimestamps)
            gpu.ResolveTimestamps(0, 2, decTsRb, 0);
    }

    private void ReadDecState(out uint curT, out uint tokenCount)
    {
        var p = (uint*)decStateRb.Map(forRead: true);
        curT = p[0];
        tokenCount = p[2];
        decStateRb.Unmap(wasWritten: false);
    }

    private void RecordZeroBuffer(GpuBuffer buf, int floatCount)
    {
        if (buf.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(buf);
        else gpu.Transition(buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoZero);
        Span<uint> c = [(uint)floatCount];
        gpu.SetRootConstants(c);
        gpu.SetUav(0, buf);
        Dispatch(gpu.CommandList, (uint)((floatCount + 255) / 256), 1, 1);
    }

    private void Dispatch1D(ulong totalGroups)
    {
        if (totalGroups == 0)
            return;
        if (totalGroups <= MaxDispatchX)
        {
            Dispatch(gpu.CommandList, (uint)totalGroups, 1, 1);
            return;
        }
        uint gy = (uint)((totalGroups + MaxDispatchX - 1) / MaxDispatchX);
        Dispatch(gpu.CommandList, MaxDispatchX, gy, 1);
    }

    private void RecordSubConv0Relu(GpuBuffer input, GpuBuffer output, int H, int W, int H1, int W1)
    {
        gpu.Transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (output.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(output);
        else gpu.Transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoSubConv0);
        Span<uint> c = [(uint)H, (uint)W, (uint)H1, (uint)W1, (uint)subCh];
        gpu.SetRootConstants(c);
        gpu.SetSrv(0, input);
        gpu.SetSrv(1, floatMega, c0wOff);
        gpu.SetSrv(2, floatMega, c0bOff);
        gpu.SetUav(0, output);
        ulong total = (ulong)H1 * (ulong)W1 * (ulong)subCh;
        Dispatch1D((total + 255UL) / 256UL);
    }

    private void RecordSubDwConv(GpuBuffer input, GpuBuffer output, ulong wOff, ulong bOff, int H0, int W0, int H2, int W2)
    {
        gpu.Transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (output.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(output);
        else gpu.Transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoSubDw);
        Span<uint> c = [(uint)H0, (uint)W0, (uint)H2, (uint)W2, (uint)subCh];
        gpu.SetRootConstants(c);
        gpu.SetSrv(0, input);
        gpu.SetSrv(1, floatMega, wOff);
        gpu.SetSrv(2, floatMega, bOff);
        gpu.SetUav(0, output);
        ulong total = (ulong)H2 * (ulong)W2 * (ulong)subCh;
        Dispatch1D((total + 255UL) / 256UL);
    }

    private void RecordF32GemmRelu(GpuBuffer input, GpuBuffer output, ulong wOff, ulong bOff, int M, int N, int K)
    {
        gpu.Transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (output.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(output);
        else gpu.Transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoF32GemmRelu);
        Span<uint> c = [(uint)M, (uint)N, (uint)K];
        gpu.SetRootConstants(c);
        gpu.SetSrv(0, input);
        gpu.SetSrv(1, floatMega, wOff);
        gpu.SetSrv(2, floatMega, bOff);
        gpu.SetUav(0, output);
        Dispatch(gpu.CommandList, (uint)((N + 63) / 64), (uint)((M + 31) / 32), 1);
    }

    private void RecordSubTranspose(GpuBuffer input, GpuBuffer output, int T, int W, int C)
    {
        gpu.Transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (output.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(output);
        else gpu.Transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoSubTranspose);
        Span<uint> c = [(uint)T, (uint)W, (uint)C];
        gpu.SetRootConstants(c);
        gpu.SetSrv(0, input);
        gpu.SetUav(0, output);
        ulong total = (ulong)T * (ulong)W * (ulong)C;
        Dispatch1D((total + 255UL) / 256UL);
    }

    private void RecordLayerNorm(GpuBuffer input, GpuBuffer output, ulong gammaOff, ulong betaOff, int T)
    {
        gpu.Transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (output.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(output);
        else gpu.Transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoLn);
        Span<uint> c = [(uint)d];
        gpu.SetRootConstants(c);
        gpu.SetSrv(0, input);
        gpu.SetSrv(1, floatMega, gammaOff);
        gpu.SetSrv(2, floatMega, betaOff);
        gpu.SetUav(0, output);
        Dispatch(gpu.CommandList, (uint)T, 1, 1);
    }

    private void RecordQ8Gemm(GpuBuffer input, uint xByteOffset, GpuBuffer output, int M, in GpuQ8Weight w, GpuEpilogue epilogue)
    {
        gpu.Transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (output.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(output);
        else gpu.Transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        DispatchQ8Gemm(input, 0, xByteOffset, output, M, w, epilogue);
    }

    private void RecordQ8GemmFromStatic(GpuBuffer staticInput, ulong srvOffset, uint xByteOffset, GpuBuffer output, int M, in GpuQ8Weight w, GpuEpilogue epilogue)
    {
        if (output.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(output);
        else gpu.Transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        DispatchQ8Gemm(staticInput, srvOffset, xByteOffset, output, M, w, epilogue);
    }

    private void DispatchQ8Gemm(GpuBuffer input, ulong inputSrvOffset, uint xByteOffset, GpuBuffer output, int M, in GpuQ8Weight w, GpuEpilogue epilogue, ulong outputUavOffset = 0)
    {
        var tile = Q8GemmTile(M);
        var pipeline = psoQ8GemmSmall;
        if (tile == Q8GemmLargeTile && largeGemmCompile.IsCompletedSuccessfully)
            pipeline = largeGemmCompile.Result;
        else
            tile = Q8GemmSmallTile;
        SetPipelineState(gpu.CommandList, pipeline);
        Span<uint> c = [(uint)M, (uint)w.Rows, (uint)w.K, w.HasBias ? 1u : 0u, (uint)epilogue, xByteOffset, 0u, 0u];
        gpu.SetRootConstants(c);
        gpu.SetSrv(0, input, inputSrvOffset);
        gpu.SetSrv(1, wqMega, w.QByteOffset);
        gpu.SetSrv(2, wdMega, w.DByteOffset);
        gpu.SetSrv(3, floatMega, w.BiasByteOffset);
        gpu.SetUav(0, output, outputUavOffset);
        Dispatch(gpu.CommandList, (uint)((w.Rows + tile - 1) / tile), (uint)((M + tile - 1) / tile), 1);
    }

    internal const int Q8GemmLargeTile = 64, Q8GemmSmallTile = 32;
    private const int RowsUpToWhichLargeTileUnderfillsTheGpu = 96;

    private static int Q8GemmTile(int rows) => rows > RowsUpToWhichLargeTileUnderfillsTheGpu ? Q8GemmLargeTile : Q8GemmSmallTile;

    private static nint WaitForLargeGemmPipelineOrZero(Task<nint>? compile)
    {
        try { return compile?.GetAwaiter().GetResult() ?? 0; }
        catch (Exception) { return 0; }
    }

    private void ProjectPositionsUpTo20sInTheirOwnSubmit()
    {
        GpuBuffer table;
        try
        {
            table = gpu.CreateDefaultBuffer((ulong)layers.Length * ProjectedPositionsBytesPerLayer, allowUav: true);
        }
        catch (Exception ex) when (ex is OutOfMemoryException or System.Runtime.InteropServices.COMException)
        {
            positionsCanBeCached = false;
            return;
        }
        gpu.ExecuteAndWait();
        gpu.BeginCommands();
        int rows = 2 * MostFramesReadingTheCachedPositions - 1;
        uint firstRowByteOffset = (uint)((long)((peRows + 1) / 2 - MostFramesReadingTheCachedPositions) * d * 4);
        gpu.Transition(table, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        for (int i = 0; i < layers.Length; i++)
            DispatchQ8Gemm(floatMega, peOffset, firstRowByteOffset, table, rows, layers[i].Pos, GpuEpilogue.Store, (ulong)i * ProjectedPositionsBytesPerLayer);
        gpu.Transition(table, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        projectedPositionsUpTo20s = table;
        gpu.ExecuteAndWait();
        gpu.BeginCommands();
    }

    private void RecordAttention(GpuBuffer qkv, GpuBuffer posP, ulong posByteOffset, GpuBuffer ctx, ulong uOff, ulong vOff, int T, uint scaleBits)
    {
        gpu.Transition(qkv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Transition(posP, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (ctx.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(ctx);
        else gpu.Transition(ctx, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoAtt);
        Span<uint> c = [(uint)T, (uint)d, (uint)dk, scaleBits];
        gpu.SetRootConstants(c);
        gpu.SetSrv(0, qkv);
        gpu.SetSrv(1, posP, posByteOffset);
        gpu.SetSrv(2, floatMega, uOff);
        gpu.SetSrv(3, floatMega, vOff);
        gpu.SetUav(0, ctx);
        Dispatch(gpu.CommandList, (uint)((T + 31) / 32), (uint)heads, 1);
    }

    private void RecordGlu(GpuBuffer pw1, GpuBuffer glu, int T)
    {
        gpu.Transition(pw1, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (glu.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(glu);
        else gpu.Transition(glu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoGlu);
        ulong total = (ulong)T * (ulong)d;
        Span<uint> c = [(uint)total, (uint)d];
        gpu.SetRootConstants(c);
        gpu.SetSrv(0, pw1);
        gpu.SetUav(0, glu);
        Dispatch1D((total + 255UL) / 256UL);
    }

    private void RecordDepthwiseConvBnSilu(GpuBuffer glu, GpuBuffer dwo, ulong dwOff, ulong bnScaleOff, ulong bnShiftOff, int T)
    {
        gpu.Transition(glu, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (dwo.CurrentState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) gpu.UavBarrier(dwo);
        else gpu.Transition(dwo, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        SetPipelineState(gpu.CommandList, psoDw);
        Span<uint> c = [(uint)T, (uint)d, (uint)kernel];
        gpu.SetRootConstants(c);
        gpu.SetSrv(0, glu);
        gpu.SetSrv(1, floatMega, dwOff);
        gpu.SetSrv(2, floatMega, bnScaleOff);
        gpu.SetSrv(3, floatMega, bnShiftOff);
        gpu.SetUav(0, dwo);
        ulong total = (ulong)T * (ulong)d;
        Dispatch1D((total + 255UL) / 256UL);
    }

    public void Dispose()
    {
        if (disposed)
            return;
        disposed = true;
        var largeGemm = WaitForLargeGemmPipelineOrZero(largeGemmCompile);
        SafeRelease(ref largeGemm); SafeRelease(ref psoQ8GemmSmall); SafeRelease(ref psoLn); SafeRelease(ref psoAtt);
        SafeRelease(ref psoGlu); SafeRelease(ref psoDw);
        SafeRelease(ref psoSubConv0); SafeRelease(ref psoSubDw);
        SafeRelease(ref psoF32GemmRelu); SafeRelease(ref psoSubTranspose);
        SafeRelease(ref psoInitDec); SafeRelease(ref psoZero); SafeRelease(ref psoLstmLayer);
        SafeRelease(ref psoQ8GemvCond); SafeRelease(ref psoArgMaxStep);
        SafeRelease(ref psoMelFftLog); SafeRelease(ref psoMelNorm);
        foreach (var b in decH) b.Dispose();
        foreach (var b in decC) b.Dispose();
        decPredProj.Dispose();
        decLogits.Dispose(); decState.Dispose(); decStateRb.Dispose(); decTsRb.Dispose();
        decTokens?.Dispose(); decTokensRb?.Dispose();
        bufAudio?.Dispose(); bufAudioUpload?.Dispose();
        bufFeats?.Dispose(); bufFeatsUpload?.Dispose();
        bufSubA?.Dispose(); bufSubB?.Dispose();
        bufX?.Dispose(); bufA?.Dispose(); bufB?.Dispose();
        bufWide?.Dispose(); bufEncProj?.Dispose();
        projectedPositionsUpTo20s?.Dispose();
        floatMega.Dispose();
        wdMega.Dispose();
        wqMega.Dispose();
        gpu.Dispose();
    }
}
