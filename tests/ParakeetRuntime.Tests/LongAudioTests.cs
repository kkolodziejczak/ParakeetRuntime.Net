using ParakeetRuntime;

namespace ParakeetRuntime.Tests;

public sealed class LongAudioTests
{
    private const int TwentySeconds = 320_000;

    [Fact]
    public async Task Audio_of_exactly_20_seconds_is_transcribed_in_one_call_as_given()
    {
        var model = new RecordingModel();
        using var engine = new SpeechEngine(model);
        var audio = Noise(TwentySeconds);

        await engine.TranscribeAsync(audio);

        Assert.Same(audio, Assert.Single(model.Pieces));
    }

    [Theory]
    [InlineData(TwentySeconds + 1)]
    [InlineData(3 * TwentySeconds + 7)]
    [InlineData(10 * 60 * 16_000)]
    public async Task Longer_audio_with_no_quiet_stretch_is_cut_into_pieces_of_20_seconds_or_less_with_no_sample_lost(int length)
    {
        var model = new RecordingModel();
        using var engine = new SpeechEngine(model);
        var audio = Noise(length);

        await engine.TranscribeAsync(audio);

        Assert.True(model.Pieces.Count > 1);
        Assert.All(model.Pieces, piece => Assert.InRange(piece.Length, 1, TwentySeconds));
        Assert.Equal(audio, model.Pieces.SelectMany(piece => piece));
    }

    [Fact]
    public async Task Long_audio_is_cut_inside_its_quietest_stretch()
    {
        var model = new RecordingModel();
        using var engine = new SpeechEngine(model);
        var audio = Noise(30 * 16_000);
        Array.Clear(audio, 14 * 16_000, 8_000);

        await engine.TranscribeAsync(audio);

        Assert.InRange(model.Pieces[0].Length, 14 * 16_000, 14 * 16_000 + 8_000);
    }

    [Fact]
    public async Task The_cut_is_at_the_centre_of_the_quietest_300_ms()
    {
        var model = new RecordingModel();
        using var engine = new SpeechEngine(model);
        var audio = Noise(30 * 16_000);
        Array.Clear(audio, 467 * 480, 10 * 480);

        await engine.TranscribeAsync(audio);

        Assert.Equal(472 * 480, model.Pieces[0].Length);
    }

    [Fact]
    public async Task Silent_audio_is_cut_as_late_as_a_whole_quiet_window_fits_in_20_seconds()
    {
        var model = new RecordingModel();
        using var engine = new SpeechEngine(model);

        await engine.TranscribeAsync(new float[30 * 16_000]);

        Assert.Equal(661 * 480, model.Pieces[0].Length);
    }

    [Fact]
    public async Task The_texts_of_the_pieces_are_joined_with_one_space_leaving_out_empty_ones()
    {
        var model = new RecordingModel(" Hello.", "", "world. ");
        using var engine = new SpeechEngine(model);

        var text = await engine.TranscribeAsync(Noise(5 * TwentySeconds / 2));

        Assert.Equal(3, model.Pieces.Count);
        Assert.Equal("Hello. world.", text);
    }

    [Fact]
    public async Task Cancelling_during_a_piece_finishes_that_piece_and_starts_no_other()
    {
        using var cancellation = new CancellationTokenSource();
        var model = new RecordingModel { WhileTranscribing = cancellation.Cancel };
        using var engine = new SpeechEngine(model);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => engine.TranscribeAsync(Noise(5 * TwentySeconds / 2), cancellation.Token));

        Assert.Single(model.Pieces);
    }

    [Fact]
    public async Task Cancelling_after_a_piece_that_lost_the_GPU_still_reports_the_CPU()
    {
        using var cancellation = new CancellationTokenSource();
        var model = new RecordingModel { IsUsingGpu = true };
        model.WhileTranscribing = () =>
        {
            model.IsUsingGpu = false;
            cancellation.Cancel();
        };
        using var engine = new SpeechEngine(model);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => engine.TranscribeAsync(Noise(5 * TwentySeconds / 2), cancellation.Token));

        Assert.False(engine.IsUsingGpu);
        Assert.Equal("CPU", engine.Backend);
    }

    [Fact]
    public async Task A_call_made_during_a_long_transcription_waits_for_all_of_its_pieces()
    {
        var firstPieceStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseFirstPiece = new ManualResetEventSlim();
        var model = new RecordingModel();
        model.WhileTranscribing = () =>
        {
            if (model.Pieces.Count == 1)
            {
                firstPieceStarted.SetResult();
                releaseFirstPiece.Wait();
            }
        };
        using var engine = new SpeechEngine(model);
        var shortAudio = Noise(16_000);

        var longCall = engine.TranscribeAsync(Noise(5 * TwentySeconds / 2));
        await firstPieceStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));
        var shortCall = engine.TranscribeAsync(shortAudio);
        releaseFirstPiece.Set();
        await Task.WhenAll(longCall, shortCall).WaitAsync(TimeSpan.FromSeconds(5));

        Assert.Same(shortAudio, model.Pieces[^1]);
        Assert.Equal(4, model.Pieces.Count);
    }

    [Fact]
    public async Task A_WAV_stream_is_read_from_its_position_and_left_open()
    {
        var model = new RecordingModel();
        using var engine = new SpeechEngine(model);
        var wav = new MemoryStream();
        wav.Write([1, 2, 3]);
        wav.Write(WavOf([0.25f, -0.5f]));
        wav.Position = 3;

        await engine.TranscribeWavAsync(wav);

        Assert.Equal([0.25f, -0.5f], Assert.Single(model.Pieces));
        Assert.True(wav.CanRead);
    }

    [Fact]
    public async Task A_WAV_file_is_transcribed_from_its_path()
    {
        var model = new RecordingModel("Hi.");
        using var engine = new SpeechEngine(model);
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.wav");
        await File.WriteAllBytesAsync(path, WavOf([0.25f]));
        try
        {
            Assert.Equal("Hi.", await engine.TranscribeWavAsync(path));
            Assert.Equal([0.25f], Assert.Single(model.Pieces));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task A_missing_WAV_file_is_reported_as_not_found()
    {
        using var engine = new SpeechEngine(new RecordingModel());

        await Assert.ThrowsAsync<FileNotFoundException>(() => engine.TranscribeWavAsync(Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.wav")));
    }

    [Fact]
    public async Task A_WAV_path_in_a_missing_folder_is_reported_as_not_found()
    {
        using var engine = new SpeechEngine(new RecordingModel());

        await Assert.ThrowsAsync<DirectoryNotFoundException>(() => engine.TranscribeWavAsync(Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}", "missing.wav")));
    }

    [Fact]
    public async Task Audio_that_is_not_WAV_is_rejected_before_anything_is_transcribed()
    {
        var model = new RecordingModel();
        using var engine = new SpeechEngine(model);

        await Assert.ThrowsAsync<InvalidDataException>(() => engine.TranscribeWavAsync(new MemoryStream("ID3 not a wav file"u8.ToArray())));

        Assert.Empty(model.Pieces);
    }

    private static float[] Noise(int length)
    {
        var random = new Random(1);
        var samples = new float[length];
        for (var i = 0; i < length; i++)
            samples[i] = (float)(random.NextDouble() - 0.5);
        return samples;
    }

    private static byte[] WavOf(float[] samples)
    {
        var data = new byte[samples.Length * 4];
        Buffer.BlockCopy(samples, 0, data, 0, data.Length);
        return [.. WavAudioTests.WavHeader(3, 32, 1, 16_000, (uint)data.Length), .. data];
    }

    private sealed class RecordingModel(params string[] texts) : IUtteranceRecognizer
    {
        public List<float[]> Pieces { get; } = [];
        public Action WhileTranscribing { get; set; } = () => { };
        public string BackendName => IsUsingGpu ? "GPU: test" : "CPU";
        public bool IsUsingGpu { get; set; }

        public string Transcribe(float[] samples16KhzMono)
        {
            lock (Pieces)
                Pieces.Add(samples16KhzMono);
            WhileTranscribing();
            return Pieces.Count <= texts.Length ? texts[Pieces.Count - 1] : "";
        }

        public void Dispose() { }
    }
}
