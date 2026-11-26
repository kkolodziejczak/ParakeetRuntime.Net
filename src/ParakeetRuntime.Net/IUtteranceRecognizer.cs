namespace ParakeetRuntime.Net;

internal interface IUtteranceRecognizer : IDisposable
{
    string BackendName { get; }
    bool IsUsingGpu { get; }
    string Transcribe(float[] samples16KhzMono);
}
