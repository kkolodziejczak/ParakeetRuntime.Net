namespace ParakeetRuntime;

internal interface IUtteranceRecognizer : IDisposable
{
    string Transcribe(float[] samples16KhzMono);
}
