namespace ParakeetRuntime;

internal interface IMicrophonePackets : IDisposable
{
    float[]? NextPacket();
}
