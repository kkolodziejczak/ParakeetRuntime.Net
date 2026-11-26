namespace ParakeetRuntime.Net;

internal interface IMicrophonePackets : IDisposable
{
    float[]? NextPacket();
}
