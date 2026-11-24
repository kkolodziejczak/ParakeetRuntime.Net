using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace ParakeetRuntime;

internal static unsafe class Wasapi
{
    [DllImport("ole32.dll")]
    private static extern int CoInitializeEx(nint reserved, uint coInit);

    [DllImport("ole32.dll")]
    private static extern void CoUninitialize();

    [DllImport("ole32.dll")]
    private static extern int CoCreateInstance(Guid* clsid, nint outer, uint clsContext, Guid* iid, nint* instance);

    [DllImport("ole32.dll")]
    private static extern void CoTaskMemFree(nint memory);

    [DllImport("ole32.dll")]
    private static extern int PropVariantClear(PropVariant* value);

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct WaveFormatEx
    {
        public ushort FormatTag, Channels;
        public uint SamplesPerSec, AvgBytesPerSec;
        public ushort BlockAlign, BitsPerSample, ExtraSize;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PropertyKey
    {
        public Guid FormatId;
        public uint PropertyId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PropVariant
    {
        public ushort Type, Reserved1, Reserved2, Reserved3;
        public nint Pointer;
        public nint Second;
    }

    private const uint ClsCtxAll = 23, MultiThreaded = 0, ActiveDevices = 1;
    private const int CaptureFlow = 1, ConsoleRole = 0, SharedMode = 0;
    private const int ElementNotFound = unchecked((int)0x80070490);
    private const int AccessDenied = unchecked((int)0x80070005);
    private const int DeviceInvalidated = unchecked((int)0x88890004);
    private const uint AutoConvertPcm = 0x80000000, DefaultQualityResampler = 0x08000000, SilentBuffer = 2;
    private const ushort IeeeFloat = 3, UnicodeStringVariant = 31;
    private const long OneSecondIn100Nanoseconds = 10_000_000;

    private static nint* Vtbl(nint comObject) => *(nint**)comObject;

    private static void Check(int hresult, string what)
    {
        if (hresult >= 0)
            return;
        throw hresult switch
        {
            AccessDenied => new UnauthorizedAccessException(
                "Windows does not let this app use the microphone. Turn on microphone access for desktop apps in Settings > Privacy & security > Microphone."),
            DeviceInvalidated => new IOException("The microphone was unplugged or turned off."),
            _ => new IOException($"{what} failed (0x{hresult:X8}).")
        };
    }

    private static void Release(nint comObject)
    {
        if (comObject != 0)
            ((delegate* unmanaged[Stdcall]<nint, uint>)Vtbl(comObject)[2])(comObject);
    }

    public static T OnItsOwnComThread<T>(Func<T> work)
    {
        T? result = default;
        ExceptionDispatchInfo? failure = null;
        var thread = new Thread(() =>
        {
            try
            {
                Check(CoInitializeEx(0, MultiThreaded), "Starting COM");
                try
                {
                    result = work();
                }
                finally
                {
                    CoUninitialize();
                }
            }
            catch (Exception exception)
            {
                failure = ExceptionDispatchInfo.Capture(exception);
            }
        });
        thread.Start();
        thread.Join();
        failure?.Throw();
        return result!;
    }

    private static nint DeviceEnumerator()
    {
        var deviceEnumeratorClass = new Guid("BCDE0395-E52F-467C-8E3D-C4579291692E");
        var deviceEnumeratorInterface = new Guid("A95664D2-9614-4F35-A746-DE8DB63617E6");
        nint enumerator;
        Check(CoCreateInstance(&deviceEnumeratorClass, 0, ClsCtxAll, &deviceEnumeratorInterface, &enumerator), "Creating the audio device enumerator");
        return enumerator;
    }

    public static IReadOnlyList<MicrophoneDevice> ActiveCaptureDevices()
    {
        nint enumerator = DeviceEnumerator(), collection = 0;
        try
        {
            Check(((delegate* unmanaged[Stdcall]<nint, int, uint, nint*, int>)Vtbl(enumerator)[3])(enumerator, CaptureFlow, ActiveDevices, &collection), "Listing microphones");
            uint count;
            Check(((delegate* unmanaged[Stdcall]<nint, uint*, int>)Vtbl(collection)[3])(collection, &count), "Counting microphones");
            var devices = new List<MicrophoneDevice>();
            for (uint index = 0; index < count; index++)
            {
                nint device = 0;
                Check(((delegate* unmanaged[Stdcall]<nint, uint, nint*, int>)Vtbl(collection)[4])(collection, index, &device), "Opening a microphone entry");
                try
                {
                    devices.Add(new MicrophoneDevice(IdOf(device), FriendlyNameOf(device)));
                }
                finally
                {
                    Release(device);
                }
            }
            return devices;
        }
        finally
        {
            Release(collection);
            Release(enumerator);
        }
    }

    private static string IdOf(nint device)
    {
        nint id;
        Check(((delegate* unmanaged[Stdcall]<nint, nint*, int>)Vtbl(device)[5])(device, &id), "Reading a microphone id");
        try
        {
            return Marshal.PtrToStringUni(id)!;
        }
        finally
        {
            CoTaskMemFree(id);
        }
    }

    private static string FriendlyNameOf(nint device)
    {
        nint properties = 0;
        Check(((delegate* unmanaged[Stdcall]<nint, uint, nint*, int>)Vtbl(device)[4])(device, 0, &properties), "Opening a microphone's properties");
        try
        {
            var friendlyName = new PropertyKey { FormatId = new Guid("A45C254E-DF1C-4EFD-8020-67D146A850E0"), PropertyId = 14 };
            PropVariant value = default;
            Check(((delegate* unmanaged[Stdcall]<nint, PropertyKey*, PropVariant*, int>)Vtbl(properties)[5])(properties, &friendlyName, &value), "Reading a microphone name");
            try
            {
                return value.Type == UnicodeStringVariant ? Marshal.PtrToStringUni(value.Pointer)! : "Microphone";
            }
            finally
            {
                PropVariantClear(&value);
            }
        }
        finally
        {
            Release(properties);
        }
    }

    public sealed class Capture : IMicrophonePackets
    {
        private nint client, capture;

        public static Capture Open(string? deviceId)
        {
            Check(CoInitializeEx(0, MultiThreaded), "Starting COM");
            var opened = new Capture();
            try
            {
                opened.Start(deviceId);
                return opened;
            }
            catch
            {
                opened.Dispose();
                throw;
            }
        }

        private void Start(string? deviceId)
        {
            nint enumerator = DeviceEnumerator(), device = 0;
            try
            {
                int found;
                if (deviceId is null)
                    found = ((delegate* unmanaged[Stdcall]<nint, int, int, nint*, int>)Vtbl(enumerator)[4])(enumerator, CaptureFlow, ConsoleRole, &device);
                else
                    fixed (char* id = deviceId)
                        found = ((delegate* unmanaged[Stdcall]<nint, char*, nint*, int>)Vtbl(enumerator)[5])(enumerator, id, &device);
                if (found == ElementNotFound)
                    throw new IOException(deviceId is null ? "Windows has no microphone to record from." : "That microphone is no longer connected.");
                Check(found, "Opening the microphone");

                var audioClientInterface = new Guid("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2");
                var captureClientInterface = new Guid("C8ADBD64-E71E-48A0-A4DE-185C395CD317");
                nint activated;
                Check(((delegate* unmanaged[Stdcall]<nint, Guid*, uint, nint, nint*, int>)Vtbl(device)[3])(device, &audioClientInterface, ClsCtxAll, 0, &activated), "Activating the microphone");
                client = activated;
                var mono16KhzFloat = new WaveFormatEx { FormatTag = IeeeFloat, Channels = 1, SamplesPerSec = Wav.SampleRate, AvgBytesPerSec = Wav.SampleRate * 4, BlockAlign = 4, BitsPerSample = 32 };
                Check(((delegate* unmanaged[Stdcall]<nint, int, uint, long, long, WaveFormatEx*, Guid*, int>)Vtbl(client)[3])(
                    client, SharedMode, AutoConvertPcm | DefaultQualityResampler, OneSecondIn100Nanoseconds, 0, &mono16KhzFloat, null), "Asking Windows for 16 kHz mono audio");
                nint service;
                Check(((delegate* unmanaged[Stdcall]<nint, Guid*, nint*, int>)Vtbl(client)[14])(client, &captureClientInterface, &service), "Opening the capture service");
                capture = service;
                Check(((delegate* unmanaged[Stdcall]<nint, int>)Vtbl(client)[10])(client), "Starting the microphone");
            }
            finally
            {
                Release(device);
                Release(enumerator);
            }
        }

        public float[]? NextPacket()
        {
            uint nextPacketFrames;
            Check(((delegate* unmanaged[Stdcall]<nint, uint*, int>)Vtbl(capture)[5])(capture, &nextPacketFrames), "Reading the microphone");
            if (nextPacketFrames == 0)
                return null;
            byte* data;
            uint frames, flags;
            Check(((delegate* unmanaged[Stdcall]<nint, byte**, uint*, uint*, ulong*, ulong*, int>)Vtbl(capture)[3])(capture, &data, &frames, &flags, null, null), "Reading the microphone");
            var packet = (flags & SilentBuffer) != 0 ? new float[frames] : new ReadOnlySpan<float>(data, (int)frames).ToArray();
            Check(((delegate* unmanaged[Stdcall]<nint, uint, int>)Vtbl(capture)[4])(capture, frames), "Reading the microphone");
            return packet;
        }

        public void Dispose()
        {
            if (client != 0)
                ((delegate* unmanaged[Stdcall]<nint, int>)Vtbl(client)[11])(client);
            Release(capture);
            Release(client);
            capture = client = 0;
            CoUninitialize();
        }
    }
}
