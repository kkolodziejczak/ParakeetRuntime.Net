using System.Reflection;

var version = typeof(Program).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";

if (args is ["--version"])
{
    Console.WriteLine(version);
    return 0;
}

Console.Error.WriteLine($"parakeet-runtime {version}");
Console.Error.WriteLine("Usage: parakeet-runtime --version");
return args.Length == 0 ? 0 : 2;
