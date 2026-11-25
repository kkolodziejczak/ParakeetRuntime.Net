using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace ParakeetRuntime.Cli;

internal static partial class Page
{
    public static readonly byte[] Html = ReadEmbeddedPage();

    public static readonly string Policy =
        $"default-src 'none'; script-src '{ScriptHash()}'; style-src 'unsafe-inline'; connect-src 'self'; " +
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

    private static byte[] ReadEmbeddedPage()
    {
        using var resource = typeof(Page).Assembly.GetManifestResourceStream("ParakeetRuntime.Cli.Page.html")!;
        using var copy = new MemoryStream();
        resource.CopyTo(copy);
        return copy.ToArray();
    }

    private static string ScriptHash()
    {
        var script = TheOnlyScript().Match(Encoding.UTF8.GetString(Html)).Groups[1].Value;
        return $"sha256-{Convert.ToBase64String(SHA256.HashData(Encoding.UTF8.GetBytes(script)))}";
    }

    [GeneratedRegex("<script>(.*?)</script>", RegexOptions.Singleline)]
    private static partial Regex TheOnlyScript();
}
