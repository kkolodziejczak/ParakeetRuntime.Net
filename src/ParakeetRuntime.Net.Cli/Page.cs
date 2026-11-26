using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace ParakeetRuntime.Net.Cli;

internal static partial class Page
{
    public static readonly byte[] Html = ReadEmbeddedPage();

    public static readonly string Policy = PolicyFor(Encoding.UTF8.GetString(Html));

    public static string PolicyFor(string html) =>
        $"default-src 'none'; script-src '{ScriptHash(html)}'; style-src 'unsafe-inline'; connect-src 'self'; " +
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

    private static byte[] ReadEmbeddedPage()
    {
        using var resource = typeof(Page).Assembly.GetManifestResourceStream("ParakeetRuntime.Net.Cli.Page.html")!;
        using var copy = new MemoryStream();
        resource.CopyTo(copy);
        return copy.ToArray();
    }

    private static string ScriptHash(string html)
    {
        var script = TheOnlyScript().Match(html).Groups[1].Value;
        return $"sha256-{Convert.ToBase64String(SHA256.HashData(Encoding.UTF8.GetBytes(WithNewlinesAsBrowsersReadThem(script))))}";
    }

    private static string WithNewlinesAsBrowsersReadThem(string text) => text.Replace("\r\n", "\n").Replace('\r', '\n');

    [GeneratedRegex("<script>(.*?)</script>", RegexOptions.Singleline)]
    private static partial Regex TheOnlyScript();
}
