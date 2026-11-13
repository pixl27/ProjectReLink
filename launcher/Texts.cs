using System.Globalization;

namespace ReLink;

/// The launcher's words: French on a French Windows, English elsewhere (RELINK_LANG=fr or en chooses, for tests). And the names
/// and addresses the project goes by.
static class Texts
{
    public static readonly bool French =
        (Environment.GetEnvironmentVariable("RELINK_LANG") is { Length: > 0 } forced ? forced : CultureInfo.CurrentUICulture.TwoLetterISOLanguageName)
        .StartsWith("fr", StringComparison.OrdinalIgnoreCase);

    public static string L(string fr, string en) => French ? fr : en;

    public const string Name = "Project ReLink";
    public const string Repo = "pixl27/ProjectReLink";
    public const string RepoPage = "https://github.com/" + Repo;
    public const string ReleasesPage = RepoPage + "/releases/latest";
    public const string HelpPage = RepoPage + "#readme";

    /// The game's languages: the code language.txt takes, and the language's own name
    public static readonly (string Code, string Name)[] GameLanguages =
    {
        ("en-US", "English (US)"), ("en-GB", "English (UK)"), ("fr", "Français (France)"), ("fr-CA", "Français (Canada)"),
        ("de", "Deutsch"), ("it", "Italiano"), ("es", "Español (España)"), ("es-419", "Español (Latinoamérica)"),
        ("pt-BR", "Português (Brasil)"), ("nl", "Nederlands"), ("pl", "Polski"), ("ru", "Русский"),
        ("ja", "日本語"), ("ko", "한국어"), ("zh-Hans", "中文 (简体)"), ("zh-Hant", "中文 (繁體)"),
    };

    /// The game language a first install starts with: Windows' own, when the game has it
    public static string GameLanguageOfWindows()
    {
        var c = CultureInfo.CurrentUICulture;
        var two = c.TwoLetterISOLanguageName;
        var name = c.Name;
        if (two == "fr") return name.EndsWith("-CA", StringComparison.OrdinalIgnoreCase) ? "fr-CA" : "fr";
        if (two == "en") return name.EndsWith("-US", StringComparison.OrdinalIgnoreCase) ? "en-US" : "en-GB";
        if (two == "es") return name.EndsWith("-ES", StringComparison.OrdinalIgnoreCase) ? "es" : "es-419";
        if (two == "pt") return "pt-BR";
        if (two == "zh") return name.Contains("Hant", StringComparison.OrdinalIgnoreCase) || name.EndsWith("-TW", StringComparison.OrdinalIgnoreCase)
                                || name.EndsWith("-HK", StringComparison.OrdinalIgnoreCase) ? "zh-Hant" : "zh-Hans";
        return GameLanguages.Any(l => l.Code == two) ? two : "en-US";
    }

    public static string Bytes(long bytes) =>
        bytes >= 1L << 30 ? $"{bytes / (double)(1L << 30):0.0} {L("Go", "GB")}" : $"{bytes >> 20} {L("Mo", "MB")}";

    public static string Lasted(TimeSpan t) =>
        t.TotalHours >= 1 ? $"{(int)t.TotalHours} h {t.Minutes} min"
        : t.TotalMinutes >= 1 ? $"{(int)t.TotalMinutes} {L("minutes", "minutes")}"
        : $"{t.Seconds} {L("secondes", "seconds")}";
}
