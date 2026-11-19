using System.Windows;
using System.Windows.Media;

namespace ReLink;

/// The launcher's look: light (the cartridge's cream label, the default) or dark (the same cartridge in dark plastic). Chosen on
/// the Game page and kept in launcher.ini (theme=); RELINK_THEME=light or dark chooses for one run, for tests and pictures. Read once, when the
/// launcher opens: App.xaml's colours are given before the window is made, so a change shows the next time ReLink opens, as
/// the launcher's language does. (Not live: a style seals the brushes its setters hold, so they cannot change colour after.)
static class Theme
{
    public const string Light = "light", Dark = "dark";

    public static readonly string Chosen = Read();

    static string Read()
    {
        static string Known(string v) => v == Dark ? v : Light;
        if (Environment.GetEnvironmentVariable("RELINK_THEME") is { Length: > 0 } forced) return Known(forced.Trim().ToLowerInvariant());
        try
        {
            foreach (var line in System.IO.File.ReadAllLines(System.IO.Path.Combine(Storage.Base, "launcher.ini")))
                if (line.StartsWith("theme=")) return Known(line[6..].Trim());
        }
        catch (Exception) { }
        return Light;
    }

    /// The dark cartridge: a dark grey shell, a near-black label, the side darker still; the words cream; the controller's
    /// colours a little lighter, their tints deep (a chosen row: a dark tint of its page's colour, its words light). Measured
    /// (WCAG contrast, on the label): the words 13.4, the choices 7.4, the section titles 4.6, the hints 9.3; red, green and blue
    /// as words (links) 4.5, white on them (a chosen pill) 3.7; the label against the shell 1.45, every one as high or higher
    /// than the light theme's but the pills' (light: 4.2 to 6.0, its links 3.4 to 4.9)
    static readonly Dictionary<string, string> DarkColours = new()
    {
        ["Shell"] = "#3A3842", ["Groove"] = "#4A4753", ["Label"] = "#1E1D23", ["Side"] = "#16151B", ["SideHover"] = "#2A2833",
        ["SideText"] = "#ECE6D8", ["SideSoft"] = "#9E98AB", ["StatusCard"] = "#26242C",
        ["Bone"] = "#ECE6D8", ["BoneSoft"] = "#B3AB9C", ["BoneFaint"] = "#8C8579", ["HelpInk"] = "#C8C0B1",
        ["Rule"] = "#33313A", ["Warn"] = "#E06A5E", ["Off"] = "#4A4752", ["Track"] = "#3D3A45",
        ["Card"] = "#28262E", ["Field"] = "#28262E",
        ["HoverRow"] = "#10FFFFFF", ["HoverPill"] = "#1CFFFFFF", ["HoverChrome"] = "#26FFFFFF",
        ["Red"] = "#D36257", ["RedTint"] = "#47292A", ["RedEdge"] = "#8E4A44",
        ["Yellow"] = "#D4A532", ["YellowTint"] = "#463C22", ["YellowEdge"] = "#8A7234",
        ["Green"] = "#459555", ["GreenTint"] = "#243B2A", ["GreenEdge"] = "#4D7E57",
        ["Blue"] = "#6682D0", ["BlueTint"] = "#283049", ["BlueEdge"] = "#4F6299",
        ["Accent"] = "#D36257", ["AccentTint"] = "#47292A", ["AccentEdge"] = "#8E4A44",
    };


    /// App.xaml's colours replaced by the dark theme's, before the window is made
    public static void Apply(ResourceDictionary r)
    {
        if (Chosen != Dark) return;
        foreach (var (key, hex) in DarkColours)
            r[key] = new SolidColorBrush((Color)ColorConverter.ConvertFromString(hex));
    }
}
