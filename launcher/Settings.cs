using System.IO;
using System.Text;

namespace ReLink;

/// A game folder: where its program is, which release built it, and whether it can be updated.
static class Game
{
    /// The game's program: bin\relink-game.exe below the folder's small launcher, or relink-game.exe in a folder built from the sources
    public static string Exe(string folder)
    {
        var inBin = Path.Combine(folder, "bin", "relink-game.exe");
        return File.Exists(inBin) ? inBin : Path.Combine(folder, "relink-game.exe");
    }

    public static bool Installed(string? folder)
    {
        try
        {
            return folder is { Length: > 2 } && File.Exists(Path.Combine(folder, "image.bin")) && File.Exists(Exe(folder));
        }
        catch (Exception) { return false; }
    }

    /// The release that built the folder (its version file); null for a folder built from the sources
    public static string? Version(string folder)
    {
        try
        {
            foreach (var line in File.ReadAllLines(Path.Combine(folder, "relink_version.txt")))
                if (line.StartsWith("version=")) return line[8..].Trim() is { Length: > 0 } v ? v : null;
        }
        catch (Exception) { }
        return null;
    }

    /// The folder keeps the game files it was built from: an update needs nothing else
    public static bool KeepsGameFiles(string folder) =>
        Directory.Exists(Path.Combine(folder, ".relink_exefs")) && Directory.Exists(Path.Combine(folder, "romfs"));
}

/// The options: the game's own file (relink_options.ini in the game folder, the rows of the in-game menu: the game reads it at
/// its start and writes it when the menu changes something), language.txt, and the launcher's own (launcher.ini in ReLink's
/// folder).
sealed class Settings
{
    // the game's rows (gpu/vk/vk_menu.cpp: ini keys are the row names, lower case, spaces as underscores)
    public int Fullscreen = 1, FrameGeneration = 0, FramePacing = 0, FrameRate = 0, Resolution = 1, Upscaling = 0, Sharpness = 0,
               GameSpeed = 0, Handheld = 0, Ultrawide = 0, FpsCounter = 0;
    // the launcher's
    public bool Display60 = true, Audio = true, InvertY = false;
    /// The launcher's own language: en (the default) or fr (Texts.cs reads it when the launcher opens)
    public string UiLanguage = Texts.French ? "fr" : "en";
    public string Language = Texts.GameLanguageOfWindows();
    /// The game folder (empty: none yet)
    public string Folder = "";

    readonly Dictionary<string, string> other = new();   // (keys of the game's file this launcher does not know: kept)

    static string LauncherIni => Path.Combine(Storage.Base, "launcher.ini");
    string GameIni => Path.Combine(Folder, "relink_options.ini");
    string LanguageFile => Path.Combine(Folder, "language.txt");

    static Dictionary<string, string> ReadIni(string path)
    {
        var d = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        try
        {
            if (!File.Exists(path)) return d;
            foreach (var raw in File.ReadAllLines(path))
            {
                var line = raw.Trim().TrimStart('﻿');
                int eq = line.IndexOf('=');
                if (eq <= 0 || line.StartsWith('#') || line.StartsWith(';')) continue;
                d[line[..eq].Trim()] = line[(eq + 1)..].Trim();
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
        return d;
    }
    static int Int(Dictionary<string, string> d, string k, int def, int lo, int hi) =>
        d.TryGetValue(k, out var v) && int.TryParse(v, out var n) ? Math.Clamp(n, lo, hi) : def;
    static bool Bool(Dictionary<string, string> d, string k, bool def) =>
        d.TryGetValue(k, out var v) && int.TryParse(v, out var n) ? n != 0 : def;

    /// The launcher's own choices; the game folder is the one given (--launch), else the one remembered, else one found around
    /// this app, else the default place
    public static Settings Load(string? folder)
    {
        var s = new Settings();
        var l = ReadIni(LauncherIni);
        // (an earlier launcher kept the same three choices in lanceur.ini beside the game: taken once, while this one has none yet)
        if (l.Count == 0)
            foreach (var beside in new[] { folder, AppContext.BaseDirectory })
            {
                if (string.IsNullOrEmpty(beside)) continue;
                l = ReadIni(Path.Combine(beside, "lanceur.ini"));
                if (l.Count > 0) break;
            }
        s.Display60 = Bool(l, "display_60hz", true);
        s.Audio = Bool(l, "audio", true);
        s.InvertY = Bool(l, "invert_y", false);
        if (l.TryGetValue("ui_language", out var ui) && ui is "en" or "fr") s.UiLanguage = ui;
        var here = AppContext.BaseDirectory.TrimEnd('\\', '/');
        var candidates = new[]
        {
            folder, l.TryGetValue("game_folder", out var remembered) ? remembered : null, here, Path.GetDirectoryName(here), DefaultFolder,
        };
        s.Folder = candidates.FirstOrDefault(Game.Installed) ?? "";
        if (s.Folder != "") s.LoadGame();
        return s;
    }

    public static string DefaultFolder => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Games", "ReLink");

    /// The game's own file and language, read again (the in-game menu may have changed them)
    public void LoadGame()
    {
        var g = ReadIni(GameIni);
        Fullscreen = Int(g, "fullscreen", 1, 0, 1);
        FrameGeneration = Int(g, "frame_generation", 0, 0, 4);
        if (FrameGeneration == 1) FrameGeneration = 2;
        FramePacing = Int(g, "frame_pacing", 0, 0, 1);
        FrameRate = Int(g, "frame_rate", 0, 0, 1);
        // (the game's Render resolution row: 0 Auto, 1..6 = 100 %, 75 %, 67 %, 50 %, 58 %, 33 % of the output; 1, native, by default)
        Resolution = Int(g, "resolution", 1, 0, 6);
        Upscaling = Int(g, "upscaling", 0, 0, 2);
        Sharpness = Int(g, "sharpness", 0, 0, 100);
        GameSpeed = Int(g, "game_speed", 0, 0, 1);
        Handheld = Int(g, "handheld_profile", 0, 0, 1);
        Ultrawide = Int(g, "ultrawide", 0, 0, 1);
        FpsCounter = Int(g, "fps_counter", 0, 0, 1);
        other.Clear();
        foreach (var (k, v) in g)
            if (!Known.Contains(k)) other[k] = v;
        try
        {
            if (File.Exists(LanguageFile))
                foreach (var raw in File.ReadAllLines(LanguageFile))
                {
                    var line = raw.Trim();
                    if (line.Length == 0 || line.StartsWith('#')) continue;
                    Language = line;
                    break;
                }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
    }

    static readonly HashSet<string> Known = new(StringComparer.OrdinalIgnoreCase)
    {
        "fullscreen", "frame_generation", "frame_pacing", "frame_rate", "resolution", "upscaling", "sharpness", "game_speed",
        "handheld_profile", "ultrawide", "fps_counter",
    };

    public void Save()
    {
        var sb = new StringBuilder();
        sb.AppendLine("# Project ReLink: the launcher's own choices");
        sb.AppendLine($"display_60hz={(Display60 ? 1 : 0)}");
        sb.AppendLine($"audio={(Audio ? 1 : 0)}");
        sb.AppendLine($"invert_y={(InvertY ? 1 : 0)}");
        sb.AppendLine($"ui_language={UiLanguage}");
        sb.AppendLine($"game_folder={Folder}");
        Directory.CreateDirectory(Storage.Base);
        File.WriteAllText(LauncherIni, sb.ToString());
        if (Folder == "" || !Directory.Exists(Folder)) return;

        var g = new StringBuilder();
        void Row(string k, int v) => g.Append(k).Append('=').Append(v).Append('\n');
        Row("fullscreen", Fullscreen);
        Row("frame_generation", FrameGeneration);
        Row("frame_pacing", FramePacing);
        Row("frame_rate", FrameRate);
        Row("resolution", Resolution);
        Row("upscaling", Upscaling);
        Row("sharpness", Sharpness);
        Row("game_speed", GameSpeed);
        Row("handheld_profile", Handheld);
        Row("ultrawide", Ultrawide);
        Row("fps_counter", FpsCounter);
        foreach (var (k, v) in other) g.Append(k).Append('=').Append(v).Append('\n');
        File.WriteAllText(GameIni, g.ToString());

        // language.txt: its comment lines kept, the code line replaced
        var lines = File.Exists(LanguageFile) ? File.ReadAllLines(LanguageFile).ToList() : new List<string>();
        var outLines = new List<string>();
        bool done = false;
        foreach (var raw in lines)
        {
            if (!done && raw.Trim().Length > 0 && !raw.TrimStart().StartsWith('#')) { outLines.Add(Language); done = true; }
            else outLines.Add(raw);
        }
        if (!done)
        {
            if (outLines.Count == 0) outLines.Add("# The game's language (written by the launcher).");
            outLines.Add(Language);
        }
        File.WriteAllLines(LanguageFile, outLines);
    }
}
