using System.Diagnostics;
using System.IO;
using static ReLink.Texts;

namespace ReLink;

/// Where nx2/make_game.py and its Python are: a "tools" folder beside the app when installed, or the source tree it sits in.
sealed record Tools(string Root, string Python, string[]? PythonArgs = null)
{
    /// Which release these tools are (tools/version.txt, written when the release is packed); null from the source tree
    public string? Version => File.Exists(Path.Combine(Root, "version.txt")) ? File.ReadAllText(Path.Combine(Root, "version.txt")).Trim() : null;

    public static Tools? Find()
    {
        var at = AppContext.BaseDirectory;
        var bundled = Path.Combine(at, "tools");
        if (File.Exists(Path.Combine(bundled, "nx2", "make_game.py")))
        {
            var py = Path.Combine(bundled, "python", "python.exe");
            return new Tools(bundled, File.Exists(py) ? py : "python");
        }
        for (var d = new DirectoryInfo(at); d != null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "nx2", "make_game.py")))
            {
                var (exe, args) = PythonOnPath();
                return new Tools(d.FullName, exe, args);
            }
        return null;
    }

    /// The Python for a source tree: Windows' Store shortcut is not an interpreter (it only opens the Store), so the py launcher
    /// is tried when "python" is that.
    static (string Exe, string[]? Args) PythonOnPath()
    {
        var python = FirstOnPath("python.exe");
        if (python != null && !IsStoreShortcut(python)) return (python, null);
        var py = FirstOnPath("py.exe");
        if (py != null) return (py, new[] { "-3" });
        var later = AllOnPath("python.exe").FirstOrDefault(p => !IsStoreShortcut(p));
        return (later ?? python ?? "python", null);
    }

    static IEnumerable<string> AllOnPath(string name)
    {
        foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
            if (!string.IsNullOrWhiteSpace(dir))
            {
                var path = Path.Combine(dir.Trim('"'), name);
                if (File.Exists(path)) yield return path;
            }
    }

    static string? FirstOnPath(string name) => AllOnPath(name).FirstOrDefault();

    static bool IsStoreShortcut(string path) => path.Contains(@"\Microsoft\WindowsApps\", StringComparison.OrdinalIgnoreCase);
}

/// nx2/make_game.py as a child process: its lines as they come, and how it ended.
sealed class Runner
{
    readonly Tools tools;
    readonly List<string> args;
    Process? process;
    string? error;
    bool cancelled;

    public event Action<string>? Line;
    public event Action<string?>? Finished;

    public Runner(Tools tools, List<string> args)
    {
        this.tools = tools;
        this.args = args;
    }

    static ProcessStartInfo Info(Tools tools, IEnumerable<string> args)
    {
        var si = new ProcessStartInfo(tools.Python)
        {
            WorkingDirectory = tools.Root,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        si.Environment["PYTHONIOENCODING"] = "utf-8";
        // (written into the game folder's version file: the game starts this app again to update, host/hos_update.c)
        if (Environment.ProcessPath != null) si.Environment["HOOKSHOT_BUILDER_EXE"] = Environment.ProcessPath;
        si.Environment.Remove("HOOKSHOT_NGX_SDK");
        // (the folder's own compiler, headers and libraries, when it has them: nothing of the PC's but Windows' own)
        var sets = new[] { ("EMSDK", "emsdk"), ("RX_CPM", "cpm"), ("WGPU_NATIVE", "wgpu-native") };
        foreach (var (name, folder) in sets)
        {
            var path = Path.Combine(tools.Root, folder);
            if (Directory.Exists(path)) si.Environment[name] = path;
        }
        foreach (var a in tools.PythonArgs ?? Array.Empty<string>()) si.ArgumentList.Add(a);
        si.ArgumentList.Add("-I");
        si.ArgumentList.Add(Path.Combine(tools.Root, "nx2", "make_game.py"));
        foreach (var a in args) si.ArgumentList.Add(a);
        return si;
    }

    public void Start()
    {
        Task.Run(() =>
        {
            try
            {
                using var p = Process.Start(Info(tools, args))!;
                process = p;
                p.ErrorDataReceived += (_, e) => { if (e.Data != null) Line?.Invoke(e.Data); };
                p.BeginErrorReadLine();
                try { p.PriorityClass = ProcessPriorityClass.BelowNormal; } catch (Exception) { }
                while (p.StandardOutput.ReadLine() is { } line)
                {
                    if (line.StartsWith("@@error ")) error = line[8..];
                    Line?.Invoke(line);
                }
                p.WaitForExit();
                Finished?.Invoke(cancelled ? "cancelled" : p.ExitCode == 0 ? null
                    : error ?? L("Les outils se sont arrêtés sans dire pourquoi. Voir les détails.", "The build tools stopped without saying why. See the details."));
            }
            catch (Exception e)
            {
                Finished?.Invoke(L("Les outils n'ont pas pu démarrer : ", "The build tools could not start: ") + e.Message);
            }
        });
    }

    public void Cancel()
    {
        cancelled = true;
        try { process?.Kill(entireProcessTree: true); } catch (Exception) { }
    }

    /// Is this the supported game and version? The answer, and the tools' own words.
    public static async Task<(bool Ok, string Text)> Check(Tools tools, IEnumerable<string> args)
    {
        try
        {
            using var p = Process.Start(Info(tools, args))!;
            string? lastError = null;
            p.ErrorDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) lastError = e.Data; };
            p.BeginErrorReadLine();
            var result = (false, L("Les fichiers du jeu n'ont pas pu être lus.", "The game files could not be read."));
            var answered = false;
            while (await p.StandardOutput.ReadLineAsync() is { } line)
            {
                if (line.StartsWith("@@check ok ")) { result = (true, line[11..]); answered = true; }
                else if (line.StartsWith("@@check bad ")) { result = (false, line[12..]); answered = true; }
                else if (line.StartsWith("@@error ")) { result = (false, line[8..]); answered = true; }
            }
            await p.WaitForExitAsync();
            // (a step that stopped before it could say why: the reason is on stderr)
            if (!answered)
            {
                if (lastError != null && lastError.Contains("Microsoft Store", StringComparison.OrdinalIgnoreCase))
                    return (false, L("Python 3 est introuvable : la commande « python » de ce PC n'ouvre que le Microsoft Store. Installez ReLink avec ReLink-Setup.exe, qui apporte le sien.",
                                     "Python 3 was not found: the 'python' command on this PC only opens the Microsoft Store. Install ReLink with ReLink-Setup.exe, which brings its own."));
                return (false, lastError ?? L($"Les outils se sont arrêtés sans dire pourquoi (code {p.ExitCode}).", $"The game tools stopped without saying why (exit code {p.ExitCode})."));
            }
            return result;
        }
        catch (Exception)
        {
            // (the package's own Python missing: an antivirus took it, most often, after a false alarm on the download)
            if (tools.Python == "python" && Directory.Exists(Path.Combine(tools.Root, "python")))
                return (false, L("Le Python de ReLink (tools/python/python.exe) a disparu. Votre antivirus l'a sans doute retiré : restaurez-le depuis la quarantaine, puis réessayez.",
                                 "ReLink's own Python (tools/python/python.exe) is missing. Your antivirus probably removed it: restore it from quarantine, then try again."));
            return (false, L("Python 3 est introuvable. Installez ReLink avec ReLink-Setup.exe, qui apporte le sien.",
                             "Python 3 was not found. Install ReLink with ReLink-Setup.exe, which brings its own."));
        }
    }
}
