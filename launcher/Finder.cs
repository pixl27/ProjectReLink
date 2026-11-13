using System.IO;
using System.Runtime.InteropServices;
using static ReLink.Texts;

namespace ReLink;

/// A .nup found on this PC
sealed record Dump(string Path, string Where, long Size, DateTime Modified);

/// The player's game files where people keep them, and the game's shortcuts.
static class Finder
{
    /// .nup files in Downloads, Desktop, Documents, the home folder and the top of each drive, one folder level down too, newest
    /// first. Only names, sizes and dates are read here: the version is checked one file at a time afterwards.
    /// (RELINK_NO_SEARCH=1: nothing is looked for; the player browses to the file)
    public static List<Dump> FindDumps()
    {
        if (Environment.GetEnvironmentVariable("RELINK_NO_SEARCH") is { Length: > 0 } off && off != "0") return new List<Dump>();
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var places = new List<(string Dir, string Name)>
        {
            (Path.Combine(home, "Downloads"), L("Téléchargements", "Downloads")),
            (Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), L("Bureau", "Desktop")),
            (Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "Documents"),
            (home, L("Dossier personnel", "Home")),
        };
        foreach (var d in DriveInfo.GetDrives())
        {
            try
            {
                if (d.IsReady && d.DriveType is DriveType.Fixed or DriveType.Removable) places.Add((d.RootDirectory.FullName, d.Name.TrimEnd('\\')));
            }
            catch (Exception) { }
        }
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var found = new List<Dump>();
        foreach (var (dir, name) in places)
        {
            if (!Directory.Exists(dir) || !seen.Add(Path.GetFullPath(dir))) continue;
            var dirs = new List<string> { dir };
            try
            {
                dirs.AddRange(Directory.EnumerateDirectories(dir)
                    .Where(s => !Path.GetFileName(s).StartsWith('.') && !Path.GetFileName(s).StartsWith('$')
                                && !string.Equals(Path.GetFileName(s), "Windows", StringComparison.OrdinalIgnoreCase)
                                && !Path.GetFileName(s).StartsWith("Program", StringComparison.OrdinalIgnoreCase))
                    .Take(200));
            }
            catch (Exception) { }
            foreach (var sub in dirs)
            {
                try
                {
                    foreach (var f in Directory.EnumerateFiles(sub, "*.nup"))
                    {
                        var info = new FileInfo(f);
                        if (seen.Add(info.FullName)) found.Add(new Dump(info.FullName, name, info.Length, info.LastWriteTime));
                    }
                }
                catch (Exception) { }
            }
        }
        return found.OrderByDescending(d => d.Modified).ToList();
    }

    public static string Ago(DateTime t)
    {
        var d = DateTime.Now - t;
        if (d.TotalDays < 1 && t.Date == DateTime.Today) return L("modifié aujourd'hui", "modified today");
        if (d.TotalDays < 2) return L("modifié hier", "modified yesterday");
        if (d.TotalDays < 14) return L($"il y a {(int)d.TotalDays} jours", $"{(int)d.TotalDays} days ago");
        if (d.TotalDays < 60) return L($"il y a {(int)(d.TotalDays / 7)} semaines", $"{(int)(d.TotalDays / 7)} weeks ago");
        return t.ToString("d MMM yyyy");
    }

    // ---- shortcuts to the installed game ----

    public static string DesktopLink => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Project ReLink.lnk");
    public static string StartLink => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), "Project ReLink.lnk");

    /// The shortcuts the player has, made again (after an update); none made where there was none
    public static void RefreshShortcuts(string gameFolder) => Shortcuts(gameFolder, File.Exists(DesktopLink), File.Exists(StartLink));

    /// Made or removed as the two switches say (Windows' own shell object writes the .lnk). They open ReLink.exe in the game folder
    /// (the small launcher the build puts there, which opens this app for that folder); a folder without it gets this app itself,
    /// told which folder to open.
    public static void Shortcuts(string gameFolder, bool desktop, bool start)
    {
        var stub = Path.Combine(gameFolder, "ReLink.exe");
        var self = Environment.ProcessPath ?? "";
        var (target, args) = File.Exists(stub) ? (stub, "") : (self, $"--launch \"{gameFolder}\"");
        foreach (var (link, want) in new[] { (DesktopLink, desktop), (StartLink, start) })
        {
            try
            {
                if (!want)
                {
                    if (File.Exists(link)) File.Delete(link);
                    continue;
                }
                if (!File.Exists(target)) continue;
                var sl = (IShellLinkW)new ShellLink();
                sl.SetPath(target);
                if (args.Length > 0) sl.SetArguments(args);
                sl.SetWorkingDirectory(gameFolder);
                sl.SetIconLocation(File.Exists(self) ? self : target, 0);
                sl.SetDescription(L("Ocarina of Time sur PC, avec Project ReLink", "Ocarina of Time on PC, with Project ReLink"));
                Directory.CreateDirectory(Path.GetDirectoryName(link)!);
                ((System.Runtime.InteropServices.ComTypes.IPersistFile)sl).Save(link, true);
            }
            catch (Exception e)
            {
                ShortcutError = $"{Path.GetFileName(link)}: {e.Message}";
            }
        }
    }

    /// The last shortcut that could not be made, for the log
    public static string? ShortcutError;

    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    class ShellLink { }

    [ComImport, Guid("000214F9-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IShellLinkW
    {
        void GetPath([MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder file, int max, IntPtr fd, uint flags);
        void GetIDList(out IntPtr ppidl);
        void SetIDList(IntPtr pidl);
        void GetDescription([MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder name, int max);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder dir, int max);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string dir);
        void GetArguments([MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder args, int max);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string args);
        void GetHotkey(out short hotkey);
        void SetHotkey(short hotkey);
        void GetShowCmd(out int show);
        void SetShowCmd(int show);
        void GetIconLocation([MarshalAs(UnmanagedType.LPWStr)] System.Text.StringBuilder path, int max, out int index);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string path, int index);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string path, uint reserved);
        void Resolve(IntPtr hwnd, uint flags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string file);
    }
}
