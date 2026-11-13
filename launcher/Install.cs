using System.Diagnostics;
using System.IO;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using Microsoft.Win32;
using static ReLink.Texts;

namespace ReLink;

/// The install and the update: the player's copy of the game is chosen and checked, nx2/make_game.py builds the game folder
/// from it (Runner.cs), and the page follows its steps. An update (Updater.cs) uses the same page with one more step.
public partial class MainWindow
{
    readonly DispatcherTimer clock = new() { Interval = TimeSpan.FromSeconds(1) };
    readonly Stopwatch elapsed = new();
    readonly Stopwatch compileClock = new();
    readonly Dictionary<string, StepRow> rows = new();
    readonly Dictionary<string, (bool Ok, string Text)?> checks = new(StringComparer.OrdinalIgnoreCase);
    readonly List<Dump> dumps = new();
    readonly System.Text.StringBuilder log = new();
    Runner? runner;
    string? current;
    string finished = "";
    string outPath = Settings.DefaultFolder;
    string? chosenNup, romfs, exefs, foldersNote;
    bool updating, checking, launcherIsOld;
    double compiled;
    int filesDone, filesOf, filesChanged = -1, filesCompiled;
    double compileEta = -1;
    long thisCompile, previousTotal = -1;
    string lastFailure = "";
    string? latestTag, updateTarget;

    // ---- what the player chose last time (the install folder, the game files): install.json in ReLink's folder ----
    sealed class InstallChoices
    {
        public string? Out { get; set; }
        public string? Nup { get; set; }
        public string? Romfs { get; set; }
        public string? Exefs { get; set; }
    }
    static string ChoicesPath => Path.Combine(Storage.Base, "install.json");

    void LoadInstallChoices()
    {
        InstallChoices? c = null;
        try { c = System.Text.Json.JsonSerializer.Deserialize<InstallChoices>(File.ReadAllText(ChoicesPath)); } catch (Exception) { }
        if (s.Folder != "") outPath = s.Folder;
        if (c == null) return;
        if (s.Folder == "" && !string.IsNullOrEmpty(c.Out)) outPath = c.Out;
        if (c.Romfs != null && c.Exefs != null && Directory.Exists(c.Romfs) && File.Exists(Path.Combine(c.Exefs, "main")))
        {
            romfs = c.Romfs;
            exefs = c.Exefs;
            foldersNote = L($"Dossiers utilisés : {romfs} et {exefs} (comme la dernière fois)", $"Using the folders {romfs} and {exefs} (as last time)");
        }
        else if (c.Nup != null && File.Exists(c.Nup))
        {
            var info = new FileInfo(c.Nup);
            dumps.Insert(0, new Dump(info.FullName, L("La dernière fois", "Last time"), info.Length, info.LastWriteTime));
            chosenNup = c.Nup;
        }
    }

    void SaveInstallChoices()
    {
        try
        {
            var c = new InstallChoices { Out = outPath, Nup = romfs == null ? chosenNup : null, Romfs = romfs, Exefs = exefs };
            Directory.CreateDirectory(Storage.Base);
            File.WriteAllText(ChoicesPath, System.Text.Json.JsonSerializer.Serialize(c));
        }
        catch (Exception) { }
    }

    // ================= 1: the player's copy =================

    void ShowInstall()
    {
        updating = false;
        SetStepNotes(update: false);
        Show(View.Install);
        var tools = Tools.Find();
        ToolsNote.Visibility = tools == null ? Visibility.Visible : Visibility.Collapsed;
        ToolsNote.Text = L("Les outils de ReLink manquent dans son dossier. Lancez de nouveau ReLink-Setup.exe : il les télécharge et les vérifie.",
                           "ReLink's build tools are missing from its folder. Run ReLink-Setup.exe again: it downloads and checks them.");
        FillDumps();
        Refresh();
    }

    void SetStepNotes(bool update)
    {
        var steps = new List<(string, string)>
        {
            ("read", L("Lecture", "Read")), ("lift", L("Traduction", "Translate")), ("compile", L("Compilation", "Compile")), ("finish", L("Assemblage", "Assemble")),
        };
        if (update) steps.Insert(0, ("download", L("Réception", "Download")));
        staff.Set(steps.ToArray());
    }

    void FillDumps()
    {
        DumpList.Children.Clear();
        if (romfs != null && exefs != null && foldersNote != null)
            DumpList.Children.Add(new TextBlock { Text = foldersNote, Style = St("Body"), Foreground = Res("Bone"), Margin = new Thickness(0, 4, 0, 8) });
        if (dumps.Count == 0)
        {
            if (romfs == null)
                DumpList.Children.Add(new TextBlock
                {
                    Text = L("Aucun fichier .nup trouvé dans Téléchargements, Bureau ou Documents. Choisissez le vôtre ci-dessous, ou déposez-le sur cette fenêtre.",
                             "No .nup file found in Downloads, Desktop or Documents. Choose yours below, or drop it on this window."),
                    Style = St("Body"), Margin = new Thickness(0, 4, 0, 4),
                });
            return;
        }
        foreach (var d in dumps)
        {
            checks.TryGetValue(d.Path, out var c);
            var (state, brush) = !checks.ContainsKey(d.Path) || c == null ? (L("Vérification…", "Checking…"), "BoneFaint")
                : c.Value.Ok ? (L("Bonne version", "Supported version"), "Tunic")
                : c.Value.Text.StartsWith("This is build") ? (L("Autre version du jeu", "Not the supported version"), "Warn")
                : (L("Illisible", "Can't be read"), "Warn");
            var name = new TextBlock { Text = Path.GetFileName(d.Path), FontSize = 15, TextTrimming = TextTrimming.CharacterEllipsis };
            var meta = new TextBlock { Text = $"{d.Where}, {Bytes(d.Size)}, {Finder.Ago(d.Modified)}", FontSize = 12.5, Foreground = Res("BoneFaint"), Margin = new Thickness(0, 2, 0, 0) };
            var texts = new StackPanel { Children = { name, meta } };
            // (why a file is refused is said in its row, not only in a tooltip)
            if (c is { Ok: false })
                texts.Children.Add(new TextBlock { Text = Why(c.Value.Text), FontSize = 12.5, Foreground = Res("Warn"), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 3, 0, 0) });
            var chip = new TextBlock { Text = state, FontSize = 13, Foreground = Res(brush), VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(14, 2, 0, 0) };
            if (c is { Ok: true }) chip.Foreground = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromRgb(0x8F, 0xD0, 0x9A));
            var grid = new Grid { ColumnDefinitions = { new ColumnDefinition(), new ColumnDefinition { Width = GridLength.Auto } } };
            Grid.SetColumn(chip, 1);
            grid.Children.Add(texts);
            grid.Children.Add(chip);
            var row = new RadioButton
            {
                Style = St("FileRow"), GroupName = "Dumps", Content = grid, ToolTip = d.Path,
                IsChecked = romfs == null && string.Equals(d.Path, chosenNup, StringComparison.OrdinalIgnoreCase),
            };
            row.Checked += (_, _) => Choose(d.Path);
            DumpList.Children.Add(row);
        }
    }

    /// The tools' own words for a refused file, in the launcher's language when it is the usual reason (another version of the game)
    static string Why(string text)
    {
        var m = Regex.Match(text, @"^This is build (\w+)\. \w+ supports build (\w+)\.");
        return m.Success ? L($"C'est la version {m.Groups[1].Value} du jeu ; ReLink prend en charge la version {m.Groups[2].Value}.",
                             $"This is build {m.Groups[1].Value} of the game; ReLink supports build {m.Groups[2].Value}.") : text;
    }

    /// Each found file's version, one at a time (the check unpacks into one probe folder); the first good one is chosen
    async void CheckAll()
    {
        if (checking) return;
        checking = true;
        var tools = Tools.Find();
        while (tools != null && dumps.FirstOrDefault(d => !checks.ContainsKey(d.Path)) is { } d)
        {
            checks[d.Path] = null;
            FillDumps();
            var r = await Runner.Check(tools, new[] { "--check", "--nup", d.Path });
            checks[d.Path] = r;
            if (!r.Ok && r.Text.Contains("antivirus", StringComparison.OrdinalIgnoreCase))
            {
                checking = false;
                ShowProblem(r.Text, tools: true);
                return;
            }
            if (chosenNup == null && romfs == null && r.Ok) chosenNup = d.Path;
            FillDumps();
            Refresh();
        }
        if (chosenNup == null && romfs == null && dumps.Count > 0 && tools != null) { chosenNup = dumps[0].Path; FillDumps(); }
        checking = false;
        Refresh();
    }

    void Choose(string path)
    {
        chosenNup = path;
        if (romfs != null)
        {
            romfs = exefs = null;
            Dispatcher.BeginInvoke(FillDumps);   // (the folders' line goes; not from inside this row's own event)
        }
        Refresh();
    }

    void AddDump(string path, string where)
    {
        var info = new FileInfo(path);
        if (!dumps.Any(d => string.Equals(d.Path, info.FullName, StringComparison.OrdinalIgnoreCase)))
            dumps.Insert(0, new Dump(info.FullName, where, info.Length, info.LastWriteTime));
        chosenNup = info.FullName;
        romfs = exefs = null;
        FillDumps();
        CheckAll();
        Refresh();
    }

    bool SourceReady => romfs != null && exefs != null
        || chosenNup != null && checks.TryGetValue(chosenNup, out var c) && c is { Ok: true };

    /// The install page's own numbers and its button: the free space, and what pressing the button would do
    void Refresh()
    {
        OutPathText.Text = outPath;
        var free = Storage.FreeOn(outPath);
        string disk;
        try { disk = (Path.GetPathRoot(Path.GetFullPath(outPath)) ?? "").TrimEnd('\\'); } catch (Exception) { disk = ""; }
        var gb = Storage.NeededBytes >> 30;
        SpaceText.Text = free < 0 ? L($"Il faut {gb} Go libres pendant l'installation.", $"The install needs {gb} GB free while it works.")
            : L($"Il faut {gb} Go libres pendant l'installation ; {disk} en a {free >> 30}.", $"The install needs {gb} GB free while it works; {disk} has {free >> 30}.");
        SpaceText.Foreground = Res(free >= 0 && free < Storage.NeededBytes ? "Warn" : "BoneFaint");
        RefreshMode();
    }

    void RefreshMode()
    {
        if (view != View.Install || updating || runner != null) return;
        var mine = Tools.Find()?.Version;
        launcherIsOld = latestTag != null && mine != null && mine != latestTag;
        if (launcherIsOld)
        {
            MainButtonText.Text = L("Mettre ReLink à jour", "Update ReLink");
            MainButton.IsEnabled = true;
            MainNote.Text = L($"ReLink {latestTag} est sorti (celui-ci est le {mine}) : mettez-le à jour, puis installez le jeu.",
                              $"ReLink {latestTag} is out (this one is {mine}): update it first, then install the game.");
            return;
        }
        MainButtonText.Text = Game.Installed(outPath) ? L("Réinstaller", "Install again") : L("Installer", "Install");
        MainButton.IsEnabled = SourceReady && outPath.Length > 2 && Tools.Find() != null;
        MainNote.Text = Tools.Find() == null ? L("Les outils de ReLink manquent : voir à droite.", "ReLink's build tools are missing: see on the right.")
            : SourceReady ? L("Tout est prêt. L'installation ne touche jamais à votre copie.", "Everything is ready. The install never changes your copy.")
            : L("Choisissez d'abord votre copie du jeu, à droite.", "First choose your copy of the game, on the right.");
    }

    void Dropped(object sender, DragEventArgs e)
    {
        if (runner != null || e.Data.GetData(DataFormats.FileDrop) is not string[] { Length: > 0 } paths) return;
        if (view != View.Install) ShowInstall();
        foreach (var p in paths)
        {
            if (p.EndsWith(".nup", StringComparison.OrdinalIgnoreCase) && File.Exists(p)) AddDump(p, L("Déposé", "Dropped"));
            else if (Directory.Exists(Path.Combine(p, "romfs")) && Directory.Exists(Path.Combine(p, "exefs")))
                UseFolders(Path.Combine(p, "romfs"), Path.Combine(p, "exefs"));
        }
    }

    void BrowseNup(object sender, RoutedEventArgs e)
    {
        var d = new OpenFileDialog { Filter = L("Copie du jeu (*.nup)|*.nup", "Game file (*.nup)|*.nup"), Title = L("Choisir votre copie du jeu", "Choose your game file") };
        if (d.ShowDialog(this) == true) AddDump(d.FileName, L("Choisi", "Chosen"));
    }

    /// The other way in: an unpacked game, its romfs and exefs folders (or one folder holding both)
    void BrowseFolders(object sender, RoutedEventArgs e)
    {
        var both = PickFolder(L("Choisir le dossier qui contient romfs et exefs (ou le dossier romfs)", "Choose the folder with romfs and exefs (or the romfs folder)"));
        if (both == null) return;
        if (Directory.Exists(Path.Combine(both, "romfs")) && Directory.Exists(Path.Combine(both, "exefs")))
        {
            UseFolders(Path.Combine(both, "romfs"), Path.Combine(both, "exefs"));
            return;
        }
        var ex = PickFolder(L("Choisir maintenant le dossier exefs", "Now choose the exefs folder"));
        if (ex != null) UseFolders(both, ex);
    }

    void UseFolders(string r, string x)
    {
        if (!File.Exists(Path.Combine(x, "main")))
        {
            MessageBox.Show(this, L("Ce dossier exefs ne contient pas de fichier nommé main.", "That exefs folder has no file named main."), Texts.Name);
            return;
        }
        romfs = r;
        exefs = x;
        chosenNup = null;
        foldersNote = L($"Dossiers utilisés : {r} et {x}", $"Using the folders {r} and {x}");
        FillDumps();
        Refresh();
    }

    static string? PickFolder(string title)
    {
        var d = new OpenFolderDialog { Title = title };
        return d.ShowDialog() == true ? d.FolderName : null;
    }

    void BrowseOut(object sender, RoutedEventArgs e)
    {
        outPath = PickFolder(L("Choisir où installer le jeu", "Choose where the game goes")) ?? outPath;
        Refresh();
    }

    // ================= 2: the install at work =================

    async void StartInstall()
    {
        SaveInstallChoices();
        if (launcherIsOld)
        {
            MainButton.IsEnabled = false;
            MainButtonText.Text = L("Téléchargement…", "Downloading…");
            try { await Updater.RunLatestSetup(this, Array.Empty<string>()); }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message + L(" Vous pouvez aussi télécharger ReLink-Setup.exe depuis la page des versions.",
                                                     " You can also download ReLink-Setup.exe from the releases page."), Texts.Name);
                RefreshMode();
            }
            return;
        }
        var tools = Tools.Find();
        if (tools == null)
        {
            ShowProblem(L("Les outils de ReLink ne sont pas dans son dossier.", "The build tools are not in ReLink's folder."), tools: true);
            return;
        }
        var args = new List<string>();
        if (romfs != null && exefs != null) args.AddRange(new[] { "--romfs", romfs, "--exefs", exefs });
        else if (chosenNup != null) args.AddRange(new[] { "--nup", chosenNup });
        else return;
        args.AddRange(new[] { "--out", outPath, "--language", s.Language });
        Run(tools, args);
    }

    static (string Id, string Build, string Update)[] Stages => new[]
    {
        ("download", "", L("Recevoir la mise à jour", "Download the update")),
        ("read", L("Lire votre copie du jeu", "Read your game files"), L("Lire les fichiers du jeu gardés dans le dossier", "Read the game files kept in the folder")),
        ("lift", L("Traduire le jeu en code PC", "Translate the game to PC code"), L("Traduire le jeu en code PC", "Translate the game to PC code")),
        ("compile", L("Compiler", "Compile"), L("Compiler ce qui a changé", "Compile what changed")),
        ("finish", L("Assembler le jeu", "Put the game together"), L("Mettre le nouveau jeu en place", "Swap in the new game")),
    };
    static readonly Dictionary<string, (double From, double Size)> Weight = new()
    {
        ["download"] = (0, 0.04), ["read"] = (0.04, 0.04), ["lift"] = (0.08, 0.14), ["compile"] = (0.22, 0.74), ["finish"] = (0.96, 0.04),
    };

    void MakeRows(bool update)
    {
        StageList.Children.Clear();
        rows.Clear();
        current = null;
        SetStepNotes(update);
        foreach (var stage in Stages.Where(x => update || x.Id != "download"))
        {
            var row = new StepRow(update ? stage.Update : stage.Build);
            rows[stage.Id] = row;
            StageList.Children.Add(row.Root);
        }
        if (update) rows["read"].Small(L("Vos fichiers du jeu sont gardés : le .nup n'est pas nécessaire", "Your game files are kept: the .nup isn't needed"));
    }

    /// A step begins: the one before it is done, on the page and on the staff
    void Stage(string id)
    {
        if (current != null)
        {
            if (rows.TryGetValue(current, out var was)) was.Done();
            staff[current]?.Done();
        }
        current = id;
        if (rows.TryGetValue(id, out var row)) row.Begin();
        staff[id]?.Active();
    }

    void Run(Tools tools, List<string> args, bool keepRows = false)
    {
        if (!keepRows)
        {
            updating = false;
            MakeRows(update: false);
            BuildTitle.Text = L("Installation en cours", "Installing the game");
            BuildSub.Text = outPath;
        }
        log.Clear();
        Log.Clear();
        finished = "";
        compiled = 0;
        filesDone = filesOf = 0;
        filesChanged = -1;
        filesCompiled = 0;
        compileEta = -1;
        thisCompile = 0;
        previousTotal = -1;
        Show(View.Build);
        CancelLink.Visibility = Visibility.Visible;
        SetProgress(updating ? Weight["read"].From : 0);
        elapsed.Restart();
        compileClock.Reset();
        clock.Start();
        Tick();
        if (Storage.WorkChosen && !args.Contains("--work")) args.AddRange(new[] { "--work", Storage.Work });   // (the build files where the player chose)
        runner = new Runner(tools, args);
        runner.Line += l => Dispatcher.BeginInvoke(() => Line(l));
        runner.Finished += failure => Dispatcher.BeginInvoke(() => Ended(failure));
        runner.Start();
    }

    static readonly Regex FilesLine = new(@"^(\d+) files, "), FileLine = new(@"^\s+\S+\s+\d+ MB\s+\d+s");

    void Line(string line)
    {
        if (line.StartsWith("@@stage "))
        {
            var id = line.Split(' ')[1];
            Stage(id);
            if (id == "compile") compileClock.Restart();
            if (Weight.TryGetValue(id, out var w)) SetProgress(w.From);
        }
        else if (line.StartsWith("@@progress "))
        {
            if (double.TryParse(line[11..], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var f) && current != null
                && Weight.TryGetValue(current, out var w))
            {
                if (current == "compile") compiled = f;
                SetProgress(w.From + w.Size * Math.Clamp(f, 0, 1));
            }
            return;
        }
        else if (line.StartsWith("@@plan ") || line.StartsWith("@@file ") || line.StartsWith("@@eta ") || line.StartsWith("@@times "))
        {
            Number(line);
            return;
        }
        else if (line.StartsWith("@@done ")) finished = line[7..];
        else if (FilesLine.Match(line) is { Success: true } m) filesOf = int.Parse(m.Groups[1].Value);
        else if (FileLine.IsMatch(line)) filesDone++;
        if (current == "compile" && filesChanged < 0 && filesOf > 0 && rows.TryGetValue("compile", out var c))
            c.Small(L($"{filesDone} fichiers sur {filesOf}", $"{filesDone} of {filesOf} files"));
        AddLog(line);
    }

    /// The compile timer's lines: build.py says which files this run really rebuilds and how long each took before (the
    /// countdown), and make_game.py gives this build's times beside the last one's
    void Number(string line)
    {
        var p = line.Split(' ');
        switch (p[0])
        {
            case "@@plan" when p.Length == 3 && int.TryParse(p[1], out var changed) && int.TryParse(p[2], out var total):
                filesChanged = changed;
                filesOf = total;
                filesCompiled = 0;
                ChangedFiles();
                break;
            case "@@file" when p.Length == 4 && (p[3] is "new" or "again"):
                filesCompiled++;
                ChangedFiles();
                break;
            case "@@eta" when p.Length == 2 && double.TryParse(p[1], System.Globalization.NumberStyles.Float,
                                                              System.Globalization.CultureInfo.InvariantCulture, out var eta):
                compileEta = eta;
                break;
            case "@@times" when p.Length == 4:
                long.TryParse(p[2], out thisCompile);
                long.TryParse(p[3], out previousTotal);
                break;
        }
    }

    void ChangedFiles()
    {
        if (filesChanged < 0 || !rows.TryGetValue("compile", out var row)) return;
        row.Small(filesChanged == 0 ? L("Tout est déjà compilé", "Everything is already built")
                : filesChanged >= filesOf ? L($"{filesCompiled} fichiers sur {filesChanged}", $"{filesCompiled} of {filesChanged} files")
                : L($"{filesCompiled} fichiers changés sur {filesChanged}", $"{filesCompiled} of {filesChanged} changed files"));
    }

    void AddLog(string line)
    {
        log.AppendLine(line);
        if (log.Length > 600_000) log.Remove(0, log.Length - 400_000);
        if (Log.Visibility == Visibility.Visible)
        {
            Log.AppendText(line + "\n");
            Log.ScrollToEnd();
        }
    }

    void SetProgress(double f) => PercentText.Text = $"{(int)(Math.Clamp(f, 0, 1) * 100)} %";

    void Tick()
    {
        foreach (var r in rows.Values) r.Tick();
        if (runner == null) return;
        var tail = updating ? L(" Le jeu actuel reste jouable jusqu'à la fin.", " Your current game stays playable until the end.")
                            : L(" Le PC reste utilisable.", " Your PC stays usable.");
        if (current == "compile")
        {
            var left = compileEta >= 0 ? TimeLeft(compileEta)
                : compiled > 0.03 ? TimeLeft(compileClock.Elapsed.TotalSeconds / compiled * (1 - compiled) + 60)
                : L("Estimation en cours.", "Estimating.");
            BuildNote.Text = left + tail;
            return;
        }
        BuildNote.Text = (updating ? L("Mise à jour en cours.", "Updating.") : L("Compter 10 à 45 minutes en tout.", "Allow 10 to 45 minutes in all.")) + tail;
    }

    static string TimeLeft(double seconds)
    {
        if (seconds < 45) return L("Moins d'une minute.", "Less than a minute left.");
        if (seconds < 90) return L("Encore une minute environ.", "About a minute left.");
        var minutes = (int)Math.Round(seconds / 60);
        return minutes >= 60 ? L($"Encore {minutes / 60} h {minutes % 60} min environ.", $"About {minutes / 60} h {minutes % 60} min left.")
                             : L($"Encore {minutes} minutes environ.", $"About {minutes} minutes left.");
    }

    void Ended(string? failure)
    {
        clock.Stop();
        elapsed.Stop();
        runner = null;
        if (finished.Length > 0)
        {
            foreach (var r in rows.Values) r.Done();
            foreach (var n in staff.Notes) n.Done();
            SetProgress(1);
            ShowReady(preview: false);
            return;
        }
        if (failure == "cancelled")
        {
            if (updating && Game.Installed(outPath)) ShowHome();
            else ShowInstall();
            return;
        }
        if (current != null)
        {
            if (rows.TryGetValue(current, out var row)) row.Stopped();
            staff[current]?.Problem();
        }
        ShowProblem(failure ?? L("Les outils se sont arrêtés sans dire pourquoi.", "The build tools stopped without saying why."),
                    tools: failure?.Contains("antivirus", StringComparison.OrdinalIgnoreCase) == true);
    }

    void LogToggled(object sender, RoutedEventArgs e)
    {
        var show = Log.Visibility != Visibility.Visible;
        Log.Visibility = show ? Visibility.Visible : Visibility.Collapsed;
        StageList.Visibility = show ? Visibility.Collapsed : Visibility.Visible;
        LogLink.Content = show ? L("Masquer les détails", "Hide details") : L("Afficher les détails", "Show details");
        if (show)
        {
            Log.Text = log.ToString();
            Log.ScrollToEnd();
        }
    }

    void CancelBuild(object sender, RoutedEventArgs e)
    {
        download?.Cancel();
        runner?.Cancel();
    }

    // ================= 3: the game is ready =================

    CheckBox DesktopTick = null!, StartTick = null!;
    TextBlock StorageNote = null!;
    Button FreeLink = null!;
    Border StorageRow = null!;

    void BuildReadyRows()
    {
        SwitchRow(ReadyRows, L("Raccourci sur le bureau", "Desktop shortcut"), null, out DesktopTick, out _);
        SwitchRow(ReadyRows, L("Entrée dans le menu Démarrer", "Start menu entry"), null, out StartTick, out _);
        DesktopTick.IsChecked = StartTick.IsChecked = true;
        DesktopTick.Click += ShortcutsChanged;
        StartTick.Click += ShortcutsChanged;
        StorageNote = new TextBlock { Style = St("RowHint"), Margin = new Thickness(0) };
        FreeLink = new Button { Style = St("Link"), HorizontalAlignment = HorizontalAlignment.Left, Margin = new Thickness(0, 8, 0, 0) };
        FreeLink.Click += FreeSpace;
        StorageRow = new Border { Style = St("Row"), Child = new StackPanel { Children = { StorageNote, FreeLink } }, Visibility = Visibility.Collapsed };
        ReadyRows.Children.Add(StorageRow);
        ReadyRows.Children.Add(new TextBlock
        {
            Text = L("Le premier lancement est long, une seule fois (environ un quart d’heure sur un PC moyen) : le jeu se prépare pour votre carte graphique, avec un écran de chargement.",
                     "The first start is long, once (about a quarter of an hour on a mid-range PC): the game gets ready for your graphics card, with a loading screen."),
            Style = St("Body"), Margin = new Thickness(0, 16, 0, 0),
        });
    }

    void ShowReady(bool preview)
    {
        var time = elapsed.Elapsed;
        ReadyTitle.Text = updating ? L("Le jeu est à jour", "Your game is updated") : L("Le jeu est prêt", "Your game is ready");
        var compiledNote = thisCompile > 0 ? L($" (dont {Lasted(TimeSpan.FromSeconds(thisCompile))} de compilation)", $" (compiled in {Lasted(TimeSpan.FromSeconds(thisCompile))})") : "";
        ReadyNote.Text = preview ? L($"Installé en 31 minutes dans {finished}.", $"Installed in 31 minutes in {finished}.")
            : updating ? L($"Mis à jour en {Lasted(time)}{compiledNote} dans {finished}.", $"Updated in {Lasted(time)}{compiledNote} in {finished}.")
            : L($"Installé en {Lasted(time)}{compiledNote} dans {finished}.", $"Installed in {Lasted(time)}{compiledNote} in {finished}.");
        if (!preview)
        {
            if (!updating) Finder.Shortcuts(finished, DesktopTick.IsChecked == true, StartTick.IsChecked == true);
            else
            {   // (the ones the player has, made again; the switches say which there are)
                DesktopTick.IsChecked = File.Exists(Finder.DesktopLink);
                StartTick.IsChecked = File.Exists(Finder.StartLink);
                Finder.RefreshShortcuts(finished);
            }
            if (Finder.ShortcutError != null) AddLog("shortcut not made: " + Finder.ShortcutError);
            // the game folder is the launcher's from now on: its settings page reads and writes it
            s.Folder = finished;
            try { s.LoadGame(); s.Save(); } catch (Exception) { }
            ShowSettings();
            _ = FreeAfterBuild();
        }
        Show(View.Ready);
        MainButton.IsEnabled = preview || Game.Installed(finished);
        MainNote.Text = PlaySummary();
    }

    void ShortcutsChanged(object sender, RoutedEventArgs e)
    {
        if (finished.Length > 0) Finder.Shortcuts(finished, DesktopTick.IsChecked == true, StartTick.IsChecked == true);
    }

    void ReadyToHome(object sender, RoutedEventArgs e)
    {
        ShowHome();
        CheckLatest();
    }

    /// What ReLink keeps on the PC, said once the game is ready, with the way to let it go
    async Task ShowStorageUse()
    {
        StorageRow.Visibility = Visibility.Collapsed;
        var (total, freeable) = await Task.Run(Storage.Measure);
        if (total < 100L << 20) return;
        var where = Storage.WorkChosen ? L($"{Storage.Base} et {Storage.Work}", $"{Storage.Base} and {Storage.Work}") : Storage.Base;
        StorageNote.Text = freeable >= 100L << 20
            ? L($"ReLink garde {Bytes(total)} sur ce PC ({where}). {Bytes(freeable)} ne servent qu'à rendre la prochaine mise à jour plus rapide et peuvent être libérés : le jeu reste jouable.",
                $"ReLink keeps {Bytes(total)} on this PC ({where}). {Bytes(freeable)} of it only makes the next update quicker and can be freed: your game keeps playing.")
            : L($"ReLink garde {Bytes(total)} sur ce PC ({where}) : le lanceur et ses outils.", $"ReLink keeps {Bytes(total)} on this PC ({where}): the launcher and its tools.");
        FreeLink.Visibility = freeable >= 100L << 20 ? Visibility.Visible : Visibility.Collapsed;
        FreeLink.IsEnabled = true;
        FreeLink.Content = L("Libérer cet espace", "Free up this space");
        StorageRow.Visibility = Visibility.Visible;
    }

    /// The build files (translated code, compiled objects), the downloads and older launcher versions go as soon as the game is
    /// ready; the next update then translates and compiles again. RELINK_KEEP_BUILD_FILES=1 keeps them for quick updates.
    async Task FreeAfterBuild()
    {
        if (!Storage.KeepBuildFiles)
        {
            var left = await Task.Run(Storage.Free);
            if (left >= 100L << 20) AddLog($"build files: {Bytes(left)} could not be deleted (a file in use); they go after the next build");
        }
        await ShowStorageUse();
    }

    async void FreeSpace(object sender, RoutedEventArgs e)
    {
        FreeLink.IsEnabled = false;
        FreeLink.Content = L("Libération…", "Freeing…");
        var left = await Task.Run(Storage.Free);
        await ShowStorageUse();
        if (left >= 100L << 20)
            StorageNote.Text += L($" ({Bytes(left)} n'ont pas pu être supprimés : un fichier est utilisé. Fermez le jeu et réessayez.)",
                                  $" ({Bytes(left)} could not be deleted: a file is in use. Close the game and try again.)");
    }

    // ================= when something goes wrong =================

    void ShowProblem(string why, bool tools)
    {
        lastFailure = why;
        ProblemSteps.Children.Clear();
        ProblemSub.Text = updating ? L("Votre jeu actuel n'a pas changé et reste jouable.", "Your current game is unchanged and still plays.")
                                   : L("Rien n'a été modifié dans le dossier du jeu.", "Nothing was changed in your game folder.");
        if (tools)
        {
            ProblemTitle.Text = L("Les outils sont à réparer", "The build tools need repairing");
            ProblemHead.Text = L("Votre antivirus a retiré une partie des outils", "Your antivirus removed part of the build tools");
            ProblemText.Text = why + "\n\n" + L("Les antivirus prennent parfois par erreur des outils de développement non signés pour une menace.",
                                                "Antivirus programs sometimes flag unsigned developer tools by mistake.");
            var steps = new[]
            {
                L("Ouvrez Sécurité Windows, puis Historique de protection, et restaurez les fichiers retirés.", "Open Windows Security, then Protection history, and restore the removed files."),
                L($"Ou ajoutez {Storage.Base} aux exclusions.", $"Or add {Storage.Base} to its exclusions."),
                L("Puis choisissez Réparer : ReLink télécharge de nouveau ses outils et vérifie chacun.", "Then press Repair: ReLink downloads its tools again and checks each one."),
            };
            for (var i = 0; i < steps.Length; i++)
                ProblemSteps.Children.Add(new TextBlock { Text = $"{i + 1}.  {steps[i]}", Style = St("Body"), Margin = new Thickness(0, 6, 0, 0) });
        }
        else
        {
            ProblemTitle.Text = updating ? L("La mise à jour s'est arrêtée", "The update stopped") : L("L'installation s'est arrêtée", "The install stopped");
            ProblemHead.Text = L("Ce qui s'est passé", "What went wrong");
            ProblemText.Text = why;
        }
        RepairLink.Visibility = tools ? Visibility.Visible : Visibility.Collapsed;
        DetailsLink.Content = L("Afficher les détails", "Show details");
        CopyLink.Content = L("Copier pour un rapport", "Copy for a report");
        Show(View.Problem);
    }

    void ProblemShowDetails(object sender, RoutedEventArgs e)
    {
        var lines = log.ToString().Split('\n');
        ProblemSteps.Children.Clear();
        ProblemSteps.Children.Add(new TextBox
        {
            Text = string.Join("\n", lines.Skip(Math.Max(0, lines.Length - 80))), IsReadOnly = true, FontFamily = (System.Windows.Media.FontFamily)FindResource("Mono"),
            FontSize = 11.5, Foreground = Res("BoneSoft"), Background = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromArgb(0x66, 0, 0, 0)),
            BorderThickness = new Thickness(0), Padding = new Thickness(8), TextWrapping = TextWrapping.Wrap,
        });
    }

    void CopyForReport(object sender, RoutedEventArgs e)
    {
        var lines = log.ToString().Split('\n');
        var text = $"Project ReLink {Tools.Find()?.Version ?? "(source)"} on Windows {Environment.OSVersion.Version}\n" +
                   $"{ProblemTitle.Text}: {lastFailure}\n\n" + string.Join("\n", lines.Skip(Math.Max(0, lines.Length - 200)));
        try { Clipboard.SetText(text); CopyLink.Content = L("Copié", "Copied"); } catch (Exception) { }
    }

    void Back(object sender, RoutedEventArgs e)
    {
        if (updating && Game.Installed(outPath)) { ShowHome(); CheckLatest(); }
        else ShowInstall();
    }

    /// The setup (ReLink-Setup.exe beside this app) checks every part against its checksum and downloads the missing ones
    void Repair(object sender, RoutedEventArgs e)
    {
        var setup = Path.Combine(AppContext.BaseDirectory, "ReLink-Setup.exe");
        if (File.Exists(setup))
        {
            Process.Start(new ProcessStartInfo(setup, "--repair") { UseShellExecute = false, WorkingDirectory = AppContext.BaseDirectory });
            Application.Current.Shutdown();
            return;
        }
        Open(ReleasesPage);
    }

    // ================= the update (Updater.cs) =================

    CancellationTokenSource? download;

    public void BeginUpdate(string folder, string? tag)
    {
        updating = true;
        outPath = folder;
        finished = "";
        var to = tag ?? L("la dernière version", "the latest version");
        var from = Game.Version(folder) ?? Tools.Find()?.Version;
        MakeRows(update: true);
        BuildTitle.Text = L("Mise à jour du jeu", "Updating your game");
        BuildSub.Text = from != null && from != to ? L($"De {from} à {to}", $"From {from} to {to}") : to;
        BuildNote.Text = L("Réception de la mise à jour.", "Getting the update.");
        Show(View.Build);
        SetProgress(0);
        Stage("download");
        CancelLink.Visibility = Visibility.Visible;
    }

    public CancellationToken DownloadToken => (download = new CancellationTokenSource()).Token;

    public void DownloadProgress(double f)
    {
        rows.GetValueOrDefault("download")?.Small($"{(int)(f * 100)} %");
        SetProgress(Weight["download"].Size * f);
    }

    public void Note(string line)
    {
        AddLog(line);
        rows.GetValueOrDefault("download")?.Small(line);
    }

    public void DownloadFailed(string why)
    {
        download = null;
        rows.GetValueOrDefault("download")?.Stopped();
        staff["download"]?.Problem();
        ShowProblem(why, tools: false);
        ProblemTitle.Text = L("La mise à jour n'a pas pu être téléchargée", "The update could not be downloaded");
    }

    /// The new tools are here: the folder built again with them (only what changed is compiled)
    internal void UpdateBuild(Tools tools, string folder)
    {
        download = null;
        rows.GetValueOrDefault("download")?.Small(L("Outils vérifiés", "Build tools checked"));
        Run(tools, new List<string> { "--update", folder }, keepRows: true);
    }
}
