using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using static ReLink.Texts;

namespace ReLink;

/// One window for everything a player does: install the game from their own copy, update it, set it up and start it.
/// On the left, the name, the staff and what to do next; on the right, the tablet, which shows one page at a time (Install.cs:
/// the install and the update; Home.cs: the settings and the game itself).
public partial class MainWindow : Window
{
    enum View { Install, Build, Ready, Problem, Home }

    Settings s = new();
    readonly Staff staff;
    readonly DispatcherTimer skyTimer = new() { Interval = TimeSpan.FromMinutes(5) };
    View view = View.Install;
    bool snapshot;

    static Brush Res(string key) => (Brush)Application.Current.FindResource(key);
    static Style St(string key) => (Style)Application.Current.FindResource(key);

    public MainWindow()
    {
        InitializeComponent();
        staff = new Staff(StaffCanvas);
        Scene.Draw(Sky, 1120, 700, Scene.Hour());
        skyTimer.Tick += (_, _) => Scene.Draw(Sky, 1120, 700, Scene.Hour());
        skyTimer.Start();
        clock.Tick += (_, _) => Tick();
        padTimer.Tick += (_, _) => UpdatePads();
        ApplyTexts();
        BuildHomeRows();
        BuildReadyRows();
        // (the window has no title bar: it moves by any place that is not a control)
        MouseLeftButtonDown += (_, e) =>
        {
            if (e.ButtonState != MouseButtonState.Pressed) return;
            try { DragMove(); } catch (InvalidOperationException) { }
        };
        Drop += Dropped;
        Loaded += (_, _) => Start();
        Closing += (_, _) => Native.RestoreRefresh();
        AppDomain.CurrentDomain.UnhandledException += (_, _) => Native.RestoreRefresh();
    }

    static string? Arg(string name)
    {
        var a = Environment.GetCommandLineArgs();
        var i = Array.IndexOf(a, name);
        return i >= 0 && i + 1 < a.Length ? a[i + 1] : null;
    }

    void ApplyTexts()
    {
        Subtitle.Text = L("Ocarina of Time, recompilé pour votre PC", "Ocarina of Time, recompiled for your PC");
        HelpLink.Content = L("Aide", "Help");
        FolderLink.Content = L("Dossier du jeu", "Game folder");
        UpdateLink.Content = L("Mettre à jour", "Update");

        InstallTitle.Text = L("Installer le jeu", "Install the game");
        InstallSub.Text = L("ReLink fabrique le jeu sur ce PC à partir de votre propre copie. Il faut le jeu avec sa mise à jour v131072 (build e0f3ccfa) : les autres versions ne fonctionnent pas. Cela se fait une fois et prend de 10 à 45 minutes.",
                            "ReLink builds the game on this PC from your own copy. It needs the game with its update v131072 (build e0f3ccfa): other versions do not work. It happens once and takes 10 to 45 minutes.");
        CopyHead.Text = L("Votre copie du jeu", "Your copy of the game");
        BrowseNupLink.Content = L("Choisir un fichier .nup…", "Choose a .nup file…");
        BrowseFoldersLink.Content = L("Choisir les dossiers romfs et exefs…", "Choose the romfs and exefs folders…");
        OutHead.Text = L("Dossier d'installation", "Install folder");
        OutChange.Content = L("Changer…", "Change…");

        LogLink.Content = L("Afficher les détails", "Show details");
        CancelLink.Content = L("Annuler", "Cancel");
        ReadySettings.Content = L("Régler le jeu", "Set up the game");
        BackLink.Content = L("Retour", "Back");
        RepairLink.Content = L("Réparer", "Repair");
        DetailsLink.Content = L("Afficher les détails", "Show details");
        CopyLink.Content = L("Copier pour un rapport", "Copy for a report");

        TabDisplay.Content = L("Affichage", "Display");
        TabSmooth.Content = L("Fluidité", "Smoothness");
        TabGame.Content = L("Jeu", "Game");
        TabControls.Content = L("Contrôles", "Controls");
        HomeFoot.Text = L("Le menu en jeu (Inser ou F1) change aussi ces réglages.", "The in-game menu (Insert or F1) changes these settings too.");

        PlayingTitle.Text = L("Bonne partie", "Enjoy the game");
        RestoreNow.Content = L("Remettre la fréquence de l'écran maintenant", "Put the display's refresh rate back now");
    }

    // ================= the start =================

    async void Start()
    {
        var a = Environment.GetCommandLineArgs();
        snapshot = Arg("--snapshot") != null;
        s = Settings.Load(Arg("--launch")?.TrimEnd('\\', '/'));
        LoadInstallChoices();
        ShowSettings();
        UpdateScreen();
        UpdatePads();
        padTimer.Start();
        if (snapshot)
        {
            Preview(Arg("--view") ?? "home");
            TakeSnapshot(Arg("--snapshot")!);
            return;
        }
        if (Arg("--check") is { } report)   // (tests: what the launcher sees, without touching the display)
        {
            var (w, h, hz) = Native.CurrentMode();
            var ok = Native.SetRefresh(60, true, out var why);
            File.WriteAllLines(report, new[]
            {
                $"folder={s.Folder}", $"game={(s.Folder == "" ? "" : Game.Exe(s.Folder))}", $"version={(s.Folder == "" ? "" : Game.Version(s.Folder))}",
                $"tools={Tools.Find()?.Root}", $"mode={w}x{h}@{hz}", $"rates={string.Join(",", Native.RatesAt(w, h))}",
                $"set60_test={(ok ? "ok" : "no")}: {why}", $"pads={Native.PadsConnected()}", $"language={s.Language}",
            });
            Close();
            return;
        }
        if (a.Contains("--update"))
        {
            Updater.FromCommandLine(this);
            return;
        }
        if (s.Folder != "" && !a.Contains("--install"))
        {
            ShowHome();
            staff.Play();
            CheckLatest();
            if (a.Contains("--play")) Play();   // (a shortcut that starts the game at once: ReLink.exe --launch <folder> --play)
            return;
        }
        ShowInstall();
        staff.Play();
        var found = await Task.Run(Finder.FindDumps);
        foreach (var d in found)
            if (!dumps.Any(x => string.Equals(x.Path, d.Path, StringComparison.OrdinalIgnoreCase))) dumps.Add(d);
        FillDumps();
        CheckAll();
        latestTag = await Updater.LatestTag();
        RefreshMode();
    }

    // ================= the pages =================

    void Show(View v)
    {
        view = v;
        var pages = new (View Is, FrameworkElement Page)[]
        {
            (View.Install, InstallView), (View.Build, BuildView), (View.Ready, ReadyView), (View.Problem, ProblemView), (View.Home, HomeView),
        };
        foreach (var (k, page) in pages)
        {
            page.Visibility = k == v ? Visibility.Visible : Visibility.Collapsed;
            if (k != v || snapshot) continue;
            page.Opacity = 0;
            page.BeginAnimation(OpacityProperty, new DoubleAnimation(1, TimeSpan.FromMilliseconds(220)));
        }
        // the left column says the same thing as the page: a button when there is something to press, the percentage while it works
        ProgressBlock.Visibility = v == View.Build ? Visibility.Visible : Visibility.Collapsed;
        MainButton.Visibility = v is View.Install or View.Ready or View.Home ? Visibility.Visible : Visibility.Collapsed;
        MainNote.Visibility = MainButton.Visibility;
        StatusRow.Visibility = v == View.Home ? Visibility.Visible : Visibility.Collapsed;
        FolderLink.Visibility = v is View.Ready or View.Home ? Visibility.Visible : Visibility.Collapsed;
        if (v is View.Ready or View.Home) MainButtonText.Text = L("Jouer", "Play");
    }

    void MainClick(object sender, RoutedEventArgs e)
    {
        if (view == View.Install) StartInstall();
        else if (view is View.Ready or View.Home) Play();
    }

    void TabChanged(object sender, RoutedEventArgs e)
    {
        if (PageDisplay == null) return;   // (the first tab is checked while the window is still being made)
        foreach (var (tab, page) in new[] { (TabDisplay, PageDisplay), (TabSmooth, PageSmooth), (TabGame, PageGame), (TabControls, PageControls) })
            page.Visibility = tab.IsChecked == true ? Visibility.Visible : Visibility.Collapsed;
    }

    // ================= links =================

    static void Open(string what)
    {
        try { Process.Start(new ProcessStartInfo(what) { UseShellExecute = true }); } catch (Exception) { }
    }

    void OpenHelp(object sender, RoutedEventArgs e) => Open(HelpPage);

    void OpenGameFolder(object sender, RoutedEventArgs e)
    {
        var folder = view == View.Ready && finished.Length > 0 ? finished : s.Folder;
        if (folder != "" && Directory.Exists(folder)) Open(folder);
    }

    // ================= the window =================

    void Minimize(object sender, RoutedEventArgs e) => WindowState = WindowState.Minimized;

    void CloseWindow(object sender, RoutedEventArgs e)
    {
        if (runner != null && MessageBox.Show(this, L("Arrêter l'installation et fermer ?", "Stop the install and close?"), Texts.Name, MessageBoxButton.YesNo) != MessageBoxResult.Yes) return;
        runner?.Cancel();
        Close();
    }

    // ================= pictures of the window, for checking it (nothing is run) =================
    //   ReLink.exe --snapshot out.png --view install|build|update|ready|problem|home|home-update|smooth|game|controls|playing

    void Preview(string what)
    {
        switch (what)
        {
            case "install":
                dumps.Clear();
                dumps.Add(new Dump(@"C:\Users\player\Downloads\ocarina-of-time.nup", L("Téléchargements", "Downloads"), 31L << 30, DateTime.Now.AddDays(-3)));
                dumps.Add(new Dump(@"D:\dumps\ocarina-v0.nup", "D:", 29L << 30, DateTime.Now.AddDays(-40)));
                checks[dumps[0].Path] = (true, "This is the supported version.");
                checks[dumps[1].Path] = (false, "This is build 1a2b3c4d. ReLink supports build e0f3ccfa.");
                chosenNup = dumps[0].Path;
                outPath = @"E:\Games\ReLink";
                ShowInstall();
                break;
            case "build":
                outPath = @"E:\Games\ReLink";
                MakeRows(update: false);
                Show(View.Build);
                BuildTitle.Text = L("Installation en cours", "Installing the game");
                BuildSub.Text = outPath;
                Stage("read"); rows["read"].Small(L("Version vérifiée", "Version checked"));
                Stage("lift");
                Stage("compile"); rows["compile"].Small(L("31 fichiers sur 58", "31 of 58 files"));
                SetProgress(0.54);
                BuildNote.Text = L("Encore 19 minutes environ. Le PC reste utilisable.", "About 19 minutes left. Your PC stays usable.");
                break;
            case "update":
                BeginUpdate(outPath, "v0.7.0");
                Stage("read"); rows["download"].Small(L("3 Mo, outils inchangés", "3 MB, build tools unchanged"));
                Stage("lift");
                Stage("compile"); rows["compile"].Small(L("7 fichiers changés sur 516", "7 of 516 files changed"));
                SetProgress(0.62);
                BuildNote.Text = L("Encore 3 minutes environ. Le jeu actuel reste jouable jusqu'à la fin.", "About 3 minutes left. Your current game stays playable until the end.");
                break;
            case "ready":
                MakeRows(update: false);
                foreach (var id in new[] { "read", "lift", "compile", "finish" }) Stage(id);
                finished = outPath;
                elapsed.Restart();
                ShowReady(preview: true);
                break;
            case "problem":
                MakeRows(update: false);
                Stage("read"); Stage("lift"); Stage("compile");
                ShowProblem(L("Des fichiers du compilateur de ReLink ont disparu (clang.exe dans tools\\llvm-mingw\\bin).",
                              "Files of ReLink's own compiler are missing (clang.exe in tools\\llvm-mingw\\bin)."), tools: true);
                break;
            case "playing":
                ShowHome();
                ShowPlaying(L("L'écran est à 60 Hz ; il revient à 100 Hz quand vous quittez le jeu.", "The display is at 60 Hz; it goes back to 100 Hz when you leave the game."));
                break;
            default:
                ShowHome();
                StatusText.Text = L("Version v0.6.1, à jour.", "Version v0.6.1, up to date.");
                if (what == "home-update")
                {
                    StatusText.Text = L("Version v0.6.1. La v0.7.0 est disponible.", "Version v0.6.1. v0.7.0 is available.");
                    UpdateLink.Visibility = Visibility.Visible;
                }
                if (what == "smooth") TabSmooth.IsChecked = true;
                else if (what == "game") TabGame.IsChecked = true;
                else if (what == "controls") TabControls.IsChecked = true;
                break;
        }
    }

    void TakeSnapshot(string path)
    {
        Dispatcher.BeginInvoke(() =>
        {
            UpdateLayout();
            var dpi = VisualTreeHelper.GetDpi(this);
            var bmp = new RenderTargetBitmap((int)(ActualWidth * dpi.DpiScaleX), (int)(ActualHeight * dpi.DpiScaleY), dpi.PixelsPerInchX, dpi.PixelsPerInchY, PixelFormats.Pbgra32);
            bmp.Render(Root);
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(bmp));
            using (var f = File.Create(path)) enc.Save(f);
            Close();
        }, DispatcherPriority.ApplicationIdle);
    }
}
