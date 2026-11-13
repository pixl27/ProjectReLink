using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Animation;
using System.Windows.Threading;
using static ReLink.Texts;

namespace ReLink;

/// Home: the game's settings (Settings.cs: the game's own options file, shared with its in-game menu), what the game needs
/// (written on the staff), and the game itself.
public partial class MainWindow
{
    readonly DispatcherTimer padTimer = new() { Interval = TimeSpan.FromMilliseconds(700) };
    Process? game;
    bool loading = true;
    int origHz;
    bool display60Possible;

    CheckBox Fullscreen = null!, Display60 = null!, FpsCounter = null!, Ultrawide = null!, FramePacing = null!, Audio = null!, Handheld = null!, InvertY = null!;
    WrapPanel ResolutionGroup = null!, UpscaleGroup = null!, FrameRateGroup = null!, SpeedGroup = null!, FramegenGroup = null!;
    Slider Sharpness = null!;
    TextBlock SharpnessText = null!, Display60Hint = null!, FolderText = null!;
    ComboBox LanguageBox = null!;

    sealed record LangItem(string Code, string Name)
    {
        public override string ToString() => Name;
    }

    // ---------------------------------------------------------------- the rows
    Border SwitchRow(Panel into, string label, string? hint, out CheckBox box, out TextBlock hintBlock)
    {
        box = new CheckBox { Style = St("Switch") };
        hintBlock = new TextBlock { Text = hint ?? "", Style = St("RowHint"), Visibility = hint == null ? Visibility.Collapsed : Visibility.Visible };
        var texts = new StackPanel { Children = { new TextBlock { Text = label, Style = St("RowLabel") }, hintBlock } };
        var grid = new Grid { ColumnDefinitions = { new ColumnDefinition(), new ColumnDefinition { Width = GridLength.Auto } } };
        Grid.SetColumn(box, 1);
        grid.Children.Add(texts);
        grid.Children.Add(box);
        var row = new Border { Style = St("Row"), Child = grid };
        into.Children.Add(row);
        return row;
    }

    WrapPanel ChoiceRow(Panel into, string label, string? hint, string group, params (int Value, string Text)[] choices)
    {
        var panel = new WrapPanel { Margin = new Thickness(0, 7, 0, 0) };
        foreach (var (value, text) in choices)
            panel.Children.Add(new RadioButton { Style = St("Choice"), GroupName = group, Content = text, Tag = value });
        var stack = new StackPanel { Children = { new TextBlock { Text = label, Style = St("RowLabel") }, panel } };
        if (hint != null) stack.Children.Add(new TextBlock { Text = hint, Style = St("RowHint") });
        into.Children.Add(new Border { Style = St("Row"), Child = stack });
        return panel;
    }

    void BuildHomeRows()
    {
        // ---- display
        SwitchRow(RowsDisplay, L("Plein écran", "Full screen"), L("F11 bascule aussi pendant le jeu.", "F11 switches it while you play, too."), out Fullscreen, out _);
        SwitchRow(RowsDisplay, L("Écran à 60 Hz pendant le jeu", "Display at 60 Hz while you play"),
                  L("Sans synchronisation adaptative, un écran à 100 ou 144 Hz montre 60 images par seconde par à-coups. À 60 Hz chaque image dure le même temps. La fréquence d'origine revient quand le jeu se ferme.",
                    "Without adaptive sync, a 100 or 144 Hz display shows 60 frames a second unevenly. At 60 Hz every frame lasts the same time. The display's own rate comes back when the game closes."),
                  out Display60, out Display60Hint);
        ResolutionGroup = ChoiceRow(RowsDisplay, L("Résolution du rendu 3D", "3D render resolution"),
                  L("Native est la plus nette. Les pourcentages allègent la carte graphique ; le jeu agrandit ensuite l'image.",
                    "Native is the sharpest. The percentages lighten the graphics card's work; the game scales the picture up afterwards."),
                  "Resolution", (1, L("Native", "Native")), (0, L("Automatique", "Automatic")), (2, "75 %"), (3, "67 %"), (5, "58 %"), (4, "50 %"));
        UpscaleGroup = ChoiceRow(RowsDisplay, L("Mise à l'échelle", "Upscaling"), null, "Upscale", (0, L("Simple", "Simple")), (1, "Catmull-Rom"), (2, "Lanczos"));

        Sharpness = new Slider { Style = St("Line"), Minimum = 0, Maximum = 100, TickFrequency = 5, IsSnapToTickEnabled = true, Width = 210, LargeChange = 10, SmallChange = 5 };
        SharpnessText = new TextBlock { Width = 48, TextAlignment = TextAlignment.Right, VerticalAlignment = VerticalAlignment.Center, Foreground = Res("BoneSoft") };
        var sharp = new Grid { ColumnDefinitions = { new ColumnDefinition(), new ColumnDefinition { Width = GridLength.Auto }, new ColumnDefinition { Width = GridLength.Auto } } };
        sharp.Children.Add(new TextBlock { Text = L("Netteté", "Sharpness"), Style = St("RowLabel") });
        Grid.SetColumn(Sharpness, 1);
        Grid.SetColumn(SharpnessText, 2);
        sharp.Children.Add(Sharpness);
        sharp.Children.Add(SharpnessText);
        RowsDisplay.Children.Add(new Border { Style = St("Row"), Child = sharp });

        SwitchRow(RowsDisplay, L("Compteur d'images par seconde", "Frames-per-second counter"), null, out FpsCounter, out _);
        SwitchRow(RowsDisplay, L("Ultra-large", "Ultrawide"), L("La caméra prend le format de l'écran s'il est plus large que 16:9.", "The camera takes the display's shape when it is wider than 16:9."),
                  out Ultrawide, out _);

        // ---- smoothness
        FrameRateGroup = ChoiceRow(RowsSmooth, L("Images par seconde", "Frames per second"),
                  L("Débloqué : jusqu'à la fréquence de l'écran, et le jeu suit le temps réel. Sur un écran à fréquence fixe, 60 est plus régulier.",
                    "Unlocked: up to the display's rate, and the game follows real time. On a fixed-rate display, 60 is steadier."),
                  "FrameRate", (0, L("60, comme la console", "60, as on the console")), (1, L("Débloqué", "Unlocked")));
        SpeedGroup = ChoiceRow(RowsSmooth, L("Vitesse du jeu", "Game speed"),
                  L("Temps réel : si une image tarde, le jeu ne ralentit pas.", "Real time: when a frame is late, the game does not slow down."),
                  "Speed", (0, L("Par image", "By frame")), (1, L("Temps réel", "Real time")));
        FramegenGroup = ChoiceRow(RowsSmooth, L("Génération d'images", "Frame generation"),
                  L("Des images intercalées : plus fluide à l'œil, pas plus réactif. L'écran garde alors sa fréquence.",
                    "Frames made in between: smoother to the eye, not more responsive. The display keeps its own rate then."),
                  "Framegen", (0, L("Aucune", "None")), (2, "×2"), (3, "×3"), (4, "×4"));
        SwitchRow(RowsSmooth, L("Rythme régulier des images générées", "Even pacing of generated frames"),
                  L("Avec la génération d'images seulement.", "With frame generation only."), out FramePacing, out _);

        // ---- game
        LanguageBox = new ComboBox { Style = St("List"), Width = 230, ItemsSource = GameLanguages.Select(l => new LangItem(l.Code, l.Name)).ToList() };
        var lang = new Grid { ColumnDefinitions = { new ColumnDefinition(), new ColumnDefinition { Width = GridLength.Auto } } };
        lang.Children.Add(new TextBlock { Text = L("Langue du jeu", "Game language"), Style = St("RowLabel") });
        Grid.SetColumn(LanguageBox, 1);
        lang.Children.Add(LanguageBox);
        RowsGame.Children.Add(new Border { Style = St("Row"), Child = lang });
        SwitchRow(RowsGame, L("Son", "Sound"), null, out Audio, out _);
        SwitchRow(RowsGame, L("Profil portable", "Handheld profile"),
                  L("Les réglages graphiques du mode portable : plus léger, plus flou.", "The handheld mode's graphics settings: lighter, softer."), out Handheld, out _);
        FolderText = new TextBlock { Style = St("RowHint"), TextTrimming = TextTrimming.CharacterEllipsis, TextWrapping = TextWrapping.NoWrap };
        var open = new Button { Style = St("Link"), Content = L("Ouvrir", "Open"), Margin = new Thickness(0, 0, 22, 0) };
        open.Click += OpenGameFolder;
        var again = new Button { Style = St("Link"), Content = L("Installer de nouveau…", "Install again…") };
        again.Click += (_, _) => { if (game == null) { ShowInstall(); CheckAll(); } };
        RowsGame.Children.Add(new Border
        {
            Style = St("Row"),
            Child = new StackPanel
            {
                Children =
                {
                    new TextBlock { Text = L("Dossier du jeu", "Game folder"), Style = St("RowLabel") }, FolderText,
                    new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 8, 0, 0), Children = { open, again } },
                },
            },
        });

        // ---- controls
        RowsControls.Children.Add(new TextBlock
        {
            Text = L("Manette Xbox, ou toute manette XInput ; une manette PlayStation ou Switch passe par Steam Input ou DS4Windows. Les boutons suivent leur position : celui du bas est le B de la console.",
                     "An Xbox controller, or any XInput one; a PlayStation or Switch controller goes through Steam Input or DS4Windows. Buttons follow their place: the bottom one is the console's B."),
            Style = St("Body"), Margin = new Thickness(0, 10, 0, 4),
        });
        SwitchRow(RowsControls, L("Souris : inverser l'axe vertical", "Mouse: invert the vertical axis"),
                  L("La souris tourne la caméra après un clic dans la fenêtre du jeu ; Échap la libère.", "The mouse turns the camera after a click in the game's window; Escape lets it go."),
                  out InvertY, out _);
        static string K(char c) => Native.KeyLabel(c);
        Keys(L("Manette", "Controller"), new[]
        {
            (L("A (bas)", "A (bottom)"), L("B de la console", "the console's B")), (L("B (droite)", "B (right)"), L("A de la console", "the console's A")),
            (L("X (gauche)", "X (left)"), L("Y de la console", "the console's Y")), (L("Y (haut)", "Y (top)"), L("X de la console", "the console's X")),
            ("LB, RB", "L, R"), ("LT, RT", "ZL, ZR"), ("Start", "+"), ("Back", "−"),
            (L("Sticks", "Sticks"), L("les sticks (clic : L3, R3)", "the sticks (click: L3, R3)")), (L("Croix", "D-pad"), L("la croix directionnelle", "the D-pad")),
        });
        Keys(L("Clavier et souris", "Keyboard and mouse"), new[]
        {
            ($"{K('W')} {K('A')} {K('S')} {K('D')}", L("stick gauche", "left stick")),
            (L($"Flèches, ou {K('I')} {K('J')} {K('K')} {K('L')}", $"Arrows, or {K('I')} {K('J')} {K('K')} {K('L')}"), L("caméra (stick droit)", "camera (right stick)")),
            (L("Espace", "Space"), "X"), (K('E'), "A"), (L("Maj", "Shift"), "B"), ($"{K('Q')}, {K('R')}", "L, R"),
            (L($"{K('Z')}, {K('X')} ou {K('F')}", $"{K('Z')}, {K('X')} or {K('F')}"), "ZL, ZR"),
            ($"Ctrl, {K('C')}", L("clic du stick gauche, droit", "left, right stick click")),
            (L($"Tab ou Entrée, {K('M')}", $"Tab or Enter, {K('M')}"), "+, −"), ("1 2 3 4", L("la croix", "the D-pad")),
            (L("Clic gauche, droit", "Left, right click"), "Y, ZL"), ("F11", L("plein écran", "full screen")),
            (L("Inser ou F1", "Insert or F1"), L("le menu des réglages en jeu", "the in-game settings menu")),
        });

        Wire();
    }

    void Keys(string title, (string Key, string What)[] list)
    {
        RowsControls.Children.Add(new TextBlock { Text = title, Style = St("Section") });
        foreach (var (key, what) in list)
        {
            var g = new Grid { Margin = new Thickness(0, 3, 0, 3), ColumnDefinitions = { new ColumnDefinition { Width = new GridLength(190) }, new ColumnDefinition() } };
            g.Children.Add(new TextBlock { Text = key, FontSize = 14 });
            var t = new TextBlock { Text = what, FontSize = 14, Foreground = Res("BoneSoft"), TextWrapping = TextWrapping.Wrap };
            Grid.SetColumn(t, 1);
            g.Children.Add(t);
            RowsControls.Children.Add(g);
        }
    }

    // ---------------------------------------------------------------- settings <-> controls
    static IEnumerable<RadioButton> Radios(Panel p) => p.Children.OfType<RadioButton>();
    static void Pick(Panel p, int v)
    {
        foreach (var r in Radios(p)) r.IsChecked = (int)r.Tag == v;
    }
    void OnPick(Panel p, Action<int> set)
    {
        foreach (var r in Radios(p))
            r.Checked += (_, _) => { if (!loading) { set((int)r.Tag); Changed(); } };
    }
    void OnSwitch(CheckBox c, Action<bool> set)
    {
        c.Checked += (_, _) => { if (!loading) { set(true); Changed(); } };
        c.Unchecked += (_, _) => { if (!loading) { set(false); Changed(); } };
    }

    void Wire()
    {
        OnSwitch(Fullscreen, v => s.Fullscreen = v ? 1 : 0);
        OnSwitch(Display60, v => s.Display60 = v);
        OnSwitch(FpsCounter, v => s.FpsCounter = v ? 1 : 0);
        OnSwitch(Ultrawide, v => s.Ultrawide = v ? 1 : 0);
        OnSwitch(FramePacing, v => s.FramePacing = v ? 1 : 0);
        OnSwitch(Audio, v => s.Audio = v);
        OnSwitch(Handheld, v => s.Handheld = v ? 1 : 0);
        OnSwitch(InvertY, v => s.InvertY = v);
        OnPick(ResolutionGroup, v => s.Resolution = v);
        OnPick(UpscaleGroup, v => s.Upscaling = v);
        OnPick(FrameRateGroup, v => s.FrameRate = v);
        OnPick(SpeedGroup, v => s.GameSpeed = v);
        OnPick(FramegenGroup, v => s.FrameGeneration = v);
        Sharpness.ValueChanged += (_, e) =>
        {
            SharpnessText.Text = $"{(int)e.NewValue} %";
            if (!loading) { s.Sharpness = (int)e.NewValue; Changed(); }
        };
        LanguageBox.SelectionChanged += (_, _) =>
        {
            if (!loading && LanguageBox.SelectedItem is LangItem l) { s.Language = l.Code; Changed(); }
        };
    }

    void ShowSettings()
    {
        loading = true;
        Fullscreen.IsChecked = s.Fullscreen != 0;
        Display60.IsChecked = s.Display60;
        FpsCounter.IsChecked = s.FpsCounter != 0;
        Ultrawide.IsChecked = s.Ultrawide != 0;
        FramePacing.IsChecked = s.FramePacing != 0;
        Audio.IsChecked = s.Audio;
        Handheld.IsChecked = s.Handheld != 0;
        InvertY.IsChecked = s.InvertY;
        Pick(ResolutionGroup, s.Resolution);
        Pick(UpscaleGroup, s.Upscaling);
        Pick(FrameRateGroup, s.FrameRate);
        Pick(FramegenGroup, s.FrameGeneration);
        Sharpness.Value = s.Sharpness;
        SharpnessText.Text = $"{s.Sharpness} %";
        var items = (List<LangItem>)LanguageBox.ItemsSource;
        LanguageBox.SelectedItem = items.FirstOrDefault(l => l.Code == s.Language) ?? items[0];
        FolderText.Text = s.Folder;
        Dependents();
        loading = false;
    }

    /// Options that depend on others: unlocked runs game time on the clock; pacing only means something with made frames
    void Dependents()
    {
        SpeedGroup.IsEnabled = s.FrameRate == 0;
        Pick(SpeedGroup, s.FrameRate != 0 ? 1 : s.GameSpeed);
        FramePacing.IsEnabled = s.FrameGeneration != 0;
        // (frame generation fills the display's own refresh: no switch to 60 Hz then)
        Display60.IsEnabled = display60Possible && s.FrameGeneration < 2;
        if (view is View.Home or View.Ready) MainNote.Text = PlaySummary();
        ShowChecks();
    }

    /// What Play will do, in one line under the button
    string PlaySummary()
    {
        var parts = new List<string> { s.Fullscreen != 0 ? L("plein écran", "full screen") : L("fenêtre", "window") };
        if (s.Display60 && origHz != 60 && origHz > 0 && display60Possible && s.FrameGeneration < 2) parts.Add(L("écran à 60 Hz", "display at 60 Hz"));
        if (s.FrameGeneration >= 2) parts.Add(L($"images ×{s.FrameGeneration}", $"frames ×{s.FrameGeneration}"));
        else parts.Add(s.FrameRate != 0 ? L("images débloquées", "unlocked frame rate") : L("60 images par seconde", "60 frames a second"));
        var line = string.Join(", ", parts);
        return char.ToUpper(line[0]) + line[1..] + ".";
    }

    void Changed()
    {
        loading = true;
        Dependents();
        loading = false;
        try { s.Save(); }
        catch (Exception e) { MainNote.Text = L("Réglages non enregistrés : ", "Settings not saved: ") + e.Message; }
    }

    // ---------------------------------------------------------------- what the game needs: the notes on the staff
    void ShowHome()
    {
        updating = false;
        staff.Set(("game", L("Jeu", "Game")), ("pad", L("Manette", "Controller")), ("screen", L("Écran", "Display")));
        Show(View.Home);
        try { if (s.Folder != "") s.LoadGame(); } catch (Exception) { }
        ShowSettings();
        UpdateScreen();
        MainButton.IsEnabled = Game.Installed(s.Folder) && game == null;
        MainNote.Text = PlaySummary();
        var version = s.Folder == "" ? null : Game.Version(s.Folder);
        StatusText.Text = version != null ? L($"Version {version}.", $"Version {version}.") : L("Jeu construit depuis les sources.", "Game built from the sources.");
        UpdateLink.Visibility = Visibility.Collapsed;
    }

    void ShowChecks()
    {
        if (view != View.Home) return;
        if (staff["game"] is { } g)
        {
            var ok = Game.Installed(s.Folder);
            if (ok) g.Done(); else g.Problem();
            g.Sub = ok ? L("Prêt", "Ready") : L("Introuvable : installez-le de nouveau", "Not found: install it again");
        }
        if (staff["pad"] is { } p)
        {
            var n = Native.PadsConnected();
            if (n > 0) p.Done(); else p.Waiting();
            p.Sub = n == 0 ? L("Aucune : le clavier et la souris jouent aussi", "None: keyboard and mouse play too")
                  : n == 1 ? L("Connectée", "Connected") : L($"{n} connectées", $"{n} connected");
        }
        if (staff["screen"] is { } d)
        {
            var (w, h, hz) = Native.CurrentMode();
            d.Done();
            var to60 = s.Display60 && display60Possible && s.FrameGeneration < 2;
            d.Sub = w == 0 ? "" : to60 ? L($"{w}×{h}, {hz} Hz, 60 Hz pendant le jeu", $"{w}×{h}, {hz} Hz, 60 Hz while you play") : $"{w}×{h}, {hz} Hz";
        }
    }

    void UpdateScreen()
    {
        var (w, h, hz) = Native.CurrentMode();
        if (game == null) origHz = hz;
        var can60 = w > 0 && Native.RatesAt(w, h).Contains(60);
        display60Possible = can60 && origHz != 60;
        Display60.IsEnabled = display60Possible && s.FrameGeneration < 2;
        if (origHz == 60) Display60Hint.Text = L("L'écran est déjà à 60 Hz : rien à changer.", "The display is already at 60 Hz: nothing to change.");
        else if (!can60) Display60Hint.Text = L("Cet écran ne propose pas 60 Hz à sa résolution actuelle.", "This display has no 60 Hz mode at its current resolution.");
        ShowChecks();
    }

    int padsBefore = -1;
    void UpdatePads()
    {
        var n = Native.PadsConnected();
        if (n == padsBefore) return;
        padsBefore = n;
        ShowChecks();
    }

    /// The release that built the game beside the latest one: says which, and offers the update
    async void CheckLatest()
    {
        var folder = s.Folder;
        if (folder == "") return;
        var version = Game.Version(folder);
        latestTag = await Updater.LatestTag();
        if (view != View.Home || folder != s.Folder) return;
        updateTarget = latestTag ?? Tools.Find()?.Version;
        UpdateLink.Visibility = Visibility.Collapsed;
        if (version == null)
        {
            StatusText.Text = latestTag != null ? L($"Jeu construit depuis les sources. Dernière version publiée : {latestTag}.", $"Game built from the sources. Latest release: {latestTag}.")
                                                : L("Jeu construit depuis les sources.", "Game built from the sources.");
            return;
        }
        if (updateTarget == null) StatusText.Text = L($"Version {version}. Les mises à jour n'ont pas pu être vérifiées.", $"Version {version}. Updates could not be checked.");
        else if (version != updateTarget && Game.KeepsGameFiles(folder))
        {
            StatusText.Text = L($"Version {version}. La {updateTarget} est disponible.", $"Version {version}. {updateTarget} is available.");
            UpdateLink.Visibility = Visibility.Visible;
        }
        else if (version != updateTarget)
            StatusText.Text = L($"Version {version}. La {updateTarget} est sortie : installez de nouveau le jeu pour l'obtenir.", $"Version {version}. {updateTarget} is out: install the game again to get it.");
        else StatusText.Text = latestTag == null ? L($"Version {version}. Les mises à jour n'ont pas pu être vérifiées.", $"Version {version}. Updates could not be checked.")
                                                 : L($"Version {version}, à jour.", $"Version {version}, up to date.");
    }

    void UpdateClick(object sender, RoutedEventArgs e)
    {
        if (game != null || s.Folder == "") return;
        Updater.UpdateFolder(this, s.Folder, updateTarget, fetched: false);
    }

    // ---------------------------------------------------------------- play
    void Play()
    {
        var folder = view == View.Ready && finished.Length > 0 ? finished : s.Folder;
        if (game != null || !Game.Installed(folder)) return;
        s.Folder = folder;
        try { s.Save(); }
        catch (Exception ex) { MainNote.Text = L("Réglages non enregistrés : ", "Settings not saved: ") + ex.Message; return; }
        var screenNote = "";
        UpdateScreen();
        if (s.Display60 && display60Possible && s.FrameGeneration < 2)   // (frame generation fills the display's own refresh)
            screenNote = Native.SetRefresh(60, false, out var why)
                ? L($"L'écran est à 60 Hz ; il revient à {origHz} Hz quand vous quittez le jeu.", $"The display is at 60 Hz; it goes back to {origHz} Hz when you leave the game.")
                : L($"Écran laissé à {origHz} Hz ({why}).", $"Display left at {origHz} Hz ({why}).");
        var psi = new ProcessStartInfo(Game.Exe(folder)) { WorkingDirectory = folder, UseShellExecute = false };
        foreach (var k in psi.Environment.Keys.Where(k => k.StartsWith("HOS_", StringComparison.OrdinalIgnoreCase) ||
                                                          k.StartsWith("RX_", StringComparison.OrdinalIgnoreCase)).ToList())
            psi.Environment.Remove(k);   // (a test's leftovers must not change the game)
        psi.Environment["HOOKSHOT_NO_UPDATE_CHECK"] = "1";   // (the launcher already looked: the game does not ask again)
        if (!s.Audio) psi.Environment["HOS_AUDIO"] = "0";
        if (s.InvertY) psi.Environment["HOS_MOUSE_INVERT_Y"] = "1";
        if (s.FrameRate != 0) psi.Environment["HOS_FPS_UNLOCKED"] = "1";   // (the game reads the unlocked rate from here only: host/nn_vi.c)
        try
        {
            game = Process.Start(psi);
        }
        catch (Exception ex)
        {
            Native.RestoreRefresh();
            MainNote.Text = L("Le jeu n'a pas démarré : ", "The game did not start: ") + ex.Message;
            return;
        }
        if (game == null) { Native.RestoreRefresh(); return; }
        game.EnableRaisingEvents = true;
        game.Exited += (_, _) => Dispatcher.BeginInvoke(GameEnded);
        MainButton.IsEnabled = false;
        ShowPlaying(screenNote == "" ? L("Le jeu tourne.", "The game is running.") : screenNote);
        WindowState = WindowState.Minimized;
    }

    void ShowPlaying(string hint)
    {
        PlayingHint.Text = hint;
        RestoreNow.Visibility = game != null && Native.CurrentMode().hz != origHz ? Visibility.Visible : Visibility.Collapsed;
        // (only the sky and the hills stay behind it: nothing to press while the game runs but the window's own buttons)
        LeftColumn.Visibility = Tablet.Visibility = Visibility.Hidden;
        PlayingOverlay.Visibility = Visibility.Visible;
        if (!snapshot) PlayingOverlay.BeginAnimation(OpacityProperty, new DoubleAnimation(0, 1, TimeSpan.FromMilliseconds(250)));
    }

    void GameEnded()
    {
        Native.RestoreRefresh();
        game = null;
        PlayingOverlay.BeginAnimation(OpacityProperty, null);
        PlayingOverlay.Visibility = Visibility.Collapsed;
        LeftColumn.Visibility = Tablet.Visibility = Visibility.Visible;
        ShowHome();   // (the in-game menu may have changed the options: read again)
        CheckLatest();
        WindowState = WindowState.Normal;
        Activate();
    }

    void RestoreNowClick(object sender, RoutedEventArgs e)
    {
        Native.RestoreRefresh();
        RestoreNow.Visibility = Visibility.Collapsed;
        PlayingHint.Text = L($"Écran remis à {Native.CurrentMode().hz} Hz. Le jeu tourne toujours.", $"Display back at {Native.CurrentMode().hz} Hz. The game is still running.");
    }
}
