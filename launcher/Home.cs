using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
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

    CheckBox Fullscreen = null!, Display60 = null!, FpsCounter = null!, Ultrawide = null!, FramePacing = null!, VSync = null!, Audio = null!, Handheld = null!, InvertY = null!,
             NvidiaDlss = null!;
    WrapPanel ResolutionGroup = null!, UpscalerGroup = null!, UpscaleGroup = null!, FrameRateGroup = null!, SpeedGroup = null!, FramegenGroup = null!, UiLanguageGroup = null!, ThemeGroup = null!;
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
        UpscalerGroup = ChoiceRow(RowsDisplay, L("Agrandissement de la 3D", "Upscaler"),
                  L("Ce qui agrandit la 3D jusqu'à l'écran et lisse ses bords. AMD FSR marche sur toutes les cartes graphiques ; NVIDIA DLSS demande une carte RTX et la case NVIDIA DLSS ci-dessous. Avec FSR ou DLSS, mettez la netteté vers 80 %.",
                    "What scales the 3D up to the screen and smooths its edges. AMD FSR works on every graphics card; NVIDIA DLSS needs an RTX card and the NVIDIA DLSS switch below. With FSR or DLSS, set Sharpness to about 80 %."),
                  "Upscaler", (0, L("Celui du jeu", "The game's own")), (3, "AMD FSR"), (4, L("AMD FSR, AA natif", "AMD FSR native AA")), (1, "NVIDIA DLSS"), (2, "NVIDIA DLAA"));
        UpscaleGroup = ChoiceRow(RowsDisplay, L("Filtre d'agrandissement de la fenêtre", "Window scaling filter"), L("Quand l'image est plus petite que la fenêtre.", "When the picture is smaller than the window."), "Upscale", (0, L("Simple", "Simple")), (1, "Catmull-Rom"), (2, "Lanczos"));
        SwitchRow(RowsDisplay, L("NVIDIA DLSS (cartes RTX)", "NVIDIA DLSS (RTX cards)"),
                  L("Le jeu est reconstruit avec le SDK DLSS, téléchargé chez NVIDIA. L'activer, c'est accepter la licence du SDK NVIDIA DLSS (github.com/NVIDIA/DLSS). Ensuite, choisissez DLSS ou DLAA dans le menu du jeu (touche Insert, Upscaler).",
                    "The game is built again with the DLSS SDK, downloaded from NVIDIA. Turning this on accepts NVIDIA's DLSS SDK license (github.com/NVIDIA/DLSS). Then choose DLSS or DLAA in the game's menu (Insert key, Upscaler)."),
                  out NvidiaDlss, out _);

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
                  L("Débloqué : jusqu'à la fréquence de l'écran, et le jeu suit le temps réel. Sur un écran à fréquence fixe, 60 est plus régulier. " +
                    "30, fluide : chaque image reste affichée le même temps (synchronisée à l'écran), pour les PC qui n'atteignent pas 60.",
                    "Unlocked: up to the display's rate, and the game follows real time. On a fixed-rate display, 60 is steadier. " +
                    "30, smooth: every frame stays on screen for the same time (synced to the display), for PCs that do not reach 60."),
                  "FrameRate", (0, L("60, comme la console", "60, as on the console")), (1, L("Débloqué", "Unlocked")),
                  (2, L("30, fluide", "30, smooth")));
        SpeedGroup = ChoiceRow(RowsSmooth, L("Vitesse du jeu", "Game speed"),
                  L("Temps réel : si une image tarde, le jeu ne ralentit pas.", "Real time: when a frame is late, the game does not slow down."),
                  "Speed", (0, L("Par image", "By frame")), (1, L("Temps réel", "Real time")));
        FramegenGroup = ChoiceRow(RowsSmooth, L("Génération d'images", "Frame generation"),
                  L("Des images intercalées : plus fluide à l'œil, pas plus réactif. L'écran garde alors sa fréquence.",
                    "Frames made in between: smoother to the eye, not more responsive. The display keeps its own rate then."),
                  "Framegen", (0, L("Aucune", "None")), (2, "×2"), (3, "×3"), (4, "×4"));
        SwitchRow(RowsSmooth, L("Rythme régulier des images générées", "Even pacing of generated frames"),
                  L("Avec la génération d'images seulement.", "With frame generation only."), out FramePacing, out _);
        SwitchRow(RowsSmooth, "V-Sync",
                  L("Chaque image attend le rafraîchissement de l'écran : plus de déchirure de l'image, un peu plus de délai. Utile là où le pilote n'a pas de mode mailbox (AMD sous Windows). Le menu en jeu le change aussi, aussitôt.",
                    "Each frame waits for the display's refresh: no more tearing, a little more delay. Useful where the driver has no mailbox mode (AMD on Windows). The in-game menu changes it too, at once."),
                  out VSync, out _);

        // ---- game
        LanguageBox = new ComboBox { Style = St("List"), Width = 230, ItemsSource = GameLanguages.Select(l => new LangItem(l.Code, l.Name)).ToList() };
        var lang = new Grid { ColumnDefinitions = { new ColumnDefinition(), new ColumnDefinition { Width = GridLength.Auto } } };
        lang.Children.Add(new TextBlock { Text = L("Langue du jeu", "Game language"), Style = St("RowLabel") });
        Grid.SetColumn(LanguageBox, 1);
        lang.Children.Add(LanguageBox);
        RowsGame.Children.Add(new Border { Style = St("Row"), Child = lang });
        UiLanguageGroup = ChoiceRow(RowsGame, L("Langue du lanceur", "Launcher language"),
                  L("Le changement se voit à la prochaine ouverture de ReLink.", "The change shows the next time ReLink opens."),
                  "UiLanguage", (0, "English"), (1, "Français"));
        ThemeGroup = ChoiceRow(RowsGame, L("Apparence du lanceur", "Launcher look"),
                  L("Le menu en jeu (Insert ou F1) la suit aussi. Le lanceur change à sa prochaine ouverture.", "The in-game menu (Insert or F1) follows it too. The launcher changes the next time it opens."),
                  "Theme", (0, L("Clair", "Light")), (1, L("Sombre", "Dark")));
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
        // (the prepared shaders deleted, prepared again at the next start: objects drawn white or wrong after a start that
        // crashed while it prepared them)
        var reprepare = new Button { Style = St("Link"), Content = L("Préparer de nouveau…", "Prepare again…"), HorizontalAlignment = HorizontalAlignment.Left };
        reprepare.Click += PrepareShadersAgain;
        RowsGame.Children.Add(new Border
        {
            Style = St("Row"),
            Child = new StackPanel
            {
                Children =
                {
                    new TextBlock { Text = L("Shaders", "Shaders"), Style = St("RowLabel") },
                    new TextBlock
                    {
                        Text = L("Préparés une fois pour votre carte graphique. Si des objets s'affichent mal (tout blancs, par exemple), préparez-les de nouveau : le prochain lancement prend 10 à 15 minutes de plus. Vos sauvegardes ne sont pas touchées.",
                                 "Prepared once for your graphics card. If objects look wrong (all white, for example), prepare them again: the next start takes 10 to 15 minutes longer. Your saves are not touched."),
                        Style = St("RowHint"),
                    },
                    new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 8, 0, 0), Children = { reprepare } },
                },
            },
        });

        BuildShortcutRows();

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
        foreach (var page in new Panel[] { RowsDisplay, RowsSmooth, RowsGame, RowsControls }) WireRows(page);
    }

    // ---------------------------------------------------------------- the rows as the in-game menu has them
    // (one row selected a page: tinted in the page's colour, with what it does under it; the others one line each. A click or
    // the keyboard's focus selects one; the arrows move and change, Q and E change pages: OptionsKey)
    readonly Dictionary<Panel, Border> selectedRows = new();

    /// The controls in a row, laid out or not (its logical children)
    static IEnumerable<T> Within<T>(DependencyObject root) where T : DependencyObject
    {
        foreach (var c in LogicalTreeHelper.GetChildren(root).OfType<DependencyObject>())
        {
            if (c is T t) yield return t;
            foreach (var deeper in Within<T>(c)) yield return deeper;
        }
    }
    static IEnumerable<TextBlock> Styled(DependencyObject root, string style) => Within<TextBlock>(root).Where(t => t.Style == St(style));

    void WireRows(Panel page)
    {
        var rows = page.Children.OfType<Border>().ToList();
        foreach (var b in rows)
        {
            b.PreviewMouseLeftButtonDown += (_, _) => SelectRow(b);
            b.IsKeyboardFocusWithinChanged += (_, e) => { if ((bool)e.NewValue) SelectRow(b); };
            Look(b, false);
        }
        if (rows.Count > 0) SelectRow(rows[0]);
    }

    /// A row selected or not: the same height either way (the edge all round with its padding, or the rule under it with a
    /// little more padding), so nothing moves when the selection does. Its hints stay hidden: OptionsHelp says them
    void Look(Border b, bool on)
    {
        b.Background = on ? (Brush)b.FindResource("AccentTint") : Brushes.Transparent;
        b.BorderBrush = on ? (Brush)b.FindResource("AccentEdge") : Res("Rule");
        b.BorderThickness = on ? new Thickness(1.5) : new Thickness(0, 0, 0, 1);
        b.Padding = on ? new Thickness(12, 8, 12, 8) : new Thickness(13.5, 9.5, 13.5, 9);
        b.CornerRadius = new CornerRadius(on ? 10 : 0);
        b.Margin = new Thickness(0);
        foreach (var h in Styled(b, "RowHint")) h.Visibility = Visibility.Collapsed;
        foreach (var l in Styled(b, "RowLabel")) l.FontWeight = on ? FontWeights.Bold : FontWeights.Normal;
    }

    void SelectRow(Border b)
    {
        if (b.Parent is not Panel page) return;
        if (selectedRows.TryGetValue(page, out var was) && was == b) return;
        if (was != null) Look(was, false);
        selectedRows[page] = b;
        Look(b, true);
        b.BringIntoView();
        ShowRowHelp();
    }

    /// What the open page's selected row does, in the box under the rows, a bar of the page's colour at its left
    void ShowRowHelp()
    {
        if (OptionsHelpText == null || !selectedRows.TryGetValue(OpenPage(), out var row)) return;
        OptionsHelpText.Text = string.Join("\n", Styled(row, "RowHint").Select(t => t.Text).Where(t => t.Length > 0));
        OptionsHelp.BorderBrush = (Brush)row.FindResource("Accent");
        OptionsHelp.Visibility = OptionsHelpText.Text.Length > 0 ? Visibility.Visible : Visibility.Hidden;
    }

    Panel OpenPage() => PageDisplay.Visibility == Visibility.Visible ? RowsDisplay : PageSmooth.Visibility == Visibility.Visible ? RowsSmooth
                      : PageGame.Visibility == Visibility.Visible ? RowsGame : RowsControls;

    /// The keyboard's focus on the open page's selected row, its first control (the arrows then work at once)
    void FocusSelectedRow()
    {
        if (!selectedRows.TryGetValue(OpenPage(), out var row)) return;
        Within<Control>(row).FirstOrDefault(c => c.Focusable && c.IsEnabled && c.IsVisible)?.Focus();
    }

    void OptionsKey(object sender, KeyEventArgs e)
    {
        if (e.OriginalSource is ComboBox { IsDropDownOpen: true } || Keyboard.FocusedElement is ComboBoxItem) return;
        var tabs = new[] { TabDisplay, TabSmooth, TabGame, TabControls };
        if (e.Key is Key.Q or Key.E)
        {
            var at = Array.FindIndex(tabs, t => t.IsChecked == true);
            tabs[(at + (e.Key == Key.E ? 1 : tabs.Length - 1)) % tabs.Length].IsChecked = true;
            e.Handled = true;
            return;
        }
        var page = OpenPage();
        if (!selectedRows.TryGetValue(page, out var row)) return;
        if (e.Key is Key.Up or Key.Down)
        {
            var rows = page.Children.OfType<Border>().Where(b => b.IsVisible).ToList();
            var i = rows.IndexOf(row);
            if (i < 0) return;
            SelectRow(rows[(i + (e.Key == Key.Down ? 1 : rows.Count - 1)) % rows.Count]);
            FocusSelectedRow();
            e.Handled = true;
        }
        else if (e.Key is Key.Left or Key.Right)
        {
            var dir = e.Key == Key.Right ? 1 : -1;
            if (Within<CheckBox>(row).FirstOrDefault(c => c.IsEnabled) is { } box) box.IsChecked = dir > 0;
            else if (Within<Slider>(row).FirstOrDefault() is { } slider) slider.Value = Math.Clamp(slider.Value + dir * slider.SmallChange, slider.Minimum, slider.Maximum);
            else if (Within<ComboBox>(row).FirstOrDefault() is { } list) list.SelectedIndex = Math.Clamp(list.SelectedIndex + dir, 0, list.Items.Count - 1);
            else
            {
                var choices = Within<RadioButton>(row).Where(r => r.IsEnabled && r.Visibility == Visibility.Visible).ToList();
                var at = choices.FindIndex(r => r.IsChecked == true);
                if (choices.Count > 0) choices[Math.Clamp(at + dir, 0, choices.Count - 1)].IsChecked = true;
            }
            e.Handled = true;
        }
    }

    // ---------------------------------------------------------------- Home's cards: what Play will show, in a line each
    void UpdateHero()
    {
        var res = s.Resolution switch { 1 => L("Native", "Native"), 0 => L("Automatique", "Automatic"), 2 => "75 %", 3 => "67 %", 5 => "58 %", 4 => "50 %", _ => "" };
        var up = s.Upscaler switch { 3 => "AMD FSR", 4 => L("AMD FSR, AA natif", "AMD FSR native AA"), 1 => "NVIDIA DLSS", 2 => "NVIDIA DLAA", _ => L("Celui du jeu", "The game's own") };
        LooksText.Text = $"{(s.Fullscreen != 0 ? L("Plein écran", "Full screen") : L("Fenêtre", "Window"))} · {res} · {up}";
        var rate = s.FrameRate switch { 1 => L("FPS débloqués", "Unlocked fps"), 2 => L("30 fps réguliers", "Even 30 fps"), _ => "60 fps" };
        MovesText.Text = s.FrameGeneration >= 2 ? $"{rate} · ×{s.FrameGeneration}" : rate;
        LooksSub.Text = L($"Netteté {s.Sharpness} %", $"Sharpness {s.Sharpness} %") + (s.Ultrawide != 0 ? L(" · ultra-large", " · ultrawide") : "");
        MovesSub.Text = (s.FrameGeneration >= 2 ? L("Génération d'images", "Frame generation") : L("Sans génération d'images", "No frame generation")) +
                        (s.VSync != 0 ? " · V-Sync" : "");
        // (the sky's line: once, after an install or an update, what the first start does; the rest of the time the name's)
        HeroSub.Text = justBuilt ? L("Prêt à jouer ! Le premier lancement prépare les shaders pour votre carte graphique : une fois, environ 15 minutes.",
                                     "Ready to play! The first start prepares the shaders for your graphics card: once, about 15 minutes.")
                                 : L("recompilé pour votre PC", "recompiled for your PC");
        UpdateHomeCards();
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
        OnSwitch(VSync, v => s.VSync = v ? 1 : 0);
        OnSwitch(Audio, v => s.Audio = v);
        OnSwitch(Handheld, v => s.Handheld = v ? 1 : 0);
        OnSwitch(InvertY, v => s.InvertY = v);
        OnSwitch(NvidiaDlss, v =>
        {
            s.NvidiaDlss = v;
            // (turned on with a game installed: built again now, NVIDIA's SDK fetched first; off: the next build is without it)
            var tools = Tools.Find();
            if (v && tools != null && game == null && Game.Installed(s.Folder))
                Dispatcher.BeginInvoke(() => Rebuild(tools, s.Folder));
        });
        OnPick(ResolutionGroup, v => s.Resolution = v);
        OnPick(UpscalerGroup, v => s.Upscaler = v);
        OnPick(UpscaleGroup, v => s.Upscaling = v);
        OnPick(FrameRateGroup, v => s.FrameRate = v);
        OnPick(SpeedGroup, v => s.GameSpeed = v);
        OnPick(FramegenGroup, v => s.FrameGeneration = v);
        OnPick(UiLanguageGroup, v => s.UiLanguage = v == 1 ? "fr" : "en");
        OnPick(ThemeGroup, v => { s.LauncherTheme = v == 1 ? Theme.Dark : Theme.Light; s.MenuLook = v; });   // (the in-game menu too)
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
        VSync.IsChecked = s.VSync != 0;
        Audio.IsChecked = s.Audio;
        Handheld.IsChecked = s.Handheld != 0;
        InvertY.IsChecked = s.InvertY;
        NvidiaDlss.IsChecked = s.NvidiaDlss;
        Pick(ResolutionGroup, s.Resolution);
        Pick(UpscalerGroup, s.Upscaler);
        Pick(UpscaleGroup, s.Upscaling);
        Pick(FrameRateGroup, s.FrameRate);
        Pick(FramegenGroup, s.FrameGeneration);
        Pick(UiLanguageGroup, s.UiLanguage == "fr" ? 1 : 0);
        Pick(ThemeGroup, s.LauncherTheme == Theme.Dark ? 1 : 0);
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
        // (NVIDIA DLSS and DLAA to choose only in a game built with them: the NVIDIA DLSS switch)
        foreach (var r in Radios(UpscalerGroup))
            if ((int)r.Tag is 1 or 2) r.Visibility = s.NvidiaDlss || s.Upscaler is 1 or 2 ? Visibility.Visible : Visibility.Collapsed;
        // (frame generation fills the display's own refresh: no switch to 60 Hz then)
        Display60.IsEnabled = display60Possible && s.FrameGeneration < 2;
        UpdateHero();
        ShowRowHelp();
        ShowChecks();
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
        MainNote.Text = "";
        var version = s.Folder == "" ? null : Game.Version(s.Folder);
        if (version != null) Card(version);   // (its state, up to date or not, once CheckLatest knows it)
        else ShowVersion(null, null, false, true);
        UpdateSide();
        ShowMods();
    }

    /// Exefs mods (ExefsMods.cs): the code patches in the folder's mods take effect when the game is built. When they are not
    /// the ones it was built with, or the last build left some without effect, a badge on Mods says so; the Mods page says what,
    /// with Build again. preview: the snapshot's (2 patch files)
    void ShowMods(int preview = 0)
    {
        var folder = s.Folder;
        var changed = preview > 0 || (folder != "" && Game.Installed(folder) && ExefsMods.Changed(folder, out _));
        var problems = preview == 0 && folder != "" && PatchProblems(folder).Count > 0;
        ModsBadge.Visibility = changed || problems ? Visibility.Visible : Visibility.Collapsed;
        modsToBuild = changed;
        UpdateHomeCards();
    }

    bool modsToBuild;

    void RebuildClick(object sender, RoutedEventArgs e)
    {
        if (game != null || runner != null || !Game.Installed(s.Folder) || !Game.KeepsGameFiles(s.Folder)) return;
        var tools = Tools.Find();
        if (tools == null)
        {
            MainNote.Text = L("Les outils de construction de ReLink sont introuvables : installez de nouveau ReLink.",
                              "ReLink's build tools were not found: install ReLink again.");
            return;
        }
        Rebuild(tools, s.Folder, L("Reconstruction du jeu avec vos mods", "Building your game again with your mods"));
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
        MainNote.Text = "";
        ShowVersion(version, version == null ? latestTag : updateTarget, version != null && Game.KeepsGameFiles(folder), @checked: latestTag != null || version == null);
    }

    void UpdateClick(object sender, RoutedEventArgs e)
    {
        if (game != null || s.Folder == "") return;
        Updater.UpdateFolder(this, s.Folder, updateTarget, fetched: false);
    }

    // ---------------------------------------------------------------- play
    void Play()
    {
        var folder = s.Folder;
        if (game != null || runner != null || !Game.Installed(folder)) return;
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
        psi.Environment["RELINK_NO_UPDATE_CHECK"] = "1";   // (the launcher already looked: the game does not ask again)
        if (!s.Audio) psi.Environment["HOS_AUDIO"] = "0";
        if (s.InvertY) psi.Environment["HOS_MOUSE_INVERT_Y"] = "1";
        if (s.FrameRate == 1) psi.Environment["HOS_FPS_UNLOCKED"] = "1";   // (the game reads the unlocked rate from here only: host/nn_vi.c)
        if (s.FrameRate == 2) psi.Environment["HOS_FPS_CAP"] = "30";       // (30, smooth: host/nn_vi.c, gpu/vk/vk_main.cpp low_cap_repeats)
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
        justBuilt = false;
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
