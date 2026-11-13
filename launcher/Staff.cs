using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;

namespace ReLink;

/// One note on the staff: something that is ready or not (the game, the controller, the display), or one step of an install.
/// Hollow while it waits, breathing while it is the step at work, gold when done.
sealed class Note
{
    readonly Ellipse head = new() { Width = 15, Height = 11, StrokeThickness = 1.6, RenderTransformOrigin = new Point(0.5, 0.5), RenderTransform = new RotateTransform(-20) };
    readonly Line stem = new() { StrokeThickness = 1.6 };
    readonly TextBlock label = new() { TextAlignment = TextAlignment.Center, TextTrimming = TextTrimming.CharacterEllipsis };
    readonly TextBlock sub = new() { TextAlignment = TextAlignment.Center, TextWrapping = TextWrapping.Wrap, FontSize = 12.5, LineHeight = 16 };
    public readonly UIElement[] Parts;

    static Brush B(string key) => (Brush)Application.Current.FindResource(key);

    public Note(double x, double y, double textTop, double width, string text, double size)
    {
        Canvas.SetLeft(head, x - 7.5);
        Canvas.SetTop(head, y - 5.5);
        stem.X1 = stem.X2 = x + 6.6;
        stem.Y1 = y - 2;
        stem.Y2 = y - 36;
        label.Text = text;
        label.FontSize = size;
        label.Width = sub.Width = width;
        Canvas.SetLeft(label, x - width / 2);
        Canvas.SetTop(label, textTop);
        Canvas.SetLeft(sub, x - width / 2);
        Canvas.SetTop(sub, textTop + size + 7);
        sub.Foreground = B("BoneFaint");
        Parts = new UIElement[] { stem, head, label, sub };
        Waiting();
    }

    public string Sub
    {
        set => sub.Text = value;
    }

    void Still()
    {
        head.BeginAnimation(UIElement.OpacityProperty, null);
        head.Opacity = 1;
    }

    public void Waiting()
    {
        Still();
        head.Fill = Brushes.Transparent;
        head.Stroke = stem.Stroke = B("BoneFaint");
        label.Foreground = B("BoneFaint");
    }

    public void Active()
    {
        head.Fill = head.Stroke = stem.Stroke = B("Gold");
        label.Foreground = B("Bone");
        head.BeginAnimation(UIElement.OpacityProperty, new DoubleAnimation(1, 0.35, TimeSpan.FromSeconds(0.9))
            { AutoReverse = true, RepeatBehavior = RepeatBehavior.Forever, EasingFunction = new SineEase { EasingMode = EasingMode.EaseInOut } });
    }

    public void Done()
    {
        Still();
        head.Fill = head.Stroke = stem.Stroke = B("Gold");
        label.Foreground = B("Bone");
    }

    public void Problem()
    {
        Still();
        head.Fill = head.Stroke = stem.Stroke = B("Warn");
        label.Foreground = B("Bone");
    }
}

/// The staff under the title: five lines, and the notes written on them. On the home page the notes are what the game needs
/// (the game itself, a controller, the display); during an install they are its steps, filled in one after the other.
sealed class Staff
{
    const double Width = 424, Top = 40, Gap = 10;
    readonly Canvas canvas;
    readonly Dictionary<string, Note> notes = new();

    public Staff(Canvas canvas) => this.canvas = canvas;

    public Note? this[string id] => notes.GetValueOrDefault(id);
    public IEnumerable<Note> Notes => notes.Values;

    public void Set(params (string Id, string Label)[] items)
    {
        canvas.Children.Clear();
        notes.Clear();
        var line = (Brush)Application.Current.FindResource("BoneFaint");
        for (var i = 0; i < 5; i++)
            canvas.Children.Add(new Line { X1 = 0, X2 = Width, Y1 = Top + i * Gap, Y2 = Top + i * Gap, Stroke = line, StrokeThickness = 1, Opacity = 0.55, SnapsToDevicePixels = true });
        // the bar lines: one where the staff begins, the double one that ends a piece
        foreach (var (x, thick) in new[] { (0.5, 1.0), (Width - 6.5, 1.0), (Width - 1.5, 3.0) })
            canvas.Children.Add(new Line { X1 = x, X2 = x, Y1 = Top, Y2 = Top + 4 * Gap, Stroke = line, StrokeThickness = thick, Opacity = 0.55, SnapsToDevicePixels = true });

        var n = items.Length;
        var few = n <= 3;
        var first = few ? 66.0 : 50.0;
        var step = n > 1 ? (Width - 2 * first) / (n - 1) : 0;
        var textWidth = few ? 132 : step - 6;
        for (var i = 0; i < n; i++)
        {
            // (rising, each note on the next line or space up: the last step lands at the top)
            var y = Top + 4 * Gap - (few ? 10 + i * 10 : (n == 4 ? 5 : 0) + i * 10);
            var note = new Note(first + i * step, y, Top + 4 * Gap + 20, textWidth, items[i].Label, few ? 15 : n == 4 ? 13.5 : 12.5);
            notes[items[i].Id] = note;
            foreach (var p in note.Parts) canvas.Children.Add(p);
        }
    }

    /// The notes come in one after the other, once, when the window opens
    public void Play()
    {
        var i = 0;
        foreach (var note in notes.Values)
        {
            foreach (var p in note.Parts.Where(p => p is not Ellipse))
            {
                p.Opacity = 0;
                p.BeginAnimation(UIElement.OpacityProperty, new DoubleAnimation(0, 1, TimeSpan.FromMilliseconds(320)) { BeginTime = TimeSpan.FromMilliseconds(180 + i * 150) });
            }
            i++;
        }
    }
}

/// One step of an install, in the tablet: its mark (waiting, at work, done, stopped), its name, a line under it and its time.
sealed class StepRow
{
    public readonly Grid Root = new() { Margin = new Thickness(0, 9, 0, 9) };
    readonly Rectangle mark = new() { Width = 9, Height = 9, StrokeThickness = 1.4, RenderTransformOrigin = new Point(0.5, 0.5), RenderTransform = new RotateTransform(45),
                                      VerticalAlignment = VerticalAlignment.Top, HorizontalAlignment = HorizontalAlignment.Left, Margin = new Thickness(3, 7, 0, 0) };
    readonly TextBlock name = new() { FontSize = 15.5 };
    readonly TextBlock small = new() { FontSize = 12.5, Visibility = Visibility.Collapsed, TextTrimming = TextTrimming.CharacterEllipsis, Margin = new Thickness(0, 2, 0, 0) };
    readonly TextBlock time = new() { FontSize = 13, Text = "", VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0) };
    readonly Stopwatch watch = new();
    bool running;

    static Brush B(string key) => (Brush)Application.Current.FindResource(key);

    public StepRow(string text)
    {
        Root.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(30) });
        Root.ColumnDefinitions.Add(new ColumnDefinition());
        Root.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var texts = new StackPanel { Children = { name, small } };
        Grid.SetColumn(texts, 1);
        Grid.SetColumn(time, 2);
        Root.Children.Add(mark);
        Root.Children.Add(texts);
        Root.Children.Add(time);
        name.Text = text;
        small.Foreground = B("BoneFaint");
        Wait();
    }

    public void Small(string text)
    {
        small.Text = text;
        small.Visibility = Visibility.Visible;
    }

    void Wait()
    {
        mark.Fill = Brushes.Transparent;
        mark.Stroke = B("BoneFaint");
        name.Foreground = B("BoneFaint");
        time.Foreground = B("BoneFaint");
    }

    public void Begin()
    {
        running = true;
        watch.Restart();
        mark.Fill = Brushes.Transparent;
        mark.Stroke = B("Gold");
        name.Foreground = B("Bone");
        time.Foreground = B("Bone");
        Tick();
    }

    public void Tick()
    {
        if (running) time.Text = $"{(int)watch.Elapsed.TotalMinutes}:{watch.Elapsed.Seconds:00}";
    }

    public void Done()
    {
        if (running) Tick();
        running = false;
        watch.Stop();
        mark.Fill = mark.Stroke = B("Gold");
        name.Foreground = B("Bone");
        time.Foreground = B("BoneFaint");
    }

    public void Stopped()
    {
        running = false;
        watch.Stop();
        mark.Fill = mark.Stroke = B("Warn");
        name.Foreground = B("Bone");
    }
}
