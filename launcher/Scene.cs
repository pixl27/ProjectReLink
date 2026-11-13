using System.Globalization;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace ReLink;

/// The sky and the hills behind the window. They follow the player's clock, as the game's own day does: night, dawn, day and
/// dusk, with the sun or the moon as high as the hour puts it. The top of the sky stays dark at every hour: the title is read
/// there. (RELINK_HOUR=13.5 sets the hour, for tests)
static class Scene
{
    readonly record struct Key(double Hour, Color Top, Color Low);

    static Color C(string hex) => (Color)ColorConverter.ConvertFromString("#" + hex);

    static readonly Key[] Keys =
    {
        new(0, C("0A1424"), C("16303F")),
        new(4.75, C("0A1424"), C("16303F")),
        new(6.25, C("26375A"), C("D49A6A")),   // dawn
        new(8.5, C("235A8E"), C("9CCBD9")),    // day
        new(16.5, C("235A8E"), C("9CCBD9")),
        new(18.5, C("2A2F57"), C("DA8A52")),   // dusk
        new(20.25, C("0A1424"), C("16303F")),
        new(24, C("0A1424"), C("16303F")),
    };

    public static double Hour()
    {
        if (Environment.GetEnvironmentVariable("RELINK_HOUR") is { Length: > 0 } forced &&
            double.TryParse(forced, NumberStyles.Float, CultureInfo.InvariantCulture, out var h))
            return Math.Clamp(h, 0, 23.99);
        var now = DateTime.Now;
        return now.Hour + now.Minute / 60.0;
    }

    static Color Mix(Color a, Color b, double t) => Color.FromRgb(
        (byte)(a.R + (b.R - a.R) * t), (byte)(a.G + (b.G - a.G) * t), (byte)(a.B + (b.B - a.B) * t));

    static (Color Top, Color Low) Sky(double hour)
    {
        for (var i = 1; i < Keys.Length; i++)
            if (hour <= Keys[i].Hour)
            {
                var (a, b) = (Keys[i - 1], Keys[i]);
                var t = b.Hour > a.Hour ? (hour - a.Hour) / (b.Hour - a.Hour) : 0;
                return (Mix(a.Top, b.Top, t), Mix(a.Low, b.Low, t));
            }
        return (Keys[0].Top, Keys[0].Low);
    }

    public static void Draw(Canvas canvas, double w, double h, double hour)
    {
        canvas.Children.Clear();
        var (top, low) = Sky(hour);
        var sky = new LinearGradientBrush { StartPoint = new(0, 0), EndPoint = new(0, 1) };
        sky.GradientStops.Add(new GradientStop(top, 0));
        sky.GradientStops.Add(new GradientStop(top, 0.17));
        sky.GradientStops.Add(new GradientStop(low, 0.36));
        sky.GradientStops.Add(new GradientStop(low, 1));
        canvas.Children.Add(new Rectangle { Width = w, Height = h, Fill = sky });

        // how much of the night is in this hour: the stars' light
        var sunUp = hour > 5.5 && hour < 19.5;
        var night = hour < 5 || hour > 20.25 ? 1 : hour < 6.5 ? (6.5 - hour) / 1.5 : hour > 18.75 ? (hour - 18.75) / 1.5 : 0;
        if (night > 0.02)
        {
            var rnd = new Random(1998);
            for (var i = 0; i < 54; i++)
            {
                var size = 1 + rnd.NextDouble() * 1.4;
                var star = new Ellipse { Width = size, Height = size, Fill = Brushes.White, Opacity = night * (0.3 + rnd.NextDouble() * 0.6) };
                Canvas.SetLeft(star, rnd.NextDouble() * w);
                Canvas.SetTop(star, 8 + rnd.NextDouble() * 200);
                canvas.Children.Add(star);
            }
        }

        // the sun by day, the moon by night, over the right of the left column: as high as the hour puts it
        const double x = 420;
        if (sunUp)
        {
            var up = Math.Sin(Math.PI * (hour - 5.5) / 14);
            var y = 206 - up * 136;
            var colour = Mix(C("F2B56B"), C("FBF1CF"), up);
            Disc(canvas, x, y, 46, colour, 0.10);
            Disc(canvas, x, y, 21, colour, 1);
        }
        else
        {
            var since = hour >= 19.5 ? hour - 19.5 : hour + 4.5;
            var up = Math.Sin(Math.PI * since / 10);
            var y = 206 - up * 136;
            Disc(canvas, x, y, 17, C("E9E6D8"), 1);
            Disc(canvas, x + 7.5, y - 4, 15, top, 1);   // (the sky's own colour over it: a crescent)
        }

        // three ridges, the far one taking the horizon's light
        Hill(canvas, "M0,226 C120,194 230,200 330,216 C430,232 520,208 640,200 C780,190 900,226 1120,204 L1120,700 L0,700 Z", Mix(low, C("1E3B30"), 0.58));
        Hill(canvas, "M0,262 C90,246 190,234 300,250 C400,264 470,274 600,262 C760,246 900,272 1120,246 L1120,700 L0,700 Z", Mix(low, C("183226"), 0.82));
        Hill(canvas, "M0,292 C140,274 260,284 380,298 C520,314 700,288 860,294 C980,298 1060,288 1120,284 L1120,700 L0,700 Z", Mix(top, C("11241B"), 0.92));
    }

    static void Disc(Canvas canvas, double x, double y, double r, Color colour, double opacity)
    {
        var e = new Ellipse { Width = 2 * r, Height = 2 * r, Fill = new SolidColorBrush(colour), Opacity = opacity };
        Canvas.SetLeft(e, x - r);
        Canvas.SetTop(e, y - r);
        canvas.Children.Add(e);
    }

    static void Hill(Canvas canvas, string data, Color colour) =>
        canvas.Children.Add(new Path { Data = Geometry.Parse(data), Fill = new SolidColorBrush(colour) });
}
