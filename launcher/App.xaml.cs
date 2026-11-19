namespace ReLink;

public partial class App : System.Windows.Application
{
    /// The chosen theme's colours (Theme.cs) before StartupUri makes the window: what it looks up is already theirs
    protected override void OnStartup(System.Windows.StartupEventArgs e)
    {
        Theme.Apply(Resources);
        base.OnStartup(e);
    }
}
