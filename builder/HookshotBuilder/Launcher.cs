using System.Diagnostics;
using System.IO;

namespace HookshotBuilder;

/// Opt-in launcher routes; starting the program without arguments still opens the Builder.
static class Launcher
{
    public static string? BuilderHome()
    {
        var data = Environment.GetEnvironmentVariable("RECOMPLAUNCHER_DATA");
        return string.IsNullOrEmpty(data) ? null : Path.Combine(data, "cache", "builder");
    }

    public static int Play(string[] args)
    {
        try
        {
            if (args.Length != 1) throw new ArgumentException("Use --play followed by the built game folder.");
            var folder = Path.GetFullPath(args[0]);
            var executable = Path.Combine(folder, "bin", "hookshot.exe");
            if (!File.Exists(executable)) throw new IOException("No built game was found. Run Set Up again.");
            var info = new ProcessStartInfo(executable) { WorkingDirectory = folder, UseShellExecute = false };
            info.Environment["HOOKSHOT_NO_UPDATE_CHECK"] = "1";
            // Windows already uses this save folder by default; pass it explicitly for launcher sessions.
            if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("RECOMPLAUNCHER_DATA")))
            {
                info.ArgumentList.Add("--user");
                info.ArgumentList.Add(Path.Combine(folder, "user"));
            }
            using var game = Process.Start(info) ?? throw new IOException("The game process could not be started.");
            game.WaitForExit();
            return game.ExitCode;
        }
        catch (Exception error) when (error is IOException or ArgumentException or System.ComponentModel.Win32Exception)
        {
            Console.Error.WriteLine("error " + error.Message);
            return 1;
        }
    }
}
