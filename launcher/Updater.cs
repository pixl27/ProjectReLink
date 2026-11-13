using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text.Json;
using System.Windows;
using static ReLink.Texts;

namespace ReLink;

/// Updates. A release holds the small setup (ReLink-Setup.exe) and the list of parts it reads (relink-setup.txt). The setup
/// downloads only the parts that changed since the version installed, checks them, and starts the new launcher with the
/// arguments it was given; the launcher then builds the game folder again from the game files the folder keeps
/// (nx2/make_game.py --update), compiling only what changed.
static class Updater
{
    const string Latest = "https://api.github.com/repos/" + Repo + "/releases/latest";
    const string SetupAsset = "ReLink-Setup.exe";
    const string SetupList = "relink-setup.txt";
    static readonly TimeSpan[] RetryDelays = { TimeSpan.FromSeconds(3), TimeSpan.FromSeconds(6), TimeSpan.FromSeconds(12), TimeSpan.FromSeconds(20) };

    static HttpClient Client(TimeSpan timeout)
    {
        var http = new HttpClient { Timeout = timeout };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("ReLink");
        return http;
    }

    public static void FromCommandLine(MainWindow w)
    {
        var a = Environment.GetCommandLineArgs();
        string? folder = null, tag = null;
        var fetched = false;
        for (var i = 1; i < a.Length; i++)
        {
            if (a[i] == "--update" && i + 1 < a.Length) folder = a[++i];
            else if (a[i] == "--tag" && i + 1 < a.Length) tag = a[++i];
            else if (a[i] == "--fetched") fetched = true;
        }
        if (folder == null) return;
        UpdateFolder(w, folder, tag, fetched);
    }

    /// The latest release's tag, quietly (null: GitHub did not answer, or there is no release yet); a few tries, 20 s at most.
    /// (RELINK_LATEST_TAG=<tag> or none: no network, for tests)
    public static async Task<string?> LatestTag()
    {
        if (Environment.GetEnvironmentVariable("RELINK_LATEST_TAG") is { Length: > 0 } fake) return fake == "none" ? null : fake;
        using var http = Client(TimeSpan.FromSeconds(8));
        for (var attempt = 0; attempt < 3; attempt++)
        {
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, Latest);
                request.Headers.Accept.ParseAdd("application/vnd.github+json");
                using var response = await http.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
                    return doc.RootElement.GetProperty("tag_name").GetString();
                }
                if ((int)response.StatusCode < 500) break;   // (the API's limit, 60 an hour a PC, or no release: the list below instead)
            }
            catch (Exception) { }
            await Task.Delay(TimeSpan.FromSeconds(2 << attempt));
        }
        // (the release's setup list names its tag in its first line, and comes from GitHub's download service, not the API)
        try
        {
            var list = await http.GetStringAsync(RepoPage + "/releases/latest/download/" + SetupList);
            foreach (var line in list.Split('\n'))
                if (line.StartsWith("tag ")) return line[4..].Trim();
        }
        catch (Exception) { }
        return null;
    }

    /// The latest release's setup downloaded and started with args: it installs the new launcher (only the parts that changed) and
    /// starts it with the same args; this one closes.
    public static async Task RunLatestSetup(MainWindow w, IEnumerable<string> args)
    {
        var token = w.DownloadToken;
        using var http = Client(Timeout.InfiniteTimeSpan);
        w.Note(L("Recherche de la dernière version sur GitHub", "Asking GitHub for the latest release"));
        string? setupUrl = null, setupApiUrl = null;
        long setupSize = 0;
        using (var response = await GetWithRetries(w, http, Latest, token, "application/vnd.github+json", TimeSpan.FromSeconds(30)))
        using (var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(token)))
            foreach (var asset in doc.RootElement.GetProperty("assets").EnumerateArray())
                if (asset.GetProperty("name").GetString() == SetupAsset)
                {
                    setupUrl = asset.GetProperty("browser_download_url").GetString();
                    setupApiUrl = asset.GetProperty("url").GetString();
                    setupSize = asset.GetProperty("size").GetInt64();
                }
        if (setupUrl == null) throw new InvalidOperationException(L($"La dernière version n'a pas de {SetupAsset}.", $"The latest release has no {SetupAsset}."));
        var cache = Path.Combine(Storage.Base, "update");
        Directory.CreateDirectory(cache);
        var setup = Path.Combine(cache, SetupAsset);
        w.Note(L("Téléchargement du nouveau ReLink", "Downloading the new ReLink"));
        await DownloadWithRetries(w, http, setupUrl, setupApiUrl, setup, setupSize, token);
        var si = new ProcessStartInfo(setup) { UseShellExecute = false, WorkingDirectory = cache };
        foreach (var arg in args) si.ArgumentList.Add(arg);
        Process.Start(si);
        Application.Current.Shutdown();
    }

    /// The game in folder built again with the latest release: here when these tools are it (or fetched), else after the setup
    /// installed it (it starts the new launcher with --update folder --tag t --fetched)
    public static async void UpdateFolder(MainWindow w, string folder, string? tag, bool fetched)
    {
        w.BeginUpdate(folder, tag);
        var tools = Tools.Find();
        if (tools == null)
        {
            w.DownloadFailed(L("Les outils de ReLink sont introuvables à côté de cette application.", "ReLink's tools were not found next to this app."));
            return;
        }
        // (just downloaded, but another release than asked for: GitHub's latest link lags right after a publish, and building here
        // would keep players on the old version while the launcher said it updated)
        if (fetched && tag != null && tools.Version != null && tools.Version != tag)
        {
            w.DownloadFailed(L($"La mise à jour a installé {tools.Version}, pas {tag} : les liens de GitHub peuvent être en retard juste après une sortie. Réessayez dans quelques minutes.",
                               $"The update installed {tools.Version}, not {tag}: GitHub's download links can lag right after a release. Try Update again in a few minutes."));
            return;
        }
        // (already the new release, or just downloaded: no second download)
        if (fetched || (tag != null && tools.Version == tag))
        {
            w.UpdateBuild(tools, folder);
            return;
        }
        try
        {
            var token = w.DownloadToken;
            using var http = Client(Timeout.InfiniteTimeSpan);
            w.Note(L("Recherche de la dernière version sur GitHub", "Asking GitHub for the latest release"));
            string? newTag = null, setupUrl = null, setupApiUrl = null;
            long setupSize = 0;
            using (var response = await GetWithRetries(w, http, Latest, token, "application/vnd.github+json", TimeSpan.FromSeconds(30)))
            using (var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(token)))
            {
                newTag = doc.RootElement.GetProperty("tag_name").GetString();
                foreach (var asset in doc.RootElement.GetProperty("assets").EnumerateArray())
                    if (asset.GetProperty("name").GetString() == SetupAsset)
                    {
                        setupUrl = asset.GetProperty("browser_download_url").GetString();
                        setupApiUrl = asset.GetProperty("url").GetString();
                        setupSize = asset.GetProperty("size").GetInt64();
                    }
            }
            if (newTag == null || setupUrl == null)
                throw new InvalidOperationException(L($"La dernière version n'a pas de {SetupAsset}.", $"The latest release has no {SetupAsset}."));
            if (newTag == tools.Version)
            {
                w.UpdateBuild(tools, folder);
                return;
            }
            var cache = Path.Combine(Storage.Base, "update");
            Directory.CreateDirectory(cache);
            w.Note(L($"Téléchargement de la mise à jour {newTag}", $"Downloading the update for {newTag}"));
            var setup = Path.Combine(cache, SetupAsset);
            await DownloadWithRetries(w, http, setupUrl, setupApiUrl, setup, setupSize, token, w.DownloadProgress);
            var si = new ProcessStartInfo(setup) { UseShellExecute = false, WorkingDirectory = cache };
            foreach (var arg in new[] { "--update", folder, "--tag", newTag, "--fetched" }) si.ArgumentList.Add(arg);
            Process.Start(si);
            Application.Current.Shutdown();
        }
        catch (OperationCanceledException)
        {
            w.DownloadFailed(L("Le téléchargement a été annulé. Votre jeu n'a pas changé.", "The download was cancelled. Your game folder is unchanged."));
        }
        catch (Exception e)
        {
            w.DownloadFailed(e.Message + L(" Votre jeu n'a pas changé.", " Your game folder is unchanged."));
        }
    }

    static async Task<HttpResponseMessage> GetWithRetries(MainWindow w, HttpClient http, string url, CancellationToken token,
        string? accept = null, TimeSpan? timeout = null)
    {
        Exception? last = null;
        for (var attempt = 0; attempt <= RetryDelays.Length; attempt++)
        {
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, url);
                if (accept != null) request.Headers.Accept.ParseAdd(accept);
                using var ask = timeout is null ? null : new CancellationTokenSource(timeout.Value);
                using var both = ask is null ? null : CancellationTokenSource.CreateLinkedTokenSource(token, ask.Token);
                var response = await http.SendAsync(request, HttpCompletionOption.ResponseContentRead, both?.Token ?? token);
                if (response.IsSuccessStatusCode) return response;

                var failure = HttpFailure(response);
                response.Dispose();
                if (!IsTransient(failure)) throw failure;
                last = failure;
            }
            catch (OperationCanceledException e) when (!token.IsCancellationRequested && timeout is not null)
            {
                last = e;
            }
            catch (HttpRequestException e) when (IsTransient(e))
            {
                last = e;
            }

            if (attempt == RetryDelays.Length) break;
            var wait = (int)RetryDelays[attempt].TotalSeconds;
            w.Note(L($"GitHub n'a pas répondu{StatusDetail(last)}. Nouvel essai dans {wait} s…", $"GitHub did not answer{StatusDetail(last)}. Retrying in {wait} s…"));
            await Task.Delay(RetryDelays[attempt], token);
        }
        throw new HttpRequestException(L($"GitHub n'a pas répondu après {RetryDelays.Length + 1} essais : {last?.Message}",
                                         $"GitHub did not respond after {RetryDelays.Length + 1} attempts: {last?.Message}"),
            last, (last as HttpRequestException)?.StatusCode);
    }

    static async Task DownloadWithRetries(MainWindow w, HttpClient http, string url, string? apiUrl, string path, long size,
        CancellationToken token, Action<double>? progress = null)
    {
        Exception? last = null;
        for (var attempt = 0; attempt <= RetryDelays.Length; attempt++)
        {
            try
            {
                await DownloadOnce(http, url, path, size, token, progress, apiAsset: false);
                return;
            }
            catch (Exception e) when (IsTransient(e))
            {
                last = e;
            }

            if (apiUrl != null)
            {
                try
                {
                    await DownloadOnce(http, apiUrl, path, size, token, progress, apiAsset: true);
                    return;
                }
                catch (Exception e) when (IsTransient(e))
                {
                    last = e;
                }
            }

            if (attempt == RetryDelays.Length) break;
            var wait = (int)RetryDelays[attempt].TotalSeconds;
            w.Note(L($"Le téléchargement n'a pas abouti{StatusDetail(last)}. Nouvel essai dans {wait} s…",
                     $"GitHub's download servers did not answer{StatusDetail(last)}. Retrying in {wait} s…"));
            await Task.Delay(RetryDelays[attempt], token);
        }
        throw new HttpRequestException(L($"Le téléchargement a échoué après {RetryDelays.Length + 1} essais : {last?.Message}",
                                         $"The download failed after {RetryDelays.Length + 1} attempts: {last?.Message}"),
            last, (last as HttpRequestException)?.StatusCode);
    }

    static async Task DownloadOnce(HttpClient http, string url, string path, long size, CancellationToken token,
        Action<double>? progress, bool apiAsset)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (apiAsset) request.Headers.Accept.ParseAdd("application/octet-stream");
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, token);
        if (!response.IsSuccessStatusCode) throw HttpFailure(response);

        var total = response.Content.Headers.ContentLength ?? size;
        await using var from = await response.Content.ReadAsStreamAsync(token);
        await using var to = File.Create(path);
        var buffer = new byte[1 << 20];
        long have = 0;
        var shown = DateTime.MinValue;
        int n;
        while (true)
        {
            try { n = await from.ReadAsync(buffer, token); }
            catch (IOException e) { throw new HttpRequestException(L("La connexion s'est interrompue.", "The download connection ended unexpectedly."), e); }
            if (n == 0) break;
            await to.WriteAsync(buffer.AsMemory(0, n), token);
            have += n;
            if (progress != null && total > 0 && DateTime.UtcNow - shown > TimeSpan.FromMilliseconds(200))
            {
                shown = DateTime.UtcNow;
                progress(Math.Min(1.0, (double)have / total));
            }
        }
        if (total > 0 && have != total)
            throw new HttpRequestException(L($"Le téléchargement s'est arrêté trop tôt ({have >> 20} sur {total >> 20} Mo).", $"The download ended early ({have >> 20} of {total >> 20} MB)."));
    }

    static HttpRequestException HttpFailure(HttpResponseMessage response) =>
        new(L($"GitHub a répondu HTTP {(int)response.StatusCode} ({response.ReasonPhrase}).", $"GitHub returned HTTP {(int)response.StatusCode} ({response.ReasonPhrase})."),
            null, response.StatusCode);

    static bool IsTransient(Exception error) =>
        error is HttpRequestException request && (request.StatusCode is null || (int)request.StatusCode >= 500);

    static string StatusDetail(Exception? error) =>
        error is HttpRequestException { StatusCode: { } status } ? $" (HTTP {(int)status})" : "";
}
