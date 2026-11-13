using System.Runtime.InteropServices;
using static ReLink.Texts;

namespace ReLink;

/// The display's refresh rate (a temporary mode change while the game runs), the keyboard's labels and the XInput pads.
static class Native
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct DEVMODE
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
        public short dmSpecVersion, dmDriverVersion, dmSize, dmDriverExtra;
        public int dmFields;
        public int dmPositionX, dmPositionY, dmDisplayOrientation, dmDisplayFixedOutput;
        public short dmColor, dmDuplex, dmYResolution, dmTTOption, dmCollate;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
        public short dmLogPixels;
        public int dmBitsPerPel, dmPelsWidth, dmPelsHeight, dmDisplayFlags, dmDisplayFrequency;
        public int dmICMMethod, dmICMIntent, dmMediaType, dmDitherType, dmReserved1, dmReserved2, dmPanningWidth, dmPanningHeight;
    }

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    static extern bool EnumDisplaySettingsW(string? device, int mode, ref DEVMODE dm);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    static extern int ChangeDisplaySettingsExW(string? device, ref DEVMODE dm, IntPtr hwnd, int flags, IntPtr param);
    [DllImport("user32.dll", CharSet = CharSet.Unicode, EntryPoint = "ChangeDisplaySettingsExW")]
    static extern int ChangeDisplaySettingsExNull(string? device, IntPtr dm, IntPtr hwnd, int flags, IntPtr param);

    const int ENUM_CURRENT_SETTINGS = -1;
    const int CDS_FULLSCREEN = 0x4, CDS_TEST = 0x2;
    const int DM_PELSWIDTH = 0x80000, DM_PELSHEIGHT = 0x100000, DM_DISPLAYFREQUENCY = 0x400000;

    static DEVMODE NewMode()
    {
        var dm = new DEVMODE { dmDeviceName = "", dmFormName = "" };
        dm.dmSize = (short)Marshal.SizeOf<DEVMODE>();
        return dm;
    }

    /// The primary display's mode now: width, height, refresh (0 when unknown)
    public static (int w, int h, int hz) CurrentMode()
    {
        var dm = NewMode();
        return EnumDisplaySettingsW(null, ENUM_CURRENT_SETTINGS, ref dm) ? (dm.dmPelsWidth, dm.dmPelsHeight, dm.dmDisplayFrequency) : (0, 0, 0);
    }

    /// The refresh rates the primary display offers at this size
    public static SortedSet<int> RatesAt(int w, int h)
    {
        var rates = new SortedSet<int>();
        var dm = NewMode();
        for (int i = 0; EnumDisplaySettingsW(null, i, ref dm); i++)
        {
            if (dm.dmPelsWidth == w && dm.dmPelsHeight == h && dm.dmDisplayFrequency > 1) rates.Add(dm.dmDisplayFrequency);
            dm = NewMode();
        }
        return rates;
    }

    static bool changed;

    /// The primary display at `hz` at its current size, as a temporary mode (CDS_FULLSCREEN: not written to the registry, so
    /// Windows's own setting stays what it was). test: only asks whether it would work.
    public static bool SetRefresh(int hz, bool test, out string why)
    {
        var (w, h, cur) = CurrentMode();
        if (w == 0) { why = L("mode actuel inconnu", "current mode unknown"); return false; }
        if (cur == hz) { why = L($"déjà à {hz} Hz", $"already at {hz} Hz"); return true; }
        if (!RatesAt(w, h).Contains(hz)) { why = L($"l'écran ne propose pas {hz} Hz en {w}×{h}", $"the display has no {hz} Hz mode at {w}×{h}"); return false; }
        var dm = NewMode();
        dm.dmPelsWidth = w;
        dm.dmPelsHeight = h;
        dm.dmDisplayFrequency = hz;
        dm.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;
        int r = ChangeDisplaySettingsExW(null, ref dm, IntPtr.Zero, test ? CDS_TEST : CDS_FULLSCREEN, IntPtr.Zero);
        if (r != 0) { why = L($"Windows refuse le changement (code {r})", $"Windows refused the change (code {r})"); return false; }
        if (!test) changed = true;
        why = $"{w}×{h}, {hz} Hz";
        return true;
    }

    /// Back to the mode Windows has in its settings (only if this launcher changed it)
    public static void RestoreRefresh()
    {
        if (!changed) return;
        ChangeDisplaySettingsExNull(null, IntPtr.Zero, IntPtr.Zero, 0, IntPtr.Zero);
        changed = false;
    }

    // ---- the label of a key by its place (the game reads letter keys by position: host/nn_hid.c pos) ----
    [DllImport("user32.dll")] static extern IntPtr GetKeyboardLayout(uint thread);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern uint MapVirtualKeyExW(uint code, uint type, IntPtr hkl);
    static readonly byte[] UsScan = { 0x1E, 0x30, 0x2E, 0x20, 0x12, 0x21, 0x22, 0x23, 0x17, 0x24, 0x25, 0x26, 0x32,
                                      0x31, 0x18, 0x19, 0x10, 0x13, 0x1F, 0x14, 0x16, 0x2F, 0x11, 0x2D, 0x15, 0x2C };
    /// What the key a US keyboard labels `us` (A-Z) shows on this keyboard
    public static string KeyLabel(char us)
    {
        if (us < 'A' || us > 'Z') return us.ToString();
        var hkl = GetKeyboardLayout(0);
        uint vk = MapVirtualKeyExW(UsScan[us - 'A'], 3 /* MAPVK_VSC_TO_VK_EX */, hkl);
        uint ch = vk != 0 ? MapVirtualKeyExW(vk, 2 /* MAPVK_VK_TO_CHAR */, hkl) & 0xFFFF : 0;
        return ch >= 32 ? char.ToUpperInvariant((char)ch).ToString() : us.ToString();
    }

    // ---- XInput: which pads are connected ----
    [StructLayout(LayoutKind.Sequential)]
    struct XINPUT_STATE { public uint packet; public ushort buttons; public byte lt, rt; public short lx, ly, rx, ry; }
    [DllImport("xinput1_4.dll", EntryPoint = "XInputGetState")]
    static extern int XInputGetState(int index, out XINPUT_STATE state);

    public static int PadsConnected()
    {
        int n = 0;
        try
        {
            for (int i = 0; i < 4; i++)
                if (XInputGetState(i, out _) == 0) n++;
        }
        catch (DllNotFoundException) { }
        catch (EntryPointNotFoundException) { }
        return n;
    }
}
