# DIAVision plug-in (work in progress)

Brings the depth viewer's live data into Delta DIAVision on Windows. The viewer streams frames
on a local TCP port (`live_depth.py --serve`); a DIAVision flow element reads the newest frame
each time the flow runs.

| File | What it is | Status |
|---|---|---|
| `DepthStreamClient.cs` | Reads frames from the stream; plain .NET | Built and run against the real stream |
| `StreamTest/` | Console program that prints frames; no DIAVision needed | Built and run against the real stream |
| `PS3EyeDepthSource.cs` | DIAVision flow element, first stage | Compiles against a stand-in for `DIAVision.Core`; not yet loaded in DIAVision |

The first-stage flow element only outputs numbers (frame number, size, centre / min / max depth
in millimetres, share of valid pixels). It exists to prove the plug-in loads and the live link
works. Passing the image, depth map and point cloud on to other flow elements needs DIAVision's
own image and point cloud types; the manual does not show how a plug-in creates them, so that
step waits on the `PluginSamples` sources.

## Steps on the DIAVision PC

1. Start the viewer with the stream on: `python live_depth.py --engine <engine> --serve`
2. Check the stream without DIAVision: `dotnet run --project StreamTest`
3. Build the plug-in: `dotnet build -c Release` (add `-p:DIAVisionDir="..."` if DIAVision is not
   in `C:\Program Files\Delta Industrial Automation\DIAVision\DMV-IVS`).
4. Install it as the manual describes (section 5.1.5): copy `PS3EyeDepth.dll` and a plug-in
   description `.json` to `C:\Users\Public\Documents\DMV-IVS\Plugin\PS3EyeDepth\`. The
   manual shows the `.json` only as a picture; copy the one from the `ImageMinMax` sample and
   change the names.
5. Start DIAVision; the element appears under the `PS3EyeDepth` category.
