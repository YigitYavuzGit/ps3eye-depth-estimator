// Console check of the depth stream, independent of DIAVision:
//   dotnet run --project StreamTest            (viewer must be running with --serve)
using System;
using System.Diagnostics;
using PS3EyeDepth;

int port = args.Length > 0 ? int.Parse(args[0]) : DepthStreamClient.DefaultPort;
int frames = args.Length > 1 ? int.Parse(args[1]) : 100;
using var client = new DepthStreamClient("127.0.0.1", port);
var clock = Stopwatch.StartNew();
for (int i = 1; i <= frames; i++)
{
    var frame = client.GetFrame(FrameBlocks.Intensity | FrameBlocks.Depth | FrameBlocks.Cloud);
    if (frame == null)
    {
        Console.WriteLine("no new frame (is the viewer running?)");
        continue;
    }
    if (i % 10 == 1)
    {
        int centre = (frame.Height / 2) * frame.Width + frame.Width / 2;
        Console.WriteLine($"seq {frame.Seq}  {frame.Width}x{frame.Height}  centre depth {frame.DepthMm[centre]} mm  " +
                          $"centre xyz ({frame.CloudMm[centre * 3]:F1}, {frame.CloudMm[centre * 3 + 1]:F1}, {frame.CloudMm[centre * 3 + 2]:F1})  " +
                          $"fx {frame.Fx:F1}  {i / clock.Elapsed.TotalSeconds:F1} frames/s");
    }
}
