// DIAVision flow element that connects to the depth viewer's stream (live_depth.py --serve).
//
// FIRST STAGE: this version only uses the plug-in API shown in the DIAVision manual (chapter 5):
// a FlowElement with int inputs and double outputs. Each run it fetches the newest frame and
// reports numbers about it, which proves the plug-in loads and the live link works.
// Handing the image, depth map and point cloud to other flow elements needs DIAVision's own
// Image / point cloud types, whose construction the manual does not show; that part is added
// once the PluginSamples sources have been checked.
using System;
using DIAVision.Core.CommonTypes;
using DIAVision.Core.FlowElements;
using DIAVision.Core.FlowElements.Attributes;

namespace PS3EyeDepth
{
    [Category("PS3EyeDepth")]
    public class PS3EyeDepthSource : FlowElement
    {
        private DepthStreamClient client;
        private int connectedPort;

        [Input]
        [Linkable]
        [InitialValue("5600")]
        public int Port => (int)GetInputValue(nameof(Port));

        [Input]
        [Linkable]
        [InitialValue("3000")]
        public int TimeoutMs => (int)GetInputValue(nameof(TimeoutMs));

        [Output]
        public double FrameNumber { get; set; }

        [Output]
        public double Width { get; set; }

        [Output]
        public double Height { get; set; }

        /// <summary>Depth at the image centre in millimetres (0 = no data).</summary>
        [Output]
        public double CentreDepthMm { get; set; }

        [Output]
        public double MinDepthMm { get; set; }

        [Output]
        public double MaxDepthMm { get; set; }

        /// <summary>Share of pixels with a depth value, 0 to 1.</summary>
        [Output]
        public double ValidRatio { get; set; }

        public override bool CanExecute()
        {
            return true;
        }

        public override bool Execute()
        {
            try
            {
                if (client == null || connectedPort != Port)
                {
                    client?.Dispose();
                    client = new DepthStreamClient("127.0.0.1", Port, TimeoutMs);
                    connectedPort = Port;
                }
                DepthFrame frame = client.GetFrame(FrameBlocks.Depth);
                if (frame == null)
                    return false;  // the viewer is running but produced no new frame in time

                ushort[] depth = frame.DepthMm;
                int valid = 0, min = int.MaxValue, max = 0;
                for (int i = 0; i < depth.Length; i++)
                {
                    int d = depth[i];
                    if (d == 0)
                        continue;
                    valid++;
                    if (d < min) min = d;
                    if (d > max) max = d;
                }
                FrameNumber = frame.Seq;
                Width = frame.Width;
                Height = frame.Height;
                CentreDepthMm = depth[(frame.Height / 2) * frame.Width + frame.Width / 2];
                MinDepthMm = valid > 0 ? min : 0;
                MaxDepthMm = max;
                ValidRatio = depth.Length > 0 ? (double)valid / depth.Length : 0;
                return true;
            }
            catch (Exception)
            {
                // Viewer not running or connection lost: drop the link and retry on the next run.
                client?.Dispose();
                client = null;
                return false;
            }
        }
    }
}
