// Reads frames from the depth viewer's stream (live_depth.py --serve).
// Plain .NET, no DIAVision types, so it can be tested on its own (see StreamTest).
// The wire format is described in frame_server.py.
using System;
using System.Buffers.Binary;
using System.IO;
using System.Net.Sockets;

namespace PS3EyeDepth
{
    [Flags]
    public enum FrameBlocks : uint
    {
        None = 0,
        Intensity = 1,  // 8-bit grey image
        Colour = 2,     // 8-bit BGR image
        Depth = 4,      // 16-bit Z in millimetres, 0 = no data
        Cloud = 8,      // float X, Y, Z in millimetres per pixel, (0,0,0) = no data
    }

    public sealed class DepthFrame
    {
        public uint Seq;
        public int Width;
        public int Height;
        public float Fx, Fy, Cx, Cy;
        public float UnitMm;
        public FrameBlocks Blocks;
        public ulong TimestampMicroseconds;
        public byte[] Intensity;   // Width * Height
        public byte[] Colour;      // Width * Height * 3
        public ushort[] DepthMm;   // Width * Height
        public float[] CloudMm;    // Width * Height * 3
    }

    public sealed class DepthStreamClient : IDisposable
    {
        public const int DefaultPort = 5600;
        private const int HeaderSize = 52;

        private readonly TcpClient tcp;
        private readonly NetworkStream stream;
        private readonly byte[] header = new byte[HeaderSize];
        private readonly byte[] request = new byte[12];
        private byte[] scratch = Array.Empty<byte>();
        private uint lastSeq;

        public DepthStreamClient(string host = "127.0.0.1", int port = DefaultPort, int timeoutMs = 3000)
        {
            tcp = new TcpClient { NoDelay = true, ReceiveTimeout = timeoutMs, SendTimeout = timeoutMs };
            tcp.Connect(host, port);
            stream = tcp.GetStream();
        }

        /// <summary>Waits for the next frame. Returns null if the viewer produced none in time.</summary>
        public DepthFrame GetFrame(FrameBlocks blocks)
        {
            request[0] = (byte)'D'; request[1] = (byte)'G'; request[2] = (byte)'E'; request[3] = (byte)'T';
            BinaryPrimitives.WriteUInt32LittleEndian(request.AsSpan(4), (uint)blocks);
            BinaryPrimitives.WriteUInt32LittleEndian(request.AsSpan(8), lastSeq);
            stream.Write(request, 0, request.Length);

            ReadExactly(header, HeaderSize);
            if (header[0] != 'D' || header[1] != 'F' || header[2] != 'R' || header[3] != 'M')
                throw new IOException("Unexpected reply from the depth stream");
            var h = header.AsSpan();
            var frame = new DepthFrame
            {
                Seq = BinaryPrimitives.ReadUInt32LittleEndian(h.Slice(8)),
                Width = (int)BinaryPrimitives.ReadUInt32LittleEndian(h.Slice(12)),
                Height = (int)BinaryPrimitives.ReadUInt32LittleEndian(h.Slice(16)),
                Fx = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(h.Slice(20))),
                Fy = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(h.Slice(24))),
                Cx = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(h.Slice(28))),
                Cy = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(h.Slice(32))),
                UnitMm = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(h.Slice(36))),
                Blocks = (FrameBlocks)BinaryPrimitives.ReadUInt32LittleEndian(h.Slice(40)),
                TimestampMicroseconds = BinaryPrimitives.ReadUInt64LittleEndian(h.Slice(44)),
            };
            if (frame.Blocks == FrameBlocks.None)
                return null;

            int pixels = frame.Width * frame.Height;
            if ((frame.Blocks & FrameBlocks.Intensity) != 0)
            {
                frame.Intensity = new byte[pixels];
                ReadExactly(frame.Intensity, pixels);
            }
            if ((frame.Blocks & FrameBlocks.Colour) != 0)
            {
                frame.Colour = new byte[pixels * 3];
                ReadExactly(frame.Colour, pixels * 3);
            }
            if ((frame.Blocks & FrameBlocks.Depth) != 0)
            {
                frame.DepthMm = new ushort[pixels];
                Buffer.BlockCopy(ReadScratch(pixels * 2), 0, frame.DepthMm, 0, pixels * 2);
            }
            if ((frame.Blocks & FrameBlocks.Cloud) != 0)
            {
                frame.CloudMm = new float[pixels * 3];
                Buffer.BlockCopy(ReadScratch(pixels * 12), 0, frame.CloudMm, 0, pixels * 12);
            }
            lastSeq = frame.Seq;
            return frame;
        }

        // The stream is little-endian, like every PC this runs on, so the bytes copy straight across.
        private byte[] ReadScratch(int count)
        {
            if (scratch.Length < count)
                scratch = new byte[count];
            ReadExactly(scratch, count);
            return scratch;
        }

        private void ReadExactly(byte[] buffer, int count)
        {
            int got = 0;
            while (got < count)
            {
                int n = stream.Read(buffer, got, count - got);
                if (n <= 0)
                    throw new IOException("The depth stream closed the connection");
                got += n;
            }
        }

        public void Dispose()
        {
            stream.Dispose();
            tcp.Dispose();
        }
    }
}
