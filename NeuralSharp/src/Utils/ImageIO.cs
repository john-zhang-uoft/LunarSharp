using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.IO;

namespace NeuralSharp
{
    public class ImageIO
    {
        // Only supported on Windows
        public static byte[] ImageToByte(Image img)
        {
            using var stream = new MemoryStream();
            img.Save(stream, ImageFormat.Png);
            return stream.ToArray();
        }

        // Only supported on Windows
        public static byte[] ImageToByteImageConverter(Image img)
        {
            ImageConverter converter = new ImageConverter();
            return (byte[])converter.ConvertTo(img, typeof(byte[]));
        }

        /// <summary>
        /// Creates Bitmap from opaque grayscale data with dimensions (height, width). This method assumes that
        /// data[i * w + j] gives the value for the pixel in the i-th row and j-th column 
        /// </summary>
        /// <param name="data"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <returns></returns>
        /// <exception cref="InvalidDataException"></exception>
        public static Bitmap ConstructGrayScaleBitMapFromData(int[] data, int height, int width)
        {
            if (data.Length != height * width)
            {
                throw new InvalidDataException(
                    $"Data array size = {data.Length} cannot be reshaped to ({height}, {width}).");
            }

            // Make an empty bitmap according to height and width
            Bitmap newBitmap = new Bitmap(width, height);

            for (int i = 0; i < newBitmap.Width; i++)
            {
                for (int j = 0; j < newBitmap.Height; j++)
                {
                    // To convert gray value to rgb: r, g, b = gray value
                    newBitmap.SetPixel(i, j,
                        Color.FromArgb(255, data[i * width + j], data[i * width + j], data[i * width + j]));
                }
            }

            return newBitmap;
        }

        public static void SaveBitmapAsPNG(Bitmap bitmap, string filePath)
        {
            bitmap.Save(filePath, ImageFormat.Png);
        }
    }
}