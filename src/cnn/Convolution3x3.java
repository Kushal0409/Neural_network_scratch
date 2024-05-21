package cnn;

import UTIL.Mat;

import javax.swing.*;

public class Convolution3x3 {
    //

    /**
     * caches the input data (the image) for use in the back-propagation phase.
     */
    public float[][] input; // shape --> [28] X [28]
    //

    /**
     * caches filters that were used in the convolution phase for use in the
     * back-propagation phase.
     */
    public float[][][] filters; // shape --> [3] X [8] X [8]

    /**
     * Convolves the image with respect to a 3X3 filter
     *
     * @param image  the image matrix with shape [28] X [28]
     * @param filter a 3X3 filter used in the convolution process.
     * @return a 2D matrix with shape [26] X [26].
     */

    public float[][] convolve3x3(float[][] image, float[][] filter) {
        input = image;
        float[][] result = new float[image.length - 2][image[0].length - 2];
        //loop through
        for (int i = 1; i < image.length - 2; i++) {
            for (int j = 1; j < image[0].length - 2; j++) {
                float[][] conv_region = Mat.m_sub(image, i - 1, i + 1, j - 1, j + 1);
                result[i][j] = Mat.mm_elsum(conv_region, filter);
            }
        }
        return result;
    }

    /**
     * the forward convolution pass that convolves the image w.r.t. each filter
     * in the filter array. No padding has been used in this case, so output matrix
     * shape decreases by 2 w.r.t row width and column height.
     *
     * @param image  the input image matrix. [28] X [28]
     * @param filter a 3D matrix containing an array of 3X3 filters ([8]X[3]X[3])
     * @return a 3D array containing an array of the convolved images w.r.t.
     * each filter. [8] X [26] X [26]
     */
    public float[][][] forward(float[][] image, float[][][] filter) {
        filters = filter; // 8 X 3 X 3
        float[][][] result = new float[8][26][26];
        for (int k = 0; k < filters.length; k++) {
            float[][] res = convolve3x3(image, filters[k]);
            result[k] = res;
        }
        return result;
    }

    /**
     * @param d_L_d_out     the input gradient matrix retrieved from the back-propagation
     *                      phase of the maximum pooling stage. shape = [8] X [26] X [26]
     * @param learning_rate the learning rate factor used in the neural network.
     * @return
     */
    public float[][][] backprop(float[][][] d_L_d_out, float learning_rate) {
        //the output gradient which is dL/dfilter= (dL/dout)*(dout/dfilter)
        float[][][] d_L_d_filters = new float[filters.length][filters[0].length][filters[0][0].length];
        //reverses the convolution phase by creating a 3X3 gradient filter
        //and assigning its elements with the input gradient values scaled by
        //the corresponding pixels of the image.

        for (int i = 1; i < input.length - 2; i++) {
            for (int j = 1; j < input[0].length - 2; j++) {
                for (int k = 0; k < filters.length; k++) {
                    //get a 3X3 region of the matrix
                    float[][] region = Mat.m_sub(input, i - 1, i + 1, j - 1, j + 1);
                    //for each 3X3 region in the input image i,j
                    // d_L_d_filter(kth filter) = d_L_d_filter(kth filter)+ d_L_d_out(k,i,j)* sub_image(3,3)i,j
                    //       [3] X [3]          =       [3] X [3]         +     gradient    *      [3] X [3]
                    //see article as to how this gradient is computed.
                    //System.out.println(d_L_d_filters.length);
                    d_L_d_filters[k] = Mat.mm_add(d_L_d_filters[k], Mat.m_scale(region, d_L_d_out[k][i - 1][j - 1]));
                }
            }
        }

        //update the filter matrix with the gradient matrix obtained above.
        for (int m = 0; m < filters.length; m++) {
            // [3] X [3]  =   [3] X [3] + -lr * [3] X [3]
            filters[m] = Mat.mm_add(filters[m], Mat.m_scale(d_L_d_filters[m], -learning_rate));
        }
        return d_L_d_filters;
    }
}
//        int imageWidth = image.length;
//        int imageHeight = image[0].length;
//        System.out.println(imageHeight);
//        System.out.println(imageWidth);
//        int kernelWidth = kernel.length;
//        int kernelHeight = kernel[0].length;
//        int kernelDepth = kernel[0][0].length;
//
//        int outputWidth = imageWidth - kernelWidth + 1;
//        int outputHeight = imageHeight - kernelHeight + 1;
//
//        // Update kernel weights
//        float[][][] deltaKernel = new float[kernelWidth][kernelHeight][kernelDepth];
//        for (int y = 0; y < outputHeight; y++) { // Iterate up to output height
//            for (int x = 0; x < outputWidth; x++) { // Iterate up to output width
//                for (int z = 0; z < kernelDepth; z++) {
//                    for (int ky = 0; ky < kernelHeight; ky++) {
//                        for (int kx = 0; kx < kernelWidth; kx++) {
//                            int imageX = x + kx;
//                            int imageY = y + ky;
//
//                            // Handle out-of-bounds pixels
//                            if (imageX < 0 || imageX >= imageWidth ||
//                                    imageY < 0 || imageY >= imageHeight) {
//                                System.out.println("Hello");
//                                continue;
//                            }
//
//                            deltaKernel[kx][ky][z] += d_L_d_out[x][y][z] * image[imageX][imageY];
//                        }
//                    }
//                }
//            }
//        }
//
//        // Update image gradients (assuming element-wise activation function)
//        float[][] deltaImage = new float[imageWidth][imageHeight];
//        for (int y = 0; y < imageHeight; y++) {
//            for (int x = 0; x < imageWidth; x++) {
//                for (int z = 0; z < kernelDepth; z++) {
//                    for (int ky = 0; ky < kernelHeight; ky++) {
//                        for (int kx = 0; kx < kernelWidth; kx++) {
//                            int outputX = x - kx + 1;
//                            int outputY = y - ky + 1;
//
//                            // Handle output bounds (assuming zero-padding)
//                            if (outputX >= 0 && outputX < outputWidth &&
//                                    outputY >= 0 && outputY < outputHeight) {
//                                deltaImage[x][y] += d_L_d_out[outputX][outputY][z] * kernel[kx][ky][z];
//                            }
//                        }
//                    }
//                }
//            }
//        }
//        return deltaKernel;
//    }
