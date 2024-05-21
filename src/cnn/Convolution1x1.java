package cnn;

import UTIL.Mat;

public class Convolution1x1 {
    public float[][] input; // shape --> [28] X [28]
    //

    /**
     * caches filters that were used in the convolution phase for use in the
     * back-propagation phase.
     */
    public float[][][] filters; // shape --> [3] X [8] X [8]

    /**
     * Convolves the image with respect to a 1x1 filter
     *
     * @param image  the image matrix with shape [height] X [width]
     * @param filter a 1x1 filter used in the convolution process.
     * @return a 2D matrix with shape [height] X [width].
     */
    public static float[][] convolve1x1(float[][] image, float[][] filter) {
        int imageHeight = image.length;
        int imageWidth = image[0].length;

        // Validate input dimensions
        if (filter.length != 1 || filter[0].length != 1) {
            System.out.println(filter.length);
            throw new IllegalArgumentException("Filter must be of size 1x1 for 1x1 convolution");
        }

        // Output array with same dimensions as input image
        float[][] result = new float[imageHeight][imageWidth];

        // Loop through each element in the image
        for (int y = 0; y < imageHeight; y++) {
            for (int x = 0; x < imageWidth; x++) {
                // Element-wise multiplication and summation
                result[y][x] = image[y][x] * filter[0][0];
            }
        }

        return result;
    }

    /**
     * Forward pass for the 1x1 convolution operation
     *
     * @param image  the input image matrix.
     * @param filter the 1x1 filter used in convolution.
     * @return the result of the convolution operation.
     */
    public static float[][] forward(float[][] image, float[][] filter) {
        return convolve1x1(image, filter);
    }

    /**
     * Backpropagation for the 1x1 convolution operation
     *
     * @param d_L_d_out     the gradient of the loss with respect to the output of convolution.
     * @param input         the input image matrix.
     * @param filter        the 1x1 filter used in convolution.
     * @param learning_rate the learning rate for updating the filter.
     * @return
     */
    public static float[][][] backprop(float[][][] d_L_d_out, float[][] input, float[][] filter, float learning_rate) {
        float[][] d_L_d_filters = new float[filter.length][filter[0].length];
        //reverses the convolution phase by creating a 3X3 gradient filter
        //and assigning its elements with the input gradient values scaled by
        //the corresponding pixels of the image.
        for (int i = 1; i < input.length - 2; i++) {
            for (int j = 1; j < input[0].length - 2; j++) {
                for (int k = 0; k < filter.length; k++) {
                    //get a 3X3 region of the matrix
                    float[] region = Mat.m_sub1x1(input, i - 1, i + 1);
                    //for each 3X3 region in the input image i,j
                    // d_L_d_filter(kth filter) = d_L_d_filter(kth filter)+ d_L_d_out(k,i,j)* sub_image(3,3)i,j
                    //       [3] X [3]          =       [3] X [3]         +     gradient    *      [3] X [3]
                    //see article as to how this gradient is computed.
                    d_L_d_filters[k] = Mat.mm_add1x1(d_L_d_filters[k], Mat.m_scale1x1(region, d_L_d_out[k][i - 1][j - 1]));
                }
            }
        }
        return d_L_d_out;
    }
}