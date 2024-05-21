package cnn;


public class Convolution5x5 {
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
     * Convolves the image with respect to a 3X3 filter for each channel.
     *
     * @param image  the image matrix with shape [channels] X [28] X [28].
     * @param filter a 3D matrix containing an array of 3X3 filters ([numFilters] X [3] X [3]).
     * @return a 3D matrix containing an array of the convolved images w.r.t.
     * each filter. [numFilters] X [26] X [26].
     */
    private static float[][][] convolve3x3(float[][][] image, float[][][] filter) {
//        int outputWidth = image.length - 2;
//        int outputHeight = image.length > 0 ? image.length - 2 : 0; // Handle empty input
//
//        float[][] result = new float[outputHeight][outputWidth];
//
//        // Loop through each output pixel location
//        for (int i = 1; i < image.length - 2; i++) {
//            for (int j = 1; j < image[0].length - 2; j++) {
//                float[][] conv_region = Mat.m_sub(image, i - 1, i + 1, j - 1, j + 1);
//                result[i - 1][j - 1] = Mat.mm_elsum(conv_region, filter);
//            }
//        }
//
//        return result;
        int imageWidth = image.length;
        int imageHeight = image[0].length;
        int imageDepth = image[0][0].length;

        int kernelWidth = filter.length;
        int kernelHeight = filter[0].length;
        int kernelDepth = filter[0][0].length;

        // Validate input dimensions
        if (imageWidth < kernelWidth || imageHeight < kernelHeight || imageDepth < kernelDepth) {
            throw new IllegalArgumentException("Image dimensions must be greater than or equal to kernel dimensions");
        }

        int outputWidth = imageWidth - kernelWidth + 1;
        int outputHeight = imageHeight - kernelHeight + 1;
        int outputDepth = imageDepth - kernelDepth + 1;

        float[][][] output = new float[outputWidth][outputHeight][outputDepth];

        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                for (int z = 0; z < outputDepth; z++) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < kernelHeight; ky++) {
                        for (int kx = 0; kx < kernelWidth; kx++) {
                            for (int kz = 0; kz < kernelDepth; kz++) {
                                int imageX = x + kx;
                                int imageY = y + ky;
                                int imageZ = z + kz;

                                // Handle out-of-bounds pixels by setting them to 0
                                if (imageX < 0 || imageX >= imageWidth ||
                                        imageY < 0 || imageY >= imageHeight ||
                                        imageZ < 0 || imageZ >= imageDepth) {
                                    continue;
                                }

                                sum += image[imageX][imageY][imageZ] * filter[kx][ky][kz];
                            }
                        }
                    }
                    output[x][y][z] = sum;
                }
            }
        }

        return output;
    }
    public static float[][][] forward(float[][][] image, float[][][] filter) {
//        int numFilters = filter.length;
//
//        // Allocate result array with appropriate dimensions
//        float[][][] result = new float[8][26][26];
//
//        // Loop through each channel of the image
//        for (int c = 0; c < image.length; c++) {
//            float[][] singleChannelImage = image[c]; // Extract current channel
//
//            // Perform convolution for the current channel
//            for (int k = 0; k < numFilters; k++) {
//                float[][] res = convolve3x3(singleChannelImage, filter[k]);
//                result[k] = res;
//            }
//        }
//        return result;
        return convolve3x3(image, filter);
    }

    /**
     * Performs the convolution operation for a single channel and filter.
     *
     * @param image  a 2D matrix representing a single channel [28] X [28].
     * @param filter a 3X3 filter used in the convolution process.
     * @return a 2D matrix with shape [26] X [26].
     */

    // Backpropagation function can remain unchanged (assuming it works for single-channel gradients)
//    public static float[][][] backprop(float[][][] d_L_d_out, float[][][] input, float[][][] filter, float learning_rate) {
//        int imageDepth = input.length;
//        int imageHeight = input[0].length;
//        int imageWidth = input[0][0].length;
//
//        // Initialize the gradient matrix for the input
//        float[][][] d_L_d_input = new float[imageDepth][imageHeight][imageWidth];
//
//        // Loop through each element in the input
//        for (int d = 0; d < imageDepth; d++) {
//            for (int y = 1; y < imageHeight - 1; y++) {
//                for (int x = 1; x < imageWidth - 1; x++) {
//                    // Loop through each filter
//                    for (int k = 0; k < filter.length; k++) {
//                        // Get a 1x1 region of the input
//                        float[][][] region = new float[1][1][1];
//                        region[0][0][0] = input[d][y][x];
//
//                        // Gradient of the loss with respect to the input element
//                        d_L_d_input[d][y][x] += d_L_d_out[k][y - 1][x - 1] * filter[k][0][0];
//                    }
//                }
//            }
//        }
//
//        // Update the input using the gradient and learning rate
//        for (int d = 0; d < imageDepth; d++) {
//            for (int y = 0; y < imageHeight; y++) {
//                for (int x = 0; x < imageWidth; x++) {
//                    input[d][y][x] -= learning_rate * d_L_d_input[d][y][x];
//                }
//            }
//        }
//
//        return d_L_d_input;
//    }
//}

//    public void backprop(float[][][] d_L_d_out,float[][][] filters,float[][] input, float learning_rate){
//        //the output gradient which is dL/dfilter= (dL/dout)*(dout/dfilter)
//        float[][][] d_L_d_filters= new float[filters.length][filters[0].length][filters[0][0].length];
//        //reverses the convolution phase by creating a 3X3 gradient filter
//        //and assigning its elements with the input gradient values scaled by
//        //the corresponding pixels of the image.
//        for(int i=1;i<input.length-2;i++){
//            for(int j=1;j<input[0].length-2;j++){
//                for(int k=0;k<filters.length;k++){
//                    //get a 3X3 region of the matrix
//                    float[][] region=Mat.m_sub(input,  i - 1, i + 1, j - 1, j + 1);
//                    //for each 3X3 region in the input image i,j
//                    // d_L_d_filter(kth filter) = d_L_d_filter(kth filter)+ d_L_d_out(k,i,j)* sub_image(3,3)i,j
//                    //       [3] X [3]          =       [3] X [3]         +     gradient    *      [3] X [3]
//                    //see article as to how this gradient is computed.
//                    d_L_d_filters[k]=Mat.mm_add(d_L_d_filters[k], Mat.m_scale(region,d_L_d_out[k][i-1][j-1]));
//                }
//            }
//        }
//
//        //update the filter matrix with the gradient matrix obtained above.
//        for(int m=0;m<filters.length;m++){
//            // [3] X [3]  =   [3] X [3] + -lr * [3] X [3]
//            filters[m]= Mat.mm_add(filters[m], Mat.m_scale(d_L_d_filters[m],-learning_rate));
//        }
//    }
    public static float[][][] backprop(float[][][] deltaOutput, float[][][] image, float[][][] kernel) {
        int imageWidth = image.length;
        int imageHeight = image[0].length;
        int imageDepth = image[0][0].length;
//        System.out.println(imageHeight);
//        System.out.println(imageWidth);
//        System.out.println(imageDepth);
        int kernelWidth = kernel.length;
        int kernelHeight = kernel[0].length;
        int kernelDepth = kernel[0][0].length;
//        System.out.println(kernelWidth);
//        System.out.println(kernelHeight);
//        System.out.println(kernelDepth);
        int outputWidth = imageWidth - kernelWidth + 1;
        int outputHeight = imageHeight - kernelHeight + 1;
        int outputDepth = imageDepth - kernelDepth + 1;

        // Update kernel weights
        float[][][] deltaKernel = new float[kernelWidth][kernelHeight][kernelDepth];
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                for (int z = 0; z < outputDepth; z++) {
                    for (int ky = 0; ky < kernelHeight; ky++) {
                        for (int kx = 0; kx < kernelWidth; kx++) {
                            for (int kz = 0; kz < kernelDepth; kz++) {
                                int imageX = x + kx;
                                int imageY = y + ky;
                                int imageZ = z + kz;

                                // Handle out-of-bounds pixels
                                if (imageX < 0 || imageX >= imageWidth ||
                                        imageY < 0 || imageY >= imageHeight ||
                                        imageZ < 0 || imageZ >= imageDepth) {
                                    continue;
                                }

                                deltaKernel[kx][ky][kz] += deltaOutput[x][y][z] * image[imageX][imageY][imageZ];
                            }
                        }
                    }
                }
            }
        }

        // Update image gradients (assuming element-wise activation function)
        float[][][] deltaImage = new float[imageWidth][imageHeight][imageDepth];
        for (int y = 0; y < imageHeight; y++) {
            for (int x = 0; x < imageWidth; x++) {
                for (int z = 0; z < imageDepth; z++) {
                    for (int ky = 0; ky < kernelHeight; ky++) {
                        for (int kx = 0; kx < kernelWidth; kx++) {
                            for (int kz = 0; kz < kernelDepth; kz++) {
                                int outputX = x - kx + 1;
                                int outputY = y - ky + 1;
                                int outputZ = z - kz + 1;

                                // Handle output bounds (assuming zero-padding)
                                if (outputX >= 0 && outputX < outputWidth &&
                                        outputY >= 0 && outputY < outputHeight &&
                                        outputZ >= 0 && outputZ < outputDepth) {
                                    deltaImage[x][y][z] += deltaOutput[outputX][outputY][outputZ] * kernel[kx][ky][kz];
                                }
                            }
                        }
                    }
                }
            }
        }
        return deltaKernel;
    }
}
