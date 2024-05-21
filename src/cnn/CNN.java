
package cnn;

import UTIL.Mat;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;

public class CNN {

    /** Loads image from file and returns a bufferedImage.
     * @param src Absolute file path to image
     * @return BufferedImage loaded from file. 
     * @throws java.io.IOException 
    **/
    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
    }


    /**
     * converts a BufferedImage into a pixel array and normalizes it. 
     * @param imageToPixelate the source image to be converted.
     * @return 2D array with normalized pixel values between 0.0 and 1.0
     */
    public static float[][] img_to_mat(BufferedImage imageToPixelate) {
        int w = imageToPixelate.getWidth(), h = imageToPixelate.getHeight();
        int[] pixels = imageToPixelate.getRGB(0, 0, w, h, null, 0, w);
        float[][] dta = new float[w][h];

        for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel++) {
            dta[row][col] = (((int) pixels[pixel] >> 16 & 0xff)) / 255.0f;
            col++;
            if (col == w) {
                col = 0;
                row++;
            }
        }
        return dta;
    }

    /**
     * creates 3X3 convolution filters with random initial weights.
     * @param size number of 3X3 filters to be randomly initialized
     * @return a [size] X [3] X [3] 3d array with size filters
     */
    public static float[][][] init_filters3x3(int size) {
        float[][][] result = new float[size][3][3];
        for (int k = 0; k < size; k++) {
            result[k] = Mat.m_random3x3(3, 3);
        }
        return result;
    }

    public static float[][][] init_filters5x5(int size) {
        float[][][] result = new float[size][3][3];
        for (int k = 0; k < size; k++) {
            result[k] = Mat.m_random3x3(3, 3);
        }
        return result;
    }

    public static float[][] init_filters1x1(int size) {
        float[][] result = new float[size][1];
        for (int k = 0; k < size; k++) {
            result[k] = Mat.m_random1x1(1);
        }
        return result;
    }

    /**
     * loads a random image from a specific digit folder in the MNIST database.
     * @param label the folder label (ranges between 0 and 9)
     * @return a BufferedImage of the digit.
     * @throws IOException if the image file isn't found.
     */
    public static BufferedImage mnist_load_random(int label) throws IOException {
        String mnist_path = "data\\training";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        int random_index = new Random().nextInt(files.length);
        String final_path = mnist_path + "\\" + label + "\\" + files[random_index];
        return load_image(final_path);
    }
    
    /**
     * performs both the forward and back-propagation passes of the CNN.
     * @param training_size the number of images used for training the CNN.
     * @throws IOException if image cannot be found.
     */
    public static void train(int training_size) throws IOException {
        float[][][] filters3x3 = init_filters3x3(8);
        float[][][] filters5x5 = init_filters5x5(8);
        float[][] filters1x1 = init_filters1x1(1);
        int label_counter = 0;
        float ce_loss=0;
        int accuracy=0;
        float acc_sum=0.0f;
        float learn_rate=0.005f;
        
        //initialize layers
        Convolution1x1 conv1x1=new Convolution1x1();
        Convolution3x3 conv3x3=new Convolution3x3();
        Convolution5x5 conv5x5=new Convolution5x5();
        MaxPool pool=new MaxPool();
        SoftMax softmax=new SoftMax(13*13*8,10);

        float[][] out_l = new float[1][10];    
        for (int i = 0; i < training_size; i++) {
            //grab a random image from database.
            BufferedImage bi = mnist_load_random(label_counter);
            int correct_label = label_counter;
            if(label_counter==9){
                label_counter=0;
            }else{
                label_counter++;
            }

            //FORWARD PROPAGATION

            //convert to pixel array
            float[][] pxl = img_to_mat(bi);
            // perform convolution 28*28 --> 8x26x26
            float[][] out = Convolution1x1.forward(pxl,filters1x1);
            float[][][] output = conv3x3.forward(out, filters3x3);
            float[][][] output1 = Convolution5x5.forward(output, filters5x5);
            // perform maximum pooling  8x26x26 --> 8x13x13
            output = pool.forward(output);

            // perform softmax operation  8*13*13 --> 10
            out_l = softmax.forward(output);

            // compute cross-entropy loss
            ce_loss += (float) -Math.log(out_l[0][correct_label]);
            accuracy += correct_label == Mat.v_argmax(out_l) ? 1 : 0;

            //BACKWARD PROPAGATION --- STOCHASTIC GRADIENT DESCENT
            //gradient of the cross entropy loss
            float[][] gradient=Mat.v_zeros(10);
            gradient[0][correct_label]=-1/out_l[0][correct_label];
            float[][][] sm_gradient=softmax.backprop(gradient,learn_rate);

            float[][][] mp_gradient=pool.backprop(sm_gradient);
            conv5x5.backprop(mp_gradient, output1, filters5x5);
            conv3x3.backprop(mp_gradient, learn_rate);
            conv1x1.backprop(mp_gradient, pxl, filters1x1,  learn_rate);
            if(i % 100 == 99){
                System.out.println(" step: "+ i+ " loss: "+ce_loss/100.0+" accuracy: "+accuracy);
                ce_loss=0;
                acc_sum+=accuracy;
                accuracy=0;
            }
        }
        System.out.println("average accuracy:- "+acc_sum/training_size+"%");
    }

    public static void main(String[] args) throws IOException {      
        train(30000);
    }
}
