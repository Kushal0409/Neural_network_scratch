
package cnn;

import UTIL.Mat;

public class SoftMax {
    /**
     * weight matrix between the softmax layer and the output layer
     */
        public float[][] weights;
    /**
     * the flattened input array obtained from the max-pooling layer.
     */
        public float[][] input;
    /**
     * the bias vector of the output layer
     */
        public float[][] bias;
    /**
     * the output layer vector.
     */
        public float[][] output;
    /**
     * constructor for the softmax layer that initializes the weight matrix to
     * random values and the bias vector to zeroes.
     * @param input size of the input layer.
     * @param output size of the output layer.
     */
    public SoftMax(int input, int output) {
        weights = Mat.m_scale(Mat.m_random3x3(input, output), 1.00f / input);
        bias = Mat.v_zeros(10);
    }

    /**
     * performs the forward pass of the softmax layer.
     * @param input a [8] X [13] X [13] 3D matrix obtained from the max-pooling layer.
     * @return a [1] X [10] vector of the softmax probabilities
     */
    public float[][] forward(float[][][] input) {
        //flattens the input to [8] X [13] X [13] to a [1] X [8*13*13] vector.
        float[][] in = Mat.m_flatten(input);  //1X1342
        output = new float[1][bias.length];    //1X10
     // evaluate the total activation value --> t=[i][w]+[b] and cache the totals for backprop
     // [1] X [10] =  [1] X [1342]  * [1342] X [10] + [1] X [10]
        output = Mat.mm_add(Mat.mm_mult(in, weights), bias);
        //compute softmax probabilities.
        float[][] totals = Mat.v_exp(output);
        float inv_activation_sum = 1 / Mat.v_sum(totals);
        //cache input
        this.input = in;
        return Mat.v_scale(totals, inv_activation_sum);
    }

    /**
     * performs the back-propagation phase of the softmax layer. 
     * @param d_L_d_out the gradient vector obtained from the cross-entropy loss vector.
     * @param learning_rate the learning rate of the neural network.
     * @return a gradient matrix with the shape [8] X [13] X [13] to be fed to the
     * maxpooling layer.
     */
    public float[][][] backprop(float[][] d_L_d_out, float learning_rate) {
        //gradient of loss w.r.t. the total probabilites of the softmax layer.
        float[][] d_L_d_t = new float[1][d_L_d_out[0].length];
        //repeat softmax probability computations (caching can be used to avoid this.)
        float[][] t_exp = Mat.v_exp(output);
        float S = Mat.v_sum(t_exp);
        float[][] d_L_d_inputs=null;
        
        for (int i = 0; i < d_L_d_out[0].length; i++) {
            float grad = d_L_d_out[0][i];
            if (grad == 0) {
                continue;
            }
            //gradient of the output layer w.r.t. the totals [1] X [10]
            float[][] d_out_d_t = Mat.v_scale(t_exp, -t_exp[0][i] / (S * S));
            d_out_d_t[0][i] = t_exp[0][i] * (S - t_exp[0][i]) / (S * S);
            
            d_L_d_t = Mat.m_scale(d_out_d_t, grad); 
            //gradient of totals w.r.t weights -- [1342] X [1]
            float[][] d_t_d_weight = Mat.m_transpose(input);
            //gradient of totals w.r.t inputs -- [1342] X [10] 
            float[][] d_t_d_inputs = weights;
            //gradient of Loss w.r.t. weights ---> chain rule 
            //        [1342] X [10] = [1342] X [1] * [1] X [10]
            float[][] d_L_d_w = Mat.mm_mult(d_t_d_weight, d_L_d_t);
            //gradient of Loss w.r.t. inputs ---> chain rule
            // [1342] X [1]      [1342] X [10]    *   [10] X [1](transposed)
            d_L_d_inputs = Mat.mm_mult(d_t_d_inputs, Mat.m_transpose(d_L_d_t));
            //gradient of loss w.r.t. bias
            float[][] d_L_d_b = d_L_d_t;
            //update the weight and bias matrices.
            weights = Mat.mm_add(Mat.m_scale(d_L_d_w, -learning_rate), weights);
            bias = Mat.mm_add(Mat.m_scale(d_L_d_b, -learning_rate), bias);
        }
        // reshape the final gradient matrix to the input shape of the maxpooling layer.
        // [1] X [1342](transposed) ----> [8] X [13] X [13]
        assert d_L_d_inputs != null;
        return Mat.reshape(Mat.m_transpose(d_L_d_inputs),8, 13, 13);
    }
}
