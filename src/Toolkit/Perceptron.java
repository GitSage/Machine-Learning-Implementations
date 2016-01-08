package Toolkit;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by ben on 9/19/15.
 */
public class Perceptron extends SupervisedLearner {
    double learningRate;
    double weight;
    Random rand;
    double[] weights;
    double bias;
    boolean trained;

    public Perceptron(Random rand){
        this.rand = rand;
        learningRate = 0.1;  // arbitrary number
        trained = false;
    }

    /*
    public void train(Toolkit.Matrix features, Toolkit.Matrix labels){
        int totalEpochs = 0;
        int bestAccuracySoFar = 0;  // the number that were correct on the previous epoch
        int epochsSinceBestChanged = 0;
        int accuracyThreshold = 20;  // if we go x epochs without improving, we're done.
        double[] bestWeights = new double[features.cols()];  // will temporarily hold the best weight discovered so far
        double bestBias = 0;

        // set up weights with random values
        weights = new double[features.cols()];
        for(int i = 0; i < weights.length; i++){
            weights[i] = rand.nextDouble() * 2 - 1;
        }
        bias =  rand.nextDouble() * 2 - 1;

        while(epochsSinceBestChanged < accuracyThreshold){
            totalEpochs++;
            int epochAccuracy = 0;
            for(int r = 0; r < features.rows(); r++){
                double total = 0;
                for(int c = 0; c < features.cols(); c++){
                    total += weights[c] * features.get(r, c);
                }
                total += bias;

                int fired = total > 0 ? 1 : 0;

                // if output != expected, update
                if(fired != labels.get(r, 0)){
                    // System.out.println("Updating weights");
                    // System.out.println("Weights before: " + Arrays.toString(weights));
                    for(int c = 0; c < features.cols(); c++){
                        weights[c] += learningRate * (labels.get(r, 0) - fired) * features.get(r, c);
                    }
                    bias += learningRate * (labels.get(r, 0) - fired);
                    // System.out.println("Weights after: " + Arrays.toString(weights));
                }
                else{
                    // System.out.println("Not updating weights");
                    epochAccuracy++;
                }
            }
            if(epochAccuracy > bestAccuracySoFar){
                bestWeights = weights.clone();
                bestBias = bias;
                bestAccuracySoFar = epochAccuracy;
                epochsSinceBestChanged = 0;
            }
            else{
                epochsSinceBestChanged++;
            }
            features.shuffle(new Random(), features);
            System.out.println(totalEpochs + " " + ((double)epochAccuracy/(double)features.rows()));
        }

        // Accuracy is no longer improving, so set the best weights and finish.
        weights = bestWeights;
        bias = bestBias;
        trained = true;
        System.out.println("Done training. Total epochs: " + totalEpochs);
        System.out.println("Weights: " + Arrays.toString(weights));
        System.out.println("Bias: " + bias);
    }
    */

    public void train(Matrix features, Matrix labels){
        int totalEpochs = 0;
        double expLearningRate = 1;
        double[] bestWeights = new double[features.cols()];  // will temporarily hold the best weight discovered so far
        double bestBias = 0;

        // set up weights with random values
        weights = new double[features.cols()];
        for(int i = 0; i < weights.length; i++){
            weights[i] = rand.nextDouble() * 2 - 1;
        }
        bias =  rand.nextDouble() * 2 - 1;

        while(expLearningRate > 0.0001){
            totalEpochs++;
            int epochAccuracy = 0;
            for(int r = 0; r < features.rows(); r++){
                double total = 0;
                for(int c = 0; c < features.cols(); c++){
                    total += weights[c] * features.get(r, c);
                }
                total += bias;

                int fired = total > 0 ? 1 : 0;

                // if output != expected, update
                if(fired != labels.get(r, 0)){
                    // System.out.println("Updating weights");
                    // System.out.println("Weights before: " + Arrays.toString(weights));
                    for(int c = 0; c < features.cols(); c++){
                        weights[c] += expLearningRate * (labels.get(r, 0) - fired) * features.get(r, c);
                    }
                    bias += expLearningRate * (labels.get(r, 0) - fired);
                    // System.out.println("Weights after: " + Arrays.toString(weights));
                }
                else{
                    // System.out.println("Not updating weights");
                    epochAccuracy++;
                }
            }

            expLearningRate *= ((double)epochAccuracy/(double)features.rows());

            features.shuffle(new Random(), features);
            System.out.println(expLearningRate + " " + ((double)epochAccuracy/(double)features.rows()));
        }

        // Accuracy is no longer improving, so set the best weights and finish.
        weights = bestWeights;
        bias = bestBias;
        trained = true;
        System.out.println("Done training. Total epochs: " + totalEpochs);
        System.out.println("Weights: " + Arrays.toString(weights));
        System.out.println("Bias: " + bias);
    }

    public void predict(double[] features, double[] labels){
        if(!trained){
            System.out.println("ERROR! Attempting to predict before training.");
            return;
        }
        double total = 0;
        for(int r = 0; r < features.length; r++){
            total += weights[r] * features[r];
        }
        total += bias;
        labels[0] = total > 0 ? 1 : 0;
    }
}
