package Backpropagation;

import java.util.ArrayList;
import java.util.Random;
import Toolkit.Matrix;
import Toolkit.SupervisedLearner;


public class NeuralNet extends SupervisedLearner {

    private ArrayList<AbstractNode> inputs;
    private ArrayList<ArrayList<AbstractNode>> hLayers;
    private ArrayList<AbstractNode> outputs;

    private final double LEARNING_RATE;
    private double momentum;
    private int unitCount;
    private double weightMax;
    private double weightMin;
    private double mse;
    private double instanceNum;
    private int numCorrect;
    private Random rand;

    /**
     * Constructor for the neural net.
     * @param rand the randomizer that will be used to initialize weights.
     */
	public NeuralNet(Random rand) {
        this.rand = rand;

        inputs = new ArrayList<>();
        hLayers = new ArrayList<>();
        outputs = new ArrayList<>();

        LEARNING_RATE = 0.3;
        momentum = 0.1;
        unitCount = 0;
        weightMax = 0.05;
        weightMin = -0.05;
        mse = 0;
    }

    /**
     * Trains the neural net.
     * @param features The features used to train.
     * @param labels The targets.
     * @throws Exception
     */
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
        int numInputs = features.cols();
        int numHiddenNodes = 128;
        int numhLayers = 1;
        int numOutputs = labels.valueCount(0);

        initializeNetwork(numInputs, numHiddenNodes, numhLayers, numOutputs, features.rows());

        // System.out.println("input, hidden, hidden layers, output");
        // System.out.println(numInputs + ", " + numHiddenNodes + ", " + numhLayers + ", " + numOutputs);

        double bssf = 0;
        int bssfWindow = 50; // after this many epochs without improvement, stop.
        int epochsWithoutImprovement = 0; // the current number of epochs since last improvement
        int epochCounter = 0;
        double accuracy = 0;

        // run an epoch
        while(epochsWithoutImprovement < bssfWindow){
            epochCounter++;
            mse = 0;

            for (int e = 0; e < features.rows(); e++) {
                propForward(features.row(e), labels.row(e));
                mse += getError();
                propBack();
                updateWeights();
            }

            // check for improvement
            accuracy = getAccuracy();
            // System.out.println("Accuracy: " + accuracy);
            // System.out.println(epochCounter + " " + accuracy);

            if(accuracy > bssf){
                bssf = accuracy;
                epochsWithoutImprovement = 0;
                // System.out.println("Resetting epochs");
            }
            else{
                epochsWithoutImprovement++;
            }

            // System.out.print(i + ", ");
            // printhUnitWeights();
            // System.out.println(mse + " " + accuracy + " " + epochCounter);


        }
        System.out.println(mse + " " + accuracy + " " + epochCounter);


    }

    /**
     * Predicts the given labels using the given features.
     * @param features the features used to predict a value.
     * @param labels the value predicted using the features.
     * @throws Exception
     */
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = getTotalOutput(features);
	}

    public void initializeNetwork(int numIn, int numHidden, int numhLayers, int numOutput, double instanceNum) {
        this.instanceNum = instanceNum;

        // Initialize input layer
        for (int i = 0; i < numIn; i++) {
            AbstractNode inputNode = new InputNode();
            inputs.add(inputNode);
        }

        // Initialize hidden layers
        for (int layer = 0; layer < numhLayers; layer++) {
            ArrayList<AbstractNode> hLayer = new ArrayList<>();
            hLayers.add(hLayer);
            for (int neuron = 0; neuron < numHidden; neuron++) {
                unitCount++;
                AbstractNode hUnit = new HiddenNeuron(unitCount, LEARNING_RATE);
                hLayers.get(layer).add(hUnit);
            }
        }
        unitCount++;

        // Initialize output neurons
        for (int outputNeuron = 0; outputNeuron < numOutput; outputNeuron++) {
            AbstractNode outputUnit = new OutputNeuron(unitCount, LEARNING_RATE);
            unitCount++;
            outputs.add(outputUnit);
        }

        initializeWeights();
    }

    public void propForward(double[] inputs, double[] target) {
        sendInputs(inputs);
        sendTargets(target);

        for (int hiddenIndex = 0; hiddenIndex < hLayers.size(); hiddenIndex++) {
            ArrayList<AbstractNode> hLayer = hLayers.get(hiddenIndex);
            for (int j = 0; j < hLayer.size(); j++) {
                HiddenNeuron hNeuron = (HiddenNeuron) hLayer.get(j);
                hNeuron.receiveInputs();
                hNeuron.calcOutput();
            }
        }

        for (int outIndex = 0; outIndex < outputs.size(); outIndex++) {
            OutputNeuron neuronOut = (OutputNeuron) outputs.get(outIndex);
            neuronOut.receiveInputs();
            neuronOut.calcOutput();
        }
        double output = (double) getBestOutput();
        if (output == target[0])
            this.numCorrect++;
    }

    public double getTotalOutput(double[] myInputs) {
        sendInputs(myInputs);
        for (int i = 0; i < hLayers.size(); i++) {
            ArrayList<AbstractNode> hLayer = hLayers.get(i);
            for (int j = 0; j < hLayer.size(); j++) {
                HiddenNeuron hUnit = (HiddenNeuron) hLayer.get(j);
                hUnit.receiveInputs();
                hUnit.calcOutput();
            }
        }

        for (int i = 0; i < outputs.size(); i++) {
            OutputNeuron outNeuron = (OutputNeuron) outputs.get(i);
            outNeuron.receiveInputs();
            outNeuron.calcOutput();
        }
        return (double) getBestOutput();
    }
    
    private int getBestOutput() {
        int unitID = -1;
        double bestNet = -Double.MAX_VALUE;
        double net;

        for (int outputIndex = 0; outputIndex < outputs.size(); outputIndex++) {
            net = ((OutputNeuron) outputs.get(outputIndex)).getNet();

            if (net > bestNet) {
                bestNet = net;
                unitID = outputIndex;
            }
        }
        return unitID;
    }

    private void sendInputs(double[] inputs) {
        for (int inputIndex = 0; inputIndex < this.inputs.size(); inputIndex++) {
            InputNode inputNode = (InputNode) this.inputs.get(inputIndex);
            inputNode.setInput(inputs[inputIndex]);
        }
    }

    private void sendTargets(double[] target) {
        for (int i = 0; i < outputs.size(); i++) {
            OutputNeuron outputNeuron = (OutputNeuron) outputs.get(i);
            if (target[0] == i) // if target is a 2, when i is 2, set third outputUnit target to '1' (iris.arff)
                outputNeuron.setTarget(1);
            else
                outputNeuron.setTarget(0);
        }
    }

    public void propBack() {

        // calculate the output errors
        for (int i = 0; i < outputs.size(); i++) {
            OutputNeuron outputNeuron = (OutputNeuron) outputs.get(i);
            outputNeuron.calcErrorTerm();
            outputNeuron.distributeErrors();
        }

        // calculate the hidden errors
        for (int i = 0; i < hLayers.size() - 1; i++) {
            ArrayList<AbstractNode> hLayer = hLayers.get(hLayers.size() - (i + 1));
            for (int j = 0; j < hLayer.size(); j++) {
                HiddenNeuron hUnit = (HiddenNeuron) hLayer.get(j);
                hUnit.calcErrorTerm();
                hUnit.distributeErrors();
            }
        }
        ArrayList<AbstractNode> firsthLayer = hLayers.get(0);
        for (int i = 0; i <firsthLayer.size(); i++) {
            HiddenNeuron anotherhUnit = (HiddenNeuron) firsthLayer.get(i);
            anotherhUnit.calcErrorTerm();
        }
    }

    public void updateWeights() {
        for (int layerIndex = 0; layerIndex < hLayers.size(); layerIndex++) {
            ArrayList<AbstractNode> layer = hLayers.get(layerIndex);
            for (int neuronIndex = 0; neuronIndex < layer.size(); neuronIndex++) {
                ((Neuron) layer.get(neuronIndex)).updateWeights(momentum);
            }
        }
        for (int outputIndex = 0; outputIndex < outputs.size(); outputIndex++) {
            ((Neuron) outputs.get(outputIndex)).updateWeights(momentum);
        }
    }

    /**
     * Initialize all weights to random values between a max and min value.
     */
    private void initializeWeights () {
        // initialize output weights to random values
        ArrayList<AbstractNode> lasthLayer = hLayers.get(hLayers.size() - 1);
        for (int i = 0; i < outputs.size(); i++) {
            OutputNeuron outputNeuron = (OutputNeuron) outputs.get(i);
            for (int j = 0; j < lasthLayer.size(); j++) {
                AbstractNode hUnit = lasthLayer.get(j);
                outputNeuron.setWeightMap(hUnit, getRandomWeight());
                outputNeuron.setDeltaWeightMap(hUnit, 0);
            }
        }

        // initialize hidden weights to random values
        for (int k = 0; k < hLayers.size() - 1; k++) {
            ArrayList<AbstractNode> hLayer = hLayers.get(hLayers.size() - (k + 1));
            ArrayList<AbstractNode> nexthLayer = hLayers.get(hLayers.size() - (k + 2));
            for (int i = 0; i < hLayer.size(); i++) {
                HiddenNeuron hUnit = (HiddenNeuron) hLayer.get(i); //hidden unit from hidden layer
                for (int j = 0; j < nexthLayer.size(); j++) {
                    AbstractNode nexthUnit = nexthLayer.get(j);
                    hUnit.setWeightMap(nexthUnit, getRandomWeight());
                    hUnit.setDeltaWeightMap(nexthUnit, 0);
                }
            }
        }

        // initialize input weights
        ArrayList<AbstractNode> firsthLayer = hLayers.get(0);
        for (int i = 0; i < firsthLayer.size(); i++) {
            HiddenNeuron hUnit = (HiddenNeuron) firsthLayer.get(i);
            for (int j = 0; j < inputs.size(); j++) {
                AbstractNode inputNode = inputs.get(j);
                hUnit.setWeightMap(inputNode, getRandomWeight());
                hUnit.setDeltaWeightMap(inputNode, 0);
            }
        }
    }

    public double getAccuracy() {
        double classificationAccuracy = (numCorrect / instanceNum);
        numCorrect = 0;
        return classificationAccuracy;
    }

    private double getError() {
        double error = 0;
        for (int i = 0; i < outputs.size(); i++) {
            OutputNeuron outputUnit = (OutputNeuron) outputs.get(i);
            error += Math.pow(outputUnit.getTarget() - outputUnit.getOutput(), 2);
        }
        return error;
    }

    /**
     * Gets a random double between a max and a min.
     * @return a random double
     */
    private double getRandomWeight () {
        return weightMin + (rand.nextDouble() * (weightMax - weightMin));
    }
}
