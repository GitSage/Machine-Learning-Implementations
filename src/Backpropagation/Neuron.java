package Backpropagation;

import java.util.*;

/**
 * The base for hidden neurons and output neurons.
 * This class handles most of the functions of all neurons.
 * Stores weights and handles the passing of data from one node to the next.
 */
public abstract class Neuron extends AbstractNode {

    protected int id;
    protected double error;
    protected double net;
    protected final double LEARNING_RATE;

    protected ArrayList<Double> inputs;

    protected ArrayList<Double> weights;
    protected Map<AbstractNode, Double> nodesToWeights;
    protected double bias;
    protected double biasDelta;
    protected Map<AbstractNode, Double> nodesToDeltaWeights;

    /**
     * Sets all values to defaults.
     */
    public Neuron(int id, double learningRate) {
        this.id = id;
        net = 0;
        error = 0;
        this.LEARNING_RATE = learningRate;

		inputs = new ArrayList<>();

		weights = new ArrayList<>();
		nodesToWeights = new LinkedHashMap<>();

        bias = 0;
        biasDelta = 0;
        nodesToDeltaWeights = new LinkedHashMap<>();
    }

	public double getNet() { return net; }
    public void setWeightMap (AbstractNode node, double weight) { nodesToWeights.put(node, weight); }
    public void setDeltaWeightMap (AbstractNode node, double initWeight) { nodesToDeltaWeights.put(node, initWeight); }

    /**
     * Empties weights and inputs, then adds the node's outputs and weights to inputs and weights respectively.
     */
	public void receiveInputs() {
        weights.clear();
        inputs.clear();

        for(AbstractNode node : nodesToWeights.keySet()){
            inputs.add(node.getOutput());
            weights.add(nodesToWeights.get(node));
        }
	}

    /**
     * Calculates the output of this neuron.
     */
	public void calcOutput() {
        // get the total net
        net = 0;
        for (int i = 0; i < inputs.size(); i++) {
            net += inputs.get(i) * weights.get(i);
        }
        net += bias;

        // from the net, get the output
		output = 1 / (1 + Math.exp(-net));
	}

    /**
     * Distributes error values to all child nodes.
     */
    public void distributeErrors() {
        for (AbstractNode node : nodesToWeights.keySet()) {
            ((HiddenNeuron)node).addError(nodesToWeights.get(node), error);
        }
    }

    /**
     * Updates weights using the delta formula with the given momentum.
     * @param momentum the momentum to be used while calculating the weights.
     */
	public void updateWeights(double momentum) {
		biasDelta = (error * LEARNING_RATE) + (biasDelta * momentum);
		bias += biasDelta;
		for (AbstractNode node : nodesToWeights.keySet()) {
            double delta_weight = (LEARNING_RATE * error * node.getOutput()) + (momentum * nodesToDeltaWeights.get(node));
            nodesToWeights.put(node, nodesToWeights.get(node) + delta_weight);
			nodesToDeltaWeights.put(node, delta_weight);
		}
	}

    /**
     * Prints all weights.
     */
	public void printWeights() {
		String weights = "";
		for (AbstractNode node : nodesToWeights.keySet()) {
			double weight = nodesToWeights.get(node);
			weights += Double.toString(weight);
			weights += ", ";
		}
		System.out.println(weights);
	}

    /**
     * Calculates the error term.
     */
    public abstract void calcErrorTerm();
}
