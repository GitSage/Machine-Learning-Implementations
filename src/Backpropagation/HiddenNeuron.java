package Backpropagation;

/**
 * A hidden node. Stores an error term.
 */
public class HiddenNeuron extends Neuron {

	private double errorSum;

    /**
     * Constructor for a HiddenNeuron. Sets errorSum to zero.
     * @param id the id of this hidden neuron.
     */
	public HiddenNeuron(int id, double learningRate) {
		super(id, learningRate);
		errorSum = 0;
	}

    /**
     * Stores the error term in error. Resets errorSum.
     */
    @Override
    public void calcErrorTerm() {
        error = output * errorSum * (1 - output);
        errorSum = 0;
    }

    /**
     * Adds to the sum of error.
     * @param weight the weight to be added.
     * @param errorTerm the term of the error to be added.
     */
	public void addError(double weight, double errorTerm) {
		errorSum += weight * errorTerm;
	}
}
