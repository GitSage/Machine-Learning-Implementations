package Backpropagation;

public class OutputNeuron extends Neuron {

	double target;

    /**
     * Constructor for an output neuron.
     * @param id the id of this neuron.
     * @param learningRate the learning rate of this neuron.
     */
	public OutputNeuron(int id, double learningRate) {
        super(id, learningRate);
		target = 0;
	}

    /**
     * Calculates the error term.
     */
    @Override
    public void calcErrorTerm() {
		error = output * (target - output) * (1 - output);
	}

    /**
     * Sets the target of this neuron.
     * @param target the target of the neuron.
     */
	public void setTarget(double target) {
		this.target = target;
	}
    public double getTarget() { return target; }
}
