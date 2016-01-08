package Backpropagation;

/**
 * The abstract class for all nodes.
 */
public abstract class AbstractNode {

	protected double output;

	public AbstractNode() {
		output = 0;
	}
	
	public double getOutput() {
		return output;
	}
}
