package Backpropagation;

/**
 * An input node. Mostly a placeholder. Helps keep things organized.
 */
public class InputNode extends AbstractNode {

    /**
     * Sets the value of the input.
     * @param input the input value.
     */
	public void setInput(double input) {
		output = input;
	}
}
