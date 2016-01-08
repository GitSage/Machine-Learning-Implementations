package DecisionTree;

import java.util.HashSet;

/**
 * Wrapper class for attributes.
 * This mostly just makes it easier to keep track of the name, location, and values of each attribute.
 * Helps avoid having to recalculate these things.
 */
public class AttributeWrapper {

    private int colPos;
    private String name;
	private HashSet<Double> values;

	public AttributeWrapper(String name, HashSet<Double> values, int colPos) {
		this.name = name;
		this.values = values;
		this.colPos = colPos;
	}

    public int getColPos() {
        return colPos;
    }

    public String getName() {
        return name;
    }

	public HashSet<Double> getValues() {
		return values;
	}

	public void setValues(HashSet<Double> values) {
		this.values = values;
	}
}
