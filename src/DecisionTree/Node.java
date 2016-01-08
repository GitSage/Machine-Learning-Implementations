package DecisionTree;

import java.util.HashSet;
import java.util.LinkedHashMap;

import Toolkit.Matrix;

public class Node {

    private int nodeId;
	private LinkedHashMap<Double, Node> children;
	private AttributeWrapper attribute;
    private Label label;
    private static int nextId = 0;
	
	public Node () {
        this.nodeId = nextId++;
        children = new LinkedHashMap<>();
		attribute = new AttributeWrapper("l", new HashSet<Double>(), 0);
        label = new Label();
    }


    public AttributeWrapper getAttribute() {
        return attribute;
    }

    public void setAttribute(AttributeWrapper bestAttribute) {
        attribute = bestAttribute;
    }

    /**
     * Do a depth-first traversal of the tree, printing each node as it goes.
     */
	public void printTree(Matrix features) {
		String myLabel;
		if (children.isEmpty()) {
			myLabel = label.getStrValue();
		}
		else {
			myLabel = attribute.getName();
		}
		System.out.println("Node " + nodeId + ", label=" + myLabel);

        // if not a leaf node
		if (!children.isEmpty()) {
			for (double key : children.keySet()) {
                String edgeLabel = features.attrValue(attribute.getColPos(), (int)key);
				Node child = children.get(key);

				int childId = child.getNodeId();
                System.out.println("Edge " + nodeId + " to " + childId + ", label=" + edgeLabel);
				child.printTree(features);
			}
		}
	}

    public int getNodeId() {
        return nodeId;
    }

    /**
     * This function predicts the value of an attribute.
     * It recursively traverses the tree until the correct leaf node is found, then returns the value of the leaf node.
     * @param features
     * @param attribute
     * @return
     */
	public double predict(double[] features, int attribute) {
		double decision = 0;

        // if not a leaf
		if (!children.isEmpty()) {

            // for each child
            for (double branchVal : children.keySet()) {

                // check if the values match
                if (branchVal == features[attribute]) {
                    Node child = children.get(branchVal);
                    decision = child.predict(features, child.getAttribute().getColPos());
                }
            }
        }
        // it's a leaf, we've found our prediction!
		else {
            return label.getValue();
		}

		return decision;
	}

    public void addChild(double value, Node node) {
        children.put(value, node);
    }

    public void setLabel(String str, double val) {
        this.label = new Node.Label(str, val);
    }

    /**
     * Wrapper for a label, which consists of a string and a value.
     * The string is the string representation of the value of an attribute.
     * The value is the id of the attribute.
     */
    private class Label {

        private String str;
        private double val;

        public Label(String str, double val) {
            this.str = str;
            this.val = val;
        }

        public Label() {
            str = "n";
            val = 0;
        }

        public String getStrValue() {
            return str;
        }

        public double getValue() {
            return val;
        }
    }
}
