package DecisionTree;

import Toolkit.*;
import java.util.*;

public class DecisionTree extends SupervisedLearner {

	private Node root;

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
        root = buildTree(features, labels, null);
        // root.printTree(features);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = root.predict(features, root.getAttribute().getColPos());
	}

    public Node buildTree(Matrix features, Matrix targetAttributes, HashSet<AttributeWrapper> attributes) {

        Node root = new Node();

        // unless the attribute wrappers are given to us, set them up ourselves
        if(attributes == null){
            attributes = new HashSet<>();
            for (int i = 0; i < features.cols(); i++) {
                attributes.add(new AttributeWrapper(features.attrName(i), getAttrValue(features, i), i));
            }
        }

        if (targetAttributes.columnMax(0) == targetAttributes.columnMin(0) || attributes.isEmpty()) {
            double firstAttr = targetAttributes.get(0, 0);
            String str = targetAttributes.attrValue(0, (int)firstAttr);
            root.setLabel(str, firstAttr);
        }
        else {
            AttributeWrapper myAttr = chooseSplitAttribute(features, targetAttributes, attributes);

            root.setAttribute(myAttr);

            for (double value : myAttr.getValues()) {
                Matrix[] myFeatures = getTrimmedMatrices(myAttr.getColPos(), (int)value, features, targetAttributes);

                if (myFeatures[0].rows() == 0) {
                    Node newNode = new Node();
                    int mostCommonVal = (int)targetAttributes.mostCommonValue(0);
                    newNode.setLabel(targetAttributes.attrValue(0, mostCommonVal), mostCommonVal);
                    root.addChild(value, newNode);
                }

                else {
                    attributes.remove(myAttr);

                    root.addChild(value, buildTree(myFeatures[0], myFeatures[1], attributes));

                    attributes.add(myAttr);
                }
            }
        }
        return root;
    }

    private HashSet<Double> getAttrValue(Matrix attributes, int attribute) {
        HashSet<Double> values = new HashSet<>();
        int numVals = attributes.valueCount(attribute);
        for (int i = 0; i < attributes.rows(); i++) {
            if (attributes.get(i, attribute) != Double.MAX_VALUE)
                values.add(attributes.get(i, attribute));
            if (values.size() == numVals)
                break;
        }
        return values;
    }

    private AttributeWrapper chooseSplitAttribute(Matrix features, Matrix targets, HashSet<AttributeWrapper> attributes) {

        // ordinary find best
        double highestGain = -10;
        int highestAttribute = -10;
        for (AttributeWrapper attribute : attributes) {
            int col = attribute.getColPos();

            // calculate the gain of the attribute, then find highest gain
            double gain = getGain(col, features, targets);
            if (gain >= highestGain) {
                highestGain = gain;
                highestAttribute = col;
            }
        }

        // This is my experiment.

        /* double highestGain = 100;
        int highestAttribute = 100;
        for (AttributeWrapper attribute : attributes) {
            int attributeColPos = attribute.getColPos();
            double gain = getGain(attributeColPos, features, targets, attributes);
            if (gain <= highestGain) {
                highestGain = gain;
                highestAttribute = attributeColPos;
            }
        } */

        for (AttributeWrapper attribute : attributes) {
            if (attribute.getColPos() == highestAttribute) {
                return attribute;
            }
        }
        return null;
    }




    /////////////////////////////
    // Matrix utility functions
    /////////////////////////////


    public ArrayList<int[]> getNumTimesValueOccurs(int attrNum, Matrix matrix, Matrix targets) {
        ArrayList<int[]> numValuesOccur = new ArrayList<>();
        int num = targets.valueCount(0);
        int values = matrix.valueCount(attrNum);

        // Ugh, all the variables here gave me a headache. Did a terrible job naming them too. Ah well.
        // for each value
        for (int i = 0; i < values; i++) {
            int[] numVals = new int[num];

            // for each row in the matrix
            for (int j = 0; j < matrix.rows(); j++) {
                int targetValue = (int)targets.get(j, 0);

                // if it's missing, fill it in
                if (matrix.get(j, attrNum) == Double.MAX_VALUE) {
                    matrix.set(j, attrNum, matrix.mostCommonValue(attrNum, targets.get(j, 0)));
                }

                // if it's not missing, increment
                else if ( i == (int)(matrix.get(j, attrNum)) ) {
                    numVals[targetValue]++;
                }
            }

            numValuesOccur.add(numVals);
        }
        return numValuesOccur;
    }

    public int[] getNumTimesAttrOccurs(Matrix targets) {
        int valCount = targets.valueCount(0);
        int[] counts = new int[valCount];

        // for each target
        for (int i = 0; i < valCount; i++) {
            counts[i] = 0;

            // for each row
            for (int j = 0; j < targets.rows(); j++) {
                if ((double) i == targets.get(j, 0)) {
                    counts[i]++;
                }
            }
        }

        return counts;
    }

    public Matrix[] getTrimmedMatrices(int attrs, int val, Matrix matrix, Matrix targets) {
        Matrix[] matricesTrimmed = new Matrix[2];

        matricesTrimmed[0] = new Matrix(matrix, attrs, val, matrix.rows(), matrix.cols(), null);
        matricesTrimmed[1] = new Matrix(targets, 0, -1, targets.rows(), targets.cols(),
                matricesTrimmed[0].getToCopy());

        return matricesTrimmed;
    }


    /**
     * Calculates the gain of the algorithm.
     * Gets the entropy, then does a simple division on the entropy.
     * @param attribute
     * @param features
     * @param targets
     * @return
     */
    private double getGain(int attribute, Matrix features, Matrix targets) {
        ArrayList<int[]> numOccurrences = getNumTimesValueOccurs(attribute, features, targets);

        int[] targetOccurrences = getNumTimesAttrOccurs(targets);
        double totalOccurrences = 0.0;
        double value = 0;

        for (int i = 0; i < targetOccurrences.length; i++) {
            totalOccurrences += targetOccurrences[i];
        }

        // Getting real tired of working on this thing
        double[] valSum = new double[numOccurrences.size()];
        for (int i = 0; i < numOccurrences.size(); i++) {
            for (int j = 0; j < targetOccurrences.length; j++) {
                valSum[i] += numOccurrences.get(i)[j];
            }
        }

        for (int i = 0; i < numOccurrences.size(); i++) {
            // I screwed this equation up like four times
            value += -valSum[i] * getEntropy(numOccurrences.get(i)) / totalOccurrences;
        }
        return getEntropy(targetOccurrences) + value; // final gain calculation

    }

    /**
     * Calculates the entropy of the relevant rows of data. Uses an equation for entropy that I found on the Internet.
     * @param occurrences
     * @return
     */
    private double getEntropy(int[] occurrences) {
        double numOccurrences = 0;
        int counter = 0;
        double entropy = 0;

        for (int i = 0; i < occurrences.length; i ++) {
            if ((double)occurrences[i] == 0) {
                counter++;
            }
            numOccurrences += (double)occurrences[i];
        }

        if (counter + 1 == occurrences.length) {
            return entropy;
        }
        if (numOccurrences == 0) {
            return entropy;
        }

        for (int i = 0; i < occurrences.length; i++) {
            if (occurrences[i] != 0) {
                entropy += -(occurrences[i] / numOccurrences) *
                        (Math.log10(occurrences[i] / numOccurrences) / Math.log10(2));
            }
        }
        return entropy;
    }
}
