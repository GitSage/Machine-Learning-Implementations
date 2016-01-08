package KNearestNeighbors;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import Toolkit.Matrix;
import Toolkit.SupervisedLearner;

public class KNearestNeighbors extends SupervisedLearner{

	private Matrix features;
    private Matrix saveFeatures;
	private Matrix labels;
    boolean classified; // will be set to true for classify, or false for regression
    boolean weight;
	boolean reduce;
	private int k;

    /**
     *
     * @param k the number of neighbors
     */
	public KNearestNeighbors(int k) {
        this.k = k;
		this.k = 3;
        weight = false; // set to true for weighted, false for unweighted
		reduce = true; // set to true to reduce the algorithm.

        System.out.println("Running knn, k="+this.k);

    }

    /**
     * Trains the model.
     * @param features The features to be trained on
     * @param labels The labels to which the features belong
     * @throws Exception
     */
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		this.features = features;
        saveFeatures = new Matrix(features, 0, 0, features.rows(), features.cols());
        this.labels = labels;

        if(labels.valueCount(0) == 0){ // continuous, use regression
            System.out.println("Will use regression");
            classified = false;
        }
        else{
            System.out.println("Will use classification");
            classified = true;
        }
		if(classified && reduce){
			reduce();
		}
	}

    /**
     * Classifies an index.
     * @param testFeatures The features to be used
     * @param labels The label of the features
     * @throws Exception
     */
	@Override
	public void predict(double[] testFeatures, double[] labels) throws Exception {
        testFeatures = normalizeFeatures(testFeatures); // normalize

        if(classified){
            labels[0] = classify(testFeatures);
        }
        else{ // regression
            labels[0] = regression(testFeatures);
        }
	}

    /**
     * Classifies. Weighting is optional.
     * @param features the features to be classified on
     * @return A classification
     */
	private double classify(double[] features) {
        Instance instance = new Instance(features);
		List<PointPair> neighbors = getNeighbors(instance);

        if(weight){
            //System.out.println("Weighted classification.");
            return getWeightedClassification(neighbors);
        }
        else{
            //System.out.println("Unweighted classification.");
            return getUnweightedClassification(neighbors);
        }
	}

    /**
     * Predicts a continuous value. Weighting is optional.
     * @param features the features to be regressed on
     * @return a predicted continuous value
     */
    private double regression(double[] features){
        Instance instance = new Instance(features);
        List<PointPair> neighbors = getNeighbors(instance);

        if(weight){
            //System.out.println("Weighted regression.");
            return getWeightedRegression(neighbors);
        }
        else{
            //System.out.println("Unweighted regression.");
            return getUnweightedRegression(neighbors);
        }
    }

    /**
     * Normalize all continuous features to have a value in the range [0,1].
     * This will normalize BOTH the training set AND the test set.
     * Uses the formula fnew = (f - fmin) / (fmax - fmin)
     * @param testFeatures the features to be normalized.
     */
    private double[] normalizeFeatures(double[] testFeatures){
        features = saveFeatures;
        Matrix newFeatures = new Matrix(features, 0, 0, features.rows(), features.cols());

        double[] newTestFeatures = new double[features.cols()];
        // for each attribute
        for(int i = 0; i < features.cols(); i++){
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            // if it's not continuous, skip it
            if(features.valueCount(i) != 0){
                continue;
            }
            // find the minimum and maximum value
            for(int j = 0; j < features.rows(); j++){
                if(features.get(j, i) < min){
                    min = features.get(j, i);
                }
                if(features.get(j, i) > max){
                    max = features.get(j, i);
                }
            }

            if(testFeatures[i] < min){
                min = testFeatures[i];
            }
            if(testFeatures[i] > max){
                max = testFeatures[i];
            }

            // set to the new value according to the formula above
            double difference = max - min;
            for(int j = 0; j < features.rows(); j++) {
                double newVal = (features.get(j, i) - min) / difference;
                newFeatures.set(j, i, newVal);
            }
            newTestFeatures[i] = (testFeatures[i] - min) / difference;
        }
        features = newFeatures;

        return newTestFeatures;
    }

    /**
     * Gets an unweighted regression. Each neighbor is given equal weight, regardless of their distance
     * to the point.
     * @param neighbors a list of the k nearest neighbors and their distances.
     * @return
     */
	private double getUnweightedRegression(List<PointPair> neighbors) {
		double total = 0;
		for (PointPair pointPair : neighbors) {
			total += labels.get(pointPair.getIndex(), 0);
		}
        double average = total / k;
		return average;
	}

    /**
     * Performs a weighted regression. The strength of each neighbor depends on how far away it is.
     * This is according to the formula w = (1/dist(xq,xi))^2.
     * @param neighbors a list of the k nearest neighbors and their distances.
     * @return
     */
	private double getWeightedRegression(List<PointPair> neighbors) {
		double cw = 0;
		double cv = 0;
		for (PointPair neighbor : neighbors) {
            double w = 1 / (Math.pow(neighbor.getDistance(), 2));
			cv += w * labels.get(neighbor.getIndex(), 0);
			cw += w;
		}
		return (cv / cw);
	}

    /**
     * Performs an unweighted classification.
     * @param neighbors a list of the k nearest neighbors and their distances.
     * @return the value of the classification.
     */
	private double getUnweightedClassification(List<PointPair> neighbors) {
		HashMap<Double, Integer> freqs = new HashMap<>();

        // for each neighbor
		for (PointPair pointPair : neighbors) {
            // grab its vote
			int attr = pointPair.getIndex();
			Double vote = labels.get(attr, 0);

            // add the vote to the frequency map
			freqs.put(vote, freqs.get(vote) == null ? 1 : freqs.get(vote) + 1);
		}

        // find the maximum frequency
        int maxFreq = 0;
        double result = 0;
		for (Entry<Double, Integer> entry: freqs.entrySet()) {
			if (entry.getValue() > maxFreq) {
				maxFreq = entry.getValue();
				result = entry.getKey();
			}
		}
		
		return result;
	}

	/**
	 * Performs a weighted classification.
	 * The influence of each neighbor is inverse square on its distance.
	 * @param neighbors a list of the k nearest neighbors and their distances.
	 * @return the value of the classification.
	 */
	private double getWeightedClassification(List<PointPair> neighbors) {
		// hashmap from classification to a set of all neighbors with that classification
		HashMap<Double, HashSet<Double>>  distanceMap = new HashMap<>();

		for (PointPair neighbor : neighbors) {

			// get the classification
			double val = labels.get(neighbor.getIndex(), 0);

			// see if that classification is in the weight map already. If it isn't, add it.
			HashSet<Double> weights = distanceMap.get(val);
			if (weights == null) {
				weights = new HashSet<>();
			}

			// add the neighbor to that classification.
			weights.add(neighbor.getDistance());
			distanceMap.put(val, weights);
		}

		// find the best neighbor
		double top = 0;
		double result = 0;
		// for each classification
		for (Entry<Double, HashSet<Double>> entry : distanceMap.entrySet()) {
			// add up the distance of neighbors, using a 1/d^2 weighting function
			double currentTop = 0;
			Iterator<Double> iterator = entry.getValue().iterator();
			while (iterator.hasNext()) {
				currentTop += 1 / (Math.pow(iterator.next(), 2));
			}
			// check if it's the highest, store if it is
			if (currentTop > top) {
				top = currentTop;
				result = entry.getKey();
			}
		}
		return result;
	}

    /**
     * Gets the k nearest neighbors of the given index.
     * @param instance
     * @return a list of PointPair objects
     */
	private List<PointPair> getNeighbors(Instance instance) {
		List<PointPair> distances = new ArrayList<>();
		double distance;

        // For each feature
		for (int i = 0; i < features.rows(); i++) {
			distance = 0;
            double[] trainingInst = features.row(i);
            ArrayList<Double> instanceList = (ArrayList) instance.getValues();

            // for each attribute
			for (int j = 0; j < trainingInst.length; j++) {
				double trainingVal = trainingInst[j];
				double pointVal = instanceList.get(j);

                // if values are the same, do nothing
                if (trainingVal == pointVal) {}
                // if the value is missing, distance is 1
				else if (trainingVal == Double.MAX_VALUE || pointVal == Double.MAX_VALUE) {
					distance+=1;
                }
                // both values are present and not equal, so calculate the distance
				else {
					// for continuous values
					if (features.valueCount(j) == 0) {
                        distance += Math.abs(trainingVal - pointVal); // manhattan distance
                    }
                    // for nominal values
					else {
						distance += vdm(j, trainingVal, pointVal); // value distance metric
					}
				}
			}
			distances.add(new PointPair(distance, i));
		}

        // get the nearest neighbors
		Collections.sort(distances, new DistanceComparator());
		return distances.subList(0, k - 1);
	}

    /**
     * Gets the Value Distance Metric for nonimal values.
     * Uses the (rather annoying) formula found on the last slide of the index based learning slides.
	 * Note that this adds another loop through all rows in the training set. It's EXTREMELY slow (~x^3). A reasonable
	 * approximation for large training sets would be to simply say "if the nominal value is different, assign a
	 * distance of 1 (or .5, or similar).
     * @param attr the index of the attribute that we will perform the VDM on
     * @param trainingVal the nominal training value to be used by vdm
     * @param testVal the nominal value of the test point
     * @return
     */
	private double vdm(int attr, double trainingVal, double testVal) {
		double vdm = 0;

        // for each type of label
		for (int i = 0; i < labels.valueCount(0); i++) {
			int x = 0;
			int y = 0;
			int xc = 0;
			int yc = 0;

            // go through each row
			for (int j = 0; j < features.rows(); j++) {
				double val = features.get(j, attr);
				double outputClass = labels.get(j, 0);
				if (val == trainingVal) {
					x++;
					if((int)outputClass == i){
						xc++;
					}
				}
				if (val == testVal) {
					y++;
					if((int)outputClass == i){
						yc++;
					}
				}
			}

			// number of times value occurs / number of times it's the output class
			double ratioX = x != 0 ? xc/x : 0;
			double ratioY = y == 0 ? yc/y : 0;

			vdm += Math.pow(ratioX - ratioY, 2);
		} // end for each type of label
		return vdm;
	}

	/**
	 * Reduces the number of values in the training instance.
     * This is a leave-one-out reduction algorithm.
	 * Classifies each training value to see if it's right.
	 * If it is not, then it's excess and is removed.
	 */
	private void reduce() {
		System.out.print("Reducing... ");
		List<Integer> toRemove = new ArrayList<>();

		// for each feature
		for (int i = 0; i < features.rows(); i++) {
			// classify it. if the classification is incorrect, remove it.
			double classification = classify(features.row(i));
			if (classification != labels.get(i, 0)) {
				toRemove.add(i);
			}
		}

        System.out.println("Removed " + toRemove.size() + "instances.");

		// remove all the useless features.
		for (int i = toRemove.size() - 1; i >= 0; i--) {
			features.remove(toRemove.get(i));
			saveFeatures.remove(toRemove.get(i));
			labels.remove(toRemove.get(i));
		}
	}

	private class DistanceComparator implements Comparator<PointPair> {
		public int compare(PointPair a, PointPair b) {
			return a.getDistance().compareTo(b.getDistance());
		}
	}

	/**
     * Container to keep track of (a) the distance between two points and (b) the index of one of the points.
     */
	private class PointPair {
		private Double distance;
		private Integer index;
		
		public PointPair(Double distance, Integer index) {
			this.distance = distance;
			this.index = index;
		}

		public Double getDistance() {
			return this.distance;
		}

		public Integer getIndex() {
			return this.index;
		}
	}

    /**
     * 	An index will contain a list of attribute values
     */
    private class Instance {

        private List<Double> vals;

        public Instance(double[] values) {
            vals = new ArrayList<>();

            // add all elements
            // for some reason using Collections.addAll caused a bunch of problems.
            for (double val : values) {
                vals.add(val);
            }
        }

        public List<Double> getValues() {
            return vals;
        }

    }
}
