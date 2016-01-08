package Clustering;

import Toolkit.Matrix;

import java.text.DecimalFormat;
import java.util.HashSet;
import java.util.Set;

public class Cluster {
	public final int id;
	private double[] centroid; // the centroid. A single instance. Each value in in the array is an attribute value.
	private Set<Integer> instances;
	private static Matrix features;

	public Cluster(int id, double[] centroid, Matrix features) {
		this.id = id;
		this.centroid = centroid;
		instances = new HashSet<>();
		this.features = features;
	}

	public double[] getCentroid(){
		return centroid;
	}

	public Set<Integer> getInstances() { return instances; }

	public void addInstance(int instance) {
		instances.add(instance);
	}

	/**
	 * Calculates the best centroid for the cluster, then sets it.
     */
	public void updateCentroid() {

		int numAttributes = features.cols();

		// for each attribute, check the SSE

		for (int i = 0; i < numAttributes-1; i++) {

			int attrType = features.valueCount(i); // continuous (0) or nominal (>0)

			// if continuous, reassign the centroid to the average of everything in the attribute
			if (attrType == 0) {
				centroid[i] = features.averagedContinuousValue(i, instances);
			}
			// if nominal, reassign the centroid to the most common value in the instances of this cluster
			else {
				centroid[i] = features.mostCommonValue(i, instances);
			}
		}
	}

	public double getSSE () {
		// System.out.println("Getting SSE");
		double sse = 0;
		for (int instance : instances) {
			double[] x = features.row(instance);
			sse += KMeans.getDistance(x, centroid, features);
		}
		// System.out.println("Done getting SSE");
		return sse;
	}

	public void printCentroid(){
		String c = str(centroid[0], 0, centroid);
		for(int i = 1; i < centroid.length-1; i++){
			c += ", " + str(centroid[i], i, centroid);
		}
		System.out.println("Centroid " + id + " = " + c + ", num instances = " + instances.size());
	}

	public static void printInstance(double[] instance){
		String c = str(instance[0], 0, instance);
		for(int i = 1; i < instance.length-1; i++){
			c += ", " + str(instance[i], i, instance);
		}
		System.out.println(c);
	}

	public static String str(Double d, int col, double[] instance){
		if(d == Double.MAX_VALUE){
			return "?";
		}
		else{
			if(features.valueCount(col) == 0){
				DecimalFormat df = new DecimalFormat("#.000");
				return df.format(d).toString();
			}
			else{
				return features.attrValue(col, (int)instance[col]);
			}
		}
	}
}
