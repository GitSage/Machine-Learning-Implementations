package Clustering;

import java.util.*;
import java.util.Map.Entry;
import Toolkit.Matrix;

public class KMeans{

	/**
	 * Trains the KMeans clustering algorithm.
	 */
	public void train(Matrix features){

		int k = 6;

		// create random clusters, then sort all instances to whichever cluster is closest
		System.out.println("******************");
		System.out.println("Initial centroid assignment");
		System.out.println("******************");
		List<Cluster> clusters = initialClusterAssignment(k, features);
		for(int i = 0; i < k; i++){
			clusters.get(i).printCentroid();
		}
		// assign each instance to whatever centroid is closest
		reassignInstances(features, clusters);

		double temp = 0;
		for (Cluster cluster : clusters) {
			double cSSE = cluster.getSSE();
			temp += cSSE;
		}
		System.out.println("SSE: " + temp);

		// find sum squared distance of all clusters put together
		double totalSSE = 0;
		double prevTotalSSE;
		int numRounds = 2;
		boolean done = false;

		for (Cluster cluster : clusters) {
			totalSSE += cluster.getSSE();
		}

		// update centroid of each cluster and reassign features until no more improvement is found
		while (!done) {
			System.out.println("******************");
			System.out.println("Iteration " + numRounds++);
			System.out.println("******************");

			prevTotalSSE = totalSSE;
			totalSSE = 0;

			// update the centroid
			System.out.println("Updating Centroid");
			for (Cluster cluster : clusters) {
				cluster.updateCentroid();
			}
			for(int i = 0; i < k; i++){
				clusters.get(i).printCentroid();
			}

			// reassign all instances to whichever cluster is closest
			System.out.println("Updating Instances");
			reassignInstances(features, clusters);

			// check how much improvement we saw and whether to continue the loop
			for (Cluster cluster : clusters) {
				double cSSE = cluster.getSSE();
				System.out.println("Cluster " + cluster.id + ", " + cSSE);
				totalSSE += cSSE;
			}
			System.out.println("SSE: " + totalSSE);
			if (Math.abs(totalSSE - prevTotalSSE) < 0.00001) {
				done = true;
				System.out.println("SSE has converged, ceasing algorithm.");
			}

			// check for empty clusters
			for(Cluster cluster : clusters){
				if(cluster.getInstances().isEmpty()){
					System.out.println("Empty cluster, giving up.");
					System.exit(0);
				}
			}
		}

		printSilhouettes(clusters, features);
	}

	private void printSilhouettes(List<Cluster> clusters, Matrix features){
		System.out.println("Printing silhouettes");

		double total = 0;

		for(Cluster cluster : clusters){
			double[] centroid = cluster.getCentroid();
			double internal = 0; // avg distance of centroid to its points
			double external = 0; // avg distance of centroid to points in other clusters

			for(Cluster otherCluster : clusters){
				if(otherCluster == cluster){
					for(int instance : cluster.getInstances()){
						double distance = getDistance(features.row(instance), centroid, features);
						internal += distance;
					}
				}
				else{
					for(int instance : otherCluster.getInstances()){
						double distance = getDistance(features.row(instance), centroid, features);
						external += distance;
					}
				}
			}

			internal /= cluster.getInstances().size();
			external /= features.rows() - cluster.getInstances().size();
			total += 1-internal/external;

			System.out.println("Cluster " + cluster.id + ": " + (1-internal/external));
		}
		total /= clusters.size();
		System.out.println("Total: " + total);
	}

	/**
	 * Assigns each instance to whichever cluster is closest.
	 * @param features the features to be assigned
	 */
	private List<Cluster> initialClusterAssignment(int k, Matrix features) {

		// Define the clusters. Choose k random instances to be the centroids.
		List<Cluster> clusters = new ArrayList<>();
		for (int i = 0; i < k; i++) {
			int randInstance = (int) (Math.random() * features.rows());
			Cluster newCluster = new Cluster(i, features.row(randInstance), features);
			clusters.add(newCluster);
		}

//		for(int i = 0; i < k; i++){
//			clusters.add(new Cluster(i, features.row(i), features));
//		}

		return clusters;
	}

	/**
	 * Reassigns each instance to whichever cluster is closest.
	 * @param features the instances
	 * @param clusters the clusters
     */
	private void reassignInstances(Matrix features, List<Cluster> clusters) {
		Map<Integer, Cluster> assignment = new HashMap<>();

		// assign each instance to whatever centroid is closest
		for (int i = 0; i < features.rows(); i++) {
			int closestCluster = getClosestCluster(i, features, clusters);
			assignment.put(i, clusters.get(closestCluster));
		}

		// actually move the instances (and print)
		int format = 0;
		for(Cluster cluster : clusters){
			cluster.getInstances().clear();
		}
		for(Entry<Integer, Cluster> entry : assignment.entrySet()) {
			if (format != 0 && format % 10 == 0) {
				System.out.println();
			}
			format++;

			System.out.print(entry.getKey() + "=" + entry.getValue().id + " ");
			entry.getValue().addInstance(entry.getKey());
		}
		System.out.println();
	}

	private int getClosestCluster(int instance, Matrix features, List<Cluster> clusters){
		int correctCluster = -1;
		double min = Double.MAX_VALUE;
		for (Cluster cluster : clusters) {
			double dist = getDistance(features.row(instance), cluster.getCentroid(), features);
			if (dist < min) {
				min = dist;
				correctCluster = cluster.id;
			}
		}
		return correctCluster;
	}

	/**
	 * Calculates the euclidean distance between two sets of features.
	 * If a value is missing or the attribute is nominal, distance is 1.
	 * @param instance
	 * @param centroid
	 * @param features
     * @return a value representing the euclidean distance of the features.
     */
	public static double getDistance(double[] instance, double[] centroid, Matrix features) {
		// System.out.println("      Calcing distance");
//		Cluster.printInstance(instance);
//		Cluster.printInstance(centroid);


		double distance = 0;
		for (int i = 0; i < instance.length-1; i++) {
			if (instance[i] == Matrix.MISSING || centroid[i] == Matrix.MISSING) {
				distance += 1;
			}
			else {
				if (features.valueCount(i) == 0) {	// attribute is continuous
					// double tmp = Math.pow(instance[i] - centroid[i], 2);
					double tmp = Math.abs(instance[i] - centroid[i]);
					distance += tmp;
				}
				else {	// attribute is nominal
					if (instance[i] != centroid[i]) {
						distance += 1;
					}
				}
			}
		}
		// System.out.println("      " +distance);
		return distance;
	}

	public static void main(String[] args){
		KMeans km = new KMeans();
		Matrix mat = new Matrix();
		try{
			mat.loadArff("arff/abalone500.arff");
			mat.normalize();
			km.train(mat);
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}


}