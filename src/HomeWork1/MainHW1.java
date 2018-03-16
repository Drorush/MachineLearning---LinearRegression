package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW1 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 *
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		//load data
		Instances trainData = loadData("wind_training.txt");
		Instances testData = loadData("wind_testing.txt");

		//find best alpha and build classifier with all attributes
		LinearRegression lr = new LinearRegression();
		double bestAlpha;
		lr.buildClassifier(trainData);
		bestAlpha = lr.getAlpha();
		System.out.println("The chosen alpha is: " + bestAlpha);
		System.out.println("Training error with all features is: " + lr.calculateMSE(trainData));
		System.out.println("Test error with all features is: " + lr.calculateMSE(testData));

		//build classifiers with all 3 attributes combinations
		double minError = Double.MAX_VALUE;
		double error, testError = 0;
		String bestAttributes = "";
		// training every permutation
		for (int i = 0; i < trainData.numAttributes()-3; i++) {
			for (int j = i + 1; j < trainData.numAttributes()-2; j++) {
				for (int k = j + 1; k < trainData.numAttributes()-1; k++) {
					for (int m = 0; m < trainData.numAttributes()-1; m++) {
						if (m != i && m != j && m != k) {
							trainData.setAttributeWeight(m, 0);
							testData.setAttributeWeight(m, 0);
						} else {
							trainData.setAttributeWeight(m, 1);
							testData.setAttributeWeight(m, 1);
						}
					}
					// training a permutation with the alpha we found a step before
					lr.setAlpha(bestAlpha);
					lr.buildClassifier(trainData);
					error = lr.calculateMSE(trainData);
					System.out.println("{" + trainData.attribute(i).name() + "," + trainData.attribute(j).name() + "," + trainData.attribute(k).name() + "} Training error: "+error);
					if (error < minError) {
						bestAttributes = "{" + trainData.attribute(i).name() + "," + trainData.attribute(j).name() + "," + trainData.attribute(k).name() + "}";
						minError = error;
						testError = lr.calculateMSE(testData);
					}
				}
			}
		}
		System.out.println("Training error the features " + bestAttributes + "}: " + minError);
		System.out.println("Test error the features " + bestAttributes + "}: " + testError);
	}
}
