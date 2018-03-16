package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {

	private int m_numOfThetas;
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		if (m_alpha == 0) {
			m_truNumAttributes = trainingData.numAttributes() - 1;
			// this will be the size of m_coefficients, its the number of the true attributes (+1 for theta0)
			m_numOfThetas = trainingData.numAttributes();
			findAlpha(trainingData);
		} else {
			m_truNumAttributes = 3;
			m_numOfThetas = 15;
		}
		m_coefficients = gradientDescent(trainingData);
	}

	public void setAlpha(double alpha) {
		m_alpha = alpha;
	}

	private void findAlpha(Instances data) throws Exception {
		System.out.println("Finding best alpha ... (might take a while)");
		double[] bestErrorForAlpha = new double[18];
		double newError, oldError, minError;
		for (int i = 0; i <= 17; i++) {
			guessCoefficients();
			m_alpha = Math.pow(3,-i);
			bestErrorForAlpha[i] = Double.MAX_VALUE;
			// run gradDescent with this alpha for 20000 iterations
			for (int j = 0; j < 20000; j+=100) {
				oldError = calculateMSE(data);
				for (int k = 0; k < 100; k++) {
					gradDescent(data);
				}
				newError = calculateMSE(data);
				if (newError <= oldError ) {
					bestErrorForAlpha[i] = newError;
				}
				else break;
			}
		}
		minError = Double.POSITIVE_INFINITY;
		int minIndex = 0;
		// get's the alpha with lowest error
		for (int i = 0; i < 18; i++) {
			if (bestErrorForAlpha[i] < minError) {
				minError = bestErrorForAlpha[i];
				minIndex = i;
			}
		}
		m_alpha = Math.pow(3,-minIndex);
	}

	public double getAlpha() {
		return m_alpha;
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData)
			throws Exception {
		guessCoefficients();
		double oldError = Double.MAX_VALUE;
		double error = calculateMSE(trainingData);

		// repeat until the error is small enough
		while (Math.abs(oldError-error) >= 0.003) {
			// every 100 steps check the difference between the errors as required
			for(int i = 0; i < 100; i++){
				gradDescent(trainingData);
			}
			oldError = error;
			error = calculateMSE(trainingData);
		}
		return m_coefficients;
	}

	/** doing one step of size alpha of the gradientDescent using the formulas we saw in class */
	private void gradDescent(Instances trainingData) throws Exception {
		double[] newCoefficients = new double[m_numOfThetas];
		for (int i = 0; i < m_numOfThetas; i++) {
			double sumOfErrors = 0;
			for (int j = 0; j < trainingData.size(); j++) {
				// partial derivatives with respect to theta0 (private case)
				if (i == 0) {
					sumOfErrors += getErrorOfInstance(trainingData.get(j));
				}
				// partial derivatives with respect to thetaJ (generic case)
				else if (trainingData.attribute(i - 1).weight() != 0) {
					sumOfErrors += getErrorOfInstance(trainingData.get(j)) * trainingData.get(j).value(i - 1);
				}
			}

			// calculating the new coefficient using the formula given in class
			if (i == 0 || trainingData.attribute(i - 1).weight() != 0) {
				newCoefficients[i] = m_coefficients[i] - m_alpha * (sumOfErrors / trainingData.size());
			}
		}
		// improving the coefficients
		m_coefficients = newCoefficients;
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		double innerProduct = 0;
		double val;
		for (int i = 0; i < m_numOfThetas-1; i++) {
			val = (instance.attribute(i).weight()==0) ? 0: instance.value(i);
				innerProduct += val * m_coefficients[i + 1];
		}
		// adding theta0
		innerProduct+= m_coefficients[0];
		return innerProduct;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances data) throws Exception {
		// i assume that the data isn't an empty set
		double error = 0, errorOfInstance = 0;
		for (int i = 0; i < data.size(); i++) {
			// using private method which calculates the error for single instance
			errorOfInstance = getErrorOfInstance(data.get(i));
			error += errorOfInstance*errorOfInstance;
		}
		// returns J(theta) using the formula we saw in class
		return error/(2*data.size());
	}

	private double getErrorOfInstance(Instance inst) throws Exception {
			return (regressionPrediction(inst)- inst.value(m_ClassIndex));
	}

	// i chose to init all coefficients to 0 for better efficiency
	private void guessCoefficients() {
		m_coefficients = new double[m_numOfThetas];

	}
	/** function for debugging and checking the new coefficients */
	public double[] getCoefficients() {
		return m_coefficients.clone();
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
