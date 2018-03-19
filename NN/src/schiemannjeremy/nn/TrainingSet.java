package schiemannjeremy.nn;

import java.io.Serializable;

/**
 * A class that wraps the inputs and outputs for the training data
 * 
 * @author Jeremy Schiemann
 *
 */
public class TrainingSet implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = -2155279927467599160L;
	private double[] inputs;
	private double[] outputs;
	
	/**
	 * Constructs a training set with the given inputs and outputs </br>
	 * @param inputs - the inputs for the neural network
	 * @param outputs - the outputs the neural network should produce
	 */
	public TrainingSet(double[] inputs, double[] outputs) {
		this.inputs = inputs;
		this.outputs = outputs;
	}
	
	/**
	 * returns an array containing the values for the inputs of the neural network.
	 * @return - a double array
	 */
	public double[] getInputs() {
		return this.inputs;
	}
	
	
	/**
	 * returns an array containing the values of the outputs the neural network should produce.
	 * @return - a double array
	 */
	public double[] getOutputs() {
		return this.outputs;
	}

}
