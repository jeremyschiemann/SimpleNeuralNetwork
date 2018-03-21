package schiemannjeremy.nn;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import schiemannjeremy.linearalgebra.Matrix;


/**
 * NeuralNetwork class for constructing, training and predicting data </br>
 * @author Jeremy Schiemann
 *
 */
public class NeuralNetwork implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 6241970177328998671L;
	private final int[] LAYERS;
	private Matrix[] weights;
	private Matrix[] biases;
	
	
	private Matrix[] errors;
	private Matrix[] gradients;
	private Matrix[] outputs;
	private Matrix[] outputsWithoutActivationFunction;
	private Matrix[] weightDeltas;
	
	
	private ActivationFunction func;
	
	/**
	 * Constructs a new NeuralNetwork with any amount of layers > 2 </br>
	 * The first value will be the amount of input neurons, last value will be the amount of output neurons. <br/>
	 * Everything in between will be the amount of hidden neurons </br> </br>
	 * 
	 * By default the activation function will be sigmoid ({@link schiemannjeremy.nn.ActivationFunction#SIGMOID}). </br>
	 * Weights will be 0, but can be randomized with {@link #randomizeWeights(int, int)} </br>
	 * Biases will be 0, but can be randomized with {@link #randomizeBiases(int, int)} </br>
	 * 
	 * @param layers - an array with the number of nodes per layer, implemented argument of variable length
	 */
	public NeuralNetwork(int... layers) {
		
		if(layers.length < 2) throw new IllegalArgumentException("NeuralNetwork needs at least 2 Layers, one for input, one for output");
		
		this.LAYERS = layers;
		
		weights = new Matrix[this.LAYERS.length-1];
		biases = new Matrix[this.LAYERS.length-1];
		
		for(int i = 0; i < this.LAYERS.length-1; i++) {
			weights[i] = new Matrix(this.LAYERS[i+1], this.LAYERS[i]);
			biases[i] = new Matrix(this.LAYERS[i+1], 1);
		}
		
		this.func = ActivationFunction.SIGMOID;
	}
	
	/**
	 * Set the activation function used by the neurons </br>
	 * Most common ones are already implemented in the ActivationFunction interface itself.
	 * @see schiemannjeremy.nn.ActivationFunction
	 * @param func - an ActivationFunction 
	 */
	public void setActivationFunction(ActivationFunction func) {
		this.func = func;
	}
	
	public ActivationFunction getActivationFunction() {
		return this.func;
	}
	
	/**
	 * Randomizes the weights in the give range excluding the upper limit.
	 * @param from - lower limit
	 * @param to - upper limit
	 * @throws IllegalArgumentException when the lower limit is higher than the upper limit
	 */
	public void randomizeWeights(int from, int to) {
		
		if(from > to) throw new IllegalArgumentException("lower limit must be less than upper limit");
		
		for(Matrix m : this.weights)
			m.randomize(from, to, false);
		
	}
	
	/**
	 * Randomizes the biases in the give range excluding the upper limit.
	 * @param from - lower limit
	 * @param to - upper limit
	 * @throws IllegalArgumentException when the lower limit is higher than the upper limit
	 */
	public void randomizeBiases(int from, int to) {	
		
		if(from > to) throw new IllegalArgumentException("lower limit must be less than upper limit");
		
		for(Matrix m : this.biases)
			m.randomize(from, to, false);
	}

	/**
	 * Feeds the give data to the neural network and return the result
	 * @param input_array - an array containing every value for the inputs
	 * @return an array containing every output
	 * @throws IllegalArgumentException when the size of the input array doesnt match the inputs of the neural network
	 */
	public double[] predict(double[] input_array) {
		
		if(input_array.length != this.LAYERS[0]) throw new IllegalArgumentException("" + this.LAYERS[0] + " inputs excpected, but " + input_array.length + " received");
		
		Matrix[] outputs = new Matrix[this.LAYERS.length];
		outputs[0] = Matrix.fromArray(input_array);
		
		
		for(int i = 1; i < outputs.length; i++) {
			outputs[i] = Matrix.mul(this.weights[i-1], outputs[i-1]);
			outputs[i].add(this.biases[i-1]);
			outputs[i].map(x -> func.function(x));
		}
		
		return outputs[outputs.length-1].toArray();

	}
	
	/**
	 *  Trains the neural network using backpropagation </br>
	 * 	During each iteration the neural network gets trained with one randomly picked training set. </br>
	 * 
	 * @param trainingData - the data used for training
	 * @param learningRate - the learning rate > 0
	 * @param iterations - the amount of training iterations
	 * @throws IllegalArgumentException if the learning rate is <= 0 or if the iterations are <= 0
	 */
	public void train(TrainingData trainingData, double learningRate, int iterations) {
		
		if(learningRate <= 0) throw new IllegalArgumentException("learning rate must be >0");
		if(iterations <= 0) throw new IllegalArgumentException("must at least do one iteration");
		
		for(int i = 0; i < iterations; i++) {
			this.train(trainingData.getRandomSet(), learningRate);					
		}
	}
	
	
	/**
	 * Trains the neural network with the given training set and learning rate
	 * @param trainingSet - a training set
	 * @param learningRate - the learning rate
	 * @throws IllegalArgumentException if the length of the inputs in the training set doesnt match the length of the inputs of the neural network or if the learning rate is <= 0
	 */
	public void train(TrainingSet trainingSet, double learningRate) {
		if(trainingSet.getInputs().length != this.LAYERS[0]) throw new IllegalArgumentException("" + this.LAYERS[0] + " inputs excpected, but " + trainingSet.getInputs().length + " received");
		if(learningRate <= 0) throw new IllegalArgumentException("learning rate must be >0");
		
		Matrix targets = Matrix.fromArray(trainingSet.getOutputs());
		this.outputs = new Matrix[this.LAYERS.length];
		this.outputs[0] = Matrix.fromArray(trainingSet.getInputs());
		
		this.outputsWithoutActivationFunction = this.outputs;
		
		calcOutputs();
		calcErrors(targets);
		calcDeltaGradientsAndApply(learningRate);
	}
	


	/**
	 * Calculates the mean squared error of every TrainingSet contained in the TrainingData object
	 * @param trainingData - The data used to calculate the error
	 * @return - the error
	 */
	public double calculateError(TrainingData trainingData) {
		
		double mean = 0;
	
		for(int data = 0; data < trainingData.size(); data++) 
			mean += this.calculateError(trainingData.getTrainingSet(data));
			
		return mean/trainingData.size()	;
	}

	
	private double calculateError(TrainingSet trainingSet) {
		
		if(trainingSet.getInputs().length != this.LAYERS[0]) throw new IllegalArgumentException("" + this.LAYERS[0] + " inputs excpected, but " + trainingSet.getInputs().length + " received");

		Matrix targets = Matrix.fromArray(trainingSet.getOutputs());
		if(this.outputs == null)
			this.outputs = new Matrix[this.LAYERS.length];
		
		this.outputs[0] = Matrix.fromArray(trainingSet.getInputs());
		
		//calc outputs
		calcOutputs();
		
		//calc error
		calcErrors(targets);
		
		double sum = 0.0;
		for(int i = 0; i < this.errors[this.errors.length-1].toArray().length; i++) {
			sum += Math.pow(this.errors[this.errors.length-1].toArray()[i], 2);
		}
		
		sum /= this.errors[this.errors.length-1].toArray().length;
		 
		return Math.sqrt(sum);
	}
	
	private void calcDeltaGradientsAndApply(double learningRate) {
		
		if(this.gradients == null)
			this.gradients = new Matrix[this.outputs.length-1];
		
		if(this.weightDeltas == null)
			this.weightDeltas = new Matrix[this.outputs.length-1];
		
		for(int i = this.outputs.length-2; i >= 0; i--) {
			
			//calc gradients
			this.gradients[i] = Matrix.map(this.outputsWithoutActivationFunction[i+1], y -> func.derivatedFunction(y));
			this.gradients[i].hadamardProduct(this.errors[i]);
			this.gradients[i].mul(learningRate);
			
			//calc weight delta
			this.weightDeltas[i] = Matrix.mul(this.gradients[i], Matrix.transpose(this.outputs[i]));
			
			//adjust weight and biases
			this.weights[i].add(this.weightDeltas[i]);
			this.biases[i].add(this.gradients[i]);
			
		}
	}
	
	private void calcErrors(Matrix targets) {
		
		if(this.errors == null)
			this.errors = new Matrix[outputs.length-1];
		
		this.errors[this.errors.length-1] = Matrix.sub(targets, this.outputs[this.outputs.length-1]);
		
		for(int i = this.errors.length-1; i > 0; i--) {
			
			this.errors[i-1] = Matrix.mul(Matrix.transpose(this.weights[i]), this.errors[i]);
		}
	}
	

	private void calcOutputs() {
		
		this.outputsWithoutActivationFunction = this.outputs;
		
		for(int i = 1; i < this.outputs.length; i++) {
			this.outputs[i] = Matrix.mul(this.weights[i-1], this.outputs[i-1]);
			this.outputs[i].add(this.biases[i-1]);
			this.outputsWithoutActivationFunction[i] = this.outputs[i];
			this.outputs[i].map(x -> func.function(x));
		}
	}
	
	/**
	 * Will write the NeuralNetwork object to the given file. </br>
	 * This method will create the necessary paths to create this file.
	 * @param neuralNetwork - the object which should be stored
	 * @param file - the file where the object should be stored
	 * @throws IOException - if anything goes wrong during writing...
	 */
	public static void save(NeuralNetwork neuralNetwork, File file) throws IOException {
		
		if(!file.exists()) file.getParentFile().mkdirs();
		
		ObjectOutputStream objO = new ObjectOutputStream(new FileOutputStream(file));
		objO.writeObject(neuralNetwork);
		
		objO.close();
		
	}

	/**
	 * Restores a saved NeuralNetwork object from a File
	 * @param file - the file where the object is stored
	 * @return - the restored NeuralNetwork object
	 * @throws IOException if the file doesnt exist, something goes wrong and so on...
	 */
	public static NeuralNetwork restore(File file) throws IOException {
		
		ObjectInputStream objI = new ObjectInputStream(new FileInputStream(file));
		
		NeuralNetwork nn = null;
		try {
			nn = (NeuralNetwork)objI.readObject();
			
		}catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
		objI.close();
		
		return nn;
	}
	
	
}
