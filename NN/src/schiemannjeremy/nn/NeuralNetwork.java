package schiemannjeremy.nn;

import schiemannjeremy.linearalgebra.Matrix;


/**
 * NeuralNetwork class for constructing, training and predicting data </br>
 * @author Jeremy Schiemann
 *
 */
public class NeuralNetwork {
	
	private final int[] LAYERS;
	private Matrix[] weights;
	private Matrix[] biases;
	
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
		Matrix[] outputs = new Matrix[this.LAYERS.length];
		outputs[0] = Matrix.fromArray(trainingSet.getInputs());
		
		Matrix[] noActivationFunctionApplied = outputs;
		
		//calc outputs
		for(int i = 1; i < outputs.length; i++) {
			outputs[i] = Matrix.mul(this.weights[i-1], outputs[i-1]);
			outputs[i].add(this.biases[i-1]);
			noActivationFunctionApplied[i] = outputs[i];
			outputs[i].map(x -> func.function(x));
		}
		
		//calc error
		Matrix[] errors = new Matrix[outputs.length-1];
		errors[errors.length-1] = Matrix.sub(targets, outputs[outputs.length-1]);
		
		for(int i = errors.length-1; i > 0; i--) {
			
			errors[i-1] = Matrix.mul(Matrix.transpose(this.weights[i]), errors[i]);
		}
	
		Matrix[] gradients = new Matrix[outputs.length-1];
		Matrix[] weight_deltas = new Matrix[outputs.length-1];
		
		for(int i = outputs.length-2; i >= 0; i--) {
			
			//calc gradients
			gradients[i] = Matrix.map(noActivationFunctionApplied[i+1], y -> func.derivatedFunction(y));
			gradients[i].hadamardProduct(errors[i]);
			gradients[i].mul(learningRate);
			
			//calc weight delta
			weight_deltas[i] = Matrix.mul(gradients[i], Matrix.transpose(outputs[i]));
			
			//adjust weight and biases
			this.weights[i].add(weight_deltas[i]);
			this.biases[i].add(gradients[i]);
		}
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
		Matrix[] outputs = new Matrix[this.LAYERS.length];
		outputs[0] = Matrix.fromArray(trainingSet.getInputs());
		Matrix[] noActivationFunctionApplied = outputs;
		
		//calc outputs
		for(int i = 1; i < outputs.length; i++) {
			outputs[i] = Matrix.mul(this.weights[i-1], outputs[i-1]);
			outputs[i].add(this.biases[i-1]);
			noActivationFunctionApplied[i] = outputs[i];
			outputs[i].map(x -> func.function(x));
		}
		
		//calc error
		Matrix[] errors = new Matrix[outputs.length-1];
		errors[errors.length-1] = Matrix.sub(targets, outputs[outputs.length-1]);
		
		for(int i = errors.length-1; i > 0; i--) {
			
			errors[i-1] = Matrix.mul(Matrix.transpose(this.weights[i]), errors[i]);
		}
		
		double sum = 0.0;
		for(int i = 0; i < errors[errors.length-1].toArray().length; i++) {
			sum += Math.pow(errors[errors.length-1].toArray()[i], 2);
		}
		
		sum /= errors[errors.length-1].toArray().length;
		 
		return Math.sqrt(sum);
	}
	
}
