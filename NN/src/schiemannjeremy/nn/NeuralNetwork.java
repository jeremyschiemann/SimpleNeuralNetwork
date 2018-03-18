package schiemannjeremy.nn;

import schiemannjeremy.linearalgebra.Matrix;

public class NeuralNetwork {

//	public static void main(String[] args) {
//		
//		NeuralNetwork nn = new  NeuralNetwork(2,50,100, 50, 1);
//		nn.setActivationFunction(ActivationFunction.TAN_H);
//		
//		
//		nn.randomizeWeights(-1, 1);
//		nn.randomizeBiases(-1, 1);
//		
//		
//		
//		
//		for(double d : nn.predict(new double[] {42.3, -42.3}))
//			System.out.print("sin("+42.3+ ") equals " + d +  ", ");
//		
//		System.out.println();
//		
//
//		
//		//train
//		double[][] input_training = new double[10000][2]; 
//		double[][] output_training = new double[10000][1];
//		
//		for(int i = 0; i < input_training.length; i++) {
//			double x = Math.random();
//			input_training[i][0] = x*100;
//			input_training[i][1] = -x*100;
//			output_training[i][0] = Math.sin(x*100);
//		}
//
//
//			
//		for(int i = 0; i < 10; i ++) {
//			String d = String.format("%.16f", nn.calculateError(input_training, output_training));
//			System.out.println("" + i + ". error: " + d);
//			nn.train(input_training, output_training, 0.05, 50000);
//		}
//			
//		
//
//		for(double d : nn.predict(new double[] {42.3, -42.3}))
//			System.out.print("sin("+42.3+ ") equals " + d +  ", ");
//		
//	}
	
	private final int[] LAYERS;
	private Matrix[] weights;
	private Matrix[] biases;
	
	private ActivationFunction func;

	
	//TODO: aktivierungsfunktion änderbar machen
	//TODO: verschiedene Aktivierungsfunktionen implementieren
	
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
	
	public void setActivationFunction(ActivationFunction func) {
		this.func = func;
	}
	
	public void randomizeWeights(int from, int to) {
		
		for(Matrix m : this.weights)
			m.randomize(from, to, false);
		
	}
	
	public void randomizeBiases(int from, int to) {	
		
		for(Matrix m : this.biases)
			m.randomize(from, to, false);
	}
	
	
	
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
	

	public void train(double[][] inputs, double[][] outputs, double learningRate, int iterations) {
		
		if(inputs.length != outputs.length) throw new IllegalArgumentException("length of inputs and outputs arrays must be the same");
		if(learningRate <= 0) throw new IllegalArgumentException("learning rate must be >0");
		if(iterations <= 0) throw new IllegalArgumentException("must at least do one iteration");
		
		
		for(int i = 0; i < iterations; i++) {
				
			int index = (int)(Math.random()*inputs.length);
			this.train(inputs[index], outputs[index], learningRate);
		}
		
	}
	
	public void train(double[] input_array, double[] target_array, double learningRate) {
		
		if(input_array.length != this.LAYERS[0]) throw new IllegalArgumentException("" + this.LAYERS[0] + " inputs excpected, but " + input_array.length + " received");
		
		Matrix targets = Matrix.fromArray(target_array);
		Matrix[] outputs = new Matrix[this.LAYERS.length];
		outputs[0] = Matrix.fromArray(input_array);
		
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
	
	public double calculateError(double[][] input_data, double[][] output_data) {
		
		if(input_data.length != output_data.length) throw new IllegalArgumentException("length of inputs and outputs arrays must be the same");

		double mean = 0;
	
		for(int data = 0; data < input_data.length; data++) 
			mean += this.calculateError(input_data[data], output_data[data]);
			
		return mean/input_data.length;
	}
	
	private double calculateError(double[] input_array, double[] output_array) {
		
		if(input_array.length != this.LAYERS[0]) throw new IllegalArgumentException("" + this.LAYERS[0] + " inputs excpected, but " + input_array.length + " received");

		Matrix targets = Matrix.fromArray(output_array);
		Matrix[] outputs = new Matrix[this.LAYERS.length];
		outputs[0] = Matrix.fromArray(input_array);
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
