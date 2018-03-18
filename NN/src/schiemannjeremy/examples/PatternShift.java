package schiemannjeremy.examples;

import schiemannjeremy.nn.ActivationFunction;
import schiemannjeremy.nn.NeuralNetwork;
import schiemannjeremy.nn.TrainingData;
import schiemannjeremy.nn.TrainingSet;

/**
 * A Class to demonstrate the NN
 * @author Jeremy Schiemann
 *
 */
public class PatternShift {

	public static void main(String[] args) {
		
		//Setting up the NN to use 3 inputs, 3 outputs
		NeuralNetwork nn = new NeuralNetwork(3, 3);
		nn.randomizeWeights(0, 1);
		nn.randomizeBiases(0, 1);
		
		//Make it use the Sigmoid function
		nn.setActivationFunction(ActivationFunction.SIGMOID);
		
		
		//We want the NN to shift the numbers to the right
		
		//Creating some training data
		double[][] inputs = {
				{0,0,0},
				{1,1,1},
				{1,0,0},
			//	{0,1,0},	we use this one to test the nn later
				{0,0,1},
				{1,0,1},
				{0,1,1},
				{1,1,0}
		};
		
		//creating the output data
		double[][] outputs = {
				{0,0,0},
				{1,1,1},
				{0,1,0},
			//	{0,0,1},	same here
				{1,0,0},
				{1,1,0},
				{1,0,1},
				{0,1,1}
		};
		
		
		//adding the data to the trainingdata
		TrainingData td = new TrainingData();
		for(int i = 0; i < inputs.length; i++) {
			td.add(new TrainingSet(inputs[i], outputs[i]));
		}
		
		
		//Let the NN predict before training 
		System.out.println("Prediction for 0,0,1 before training: ");
		double[] predicted = nn.predict(new double[]{0,0,1});
		
		for(double d : predicted)
			System.out.print("  " +Math.round(d));
		System.out.println();
		
		System.out.println(String.format("Error before training: %1.5f\n", nn.calculateError(td)));
		
		//train the nn 5000 times
		nn.train(td, 0.5, 5000);
		
		//let the nn predict again
		System.out.println("Prediction for 0,0,1 after training: ");
		predicted = nn.predict(new double[]{0,0,1});
		
		for(double d : predicted)
			System.out.print("  " +Math.round(d));
		System.out.println();
		
		System.out.println(String.format("Error after training: %1.5f\n", nn.calculateError(td)));
		
	}

}
