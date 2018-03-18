package schiemannjeremy.examples;

import schiemannjeremy.nn.ActivationFunction;
import schiemannjeremy.nn.NeuralNetwork;
import schiemannjeremy.nn.TrainingData;
import schiemannjeremy.nn.TrainingSet;

public class XORProblem {

	public static void main(String[] args) {

		//create a NN with 2 inputs, 4 hidden and 1 output neuron and TanH as activation function.
		NeuralNetwork nn = new NeuralNetwork(2, 4, 1);
		nn.setActivationFunction(ActivationFunction.TAN_H);
		nn.randomizeBiases(-1, 1);
		nn.randomizeBiases(-1, 1);

		//Preparing some training data
		double[][] inputs = {
				{ 0, 0 },
				{ 1, 0 },
				{ 0, 1 },
				{ 1, 1 }
		};

		double[][] outputs = {
				{ 0 },
				{ 1 },
				{ 1 },
				{ 0 }
		};

		TrainingData td = new TrainingData();
		for (int i = 0; i < inputs.length; i++) {
			td.add(new TrainingSet(inputs[i], outputs[i]));
		}

		// Let the NN predict before training
		System.out.println("Prediction for [False, True] before training: ");
		double[] predicted = nn.predict(new double[] { 0, 1 });

		for (double d : predicted)
			System.out.print("  " + (Math.round(d) == 1.0 ? "True" : "False"));
		System.out.println();

		System.out.println(String.format("Error before training: %1.5f\n", nn.calculateError(td)));

		// train the nn 5000 times
		for(int x = 0; x < 5000; x ++)
			for(int i = 0; i < td.size(); i++)
				nn.train(td.getTrainingSet(i), 0.05);

		// let the nn predict again
		System.out.println("Prediction for [False, True] after training: ");
		predicted = nn.predict(new double[] { 0, 1 });

		for (double d : predicted)
			System.out.print("  " + (Math.round(d) == 1.0 ? "True" : "False"));
		System.out.println();

		System.out.println(String.format("Error after training: %1.5f\n", nn.calculateError(td)));

	}

}
