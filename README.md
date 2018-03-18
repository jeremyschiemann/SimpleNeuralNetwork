# SimpleNeuralNetwork
A very simple and basic neural network implemented in java </br>
Neural network is a fully connected feed forward network using matrices.

<h1> Setup </h1>
<p>
  To create a 2-layer neural network with 3 input neurons, 5 hidden neurons and 3 output neurons simply do this:

  <code></br>
    NeuralNetwork nn = new NeuralNetwork(3, 5, 2); 
  </code></br>
  
  If you want it to have more hidden layers you can just add them in the construcor like so:

  <code></br>
    NeuralNetwork nn = new NeuralNetwork(3, 5 ,5 10, 10, 2);
  </code></br>
  
  This will then crearte a NN with 3 input neurons , 2 output neurons and 4 hidden layers having 5, 5, 10, 10 neurons.
 </p>
 
 <h1>
