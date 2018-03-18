# SimpleNeuralNetwork
A very simple and basic neural network implemented in java </br>
Neural network is a fully connected feed forward network using matrices.

<h2> Setup </h1>
<h3> The Neural Network </h3>
<p>
  To create a 2-layer neural network with 3 input neurons, 5 hidden neurons and 3 output neurons simply do this:

  <code></br>
    NeuralNetwork nn = new NeuralNetwork(3, 5, 2); 
  </code></br>
  
  If you want it to have more hidden layers you can just add them in the construcor like so:

  <code></br>
    NeuralNetwork nn = new NeuralNetwork(3, 5 ,5 10, 10, 2);
  </code></br>
  
  This will then create a NN with 3 input neurons , 2 output neurons and 4 hidden layers having 5, 5, 10, 10 neurons.
  </br>
 </p>
 <h3> The Activation Function </h3>
 <p>
  The ActivationFunction can be set using:
  
  <code></br>
    nn.setActivationFunction(ActivationFunction af);
 </code></br>
 
  I have included the most common ones in the ActivationFunction interface, but you can write your own by implementing said interface. </br>
  The included ones are: Identity, Binary Step, Sigmoid, TanH, ArcTan, Softsign, ReLU, Leaky ReLU, Sinusoid, Sinc and Gaussian
</p>
