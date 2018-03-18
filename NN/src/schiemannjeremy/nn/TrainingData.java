package schiemannjeremy.nn;

import java.util.ArrayList;


/**
 * A class that contains all the training sets for training the neural network.
 * @author Jeremy Schiemann
 *
 */
public class TrainingData {
	
	private ArrayList<TrainingSet> trainingSets;
	
	/**
	 * Constructs a TrainingData object
	 */
	public TrainingData(){
		this.trainingSets = new ArrayList<>();
	}
	
	/**
	 * Constructs a TrainingData object and adds the training sets.
	 * @param training_data - the training sets which should be added.
	 */
	public TrainingData(TrainingSet[] training_data) {
		
		super();
				
		for(int i = 0; i < training_data.length; i++) {
			this.add(training_data[i]);
		}
	}
		
	/**
	 * Adds a new TrainingSet object to the collection
	 * @param trainingSet - the TrainingSet object that should be added
	 * @return - true if operation was successful
	 */
	public boolean add(TrainingSet trainingSet){
		
		if(this.trainingSets.add(trainingSet))
			return true;
		else 
			return false;
	}
	
	
	/**
	 * Returns the number of TrainingSet Objects in this list
	 * @return returns the number of TrainingSet Objects in this list
	 */
	public int size() {
		return this.trainingSets.size();
	}
	
	/**
	 * Returns the TrainingSet Object at the specified location
	 * @param index - the locations
	 * @return - the TrainingSet object
	 * @throws IndexOutOfBoundsException if the index is out of range
	 */
	public TrainingSet getTrainingSet(int index){
		
		return this.trainingSets.get(index);
	}
	
	/**
	 * returns a random TrainingSet Object.
	 * @return - the TrainingSet Object
	 */
	public TrainingSet getRandomSet(){
		
		int random = (int)(Math.random()*this.trainingSets.size());
		return this.trainingSets.get(random);	
	}

}
