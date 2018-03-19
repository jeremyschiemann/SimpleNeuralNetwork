package schiemannjeremy.nn;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.FileStore;
import java.nio.file.Files;
import java.util.ArrayList;

import javax.swing.filechooser.FileFilter;


/**
 * A class that contains all the training sets for training the neural network.
 * @author Jeremy Schiemann
 *
 */
public class TrainingData implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1862961912768893897L;
	private static final String DATA_ENDING = ".ntd";
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
	
	
	/**
	 * Will write the TrainingData object to the given file. </br>
	 * This method will create the necessary paths to create this file.
	 * @param trainingData - the object which should be stored
	 * @param file - the file where the object should be stored
	 * @throws IOException - if anything goes wrong during writing...
	 */
	public static void save(TrainingData trainingData, File file) throws IOException {
		
		if(!file.exists()) file.getParentFile().mkdirs();
		
		ObjectOutputStream objO = new ObjectOutputStream(new FileOutputStream(file));
		objO.writeObject(trainingData);
		
		objO.close();
		
	}

	/**
	 * Restores a saved TrainingData object from a File
	 * @param file - the file where the object is stored
	 * @return - the restored TrainingData object
	 * @throws IOException if the file doesnt exist, something goes wrong and so on...
	 */
	public static TrainingData restore(File file) throws IOException {
		
		ObjectInputStream objI = new ObjectInputStream(new FileInputStream(file));
		
		TrainingData td = null;
		try {
			td = (TrainingData)objI.readObject();
			
		}catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
		objI.close();
		
		return td;
	}
	
}
