package schiemannjeremy.nn;

import java.util.LinkedList;

public class TrainingData {
	
	private LinkedList<double[]> training_input_list;
	private LinkedList<double[]> training_output_list;
	
	
	public TrainingData(){
		this.training_input_list =  new LinkedList<>();
		this.training_output_list = new LinkedList<>();
	}
	
	public TrainingData(double[][] input_data, double[][] output_data) {
		
		super();
		
		if(input_data.length != output_data.length) throw new IllegalArgumentException("Input and Output data must be of same length");
		
		for(int i = 0; i < input_data.length; i++) {
			this.add(input_data[i], output_data[i]);
		}
	
	}
		
	public boolean add(double[] input_data, double[] output_data){
		
		if(this.training_input_list.add(input_data) && this.training_output_list.add(output_data))
			return true;
		else return false;
	}
	
	public double[][] getInputData(){
		
		double[][] input_data = new double[this.training_input_list.size()][];
		
		
		for(int i = 0; i < input_data.length; i++) {
			input_data[i] = this.training_input_list.get(i);
		}
		
		return input_data;
	}
	
	public double[][] getOutputData(){
		
		double[][] output_data = new double[this.training_output_list.size()][];
		
		for(int i = 0; i < output_data.length; i++) {
			output_data[i] = this.training_output_list.get(i);
		}
		
		return output_data;
	}

}
