package schiemannjeremy.linearalgebra;

import java.util.Arrays;
import java.util.function.Function;

public class Matrix {

	
//	public static void main(String[] args) {
//		Matrix m = new Matrix(2,2);
//		m.randomize(-5,5, true);
//		
//		m.print();
//		
//		Matrix n = new Matrix(2,2);
//		n.randomize(-5,5, true);
//		
//		n.print();
//		
//		
//		
//		Matrix.sub(m, n).print();
//		
//		//m.print();
//
//		
//		
//	
//	}
	
	
	//##############################################################################################################################
	
	private double[][] data;
	private int rows;
	private int columns;
	
	
	//Create a Matrix
	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		
		data = new double[rows][columns];	
	}
	
	public Matrix(double[][] data) {
		this.rows = data.length;
		this.columns = data[0].length;

		this.data = data;
	}
	
	public Matrix clone(){
		
		Matrix cloned = new Matrix(this.rows, this.columns);
		
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.columns; j++) {
				cloned.data[i][j] = this.data[i][j];
			}
		}
		
		return cloned;
	}
	
	public static Matrix fromArray(double[] array) {
		
		return new Matrix(array.length, 1).map((d, i, j) -> array[i]);
	}
	
	
	
	//nice to haves
	public Matrix randomize(int from, int to, boolean useInteger) {
		
		int delta = from-to;
		this.map((d) -> useInteger ? Math.floor(Math.random()*delta+to) : Math.random()*delta+to);
		
		return this;
	}
	
	//Mapping Functions
	public Matrix map(Function<Double, Double> function) {
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.columns; j++) {
				this.data[i][j] = function.apply(this.data[i][j]);
			}
		}
		
		return this;
	}
	
	public Matrix map(IndexedFunction<Double, Double> function) {
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.columns; j++) {
				this.data[i][j] = function.apply(this.data[i][j], i, j);
			}
		}
		
		return this;
	}
	
	public static Matrix map(Matrix m, Function<Double, Double> function) {
		
		Matrix mapped = new Matrix(m.rows, m.columns);
		
		for(int i = 0; i < m.rows; i++) {
			for(int j = 0; j < m.columns; j++) {
				mapped.data[i][j] = function.apply(m.data[i][j]);
			}
		}
		
		return mapped;
	}
	
	public static Matrix map(Matrix m, IndexedFunction<Double, Double> function) {
		
		Matrix mapped = new Matrix(m.rows, m.columns);
		
		for(int i = 0; i < m.rows; i++) {
			for(int j = 0; j < m.columns; j++) {
				mapped.data[i][j] = function.apply(m.data[i][j], i, j);
			}
		}
		
		return mapped;
	}
	
	//Object oriented mathematical Operations 
	public void add(Matrix m) {
		
		if(this.rows != m.rows || this.columns != m.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		this.map((d,i,j) -> d+m.data[i][j]);
	}
	
	public void add(int n) {
		
		this.map(d -> d+n);
	}
	
	public void add(double d) {
		
		this.map(d1 -> d1+d);
	}
	
	public void sub(Matrix m) {
		
		if(this.rows != m.rows || this.columns != m.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		this.map((d,i,j) -> d-m.data[i][j]);
	}
	
	public void sub(int n) {
		
		this.map(d -> d-n);
	}
	
	public void sub(double d) {
		
		this.map(d1 -> d1-d);
	}
	
	public void hadamardProduct(Matrix m) {
		
		if(this.rows != m.rows || this.columns != m.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		this.map((d, i, j) -> d*m.data[i][j]);
	}
	
	public void mul(int n) {
		
		this.map(d -> d*n);
	}
	
	public void mul(double d) {
		
		this.map(d1 -> d1*d);
	}
	
	//Static mathematical operations
	public static Matrix add(Matrix a, Matrix b) {
		
		if(a.rows != b.rows || a.columns != b.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		return new Matrix(a.rows, a.columns).map((d, i, j) -> a.data[i][j]+b.data[i][j]);
	}
	
	public static Matrix sub(Matrix a, Matrix b) {
		
		if(a.rows != b.rows || a.columns != b.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		return new Matrix(a.rows, a.columns).map((d, i, j) -> a.data[i][j]-b.data[i][j]);
	}
	
	public static Matrix mul(Matrix a, Matrix b) {
		
		if(a.columns != b.rows) throw new IllegalArgumentException("Incompatible matrix sizes");
		
		return new Matrix(a.rows, b.columns).map((d, i, j) ->{
			double sum = 0;
			for(int k = 0; k < a.columns; k++)
				sum += a.data[i][k] * b.data[k][j];
			
			return sum;
		});
	}
	
	public static Matrix transpose(Matrix m) {
		
		return new Matrix(m.columns, m.rows).map((d, i, j) -> m.data[j][i]);
	}
	
	//Other stuff #################################################################################################################
	public double[] toArray() {
		
		double[] arr = new double[this.rows*this.columns];
		
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.columns; j++) {
				arr[i*j+i] = this.data[i][j];
			}
		}
		
		return arr;
	}
	
	public void print() {
		System.out.println(this);
	}
	
	@Override
	public String toString() {
		
		String s = "";
		for(double[] row : this.data)
			s += Arrays.toString(row) + "\n";
		
		for(int i = 0; i < this.columns; i++)
			s += "-----";
		
		return s +"\n";
	}
	
	
}
