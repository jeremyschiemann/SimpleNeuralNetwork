package schiemannjeremy.linearalgebra;

import java.util.Arrays;
import java.util.function.Function;

/**
 * Simple class for matrix mathematics
 * @author Jeremy Schiemann
 *
 */
public class Matrix {

	private double[][] data;
	private int rows;
	private int columns;
	
	
	/**
	 * Creates a matrix with the given rows and columns
	 * @param rows - amount of rows
	 * @param columns - amount of columns
	 */
	public Matrix(int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		
		data = new double[rows][columns];	
	}
	
	/**
	 * Creates a matrix using the given double-array
	 * @param data - the array used to create the matrix
	 */
	public Matrix(double[][] data) {
		this.rows = data.length;
		this.columns = data[0].length;

		this.data = data;
	}
	
	/**
	 * clones the matrix
	 * @return - the cloned matrix
	 */
	public Matrix clone(){
		
		Matrix cloned = new Matrix(this.rows, this.columns);
		
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.columns; j++) {
				cloned.data[i][j] = this.data[i][j];
			}
		}
		
		return cloned;
	}
	
	/**
	 * Creates a matrix out of an 1D-Array </br>
	 * Matrix will be an "array.length by 1"-matrix. 
	 * @param array - the array to create a matrix
	 * @return the matrix
	 */
	public static Matrix fromArray(double[] array) {
		
		return new Matrix(array.length, 1).map((d, i, j) -> array[i]);
	}

	//#############################################################################################################################################################
	
	/**
	 * Randomizes every value in the matrix with random values between the lower limit and (excluded) upper limit
	 * @param from - lwoer limit
	 * @param to - upper limit
	 * @param useInteger - true if random numbers should be integers
	 * @return - the matrix itself
	 * @throws IllegalArgumentException if from > to
	 */
	public Matrix randomize(int from, int to, boolean useInteger) {
		
		if(from > to) throw new IllegalArgumentException("lower limit cant be higher than upper limit");
		
		int delta = from-to;
		this.map((d) -> useInteger ? Math.floor(Math.random()*delta+to) : Math.random()*delta+to);
		
		return this;
	}
	
	/**
	 * Performs the function on every element
	 * @param function - a function that accepts a double and returns one
	 * @return - the matrix itself
	 */
	public Matrix map(Function<Double, Double> function) {
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.columns; j++) {
				this.data[i][j] = function.apply(this.data[i][j]);
			}
		}
		
		return this;
	}
	
	/**
	 *  Performs the function on every element. </br>
	 *  
	 *  The function gets called like this function(d, i, j) </br>
	 *  where d is the value of the matrix's cell, </br>
	 *  i is the index of the row and j is the index of the column. 
	 *  
	 * @param function - a function that accepts a double and two integers and returns a double
	 * @return - the matrix itself
	 */
	public Matrix map(IndexedFunction<Double, Double> function) {
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.columns; j++) {
				this.data[i][j] = function.apply(this.data[i][j], i, j);
			}
		}
		
		return this;
	}
	
	/**
	 * Does the same as {@link #map(Function)} but returns a new Matrix and doesnt change the original one.
	 * @param m - the original matrix (doesnt get changed)
	 * @param function - the mapping function
	 * @return - a new matrix
	 */
	public static Matrix map(Matrix m, Function<Double, Double> function) {
		
		Matrix mapped = new Matrix(m.rows, m.columns);
		
		for(int i = 0; i < m.rows; i++) {
			for(int j = 0; j < m.columns; j++) {
				mapped.data[i][j] = function.apply(m.data[i][j]);
			}
		}
		
		return mapped;
	}
	
	/**
	 * Does the same as {@link #map(IndexedFunction)} but returns a new Matrix and doesnt change the original one.
	 * @param m - the original matrix (doesnt get changed)
	 * @param function - the mapping function
	 * @return - a new matrix
	 */
	public static Matrix map(Matrix m, IndexedFunction<Double, Double> function) {
		
		Matrix mapped = new Matrix(m.rows, m.columns);
		
		for(int i = 0; i < m.rows; i++) {
			for(int j = 0; j < m.columns; j++) {
				mapped.data[i][j] = function.apply(m.data[i][j], i, j);
			}
		}
		
		return mapped;
	}
	
	//#############################################################################################################################################################
	
	/**
	 * Adds the matrix m to this matrix elementwise
	 * @param m - the matrix to add
	 * @throws IllegalArgumentException if matrix m isnt the same size as this matrix.
	 */
	public void add(Matrix m) {
		
		if(this.rows != m.rows || this.columns != m.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		this.map((d,i,j) -> d+m.data[i][j]);
	}
	
	/**
	 * Adds an integer to every element of this matrix.
	 * @param n - the integer to add
	 */
	public void add(int n) {
		
		this.map(d -> d+n);
	}
	
	/**
	 * Adds a double to every element of this matrix.
	 * @param d - the double to add
	 */
	public void add(double d) {
		
		this.map(d1 -> d1+d);
	}
	
	/**
	 * Subtracts the matrix m from this matrix elementwise
	 * @param m - the matrix to subtract
	 * @throws IllegalArgumentException if matrix m isnt the same size as this matrix.
	 */
	public void sub(Matrix m) {
		
		if(this.rows != m.rows || this.columns != m.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		this.map((d,i,j) -> d-m.data[i][j]);
	}
	
	/**
	 * Subtracts an integer from every element of this matrix.
	 * @param n - the integer to subtract
	 */
	public void sub(int n) {
		
		this.map(d -> d-n);
	}
	
	/**
	 * Subtracts a double from every element of this matrix.
	 * @param d - the double to subtract
	 */
	public void sub(double d) {
		
		this.map(d1 -> d1-d);
	}
	
	/**
	 * Multiplies the matrix m to this matrix elementwise
	 * @param m - the matrix to multiply
	 * @throws IllegalArgumentException if the matrix m isnt the same size as this matrix
	 */
	public void hadamardProduct(Matrix m) {
		
		if(this.rows != m.rows || this.columns != m.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		this.map((d, i, j) -> d*m.data[i][j]);
	}
	
	/**
	 * Multiplies an integer with every element of this matrix.
	 * @param n - the integer to multiply
	 */
	public void mul(int n) {
		
		this.map(d -> d*n);
	}
	
	/**
	 * Multiplies a double with every element of this matrix.
	 * @param d - the double to multiply
	 */
	public void mul(double d) {
		
		this.map(d1 -> d1*d);
	}
	
	/**
	 * Adds two matrices elementwise but returns a new element instead of changing one.
	 * @param a - first matrix
	 * @param b - second matrix
	 * @return the result of a + b elementwise.
	 * @throws IllegalArgumentException if the matrices are of different sizes
	 */
	public static Matrix add(Matrix a, Matrix b) {
		
		if(a.rows != b.rows || a.columns != b.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		return new Matrix(a.rows, a.columns).map((d, i, j) -> a.data[i][j]+b.data[i][j]);
	}
	
	/**
	 * Subtracts two matrices elementwise but returns a new element instead of changing one.
	 * @param a - first matrix
	 * @param b - second matrix
	 * @return result of a - b elementwise.
	 * @throws IllegalArgumentException if the matrices are of different sizes
	 */
	public static Matrix sub(Matrix a, Matrix b) {
		
		if(a.rows != b.rows || a.columns != b.columns) throw new IllegalArgumentException("Matrices must be of same size");
		
		return new Matrix(a.rows, a.columns).map((d, i, j) -> a.data[i][j]-b.data[i][j]);
	}
	
	/**
	 * Multiplies two matrices using proper matrix multiplication
	 * @param a - first matrix
	 * @param b - second matrix
	 * @return the resulting Matrix from a x b
	 * @throws IllegalArgumentException if the matrices can be multiplied
	 */
	public static Matrix mul(Matrix a, Matrix b) {
		
		if(a.columns != b.rows) throw new IllegalArgumentException("Incompatible matrix sizes");
		
		return new Matrix(a.rows, b.columns).map((d, i, j) ->{
			double sum = 0;
			for(int k = 0; k < a.columns; k++)
				sum += a.data[i][k] * b.data[k][j];
			
			return sum;
		});
	}
	
	/**
	 * Transposes the matrix m
	 * @param m - the matrix to transpose
	 * @return a new matrix which is m transposed
	 */
	public static Matrix transpose(Matrix m) {
		
		return new Matrix(m.columns, m.rows).map((d, i, j) -> m.data[j][i]);
	}
	
	//#############################################################################################################################################################
	
	/**

	 * Flattens the matrix into a double array containing row after row
	 * @return a double-array containing the matrix
	 */
	public double[] toArray() {
		
		double[] arr = new double[this.rows*this.columns];
		
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.columns; j++) {
				arr[i*j+i] = this.data[i][j];
			}
		}
		
		return arr;
	}
	
	/**
	 * Prints this Matrix
	 */
	public void print() {
		System.out.println(this);
	}
	
	/**
	 * Converts this matrix into a string
	 * @return the string of the matrix
	 */
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
