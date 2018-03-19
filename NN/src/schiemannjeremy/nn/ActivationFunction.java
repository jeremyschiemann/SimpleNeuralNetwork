package schiemannjeremy.nn;

import java.io.Serializable;

public interface ActivationFunction extends Serializable{
	
	public static final ActivationFunction IDENTITY = new ActivationFunction(){
		/**
		 * 
		 */
		private static final long serialVersionUID = 343131331098998810L;

		@Override
		public double function(double x) {
			return x;
		}
		
		@Override
		public double derivatedFunction(double y) {
			return 1.0;
		}
	};
	
	public static final ActivationFunction BINARY_STEP = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -4703112928165586507L;

		@Override
		public double function(double x) {
			return x < 0.0 ? 0.0 : 1.0;
		}
		
		@Override
		public double derivatedFunction(double y) {
			return y != 0.0 ? 0.0 : -1000.0;
		}
	};
	
	public static final ActivationFunction SIGMOID = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = 7302154481222655379L;

		@Override
		public double function(double x) {
			return 1.0 / (1.0 + Math.exp(-x));
		}
		
		@Override
		public double derivatedFunction(double y) {
			double f = 1.0 / (1.0 + Math.exp(-y));
			return f * (1.0 - f);
		}
	};
	
	public static final ActivationFunction TAN_H = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -3485568832012538690L;

		@Override
		public double function(double x) {
			return Math.tanh(x);
		}
		
		@Override
		public double derivatedFunction(double y) {
			return 1.0 - Math.pow(Math.tanh(y), 2.0);
		}
	};
	
	public static final ActivationFunction ARC_TAN = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -3751083858351644294L;

		@Override
		public double function(double x) {
			return Math.atan(x);
		}
		
		@Override
		public double derivatedFunction(double y) {
			return 1.0 / (Math.pow(y, 2.0)+1.0);
		}
	};
	
	public static final ActivationFunction SOFTSIGN = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = 171814538462083654L;

		@Override
		public double function(double x) {
			return x / (1.0 + Math.abs(x));
		}
		
		@Override
		public double derivatedFunction(double y) {
			return y / Math.pow((1.0 + Math.abs(y)), 2.0);
		}
	};
	
	public static final ActivationFunction RELU = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -6893562610973122789L;

		@Override
		public double function(double x) {
			return x < 0.0 ? 0.0 : x; 
		}
		
		@Override
		public double derivatedFunction(double y) {
			return y < 0.0 ? 0.0 : 1.0;
		}
	};
	
	public static final ActivationFunction LEAKY_RELU = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = 1628466116427349330L;

		@Override
		public double function(double x) {
			return x < 0.0 ? 0.01*x : x; 
		}
		
		@Override
		public double derivatedFunction(double y) {
			return y < 0.0 ? 0.01 : 1.0;
		}
	};
	
	public static final ActivationFunction SINUSOID = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -8016274241633498862L;

		@Override
		public double function(double x) {
			return Math.sin(x);
		}
		
		@Override
		public double derivatedFunction(double y) {
			return Math.cos(y);
		}
	};
	
	public static final ActivationFunction SINC = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -2831356342516587053L;

		@Override
		public double function(double x) {
			return x == 0.0 ? 1.0 : (Math.sin(x) / x);
		}
		
		@Override
		public double derivatedFunction(double y) {
			return y == 0.0 ? 0.0 : ((Math.cos(y) / y) - (Math.sin(y) / Math.pow(y, 2.0)));
		}
	};
	
	public static final ActivationFunction GAUSSIAN = new ActivationFunction() {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -7857204655108665839L;

		@Override
		public double function(double x) {
			return Math.exp(-Math.pow(x, 2.0));
		}
		
		@Override
		public double derivatedFunction(double y) {
			return -2.0*y * Math.exp(-Math.pow(y, 2.0));
		}
	};
	

	public double function(double x);
	public double derivatedFunction(double y);
}
