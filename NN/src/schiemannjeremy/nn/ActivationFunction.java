package schiemannjeremy.nn;

public interface ActivationFunction {
	
	public static final ActivationFunction IDENTITY = new ActivationFunction() {
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
