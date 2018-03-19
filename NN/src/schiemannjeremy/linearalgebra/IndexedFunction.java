package schiemannjeremy.linearalgebra;

import java.io.Serializable;

public interface IndexedFunction<T, R> extends Serializable{
	R apply(T t, int n1, int n2);

}
