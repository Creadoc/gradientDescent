package gradientDescentPack;

import java.util.ArrayList;

public class Obj {
	ArrayList<Double> xs = new ArrayList<Double>();
	int locationOfY;
	Double y;
	
	public Obj(ArrayList<Double> xs, int locationOfY, double y){
		this.xs = xs;
		this.locationOfY = locationOfY;
		this.y = y;
	}
	
	public Obj(ArrayList<Double> xs){
		this.xs = xs;
	}
	
	public Obj() {
		
	}
	
}
