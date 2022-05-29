package gradientDescentPack;

//Jimmy Collins
// AI PS4
// 3/4/2021

import java.io.*;
import java.util.ArrayList;

public class PS4 {

	ArrayList<Obj> trainObjects;
	ArrayList<Obj> testObjects;
	ArrayList<ArrayList<Double>> trainVals;
	ArrayList<ArrayList<Double>> testVals;
	ArrayList<Double> trainMeans;
	ArrayList<Double> testMeans;
	ArrayList<Double> trainSigmas;
	ArrayList<Double> testSigmas;
	ArrayList<Double> trainVars;
	ArrayList<Double> testVars;
	ArrayList<ArrayList<Double>> trainZs;
	ArrayList<ArrayList<Double>> testZs;
	ArrayList<Double> trainYs;
	ArrayList<Double> testYs;
	ArrayList<ArrayList<Double>> trainWeights;
	ArrayList<ArrayList<Double>> testWeights;
	Double[] weigh;
	ArrayList<Double> costs;
	ArrayList<Double> lossList;
	double lineCount;
	int indexToRemove;
	String outputFile;
	String outputWeights;
	int numOfFeatures;
	int numOfRecords;
	int iterations;
	ArrayList<String> lfo;
	double learningRate;
	double epsilon;

	public static void main(String[] args) {

		PS4 ps = new PS4();
		ps.trainObjects = new ArrayList<Obj>();
		ps.testObjects = new ArrayList<Obj>();
		ps.trainVals = new ArrayList<ArrayList<Double>>();
		ps.testVals = new ArrayList<ArrayList<Double>>();
		ps.trainMeans = new ArrayList<Double>();
		ps.testMeans = new ArrayList<Double>();
		ps.trainSigmas = new ArrayList<Double>();
		ps.testSigmas = new ArrayList<Double>();
		ps.trainVars = new ArrayList<Double>();
		ps.testVars = new ArrayList<Double>();
		ps.trainZs = new ArrayList<ArrayList<Double>>();
		ps.testZs = new ArrayList<ArrayList<Double>>();
		ps.trainYs = new ArrayList<Double>();
		ps.testYs = new ArrayList<Double>();
		ps.lfo = new ArrayList<String>();
		ps.lineCount = 0.0;

		ps.learningRate = Double.parseDouble(args[3]);
		ps.outputFile = args[0];
		ps.outputWeights = args[2];
		ps.indexToRemove = Integer.parseInt(args[1]);
		ps.fileReader(args[0]);
		ps.epsilon = Double.parseDouble(args[4]);

		ps.process(ps.trainObjects, ps.trainVals, ps.trainMeans, ps.trainSigmas, ps.trainVars, ps.trainZs);
		ps.process(ps.testObjects, ps.testVals, ps.testMeans, ps.testSigmas, ps.testVars, ps.testZs);
		ps.numOfFeatures = ps.trainVals.get(0).size();
		ps.gradientDescent(ps.trainZs, ps.trainYs, ps.learningRate);
		ps.printFinal();
		System.out.println("END PROGRAM");
	}

	public void printFinal() {
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.printf("Number of Input Records: %d \n", numOfRecords);
		System.out.printf("Number of Features: %d \n", numOfFeatures);
		System.out.printf("Iterations: %d \n", iterations);
		System.out.printf("Loss Function output: %s \n", lfo);
	}

	public void gradientDescent(ArrayList<ArrayList<Double>> x, ArrayList<Double> y, Double a) {
		weigh = new Double[x.get(0).size()];
		costs = new ArrayList<Double>();
		lossList = new ArrayList<Double>();
		for (int i = 0; i < weigh.length; i++) {
			weigh[i] = 0.00;
		}
		boolean converged = false;
		double epsilon = this.epsilon;
		int index = 0;
		double loss = L(x, weigh, y);
		double cost = 0.0;
		lossList.add(loss);
		while (!converged) {

			for (int k = 0; k < x.size(); k++) {
				weigh[k] = weigh[k] - a * LDerive(x, weigh, y, k);
			}

			index++;

			double current = L(x, weigh, y);

			lossList.add(current);
			// System.out.println(loss + ", " + current);
			lfo.add(loss + ", " + current);
			cost = (Math.abs(loss - current) * 100.0) / loss;
			loss = current;
			costs.add(cost);
			if (cost < epsilon) {
				converged = true;
				System.out.println("index: " + index + " converged: " + converged);
				for (int i = 0; i < weigh.length; i++) {
					System.out.print(weigh[i] + ", ");
				}
				System.out.println();
				iterations = index;
				try {
					printArrayListToFile(lossList, "losses.txt");
					printArrayListToFile(costs, "costs.txt");
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

		}
	}

	public Double L(ArrayList<ArrayList<Double>> vals, Double[] weights, ArrayList<Double> ys) {
		ArrayList<Double> tempHold = new ArrayList<Double>();
		Double a = 0.0;

		// THIS IS SUM OF (x_j * Beta_j)
		for (int i = 0; i < vals.size(); i++) {
			for (int j = 0; j < vals.get(i).size(); j++) {
				a += (vals.get(i).get(j) * weights[j]);
			}
			tempHold.add(a);
			a = 0.0;
		}

		ArrayList<Double> tempo = new ArrayList<Double>();
		Double theY = 0.0;

		// THIS IS Y - SUM OF (x_j * Beta_j)
		for (int i = 0; i < ys.size(); i++) {
			for (int j = 0; j < tempHold.size(); j++) {
				theY = Math.pow(ys.get(i) - tempHold.get(j), 2);
			}
			tempo.add(theY);
		}

		Double sum = 0.0;

		// THIS IS SUM OF (Y - SUM OF (x_j * Beta_j)
		for (int i = 0; i < tempo.size(); i++) {
			sum += tempo.get(i);
		}

		Double t = (double) (2 * vals.get(0).size());
		Double fin = (1 / t) * sum;
		return fin;
	}

	public Double LDerive(ArrayList<ArrayList<Double>> vals, Double[] weigh, ArrayList<Double> ys, int index) {
		Double a = 0.0;
		for (int j = 0; j < vals.get(index).size(); j++) {
			Double tempY = (Double) ys.get(j);
			a += (tempY - vals.get(index).get(j) * weigh[j]) * -vals.get(index).get(j);
		}
		return a;
	}

	// the process method also calculates the mean of the column:
	public void process(ArrayList<Obj> o, ArrayList<ArrayList<Double>> vals, ArrayList<Double> mus,
			ArrayList<Double> sigs, ArrayList<Double> vars, ArrayList<ArrayList<Double>> zs) {
		int sizeOfMat = o.get(0).xs.size();
		Double total = 0.0;

		// calculating means
		for (int j = 0; j < sizeOfMat; j++) {
			vals.add(new ArrayList<Double>());

			for (int i = 0; i < o.size(); i++) {
				vals.get(j).add(o.get(i).xs.get(j));
				total += vals.get(j).get(i);
			}
			mus.add(total / vals.get(j).size());
			total = 0.0;
		}

		// calculating variance and sigmas
		Double sig = 0.0;
		Double sum = 0.0;
		for (int i = 0; i < vals.size(); i++) {
			for (int j = 0; j < vals.get(i).size(); j++) {
				sig = Math.pow(vals.get(i).get(j) - mus.get(i), 2);
				sum += sig;
			}
			sum = sum / vals.get(i).size();
			vars.add(sum);
			sum = Math.sqrt(sum);
			sigs.add(sum);
			sum = 0.0;
		}

		// calculating z scores
//		System.out.println("vars.size(): " + vars.size() + " here is vars: " + vars.toString());
//		System.out.println("sigs.size(): " + sigs.size() + " here is sigs: " + sigs.toString());
		Double z = 0.0;
		for (int i = 1; i < vals.size(); i++) {
			zs.add(new ArrayList<Double>());
			for (int j = 0; j < vals.get(i).size(); j++) {
				z = vals.get(i).get(j) - mus.get(i);
				z = z / sigs.get(i);
				if (Double.isNaN(z)) {
					z = 0.0;
				}
				zs.get(i - 1).add(z);
			}

		}

	}

	public void fileReader(String filename) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line;
			while ((line = br.readLine()) != null) {
				lineCount++;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("There is a problem reading the file " + filename);
		}
		double fractile1 = 0.8;
		double fractile2 = 0.2;
		fractile1 = Math.round(fractile1 * lineCount);
		fractile2 = Math.round(fractile2 * lineCount);

		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line;
			String[] words;
			int count = 1;
			int count2 = 1;
			while ((line = br.readLine()) != null) {
				words = line.split(",");
				ArrayList<Double> values = new ArrayList<Double>();
				for (int i = 0; i < words.length; i++) {
					values.add(Double.parseDouble(words[i]));
				}
				Obj o = new Obj(values);
				o.xs.add(0, 1.0);
				o.y = values.get(indexToRemove);
				o.locationOfY = indexToRemove;
				// o.xs.remove(values.get(indexToRemove-1));
				if (count <= fractile1) {
					trainObjects.add(o);
					trainYs.add(o.y);
				} else {
					testObjects.add(o);
					testYs.add(o.y);
					count2++;
				}
				count++;
			}
			numOfRecords = count;
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("There is a problem reading the file " + filename);
		}
	}

	public static void printArrayListToFile(ArrayList<Double> arrayList, String filename) throws IOException {
		FileWriter writer = new FileWriter(filename);
		int count = 0;
		for (Double dbl : arrayList) {
			writer.write(count + "," + dbl + System.lineSeparator());
			count++;
		}
		writer.close();
	}

//-------------------------------------------------------------------------------------------------------------------------

	public static void printObjListSize(ArrayList<Obj> list) {
		System.out.println("Object list size: " + list.size());
	}

	public static void printList(ArrayList<ArrayList<Double>> vals) {
		for (int i = 0; i < vals.size(); i++) {
			System.out.println("size of row: " + vals.get(i).size() + ": " + vals.get(i).toString());
			System.out.println("");
		}
	}

	public static void printListWithoutFirstElements(ArrayList<ArrayList<Double>> vals) {
		for (int i = 1; i < vals.size(); i++) {
			System.out.println(vals.get(i).toString());
			System.out.println("");
		}
	}

	public static void printYs(ArrayList<Obj> o) {
		System.out.println("The specified Y's: ");
		for (int i = 0; i < o.size(); i++) {
			System.out.print(o.get(i).y + ", and it's index: " + o.get(i).locationOfY + ".   ");
		}
		System.out.println();
	}

	public static void printObjList(ArrayList<Obj> o) {
		System.out.println("here is list: ");
		for (int a = 0; a < o.size(); a++) {
			System.out.println(o.get(a).xs.toString());

		}
	}

	public static void printIndividualObj(Obj o) {
		for (int i = 0; i < o.xs.size(); i++) {
			System.out.print(o.xs.get(i) + ", ");
		}
	}
}
