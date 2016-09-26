import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class SalesCluster {
	static final Map<String, Set<List<String>>> clusterMap = new HashMap<String, Set<List<String>>>();
 
	private static void getFeatures(Row row, double[] features) {
		int sz = row.size();
		for (int i = 2, j = 0; i < sz; ++i) {
			Object obj = row.get(i);
			if (obj instanceof Integer)
				features[j++] = ((Integer) (obj)).intValue();
			else if (obj instanceof String)
				features[j++] = Double.valueOf((String) (obj)).doubleValue();
		}
	}

	public static KMeansModel kmeans(JavaRDD<Row> salesData,
			SparkSession spark) {

		double[] weights = { 0.8, 0.2 };

		JavaRDD<Row>[] rowDataArray = salesData.randomSplit(weights);
		JavaRDD<Row> trainData = rowDataArray[0];
		JavaRDD<Row> testData = rowDataArray[1];
		trainData.cache();
		testData.cache();

		StructField[] fields = { new StructField("features", new VectorUDT(),
				false, Metadata.empty()) };
		StructType schema = new StructType(fields);

		Dataset<Row> trainDataDf = spark.createDataFrame(trainData, schema);
		//trainDataDf.show();

		// Trains a k-means model.
		KMeans kmeans = new KMeans().setK(7);
		KMeansModel model = kmeans.fit(trainDataDf);
		
		// Evaluate clustering by computing Within Set Sum of Squared Errors.
		double WSSSE = model.computeCost(trainDataDf);
		System.out.println("Within Set Sum of Squared Errors with Sales Cluster = " + WSSSE);
		System.out.println(model.explainParams());
		// Save the model
		//try {
		//	model.save("target/KMeansSalesModel");
		//} catch (Exception ex) {
		//	System.out
		//			.println("The KMeansSalesModel can not be saved ----------");
		//	ex.printStackTrace();
		//}
		// Shows the result.
		Vector[] centers = model.clusterCenters();
		System.out.println("Sales Cluster Centers: ");
		for (Vector center : centers) {
			System.out.println(center);
		}

		Dataset<Row> testDataDf = spark.createDataFrame(testData, schema);
		Dataset<Row> predictions = model.transform(testDataDf);
		Map<Tuple2<Integer, Integer>, String> dict = createDictionary();

		predictions.select("features", "prediction").javaRDD()
				.foreach(new VoidFunction<Row>() {

					@Override
					public void call(Row row) throws Exception {
						List<String> aList = new ArrayList<String>();
						Vector v = (Vector) row.get(0);
						for (int j = 0; j < v.size(); ++j) {
							if (j == 0 || j == 1) {
								Object f = dict.get(new Tuple2(j, v.apply(j)));
								aList.add((String)f);
							} else {
								aList.add(String.valueOf(v.apply(j)));
							}
						}
						if (clusterMap.containsKey(row.get(1)) == false) {
							Set<List<String>> aSet = new HashSet<List<String>>();
							aSet.add(aList);
							clusterMap.put(String.valueOf(row.get(1)), aSet);
						} else {
							clusterMap.get(row.get(1)).add(aList);
						}
					}
				});
		return model;

	}
	public static Map createDictionary() {
		Map dict = new HashMap<Tuple2<Integer, Integer>, String>();
		dict.put(new Tuple2(0, 1), "sports");
		dict.put(new Tuple2(0, 10), "Next Gen");
		dict.put(new Tuple2(0, 100), "200S");
		dict.put(new Tuple2(1, 1), "yellow");
		dict.put(new Tuple2(1, 10), "Green");
		dict.put(new Tuple2(1, 100), "Carbon Black");
		dict.put(new Tuple2(1, 1000), "red");
		dict.put(new Tuple2(1, 10000), "white");
		dict.put(new Tuple2(1, 100000), "Blue");
		return dict;
	}

}
