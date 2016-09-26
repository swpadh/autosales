import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class LeadCluster {
	public static KMeansModel kmeans(JavaRDD<Row> labeledLeadData, SparkSession spark) {

		double[] weights = { 0.8, 0.2 };

		JavaRDD<Row>[] rowDataArray = labeledLeadData.randomSplit(weights);
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
		KMeans kmeans = new KMeans().setK(7).setSeed(1L).setFeaturesCol("features")
				  .setPredictionCol("prediction");
		KMeansModel model = kmeans.fit(trainDataDf);

		// Evaluate clustering by computing Within Set Sum of Squared Errors.
		double WSSSE = model.computeCost(trainDataDf);
		System.out.println("Within Set Sum of Squared Errors with Lead Data = " + WSSSE);
		
		System.out.println(model.explainParams());
		
		// Save the model
		//try {
		//	model.save("target/KMeansLeadModel");
		//} catch (Exception ex) {
		//	System.out
		//			.println("The KMeansLeadModel can not be saved ----------");
		//	ex.printStackTrace();
		//}

		// Shows the result.
		Vector[] centers = model.clusterCenters();
		System.out.println("Lead Cluster Centers: ");
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
						List aList = new ArrayList();
						Vector v = (Vector) row.get(0);
						for (int j = 0; j < v.size(); ++j) {
							if (j == 0 || j == 12 || j == 13 || j == 15
									|| j == 16) {
								Object f = dict.get(new Tuple2(j, v.apply(j)));
								aList.add(f);
							} else {
								aList.add(v.apply(j));
							}
						}

						//System.out.println(aList + " -> " + row.get(1));
					}
				});
		return model;

	}
	public static List<Tuple2<String, String>> geClusterMap(KMeansModel leadsModel, Dataset<Row> newLeads)
	{
		List<Tuple2<String, String>> userClusters = newLeads.javaRDD()
				.map(new Function<Row, Tuple2<String, String>>() {

					public Tuple2<String, String> call(Row row)
							throws Exception {

						int sz = row.size();
						double[] features = new double[sz - 3];
						ReadLeads.getFeatures(row, features);
						Vector featureVector = Vectors.dense(features);
						return new Tuple2(String.valueOf(row.get(0)), String.valueOf(leadsModel
								.predict(featureVector)));
					}
				}).collect();
		return userClusters;
	}
	public static Map createDictionary() {
		Map dict = new HashMap<Tuple2<Integer, Integer>, String>();
		dict.put(new Tuple2(0, 1), "No");
		dict.put(new Tuple2(0, 10), "yes");
		dict.put(new Tuple2(12, 1), "Male");
		dict.put(new Tuple2(12, 10), "Female");
		dict.put(new Tuple2(13, 1), "Single");
		dict.put(new Tuple2(13, 10), "Married");
		dict.put(new Tuple2(13, 100), "Divorced");
		dict.put(new Tuple2(15, 1), "intern");
		dict.put(new Tuple2(15, 10), "Entry Level");
		dict.put(new Tuple2(15, 100), "Manager");
		dict.put(new Tuple2(15, 1000), "Director");
		dict.put(new Tuple2(15, 10000), "Sr. Manager");
		dict.put(new Tuple2(16, 0), "Less than 25,000");
		dict.put(new Tuple2(16, 1), "25,000 to 34,999");
		dict.put(new Tuple2(16, 10), "35,000 to 49,999");
		dict.put(new Tuple2(16, 100), "50,000 to 74,999");
		dict.put(new Tuple2(16, 1000), "75,000 to 99,999");
		dict.put(new Tuple2(16, 10000), "Greater than 100000");
		return dict;
	}
}
