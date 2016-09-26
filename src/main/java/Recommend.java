import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;

public class Recommend {

	public static void main(String[] args) {
		SparkSession spark = SparkSession
				.builder()
				.master("local[*]")
				.appName("Automitive360")
				.config("spark.sql.warehouse.dir",
						"file:///tmp/spark-warehouse").getOrCreate();

		spark.sparkContext().setLogLevel("ERROR");

		Dataset<Row> oldSales = ReadSales.getDataFrame(spark);

		JavaRDD<Row> salesFeatures = oldSales.javaRDD().map(
				new Function<Row, Row>() {

					public Row call(Row row) throws Exception {

						int sz = row.size();
						double[] features = new double[sz - 2];
						ReadSales.getFeatures(row, features);
						Vector featureVector = Vectors.dense(features);
						return RowFactory.create(featureVector);
					}
				});

		KMeansModel salesModel = SalesCluster.kmeans(salesFeatures,
				spark);

		
		Dataset<Row> oldLeads = ReadLeads.getDataFrame(spark, "OldLeads.csv");
		Dataset<Row> dfSuccess = oldLeads.join(oldSales, "Customer ID");
		// System.out.println("success instance count = " + dfSuccess.count());

		Dataset<Row> dfSuccess1 = dfSuccess.select("Customer ID", "First Name",
				"Middle Name", "Campaign Response",
				"Time Spent on  Company Website",
				"No of Visits Made to Company Website", "Sports- Interest",
				"Movies & Entertainment - Interest", "Technology-Interest",
				"Finance-Interest", "Politics-Interest", "Travel-Interest",
				"Business- Interest", "International-Interest", "Age",
				"Gender", "Relationship Status", "Family Size", "Job Level",
				"Family Income(Pounds)", "No of Vehicles Owned");
		// dfSuccess1.show();
		Dataset<Row> dfFailure = oldLeads.except(dfSuccess1);
		// dfFailure.show();
		// System.out.println("failure instance count = " + dfFailure.count());
		// System.out.println("total instance count = " + oldLeads.count());

		JavaRDD<Row> leadFeatures = dfSuccess1.javaRDD().map(
				new Function<Row, Row>() {

					public Row call(Row row) throws Exception {

						int sz = row.size();
						double[] features = new double[sz - 3];
						ReadLeads.getFeatures(row, features);
						Vector featureVector = Vectors.dense(features);
						return RowFactory.create(featureVector);
					}
				});

		KMeansModel leadsModel = LeadCluster.kmeans(leadFeatures,
				spark);
		Map<String, Set<String>> userSalesMap = prepareMatrix(oldSales,
				salesModel, dfSuccess1, leadsModel);
		Dataset<Row> newLeads = ReadLeads.getDataFrame(spark, "NewLeads.csv");

		Map<String, Set<List<String>>> recommendationMap = recommend(leadsModel,
				newLeads, userSalesMap);
		
		System.out.println("Recomendation Map ---------------------");
		recommendationMap.forEach(new BiConsumer<String, Set<List<String>>>() {

			@Override
			public void accept(String t, Set<List<String>> u) {
				System.out.println(t + " -> " + u);
			}
		});
    
		spark.stop();
	}

	public static void printMatrix(Map<String, Set<String>> userSalesMap,List<Tuple2<String, String>>  userClusters,Map<String, Set<List<String>>> salesClusterMap )
	{
		System.out.println("User Sales Map -----------------------------------");
		userSalesMap.forEach(new BiConsumer<String, Set<String>>() {

			@Override
			public void accept(String t, Set<String> u) {
				System.out.println(t + " -> " + u);
			}
		});
		System.out.println("User Cluster Map -----------------------------------");
		userClusters.forEach(new Consumer<Tuple2<String, String>>(){

			@Override
			public void accept(Tuple2<String, String> t) {
				System.out.println(t._1 + " -> " + t._2);
				
			}});
		System.out.println("Sales Cluster Map --------------------------------------");
		salesClusterMap.forEach(new BiConsumer<String, Set<List<String>>>() {

			@Override
			public void accept(String t, Set<List<String>> u) {
				System.out.println(t + " -> " + u);
			}
		});
		
	}
	public static Map<String, Set<List<String>>> recommend(
			KMeansModel leadsModel, Dataset<Row> newLeads,
			Map<String, Set<String>> userSalesMap) {
		// KMeansModel salesModel = KMeansModel.load("target/KMeansSalesModel");
		// KMeansModel leadModel = KMeansModel.load("target/KMeansLeadModel");
	
		List<Tuple2<String, String>>  userClusters = LeadCluster.geClusterMap(leadsModel, newLeads);
		
		Map<String, Set<List<String>>> salesClusterMap = SalesCluster.clusterMap;
	
		//printMatrix(userSalesMap,userClusters,salesClusterMap );
		
		Map<String, Set<List<String>>> recomendationMap = new HashMap<String, Set<List<String>>>();
		for (Tuple2<String, String> ucluster : userClusters) {

			Set salesCluster = userSalesMap.get(ucluster._2);
			
			if (salesCluster != null) {

				salesCluster.forEach(new Consumer<String>() {

					@Override
					public void accept(String t) {
						Set<List<String>> recList = salesClusterMap.get(t);

						if (recomendationMap.containsKey(ucluster._1) == false) {
							recomendationMap.put(ucluster._1, recList);
						} else {

							Set<List<String>> itemLists = recomendationMap
									.get(ucluster._1);
							if (recList != null && itemLists != null) {
								Iterator<List<String>> iter = recList.iterator();
								while (iter.hasNext())
									itemLists.add(iter.next());
							}

						}
					}
				});
			}
		}

		return recomendationMap;
	}

	public static Map<String, Set<String>> prepareMatrix(Dataset<Row> dfSales,
			KMeansModel salesModel, Dataset<Row> dfUser,
			KMeansModel leadsModel) {
		// KMeansModel salesModel = KMeansModel.load("target/KMeansSalesModel");
		// KMeansModel leadModel = KMeansModel.load("target/KMeansLeadModel");

		List<Tuple2<String, String>> userClusters = dfUser.javaRDD()
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

		Map<String, String> userMap = new HashMap<String, String>();
		for (Tuple2<String, String> aPair : userClusters) {
			userMap.put(aPair._1,aPair._2);
		}
		// find sales cluster for which this user cluster belongs
		List<Tuple2<String, String>> salesClusters = dfSales.javaRDD()
				.map(new Function<Row, Tuple2<String, String>>() {

					public Tuple2<String, String> call(Row row)
							throws Exception {

						int sz = row.size();
						double[] features = new double[sz - 2];
						ReadSales.getFeatures(row, features);
						Vector featureVector = Vectors.dense(features);
						return new Tuple2(String.valueOf(row.get(0)), String.valueOf(salesModel
								.predict(featureVector)));
					}
				}).collect();
		Map<String, String> salesMap = new HashMap<String, String>();
		for (Tuple2<String, String> aPair : salesClusters) {
			salesMap.put(aPair._1, aPair._2);
		}

		Map<String, Set<String>> userSalesMap = new HashMap<String, Set<String>>();
		Set<String> userIdSet = userMap.keySet();
		Iterator<String> itr = userIdSet.iterator();
		while (itr.hasNext()) {
			String uid = itr.next();
			if (userSalesMap.containsKey(userMap.get(uid)) == false) {
				Set<String> aSet = new HashSet<String>();
				aSet.add(salesMap.get(uid));
				userSalesMap.put(userMap.get(uid), aSet);
			} else {
				userSalesMap.get(userMap.get(uid)).add(salesMap.get(uid));
			}
		}
		return userSalesMap;
	}
}
