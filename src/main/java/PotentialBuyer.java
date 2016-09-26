import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;

public class PotentialBuyer {

	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder().master("local[*]").appName("PotentialBuyer").config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse").getOrCreate();
		spark.sparkContext().setLogLevel("ERROR");

		Dataset<Row> oldLeads = ReadLeads.getDataFrame(spark, "OldLeads.csv");
		Dataset<Row> newLeads = ReadLeads.getDataFrame(spark, "NewLeads.csv");
		Dataset<Row> oldSales = ReadSales.getDataFrame(spark);

		Dataset<Row> dfSuccess = oldLeads.join(oldSales, "Customer ID");
		//System.out.println("success instance count = " + dfSuccess.count());

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
		//System.out.println("failure instance count = " + dfFailure.count());
		//System.out.println("total instance count = " + oldLeads.count());

		JavaRDD<LabeledPoint> labeledSuccessData = dfSuccess1.javaRDD().map(
				new Function<Row, LabeledPoint>() {

					public LabeledPoint call(Row row) throws Exception {

						int sz = row.size();
						double[] features = new double[sz - 3];
						getFeatures(row, features);
						Vector featureVector = Vectors.dense(features);
						return new LabeledPoint(1.0, featureVector);
					}
				});
		JavaRDD<LabeledPoint> labeledFailureData = dfFailure.javaRDD().map(
				new Function<Row, LabeledPoint>() {

					public LabeledPoint call(Row row) throws Exception {

						int sz = row.size();
						double[] features = new double[sz - 3];
						getFeatures(row, features);
						Vector featureVector = Vectors.dense(features);
						return new LabeledPoint(0.0, featureVector);
					}
				});
		JavaRDD<LabeledPoint> labeledData = labeledSuccessData
				.union(labeledFailureData);

		JavaRDD<LabeledPoint> labeledTestData = newLeads.javaRDD().map(
				new Function<Row, LabeledPoint>() {

					public LabeledPoint call(Row row) throws Exception {

						int sz = row.size();
						double[] features = new double[sz - 3];
						getFeatures(row, features);
						Vector featureVector = Vectors.dense(features);
						return new LabeledPoint(0.0, featureVector);
					}
				});

		predictGBT(labeledData,spark);

		spark.stop();
	}
	private static void getFeatures(Row row, double[] features) {
		int sz = row.size();
		for (int i = 3, j = 0; i < sz; ++i) {
			Object obj = row.get(i);
			if (obj instanceof Integer)
				features[j++] = ((Integer) (obj)).intValue();
			else if (obj instanceof String)
				features[j++] = Double.valueOf((String) (obj)).doubleValue();
		}
	}

	static void predictGBT(JavaRDD<LabeledPoint> labeledData, SparkSession spark) {

		double[] weights = { 0.8, 0.2 };

		JavaRDD<LabeledPoint>[] labeledDataArray = labeledData
				.randomSplit(weights);
		JavaRDD<LabeledPoint> trainData = labeledDataArray[0];
		JavaRDD<LabeledPoint> testData = labeledDataArray[1];
		trainData.cache();
		testData.cache();

		BoostingStrategy boostingStrategy = BoostingStrategy
				.defaultParams("Classification");
		boostingStrategy.setNumIterations(10);
		boostingStrategy.getTreeStrategy().setNumClasses(2);
		boostingStrategy.getTreeStrategy().setMaxDepth(5);
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(
				categoricalFeaturesInfo);

		final GradientBoostedTreesModel gbtModel = GradientBoostedTrees.train(
				trainData, boostingStrategy);

		JavaRDD<Tuple2<Object, Object>> scoreAndLabels = testData
				.map(new Function<LabeledPoint, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(LabeledPoint p) {
						Double score = gbtModel.predict(p.features());
						return new Tuple2<Object, Object>(score, p.label());
					}
				});

		BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(
				JavaRDD.toRDD(scoreAndLabels));
		printMetrics(metrics);
	}

	public static void printMetrics(BinaryClassificationMetrics metrics) {
		// Precision by threshold
		JavaRDD<Tuple2<Object, Object>> precision = metrics
				.precisionByThreshold().toJavaRDD();
		System.out.println("Precision by threshold: " + precision.collect());

		// Recall by threshold
		JavaRDD<Tuple2<Object, Object>> recall = metrics.recallByThreshold()
				.toJavaRDD();
		System.out.println("Recall by threshold: " + recall.collect());

		// F Score by threshold
		JavaRDD<Tuple2<Object, Object>> f1Score = metrics.fMeasureByThreshold()
				.toJavaRDD();
		System.out.println("F1 Score by threshold: " + f1Score.collect());

		JavaRDD<Tuple2<Object, Object>> f2Score = metrics.fMeasureByThreshold(
				2.0).toJavaRDD();
		System.out.println("F2 Score by threshold: " + f2Score.collect());

		// Precision-recall curve
		JavaRDD<Tuple2<Object, Object>> prc = metrics.pr().toJavaRDD();
		System.out.println("Precision-recall curve: " + prc.collect());

		// Thresholds
		JavaRDD<Double> thresholds = precision
				.map(new Function<Tuple2<Object, Object>, Double>() {
					public Double call(Tuple2<Object, Object> t) {
						return new Double(t._1().toString());
					}
				});

		// ROC Curve
		JavaRDD<Tuple2<Object, Object>> roc = metrics.roc().toJavaRDD();
		System.out.println("ROC curve: " + prc.collect());

		// AUPRC
		System.out.println("Area under precision-recall curve = "
				+ metrics.areaUnderPR());

		// AUROC
		System.out.println("Area under ROC = " + metrics.areaUnderROC());
	}
}
