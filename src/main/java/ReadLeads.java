import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;

public class ReadLeads {
	public static Dataset<Row> getDataFrame(final SparkSession spark,
			String fileName) {
		
		String base = "src/main/resources/";
		final Map dict = createDictionary();
		Dataset<Row> rawdf = spark.read().option("header","true").csv(base + fileName);

		JavaRDD<Row> lstRDD = rawdf.javaRDD().map(new Function<Row, Row>() {
			public Row call(Row row) {
				int sz = row.size();
				List aList = new ArrayList();
				for (int i = 0; i < sz; ++i) {
					if (i == 3 || i == 15 || i == 16 || i == 18 || i == 19) {
						Object s = dict.get(((String) row.get(i)).trim());
						if (s != null)
							aList.add(s);
						else
							aList.add("0");
					} else {
						Object cell = row.get(i);
						if (cell instanceof Integer) {
							aList.add(((Integer) cell).intValue());
						} else if (cell instanceof String && cell.equals("NA")) {
							aList.add("0");
						} else
							aList.add(cell);

					}
				}
				return RowFactory.create(aList.toArray());
			}
		});
		Dataset<Row> df = spark.createDataFrame(lstRDD, rawdf.schema());

		// df.select("Customer ID", "Job Level", "Gender",
		// "Relationship Status","Job Level", "Family Income(Pounds)").show();
		return df;
	}

	public static Map createDictionary() {
		Map dict = new HashMap<String, String>();
		dict.put("No", "01");
		dict.put("yes", "10");
		dict.put("Male", "01");
		dict.put("Female", "10");
		dict.put("Single", "001");
		dict.put("Married", "010");
		dict.put("Divorced", "100");
		dict.put("intern", "00001");
		dict.put("Entry Level", "00010");
		dict.put("Manager", "00100");
		dict.put("Director", "01000");
		dict.put("Sr. Manager", "10000");
		dict.put("Less than 25,000", "00000");
		dict.put("25,000 to 34,999", "00001");
		dict.put("35,000 to 49,999", "00010");
		dict.put("50,000 to 74,999", "00100");
		dict.put("75,000 to 99,999", "01000");
		dict.put("Greater than 100000", "10000");
		return dict;
	}
	public static void getFeatures(Row row, double[] features) {
		int sz = row.size();
		for (int i = 3, j = 0; i < sz; ++i) {
			Object obj = row.get(i);
			if (obj instanceof Integer)
				features[j++] = ((Integer) (obj)).intValue();
			else if (obj instanceof String)
				features[j++] = Double.valueOf((String) (obj)).doubleValue();
		}
	}
}
