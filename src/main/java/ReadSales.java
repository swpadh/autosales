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

public class ReadSales {

	public static Dataset<Row> getDataFrame(final SparkSession spark) {
		
		String path = "src/main/resources/OldSales.csv";	
		final Map dict = createDictionary();
		
		Dataset<Row> rawdf = spark.read().option("header","true").csv(path);
		
		JavaRDD<Row> lstRDD = rawdf.javaRDD().map(new Function<Row, Row>() {
			public Row call(Row row) {
				int sz = row.size();
				List aList = new ArrayList();
				for (int i = 0; i < sz; ++i) {
					if (i == 2 || i == 3) {
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
		return df;
	}
	public static Map createDictionary() {
		Map dict = new HashMap<String, String>();
		dict.put("sports", "001");
		dict.put("Next Gen", "010");
		dict.put("200S", "100");
		dict.put("yellow", "000001");
		dict.put("Green", "000010");
		dict.put("Carbon Black", "000100");
		dict.put("red", "001000");
		dict.put("white", "010000");
		dict.put("Blue", "100000");
		return dict;
	}

	public static void getFeatures(Row row, double[] features) {
		int sz = row.size();
		for (int i = 2, j = 0; i < sz; ++i) {
			Object obj = row.get(i);
			if (obj instanceof Integer)
				features[j++] = ((Integer) (obj)).intValue();
			else if (obj instanceof String)
				features[j++] = Double.valueOf((String) (obj)).doubleValue();
		}
	}
}
