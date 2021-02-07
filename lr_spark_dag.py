"""Example DAG demonstrating the usage of the BashOperator."""

from datetime import timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator

from typing import List

from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline


def lr_spark_train(input_csv_file_path: str,
                   output_model_folder_path: str,
                   label_column: str,
                   cat_features: List[str]):

    print("Creating Spark session...")
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .appName("Spark Linear Regression") \
        .getOrCreate()

    print("Reading input data...")
    df_raw = (spark.read
              .format("csv")
              .option("delimiter", ",")
              .option("inferSchema", "true")
              .option("header", "true")
              .load(input_csv_file_path))

    # Clean NaNs
    df_column_names_cleaned = df_raw
    for sf in df_column_names_cleaned.schema.fields:
        df_column_names_cleaned = (
            df_column_names_cleaned.withColumnRenamed(sf.name,
                                                      sf.name.strip().lower()
                                                      .replace(' ', '_').replace('/', '_').replace('-', '_')))

    # Separate features from the label

    # features without label
    list_columns = [sf.name for sf in df_column_names_cleaned.schema.fields if sf.name != label_column]

    # Prepare Feature Engineering steps
    # Delete rows where label is missing
    df_nulls_removed = df_column_names_cleaned.where("{} is not null".format(label_column))

    # Split data into train and test sets
    splits = df_nulls_removed.randomSplit([0.8, 0.2])
    train_df = splits[0]
    test_df = splits[1]

    # Prepare column names for the preprocessing steps
    cat_features_ind = ["{}_ind".format(cf) for cf in cat_features]
    cat_features_vec = ["{}_vec".format(cf) for cf in cat_features]
    # Note: following for comprehension has time complexity of the product of lengths of lists
    list_columns_no_cat = [lc for lc in list_columns if lc not in cat_features]
    final_list_columns = list_columns_no_cat + cat_features_vec

    # set column names as variables
    features_unscaled_col = "features_unscaled"
    features_col = "features"

    # initialize preprocessing steps and the model
    indexers = [StringIndexer(inputCol=cat_features[i], outputCol=cat_features_ind[i]) for i in
                range(len(cat_features))]
    encoder = OneHotEncoder(inputCols=cat_features_ind, outputCols=cat_features_vec)
    assembler = VectorAssembler(inputCols=final_list_columns, outputCol=features_unscaled_col).setHandleInvalid("skip")
    scaler = MinMaxScaler(inputCol=features_unscaled_col, outputCol=features_col)
    lr = LinearRegression(featuresCol=features_col, labelCol=label_column)

    # collect stages and create pipeline
    stages = indexers + [encoder, assembler, scaler, lr]
    pipeline = Pipeline(stages=stages)

    print("Fitting the pipeline...")
    model = pipeline.fit(train_df)

    # assess performance of the pipeline
    # tbd

    print("Pipeline has been trained!")
    lr_model = model.stages[-1]
    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("R2: %f" % trainingSummary.r2)

    print("Saving the model...")
    model.save(output_model_folder_path)

    print("Model saved successfully!")


args = {
    'owner': 'airflow',
}

dag = DAG(
    dag_id='lr_spark',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=60)
)


lr_spark_training = PythonOperator(
    task_id='lr_spark_training',
    python_callable=lr_spark_train,
    op_kwargs={
        'input_csv_file_path': 'lr_input.csv',
        'output_model_folder_path': 'lr_output',
        'label_column': 'life_expectancy',
        'cat_features': ['country', 'year', 'status']
    },
    dag=dag
)


if __name__ == "__main__":
    dag.cli()

