from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import argparse


def loadData(observations: str, patient: str, conditions: str, target: str) -> None:
    observations_df = spark.read.option("header", "true").csv(observations)
    patients_df = (
        spark.read.option("header", "true")
        .csv(patient)
        .select(col("Id").alias("patientid"), col("BIRTHDATE").alias("dateofbirth"))
    )
    diabetics_df = (
        spark.read.option("header", "true")
        .csv(conditions)
        .filter(col("DESCRIPTION") == "Diabetes")
        .select(col("PATIENT").alias("patientid"), col("START").alias("start"))
    )
    systolic_observations_df = (
        observations_df.select("PATIENT", "DATE", "VALUE")
        .withColumnRenamed("VALUE", "systolic")
        .withColumnRenamed("PATIENT", "patientid")
        .withColumnRenamed("DATE", "dateofobservation")
        .filter((col("CODE") == "8480-6"))
    )
    diastolic_observations_df = (
        observations_df.select("PATIENT", "DATE", "VALUE")
        .withColumnRenamed("VALUE", "diastolic")
        .withColumnRenamed("PATIENT", "patientid")
        .withColumnRenamed("DATE", "dateofobservation")
        .filter((col("code") == "8462-4"))
    )

    hdl_observations_df = (
        observations_df.select("PATIENT", "DATE", "VALUE")
        .withColumnRenamed("VALUE", "hdl")
        .withColumnRenamed("PATIENT", "patientid")
        .withColumnRenamed("DATE", "dateofobservation")
        .filter((col("code") == "2085-9"))
    )

    ldl_observations_df = (
        observations_df.select("PATIENT", "DATE", "VALUE")
        .withColumnRenamed("VALUE", "ldl")
        .withColumnRenamed("PATIENT", "patientid")
        .withColumnRenamed("DATE", "dateofobservation")
        .filter((col("code") == "18262-6"))
    )

    bmi_observations_df = (
        observations_df.select("PATIENT", "DATE", "VALUE")
        .withColumnRenamed("VALUE", "bmi")
        .withColumnRenamed("PATIENT", "patientid")
        .withColumnRenamed("DATE", "dateofobservation")
        .filter((col("code") == "39156-5"))
    )
    merged_observations_df = (
        systolic_observations_df.join(
            diastolic_observations_df, ["patientid", "dateofobservation"]
        )
        .join(hdl_observations_df, ["patientid", "dateofobservation"])
        .join(ldl_observations_df, ["patientid", "dateofobservation"])
        .join(bmi_observations_df, ["patientid", "dateofobservation"])
    )
    merged_observations_with_age_df = (
        merged_observations_df.join(patients_df, "patientid")
        .withColumn("age", datediff(col("dateofobservation"), col("dateofbirth")) / 365)
        .drop("dateofbirth")
    )
    observations_and_condition_df = merged_observations_with_age_df.join(
        diabetics_df, "patientid", "left_outer"
    ).withColumn("diabetic", when(col("start").isNotNull(), 1).otherwise(0))
    observations_and_condition_df = observations_and_condition_df.filter(
        (col("diabetic") == 0) | ((col("dateofobservation") >= col("start")))
    )
    w = Window.partitionBy(observations_and_condition_df["patientid"]).orderBy(
        merged_observations_df["dateofobservation"].asc()
    )

    first_observation_df = (
        observations_and_condition_df.withColumn("rn", row_number().over(w))
        .where(col("rn") == 1)
        .drop("rn")
    )
    first_observation_df.coalesce(1).write.parquet(target)


parser = argparse.ArgumentParser(description="Data Loader")
parser.add_argument(
    "--oberservation", "-o", required=False, help="Observation file location"
)
parser.add_argument("--patient", "-p", required=False, help="Patient file location")
parser.add_argument("--condition", "-c", required=False, help="Condition file location")
parser.add_argument("--target", "-t", required=False, help="Target Location")
args = parser.parse_args()


if __name__ == "__main__":
    oberservation = args.oberservation
    patient = args.patient
    condition = args.condition
    target = args.target
    spark = SparkSession.builder.appName("data loader").getOrCreate()
    loadData(oberservation, patient, condition, target)
    spark.stop()
