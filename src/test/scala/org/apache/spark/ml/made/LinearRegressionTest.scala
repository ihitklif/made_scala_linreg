package org.apache.spark.ml.made

import breeze.linalg.DenseVector
import breeze.numerics.round
import com.google.common.io.Files
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row, functions}
import org.scalatest._
import org.scalatest.flatspec._
import org.scalatest.matchers._


object TestData extends WithSpark {
  lazy val _vectors = Seq(
    Vectors.dense(13.5, 12, 7.0),
    Vectors.dense(-1, 0, 3.2),
  )
  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }

  lazy val _rand_vals = Seq.fill(10000)(Vectors.fromBreeze(DenseVector.rand(3)))
  lazy val _rand_data: DataFrame = {
    import sqlc.implicits._
    _rand_vals.map(x => Tuple1(x)).toDF("features")
  }

}

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  val weights_inaccuracy = 0.1

  lazy val data: DataFrame = TestData._data
  lazy val vectors: Seq[Vector] = TestData._vectors

  lazy val rand_data: DataFrame = TestData._rand_data
  lazy val rand_points: Seq[Vector] = TestData._rand_vals

  "Model" should "calculate and predict" in {
    val weights: Vector = Vectors.dense(1.5, 0.3, -0.7)
    val bias: Double = 4

    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setInputCol("features")
      .setOutputCol("prediction")

    val values = model.transform(data).collect().map(_.getAs[Double](1))

    values.length should be(2)

    values(0) should be(vectors(0)(0) * weights(0) + vectors(0)(1) * weights(1) + vectors(0)(2) * weights(2) + bias +- delta)
    values(1) should be(vectors(1)(0) * weights(0) + vectors(1)(1) * weights(1) + vectors(1)(2) * weights(2) + bias +- delta)
  }

  // hidden model (1.5, 0.3, -0.7)
  "Estimator" should "produce functional model" in {
    val weights: Vector = Vectors.dense(1.5, 0.3, -0.7)
    val bias: Double = 0.8
    val true_model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setInputCol("features")
      .setOutputCol("label")

    val train = true_model
      .transform(rand_data)
      .select(functions.col("features"),
              (functions.col("label") + functions.rand() * functions.lit(0.1) - functions.lit(0.05) // add noise
                ).as("label"))

    val estimator: LinearRegression = new LinearRegression()
      .setStepSize(1)
      .setNumIterations(500)
      .setInputCol("features")
      .setOutputCol("label")
    val model = estimator.fit(train)

    model.weights(0) should be(weights(0) +- weights_inaccuracy)
    model.weights(1) should be(weights(1) +- weights_inaccuracy)
    model.weights(2) should be(weights(2) +- weights_inaccuracy)
    model.w0 should be(bias +- weights_inaccuracy)
  }

  "Estimator" should "work after re-read" in {
    val weights: Vector = Vectors.dense(1.5, 0.3, -0.7)
    val bias: Double = 0.8
    val true_model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setInputCol("features")
      .setOutputCol("label")

    val train = true_model
      .transform(rand_data)
      .select(functions.col("features"), (functions.col("label") + functions.rand() * functions.lit(0.1) - functions.lit(0.05)).as("label"))

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setStepSize(1)
        .setNumIterations(500)
        .setInputCol("features")
        .setOutputCol("label")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(train).stages(0).asInstanceOf[LinearRegressionModel]

    model.weights(0) should be(weights(0) +- weights_inaccuracy)
    model.weights(1) should be(weights(1) +- weights_inaccuracy)
    model.weights(2) should be(weights(2) +- weights_inaccuracy)
    model.w0 should be(bias +- weights_inaccuracy)
  }
}
