package org.apache.spark.ml.made

import breeze.linalg.{sum, DenseVector => BDV}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, Dataset, functions}
import org.apache.spark.sql.types.{NumericType, StructType}

trait HasInOutColumns extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(inputCol, "features")
  setDefault(outputCol, "label")
}

// via https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/regression/LinearRegression.scala

trait LinearRegressionParams extends HasInOutColumns {
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkNumericType(schema, getOutputCol)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  final val stepSize = new DoubleParam(this, "stepSize", "step")
  final val numIterations = new IntParam(this, "numIterations", "iter")
  def this() = this(Identifiable.randomUID("linearRegression"))

  def getStepSize = $(stepSize)
  def getNumIterations = $(numIterations)

  def setStepSize(step: Double) = set(stepSize, step)
  def setNumIterations(num: Int) = set(numIterations, num)

  private val gradient = new Gradient()
  private val updater = new BasicUpdater()
  private lazy val gradWeights = new GradientDescent(gradient, updater, $(inputCol), $(outputCol))
    .setStepSize(getStepSize)
    .setNumIterations(getNumIterations)

  override def setInputCol(value: String) : this.type = {
    set(inputCol, value)
    gradWeights.setInputCol($(inputCol))
    this
  }

  override def setOutputCol(value: String): this.type = {
    set(outputCol, value)
    gradWeights.setOutputCol($(outputCol))
    this
  }

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val weightInit: Vector = Vectors.dense(1.0, 1.0, 1.0, 1.0)

    val withOnes = dataset.withColumn("ones", functions.lit(1))
    val assembler = new VectorAssembler()
      .setInputCols(Array($(inputCol), "ones"))
      .setOutputCol("features_extended")

    val assembled = assembler
      .transform(withOnes)
      .select(functions.col("features_extended").as($(inputCol)), functions.col($(outputCol)))

    val weights = gradWeights.evaluate(assembled, weightInit)

    copyValues(new LinearRegressionModel(new DenseVector(weights.toArray.slice(0, 3)), weights.toArray(3))).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = ???

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]{
  def apply(s: String) = new LinearRegression(s)
}

class LinearRegressionModel private[made](
                           override val uid: String,
                           val weights: DenseVector,
                           val w0: Double) extends Model[LinearRegressionModel] with LinearRegressionParams {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = ???

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        sum(x.asBreeze *:* weights.asBreeze) + w0
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}



